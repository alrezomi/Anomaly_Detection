"""Causal GMR bag replay, calibrated with the same sampling on nominal runs.

Only nominal training may inspect complete recordings. Test observations are
held at a 10 Hz clock, without interpolation, future alignment or centered
smoothing. Stage 3 is the end marker, leaving two active task intervals.
"""
from collections import deque
import hashlib
from itertools import groupby
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from .anomaly_detection import GMRAnomalyDetector
from .threshold import build_phase_threshold_from_scores

SAMPLE_SEC = .1
PROTOCOL = 'causal_stage_1_to_3_gmr_v1'


class StageClock:
    def __init__(self):
        self.boundaries = {}
        self.stage = 0
        self.failures = []

    def observe(self, timestamp, label):
        from build_dataset_manifest import FAIL_PATTERN
        if label == '1' and self.stage == 0:
            self.stage = 1
            self.boundaries[1] = timestamp
        elif label == '2' and self.stage == 1 and timestamp > self.boundaries[1]:
            self.stage = 2
            self.boundaries[2] = timestamp
        elif label == '3' and self.stage in (1, 2) and timestamp > self.boundaries[1]:
            self.stage = 3
            self.boundaries[3] = timestamp
        elif self.stage in (1, 2) and FAIL_PATTERN.search(label):
            self.failures.append({'time': timestamp, 'label': label})

    def durations(self):
        if set(self.boundaries) != {1, 2, 3}:
            raise ValueError('Nominal GMR training requires stage 1, then 2, then 3 (end).')
        durations = [self.boundaries[2] - self.boundaries[1], self.boundaries[3] - self.boundaries[2]]
        if min(durations) <= 0:
            raise ValueError('Nominal GMR stage intervals must have positive duration.')
        return durations


def estimate_phase(stage, elapsed, durations):
    # The current test stage's end is unknown. Never use its eventual duration.
    return (stage - 1 + np.clip(np.asarray(elapsed) / durations[stage - 1], 0., 1.)) / 2.


def signal_values(kind, payload):
    if kind == 'pose':
        quaternion = np.array([payload[f'orientation_{axis}'] for axis in 'xyzw'])
        if not np.isfinite(quaternion).all() or np.linalg.norm(quaternion) < 1e-8:
            raise ValueError('Invalid TCP orientation quaternion.')
        rotation = Rotation.from_quat(quaternion).as_matrix()[:, :2].ravel()
        result = {f'position_{axis}': payload[f'position_{axis}'] for axis in 'xyz'}
        result.update({f'rotation_{i}': float(value) for i, value in enumerate(rotation)})
    elif kind == 'joints':
        result = {key: value for key, value in payload.items() if key.endswith('_position')}
    else:
        result = dict(payload)
    if not result or not np.isfinite(list(result.values())).all():
        raise ValueError(f'Missing or nonfinite {kind} values; cannot score this observation.')
    return result


class CausalSamples:
    """Merge arrived values at clock ticks. Error markers never advance phase."""
    def __init__(self, kinds, stop=None):
        self.kinds = set(kinds)
        self.stop = stop
        self.clock = StageClock()
        self.pose_frame = None
        self.columns = None

    def packets(self, events):
        latest, stamps = {}, {}
        next_tick = None
        previous = -1.

        def sample(timestamp):
            if set(latest) != self.kinds:
                return None
            # Gripper feedback can be sparse while stationary; hold its encoder.
            if any(timestamp - stamps[kind] > 1. for kind in self.kinds - {'gripper'}):
                raise ValueError('Selected TS signal is more than 1 s stale; replay stopped.')
            values = {key: value for kind in sorted(latest) for key, value in latest[kind].items()}
            columns = sorted(values)
            if self.columns is not None and self.columns != columns:
                raise ValueError('TS channel names changed during execution.')
            self.columns = columns
            stage = self.clock.stage
            return dict(time=timestamp, stage=stage,
                        stage_elapsed=timestamp - self.clock.boundaries[stage], **values)

        try:
            for timestamp, group in groupby(events, key=lambda event: event[0]):
                if self.stop is not None and self.stop.is_set():
                    return
                if not np.isfinite(timestamp) or timestamp < previous or timestamp < 0:
                    raise ValueError('TS replay requires increasing, finite bag timestamps.')
                previous = timestamp
                # Only values already received can contribute to these ticks.
                while next_tick is not None and next_tick < timestamp - 1e-9:
                    if self.stop is not None and self.stop.is_set():
                        return
                    row = sample(next_tick)
                    if row is not None:
                        yield next_tick, 'sample', row
                    next_tick += SAMPLE_SEC
                messages = list(group)
                # Markers win over signals at the exact same timestamp.
                for _, kind, payload in messages:
                    if kind != 'stage':
                        continue
                    old_stage = self.clock.stage
                    self.clock.observe(timestamp, payload)
                    if self.clock.stage != old_stage:
                        yield timestamp, 'stage', self.clock.stage
                        if self.clock.stage == 3:
                            return
                        if old_stage == 0:
                            next_tick = timestamp
                            latest.clear()
                            stamps.clear()
                for _, kind, payload in messages:
                    if kind == 'camera':
                        yield timestamp, 'camera', payload
                    elif kind in self.kinds and self.clock.stage in (1, 2):
                        if kind == 'pose':
                            frame = payload.get('pose_frame', '')
                            if self.pose_frame is not None and frame != self.pose_frame:
                                raise ValueError('TCP pose reference frame changed during execution.')
                            self.pose_frame = frame
                        latest[kind] = signal_values(kind, payload)
                        stamps[kind] = timestamp
                if next_tick is not None and next_tick <= timestamp + 1e-9:
                    row = sample(next_tick)
                    if row is not None:
                        yield next_tick, 'sample', row
                    next_tick += SAMPLE_SEC
            if self.clock.stage == 0:
                raise ValueError('No stage 1 start marker was received; no GMR execution started.')
            raise ValueError('Recording ended before stage 3 after stage 1; GMR execution is incomplete.')
        finally:
            if hasattr(events, 'close'):
                events.close()


def phase_dataframe(rows, durations):
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError('No TS samples between stage 1 and stage 3.')
    result['phase'] = [float(estimate_phase(int(row.stage), row.stage_elapsed, durations))
                       for row in result.itertuples()]
    return result


class TrailingAlarm:
    def __init__(self, smooth_window, vote_window, votes):
        self.scores = deque(maxlen=smooth_window)
        self.alarms = deque(maxlen=vote_window)
        self.votes = votes

    def append(self, epsilon):
        self.scores.append(epsilon)
        smoothed = float(np.median(self.scores))
        self.alarms.append(smoothed > 1.)
        return smoothed, sum(self.alarms) >= self.votes


def channel_diagnostics(model, scaled, component, threshold):
    """Project the joint distance boundary onto each channel in original units.

    These are diagnostic envelopes, not independently calibrated channel alarms.
    Cross-channel correlations can cause a joint alarm while every channel is
    inside its envelope. Use the same regularized covariance as distance scoring.
    """
    inputs, outputs = model._index_sets()
    mean, covariance = model._conditional_output_distribution(scaled[inputs], component)
    covariance = covariance + np.eye(len(outputs)) * model.reg_covar
    std = np.sqrt(np.maximum(np.diag(covariance), 1e-12))
    scale, offset = model.scaler.scale_[outputs], model.scaler.mean_[outputs]
    predicted = mean * scale + offset
    observed = scaled[outputs] * scale + offset
    half_width = threshold * std * scale
    residual = np.abs(scaled[outputs] - mean) / std
    if not np.isfinite([predicted, observed, half_width, residual]).all():
        raise ValueError('GMR channel diagnostics contain nonfinite values.')
    return {name: dict(observed=float(observed[i]), expected=float(predicted[i]),
                       lower=float(predicted[i] - half_width[i]), upper=float(predicted[i] + half_width[i]),
                       standardized_deviation=float(residual[i]))
            for i, name in enumerate(model.output_columns)}


class GmrOnlineDetector:
    kind = 'gmr'
    score_kind = 'gmr_distance'
    score_source = 'causal_gmr_mahalanobis'
    note = ('Causal GMR: stage 1 starts scoring, stage 2 advances progress, stage 3 ends it. '
            'Earlier 2/3 markers are ignored. Progress within a stage uses nominal durations. '
            'Signals use latest received values at 10 Hz and trailing smoothing; no future test samples. '
            'The graph shows joint distance and its nominal adaptive threshold. '
            'The alarm retains trailing epsilon smoothing and majority voting. Validate detection on held-out runs.')
    knn = True
    capture_visual = False
    frame_count = None
    start_sec = 0.
    end_sec = None
    preview_image = matched_image = None

    def __init__(self, config, progress):
        self.config = config
        self.nominals = list(dict.fromkeys(config.get('nominal_bags', [])))
        self.excluded = {Path(bag).name for bag in self.nominals}
        self.signal_mode = 'pose'
        self.path = Path(config['output_dir']) / 'online_replay' / 'gmr_online'
        self.signature = {'protocol': PROTOCOL}
        self.training_key = None

    def reset(self, bag):
        if Path(bag).name in self.excluded:
            raise ValueError('Select a bag outside GMR nominal training data.')
        from . import run_time_series_test as settings
        self.bag, self.step = bag, 0
        self.alarm = TrailingAlarm(settings.SCORE_SMOOTH_WINDOW, settings.MAJORITY_WINDOW, settings.MAJORITY_VOTES)

    def prepare_replay(self, progress, stop):
        from rosbag_io import iter_ts_replay_events
        from rynnbrain_vlm.replay_detectors import gmr_signal_topics, GMR_SIGNALS, check_stop
        from rynnbrain_vlm.visual_matching import _source_identity
        from . import run_time_series_test as settings
        if len(self.nominals) < 2:
            raise ValueError('GMR calibration requires at least two nominal_bags.')
        if self.config.get('ignore_recorded_stages', False):
            raise ValueError('Online GMR requires the recorded stage topic; ignore_recorded_stages must be false.')
        self.stage_topic = self.config.get('stage_topic', '/recording_stage')
        self.topics = gmr_signal_topics(self.nominals[0], self.signal_mode)
        for bag in [*self.nominals[1:], self.bag]:
            check_stop(stop)
            if gmr_signal_topics(bag, self.signal_mode) != self.topics:
                raise ValueError(f'{bag}: TS topics differ from nominal training topics {self.topics}.')
        names = ('N_COMPONENTS', 'COMPONENT_THRESHOLD_MARGIN', 'RANDOM_STATE', 'THRESHOLD_BINS',
                 'THRESHOLD_QUANTILE', 'THRESHOLD_MARGIN', 'MIN_POINTS_PER_BIN',
                 'THRESHOLD_FLOOR_QUANTILE', 'THRESHOLD_SMOOTH_WINDOW',
                 'SCORE_SMOOTH_WINDOW', 'MAJORITY_WINDOW', 'MAJORITY_VOTES')
        identity = dict(protocol=PROTOCOL, sample_sec=SAMPLE_SEC, topics=self.topics,
                        stage_topic=self.stage_topic, signal_mode=self.signal_mode,
                        parameters={name: getattr(settings, name) for name in names},
                        sources=[dict(bag=str(Path(b).resolve()), files=_source_identity(Path(b))) for b in self.nominals])
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        self.path.mkdir(parents=True, exist_ok=True)
        artifact = self.path / f'{key}.npz'
        if key != self.training_key:
            if artifact.is_file():
                progress('Loading calibrated nominal GMR model (test scores will be computed during replay)')
                self.load(artifact)
            else:
                runs, durations = [], []
                self.columns, self.pose_frame = None, None
                for index, bag in enumerate(self.nominals):
                    check_stop(stop)
                    progress(f'Preparing nominal GMR training data {index + 1}/{len(self.nominals)}')
                    sampler = CausalSamples(self.topics, stop)
                    rows = []
                    for _, kind, payload in sampler.packets(iter_ts_replay_events(bag, self.topics, self.stage_topic)):
                        check_stop(stop)
                        if kind == 'sample':
                            rows.append(payload)
                    check_stop(stop)
                    if not rows or {r['stage'] for r in rows} != {1, 2}:
                        raise ValueError(f'{bag}: nominal signals must cover both intervals 1→2 and 2→3.')
                    if self.columns is not None and (sampler.columns != self.columns or sampler.pose_frame != self.pose_frame):
                        raise ValueError(f'{bag}: nominal channel names or pose reference frame differ.')
                    self.columns, self.pose_frame = sampler.columns, sampler.pose_frame
                    runs.append(rows)
                    durations.append(sampler.clock.durations())
                held_scores = []
                for held in range(len(runs)):
                    check_stop(stop)
                    progress(f'Calibrating GMR on held-out nominal {held + 1}/{len(runs)}')
                    train_indices = [i for i in range(len(runs)) if i != held]
                    # Duration estimates must also exclude the held-out run.
                    fold_durations = np.median([durations[i] for i in train_indices], axis=0)
                    model = self.new_model()
                    model.fit([phase_dataframe(runs[i], fold_durations) for i in train_indices])
                    held_scores.append(model.score_dataframe(phase_dataframe(runs[held], fold_durations)))
                self.thresholds = build_phase_threshold_from_scores(pd.concat(held_scores, ignore_index=True),
                    n_bins=settings.THRESHOLD_BINS, quantile=settings.THRESHOLD_QUANTILE,
                    margin=settings.THRESHOLD_MARGIN, min_points_per_bin=settings.MIN_POINTS_PER_BIN,
                    threshold_floor_quantile=settings.THRESHOLD_FLOOR_QUANTILE,
                    smooth_threshold_window=settings.THRESHOLD_SMOOTH_WINDOW)
                check_stop(stop)
                progress('Fitting final nominal GMR model')
                self.durations = np.median(durations, axis=0)
                self.model = self.new_model()
                self.model.fit([phase_dataframe(run, self.durations) for run in runs])
                check_stop(stop)
                self.save(artifact, identity)
            self.training_key = key
        self.signal_description = GMR_SIGNALS[self.signal_mode]
        self.signature = dict(protocol=PROTOCOL, topics=self.topics, signal_mode=self.signal_mode,
                              output_columns=self.columns, pose_frame=self.pose_frame,
                              nominal_stage_durations_sec=list(self.durations), sample_hz=1 / SAMPLE_SEC,
                              model_file=str(artifact), training_key=key)
        return dict(status='ready', offline=False, **self.signature, signal_description=self.signal_description)

    def new_model(self):
        from . import run_time_series_test as settings
        return GMRAnomalyDetector(['phase'], self.columns, n_components=settings.N_COMPONENTS,
            threshold_margin=settings.COMPONENT_THRESHOLD_MARGIN, random_state=settings.RANDOM_STATE)

    def save(self, path, identity):
        metadata = dict(identity=identity, columns=self.columns, pose_frame=self.pose_frame)
        temporary = path.with_suffix('.tmp')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, metadata=json.dumps(metadata), durations=self.durations,
                mean=self.model.scaler.mean_, scale=self.model.scaler.scale_,
                weights=self.model.gmm.weights_, means=self.model.gmm.means_, covariances=self.model.gmm.covariances_,
                component_thresholds=[self.model.component_thresholds[i] for i in range(self.model.n_components)],
                phases=self.thresholds.phase_center.to_numpy(), limits=self.thresholds.threshold_mahalanobis.to_numpy())
        temporary.replace(path)

    def load(self, path):
        with np.load(path, allow_pickle=False) as data:
            if any(not np.isfinite(data[key]).all() for key in data.files if key != 'metadata'):
                raise ValueError(f'Invalid nominal GMR model: {path}')
            meta = json.loads(str(data['metadata']))
            self.columns, self.pose_frame = meta['columns'], meta['pose_frame']
            self.durations = data['durations'].copy()
            if (self.durations <= 0).any() or (data['scale'] <= 0).any():
                raise ValueError(f'Invalid GMR duration or scaling in {path}')
            self.model = self.new_model()
            self.model.scaler.mean_, self.model.scaler.scale_ = data['mean'].copy(), data['scale'].copy()
            self.model.gmm.weights_, self.model.gmm.means_ = data['weights'].copy(), data['means'].copy()
            self.model.gmm.covariances_ = data['covariances'].copy()
            self.model.component_thresholds = dict(enumerate(data['component_thresholds'].tolist()))
            self.model.fitted = True
            self.thresholds = pd.DataFrame(dict(phase_center=data['phases'], threshold_mahalanobis=data['limits']))

    def create_feed(self, publish, stop, update):
        from rosbag_io import iter_ts_replay_events
        self.sampler = CausalSamples(self.topics, stop)
        self.sampler.columns, self.sampler.pose_frame = self.columns, self.pose_frame
        cameras = self.config.get('camera_topics', [])
        events = iter_ts_replay_events(self.bag, self.topics, self.stage_topic, cameras[0] if cameras else None)
        return GmrOnlineFeed(self, self.sampler.packets(events), publish, stop, update)

    def append(self, image, timestamp, verify=False):
        started = time.monotonic()
        sample = self.pending
        phase = float(estimate_phase(sample['stage'], sample['stage_elapsed'], self.durations))
        values = np.array([phase, *[sample[column] for column in self.columns]])
        scaled = (values - self.model.scaler.mean_) / self.model.scaler.scale_
        distance, component = self.model._score_scaled_point(scaled)
        threshold = max(float(np.interp(phase, self.thresholds.phase_center,
                                       self.thresholds.threshold_mahalanobis)), 1e-8)
        epsilon = distance / threshold
        smoothed, alarm = self.alarm.append(epsilon)
        if not np.isfinite([distance, threshold, smoothed]).all():
            raise ValueError('GMR produced a nonfinite score; replay stopped.')
        channels = channel_diagnostics(self.model, scaled, component, threshold)
        row = dict(sample_index=self.step, timestamp_sec=timestamp, signal_time_sec=timestamp,
                   anomaly_score=distance, threshold=threshold, above_threshold=bool(distance > threshold),
                   alarm_filtered=bool(alarm), raw_distance=distance, adaptive_threshold=threshold,
                   epsilon_raw=epsilon, epsilon_smooth=smoothed, channels=channels,
                   phase=phase, stage=sample['stage'], component=component,
                   signal_mode=self.signal_mode, cached_feature_sec=time.monotonic() - started,
                   classifier_sec=0., verification=None, score_kind=self.score_kind,
                   vector_kind='gmr_mahalanobis_and_phase', visual_matching={'status': 'disabled'})
        self.step += 1
        return row, np.array([distance, phase]), None


class GmrOnlineFeed:
    """Pace cameras, markers and TS samples together, including before stage 1."""
    def __init__(self, detector, packets, publish, stop, update, clock=time.monotonic):
        self.detector, self.packets = detector, iter(packets)
        self.publish, self.stop, self.update, self.clock = publish, stop, update, clock
        self.origin = clock()
        self.index = 0

    def run(self):
        pass

    def next_frame(self, earliest):
        from rosbag_io import decode_image_message
        while not self.stop.is_set():
            packet = next(self.packets, None)
            if packet is None:
                return None
            timestamp, kind, payload = packet
            if self.stop.wait(max(0., self.origin + timestamp - self.clock())):
                return None
            if kind == 'camera':
                self.publish(decode_image_message(*payload), timestamp)
            elif kind == 'stage':
                self.update(gmr_stage=payload)
            else:
                self.detector.pending = payload
                self.index += 1
                return self.index - 1, timestamp, None
        return None

    def close(self):
        if hasattr(self.packets, 'close'):
            self.packets.close()
