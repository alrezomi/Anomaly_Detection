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

from .anomaly_detection import GMRAnomalyDetector
from .threshold import build_phase_threshold_from_scores

SAMPLE_SEC = .1
PROTOCOL = 'causal_stage_1_to_3_gmr_v7'  # v7: debounced release (one gripper reading must confirm the opening)
# Each sensor gets its own GMR and threshold, so one sensor's nominal variation
# (e.g. contact force spikes) cannot loosen another sensor's limit. Force is not
# a GMR sensor: subtle force changes (e.g. lifting a second instrument) overlap
# nominal runs, so force only has a safety limit on its magnitude.
SENSOR_NAMES = {'pose': 'position'}
FORCE_COLUMNS = ['force_x', 'force_y', 'force_z']
# (1 mm)^2 keeps the release covariance invertible with few, tightly clustered nominal releases.
RELEASE_VARIANCE_FLOOR = 1e-6


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
    # The current interval's end is unknown. Never use its eventual duration.
    # Intervals: stage 1->2, stage 2->3 and, when scored, the window after stage 3.
    return (stage - 1 + np.clip(np.asarray(elapsed) / durations[stage - 1], 0., 1.)) / len(durations)


def signal_values(kind, payload):
    if kind == 'pose':
        # Position only. Nominal grasps end with the wrist rotated either way
        # (R11 = +1 or -1), so orientation inflated every nominal distance.
        result = {f'position_{axis}': payload[f'position_{axis}'] for axis in 'xyz'}
    elif kind == 'force':
        result = {column: payload[column] for column in FORCE_COLUMNS}
    elif kind == 'joints':
        result = {key: value for key, value in payload.items() if key.endswith('_position')}
    else:
        result = dict(payload)
    if not result or not np.isfinite(list(result.values())).all():
        raise ValueError(f'Missing or nonfinite {kind} values; cannot score this observation.')
    return result


class CausalSamples:
    """Merge arrived values at clock ticks. Error markers never advance phase.

    Force is averaged over each tick and expressed relative to the first tick
    of stage 1, because the sensor zero drifts by >10 N between sessions.

    Sampling continues for post_sec after stage 3 (a third progress interval),
    so the release and the retreat are scored too. With release=True the
    gripper is watched even when it is not a GMR sensor: its latest reading is
    added to each sample for display, and one 'release' packet gives the TCP
    position where it first opens after a grasp. An opening must be confirmed
    by the next gripper reading (about 0.1 s later): single 0.0 readings occur
    in nominal runs and were taken as releases. Each sample also says whether
    the gripper should be holding the object (gripper_holding: closed past half
    travel, not yet released). Reading ends once the post-stage-3 window has
    passed and the release was seen, or at the end of the recording.
    """
    def __init__(self, kinds, stop=None, release=False, post_sec=0.):
        self.kinds = set(kinds)
        self.stop = stop
        self.clock = StageClock()
        self.pose_frame = None
        self.columns = None
        self.sensors = None
        self.force_bias = None
        self.track_release = release
        self.post_sec = post_sec
        self.release = None
        self.gripper = None
        self.gripper_peak = 0.
        self.opening = False  # A reading below half of the grip waits for the next reading to confirm it.

    def packets(self, events):
        latest, stamps = {}, {}
        forces = []
        next_tick = None
        previous = -1.

        def sample(timestamp):
            if forces:
                mean = {column: float(np.mean([f[column] for f in forces])) for column in forces[0]}
                if self.force_bias is None:
                    self.force_bias = mean
                latest['force'] = {column: value - self.force_bias[column] for column, value in mean.items()}
                forces.clear()
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
            self.sensors = {SENSOR_NAMES.get(kind, kind): sorted(latest[kind]) for kind in sorted(latest)}
            stage = self.clock.stage
            row = dict(time=timestamp, stage=stage,
                       stage_elapsed=timestamp - self.clock.boundaries[stage], **values)
            if self.gripper is not None and 'gripper' not in self.kinds:
                row['gripper_position'] = self.gripper  # Display only; not a GMR channel.
            if self.gripper is not None and self.track_release:
                row['gripper_holding'] = bool(self.gripper_peak > .5 and self.release is None)
            return row

        def tick():
            """Sample at next_tick, or end sampling once the post-stage-3 window has passed."""
            nonlocal next_tick
            if self.clock.stage == 3 and next_tick > self.clock.boundaries[3] + self.post_sec + 1e-9:
                next_tick = None
                return None
            timestamp, row = next_tick, sample(next_tick)
            next_tick += SAMPLE_SEC
            return None if row is None else (timestamp, 'sample', row)

        def finished():
            return (self.clock.stage == 3 and next_tick is None
                    and (not self.track_release or self.release is not None))

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
                    packet = tick()
                    if packet is not None:
                        yield packet
                if finished():
                    return
                messages = list(group)
                # Markers win over signals at the exact same timestamp.
                for _, kind, payload in messages:
                    if kind != 'stage':
                        continue
                    old_stage = self.clock.stage
                    self.clock.observe(timestamp, payload)
                    if self.clock.stage != old_stage:
                        yield timestamp, 'stage', self.clock.stage
                        if self.clock.stage == 3 and self.post_sec <= 0:
                            next_tick = None
                            if finished():
                                return
                        if old_stage == 0:
                            next_tick = timestamp
                            latest.clear()
                            stamps.clear()
                            forces.clear()
                for _, kind, payload in messages:
                    if kind == 'camera':
                        yield timestamp, 'camera', payload
                        continue
                    watched = kind in self.kinds or (kind == 'gripper' and self.track_release)
                    if not watched or self.clock.stage == 0:
                        continue
                    if kind == 'pose':
                        frame = payload.get('pose_frame', '')
                        if self.pose_frame is not None and frame != self.pose_frame:
                            raise ValueError('TCP pose reference frame changed during execution.')
                        self.pose_frame = frame
                    values = signal_values(kind, payload)
                    if kind == 'force':
                        forces.append(values)
                    elif kind in self.kinds:
                        latest[kind] = values
                    stamps[kind] = timestamp
                    if kind == 'gripper':
                        self.gripper = values['gripper_position']
                    if kind == 'gripper' and self.track_release and self.release is None:
                        # Closed past half travel, then opened below half of that grip, confirmed by the
                        # next reading; the release is reported at the confirming reading.
                        if self.gripper_peak > .5 and self.gripper < self.gripper_peak / 2 and 'pose' in latest:
                            if self.opening:
                                self.release = dict(time=timestamp, stage=self.clock.stage, **latest['pose'])
                                yield timestamp, 'release', self.release
                                if finished():
                                    return
                            self.opening = True
                        else:
                            self.opening = False
                            self.gripper_peak = max(self.gripper_peak, self.gripper)
                if next_tick is not None and next_tick <= timestamp + 1e-9:
                    packet = tick()
                    if packet is not None:
                        yield packet
                if finished():
                    return
            if self.clock.stage == 0:
                raise ValueError('No stage 1 start marker was received; no GMR execution started.')
            if self.clock.stage == 3:
                return  # Stage 3 was reached; the gripper never opened before the recording ended.
            raise ValueError('Recording ended before stage 3 after stage 1; GMR execution is incomplete.')
        finally:
            if hasattr(events, 'close'):
                events.close()


def force_magnitude(sample):
    return float(np.linalg.norm([sample[column] for column in FORCE_COLUMNS]))


def release_position(release):
    return np.array([release[f'position_{axis}'] for axis in 'xyz'])


def fit_release(positions):
    """Mean and covariance of nominal TCP positions where the gripper opened."""
    positions = np.asarray(positions, dtype=float)
    covariance = np.cov(positions.T, bias=True) + np.eye(3) * RELEASE_VARIANCE_FLOOR
    return positions.mean(axis=0), covariance


def release_distance(position, mean, covariance):
    difference = position - mean
    return float(np.sqrt(difference @ np.linalg.pinv(covariance) @ difference))


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


def channel_diagnostics(model, scaled, threshold):
    """Project the joint distance boundary onto each channel in original units.

    These are diagnostic envelopes, not independently calibrated channel alarms.
    Cross-channel correlations can cause a joint alarm while every channel is
    inside its envelope. Use the same regularized covariance as distance scoring.
    """
    inputs, outputs = model._index_sets()
    mean, covariance, _ = model._predict_output_distribution(scaled[inputs])
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
    score_kind = 'gmr_sensor_ratio'
    score_source = 'causal_gmr_mahalanobis'
    note = ('Causal GMR: stage 1 starts scoring, stage 2 advances progress, stage 3 starts a short final window. '
            'Earlier 2/3 markers are ignored. Progress within a stage uses nominal durations. '
            'Signals use latest received values at 10 Hz and trailing smoothing; no future test samples. '
            'Position (and gripper/joints in those modes) has its own GMR and nominal adaptive threshold, '
            'with trailing epsilon smoothing and majority voting. Force only has a safety limit: '
            'THRESHOLD_MARGIN times the largest force magnitude in any nominal run. Any check can raise the alarm. '
            'Scoring continues for a short window after stage 3 so the release and retreat are checked. '
            'When the gripper is recorded, the TCP position where it first opens after the grasp '
            'is also compared with the nominal release positions, and a gripper that closes to its '
            'empty value while it should hold the object raises the empty-grip alarm. '
            'Validate detection on held-out runs.')
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
        self.signal_mode = 'pose_force'
        self.path = Path(config['output_dir']) / 'online_replay' / 'gmr_online'
        self.signature = {'protocol': PROTOCOL}
        self.training_key = None
        self.force_limit = None

    def reset(self, bag):
        if Path(bag).name in self.excluded:
            raise ValueError('Select a bag outside GMR nominal training data.')
        from . import run_time_series_test as settings
        self.bag, self.step = bag, 0
        self.release_result = None
        self.alarms = {name: TrailingAlarm(settings.SCORE_SMOOTH_WINDOW, settings.MAJORITY_WINDOW, settings.MAJORITY_VOTES)
                       for name in getattr(self, 'sensors', None) or {}}

    def prepare_replay(self, progress, stop):
        from rosbag_io import iter_ts_replay_events
        from rynnbrain_vlm.replay_detectors import gmr_signal_topics, GMR_SIGNALS, GMR_OPTIONAL_SIGNALS, check_stop
        from rynnbrain_vlm.visual_matching import _source_identity
        from . import run_time_series_test as settings
        if len(self.nominals) < 2:
            raise ValueError('GMR calibration requires at least two nominal_bags.')
        if self.config.get('ignore_recorded_stages', False):
            raise ValueError('Online GMR requires the recorded stage topic; ignore_recorded_stages must be false.')
        self.stage_topic = self.config.get('stage_topic', '/recording_stage')
        bag_topics = {}
        for bag in [*self.nominals, self.bag]:
            check_stop(stop)
            bag_topics[bag] = gmr_signal_topics(bag, self.signal_mode)
        # Optional signals (force, gripper) are used only when every bag recorded them.
        missing = {}
        for kind in GMR_OPTIONAL_SIGNALS.get(self.signal_mode, []):
            names = [Path(bag).name for bag, topics in bag_topics.items() if kind not in topics]
            if names:
                missing[kind] = names
                progress(f'No {kind} topic in {len(names)} bag(s), e.g. {names[0]}; '
                         + ('scoring TCP position only.' if kind == 'force' else 'no gripper release check.'))
        bag_topics = {bag: {k: v for k, v in topics.items() if k not in missing} for bag, topics in bag_topics.items()}
        self.topics = bag_topics[self.nominals[0]]
        for bag, topics in bag_topics.items():
            if topics != self.topics:
                raise ValueError(f'{bag}: TS topics differ from nominal training topics {self.topics}.')
        # The gripper drives the release check; it is a GMR channel only in gripper modes.
        self.release_check = {'pose', 'gripper'} <= set(self.topics)
        self.sample_kinds = [kind for kind in self.topics
                             if kind != 'gripper' or self.signal_mode in ('gripper', 'pose_gripper')]
        names = ('N_COMPONENTS', 'COMPONENT_THRESHOLD_MARGIN', 'RANDOM_STATE', 'THRESHOLD_BINS',
                 'THRESHOLD_QUANTILE', 'THRESHOLD_MARGIN', 'MIN_POINTS_PER_BIN',
                 'THRESHOLD_FLOOR_QUANTILE', 'THRESHOLD_SMOOTH_WINDOW',
                 'SCORE_SMOOTH_WINDOW', 'MAJORITY_WINDOW', 'MAJORITY_VOTES', 'POST_STAGE3_SEC')
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
                runs, durations, releases = [], [], []
                self.columns, self.sensors, self.pose_frame, self.force_limit = None, None, None, None
                for index, bag in enumerate(self.nominals):
                    check_stop(stop)
                    progress(f'Preparing nominal GMR training data {index + 1}/{len(self.nominals)}')
                    sampler = CausalSamples(self.sample_kinds, stop, release=self.release_check,
                                            post_sec=settings.POST_STAGE3_SEC)
                    rows = []
                    for _, kind, payload in sampler.packets(iter_ts_replay_events(bag, self.topics, self.stage_topic)):
                        check_stop(stop)
                        if kind == 'sample':
                            rows.append(payload)
                    check_stop(stop)
                    if not rows or {r['stage'] for r in rows} != ({1, 2, 3} if settings.POST_STAGE3_SEC > 0 else {1, 2}):
                        raise ValueError(f'{bag}: nominal signals must cover intervals 1→2, 2→3 and the window after stage 3.')
                    if self.columns is not None and (sampler.columns != self.columns or sampler.pose_frame != self.pose_frame):
                        raise ValueError(f'{bag}: nominal channel names or pose reference frame differ.')
                    if self.release_check and sampler.release is None:
                        raise ValueError(f'{bag}: the gripper never opened after grasping; cannot learn the nominal release position.')
                    releases.append(sampler.release)
                    self.columns, self.pose_frame = sampler.columns, sampler.pose_frame
                    self.sensors = {name: columns for name, columns in sampler.sensors.items() if name != 'force'}
                    runs.append(rows)
                    durations.append([*sampler.clock.durations(), *([settings.POST_STAGE3_SEC] if settings.POST_STAGE3_SEC > 0 else [])])
                held_scores = {name: [] for name in self.sensors}
                for held in range(len(runs)):
                    check_stop(stop)
                    progress(f'Calibrating GMR on held-out nominal {held + 1}/{len(runs)}')
                    train_indices = [i for i in range(len(runs)) if i != held]
                    # Duration estimates must also exclude the held-out run.
                    fold_durations = np.median([durations[i] for i in train_indices], axis=0)
                    for name, columns in self.sensors.items():
                        model = self.new_model(columns)
                        model.fit([phase_dataframe(runs[i], fold_durations) for i in train_indices])
                        held_scores[name].append(model.score_dataframe(phase_dataframe(runs[held], fold_durations)))
                self.thresholds = {name: build_phase_threshold_from_scores(pd.concat(scores, ignore_index=True),
                    n_bins=settings.THRESHOLD_BINS, quantile=settings.THRESHOLD_QUANTILE,
                    margin=settings.THRESHOLD_MARGIN, min_points_per_bin=settings.MIN_POINTS_PER_BIN,
                    threshold_floor_quantile=settings.THRESHOLD_FLOOR_QUANTILE,
                    smooth_threshold_window=settings.THRESHOLD_SMOOTH_WINDOW) for name, scores in held_scores.items()}
                if self.release_check:
                    self.calibrate_release(releases, progress)
                if 'force' in self.topics:
                    self.calibrate_force_limit(runs, progress)
                check_stop(stop)
                progress('Fitting final nominal GMR models')
                self.durations = np.median(durations, axis=0)
                self.models = {}
                for name, columns in self.sensors.items():
                    self.models[name] = self.new_model(columns)
                    self.models[name].fit([phase_dataframe(run, self.durations) for run in runs])
                check_stop(stop)
                self.save(artifact, identity)
            self.training_key = key
        self.signal_description = (GMR_SIGNALS[self.signal_mode]
            + (' (force not recorded)' if 'force' in missing else '')
            + (' (gripper not recorded: no release check)' if 'gripper' in missing else
               ' + gripper release check' if self.release_check and self.signal_mode == 'pose' else ''))
        self.signature = dict(protocol=PROTOCOL, topics=self.topics, signal_mode=self.signal_mode,
                              output_columns=self.columns, sensors=self.sensors, release_check=self.release_check,
                              pose_frame=self.pose_frame,
                              nominal_stage_durations_sec=list(self.durations), sample_hz=1 / SAMPLE_SEC,
                              model_file=str(artifact), training_key=key)
        return dict(status='ready', offline=False, **self.signature, signal_description=self.signal_description)

    def calibrate_release(self, releases, progress):
        """Same calibration as the GMR thresholds: held-out nominal distances,
        THRESHOLD_QUANTILE scaled by THRESHOLD_MARGIN."""
        from . import run_time_series_test as settings
        positions = np.array([release_position(release) for release in releases])
        held_out = [release_distance(positions[i], *fit_release(np.delete(positions, i, axis=0)))
                    for i in range(len(positions))]
        self.release_mean, self.release_covariance = fit_release(positions)
        self.release_threshold = float(np.quantile(held_out, settings.THRESHOLD_QUANTILE) * settings.THRESHOLD_MARGIN)
        # One unusual nominal release (e.g. a re-grasp) sets this threshold; name it.
        worst = int(np.argmax(held_out))
        progress(f'Release check threshold {self.release_threshold:.1f}; largest held-out nominal release '
                 f'distance {held_out[worst]:.1f} ({Path(self.nominals[worst]).name}), median {np.median(held_out):.1f}')

    def calibrate_force_limit(self, runs, progress):
        """Safety limit only: well above every nominal contact, including grasp pressing."""
        from . import run_time_series_test as settings
        peaks = [max(force_magnitude(row) for row in run) for run in runs]
        worst = int(np.argmax(peaks))
        self.force_limit = float(peaks[worst] * settings.THRESHOLD_MARGIN)
        progress(f'Force safety limit {self.force_limit:.1f} N; largest nominal force '
                 f'{peaks[worst]:.1f} N ({Path(self.nominals[worst]).name})')

    def check_release(self, release):
        """Compare where the gripper first opened with the nominal release positions."""
        if release is None:
            self.release_result = dict(status='not_released', alarm=True, threshold=self.release_threshold)
        else:
            position = release_position(release)
            distance = release_distance(position, self.release_mean, self.release_covariance)
            self.release_result = dict(status='released', time_sec=release['time'], stage=release['stage'],
                position=position.tolist(), expected=self.release_mean.tolist(), distance=distance,
                threshold=self.release_threshold, alarm=bool(distance > self.release_threshold))
        return self.release_result

    def new_model(self, columns):
        from . import run_time_series_test as settings
        return GMRAnomalyDetector(['phase'], columns, n_components=settings.N_COMPONENTS,
            threshold_margin=settings.COMPONENT_THRESHOLD_MARGIN, random_state=settings.RANDOM_STATE)

    def save(self, path, identity):
        metadata = dict(identity=identity, columns=self.columns, sensors=self.sensors, pose_frame=self.pose_frame)
        arrays = {}
        for name, model in self.models.items():
            arrays.update({f'{name}__{key}': value for key, value in dict(
                mean=model.scaler.mean_, scale=model.scaler.scale_,
                weights=model.gmm.weights_, means=model.gmm.means_, covariances=model.gmm.covariances_,
                component_thresholds=[model.component_thresholds[i] for i in range(model.n_components)],
                phases=self.thresholds[name].phase_center.to_numpy(),
                limits=self.thresholds[name].threshold_mahalanobis.to_numpy()).items()})
        if self.force_limit is not None:
            arrays.update(force_limit=self.force_limit)
        if self.release_check:
            arrays.update(release_mean=self.release_mean, release_covariance=self.release_covariance,
                          release_threshold=self.release_threshold)
        temporary = path.with_suffix('.tmp')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, metadata=json.dumps(metadata), durations=self.durations, **arrays)
        temporary.replace(path)

    def load(self, path):
        with np.load(path, allow_pickle=False) as data:
            if any(not np.isfinite(data[key]).all() for key in data.files if key != 'metadata'):
                raise ValueError(f'Invalid nominal GMR model: {path}')
            meta = json.loads(str(data['metadata']))
            self.columns, self.sensors, self.pose_frame = meta['columns'], meta['sensors'], meta['pose_frame']
            self.durations = data['durations'].copy()
            if (self.durations <= 0).any():
                raise ValueError(f'Invalid GMR duration in {path}')
            self.models, self.thresholds = {}, {}
            for name, columns in self.sensors.items():
                saved = {key: data[f'{name}__{key}'].copy() for key in
                         ('mean', 'scale', 'weights', 'means', 'covariances', 'component_thresholds', 'phases', 'limits')}
                if (saved['scale'] <= 0).any():
                    raise ValueError(f'Invalid GMR scaling for {name} in {path}')
                model = self.models[name] = self.new_model(columns)
                model.scaler.mean_, model.scaler.scale_ = saved['mean'], saved['scale']
                model.gmm.weights_, model.gmm.means_, model.gmm.covariances_ = saved['weights'], saved['means'], saved['covariances']
                model.component_thresholds = dict(enumerate(saved['component_thresholds'].tolist()))
                model.fitted = True
                self.thresholds[name] = pd.DataFrame(dict(phase_center=saved['phases'], threshold_mahalanobis=saved['limits']))
            self.force_limit = float(data['force_limit']) if 'force_limit' in data.files else None
            if self.release_check:
                self.release_mean, self.release_covariance = data['release_mean'].copy(), data['release_covariance'].copy()
                self.release_threshold = float(data['release_threshold'])

    def create_feed(self, publish, stop, update):
        from rosbag_io import iter_ts_replay_events
        from . import run_time_series_test as settings
        self.sampler = CausalSamples(self.sample_kinds, stop, release=self.release_check, post_sec=settings.POST_STAGE3_SEC)
        self.sampler.columns, self.sampler.sensors, self.sampler.pose_frame = self.columns, self.sensors, self.pose_frame
        cameras = self.config.get('camera_topics', [])
        events = iter_ts_replay_events(self.bag, self.topics, self.stage_topic, cameras[0] if cameras else None)
        return GmrOnlineFeed(self, self.sampler.packets(events), publish, stop, update)

    def append(self, image, timestamp, verify=False):
        from . import run_time_series_test as settings
        started = time.monotonic()
        sample = self.pending
        phase = float(estimate_phase(sample['stage'], sample['stage_elapsed'], self.durations))
        sensors, channels = {}, {}
        for name, columns in self.sensors.items():
            model = self.models[name]
            values = np.array([phase, *[sample[column] for column in columns]])
            scaled = (values - model.scaler.mean_) / model.scaler.scale_
            distance, component = model._score_scaled_point(scaled)
            threshold = max(float(np.interp(phase, self.thresholds[name].phase_center,
                                           self.thresholds[name].threshold_mahalanobis)), 1e-8)
            alarm = self.alarms.setdefault(name, TrailingAlarm(
                settings.SCORE_SMOOTH_WINDOW, settings.MAJORITY_WINDOW, settings.MAJORITY_VOTES))
            smoothed, alarmed = alarm.append(distance / threshold)
            if not np.isfinite([distance, threshold, smoothed]).all():
                raise ValueError(f'GMR produced a nonfinite {name} score; replay stopped.')
            sensors[name] = dict(distance=distance, threshold=threshold, epsilon_raw=distance / threshold,
                                 epsilon_smooth=smoothed, alarm=bool(alarmed), component=component)
            channels.update(channel_diagnostics(model, scaled, threshold))
        if self.force_limit is not None:
            magnitude = force_magnitude(sample)
            sensors['force'] = dict(kind='limit', distance=magnitude, threshold=self.force_limit,
                                    epsilon_raw=magnitude / self.force_limit, epsilon_smooth=magnitude / self.force_limit,
                                    alarm=bool(magnitude > self.force_limit), component=None)
            for column in FORCE_COLUMNS:
                channels[column] = dict(observed=float(sample[column]), expected=None,
                                        lower=None, upper=None, standardized_deviation=None)
        if 'gripper_holding' in sample:
            # Empty grip: a rule, not a GMR. Alarm while the object should be held and the fingers read
            # their fully closed, empty value (encoder counts: readings are multiples of 1/255).
            reading, empty = float(sample['gripper_position']), settings.GRIPPER_EMPTY_COUNT / 255
            holding = bool(sample['gripper_holding'] and sample['stage'] < 3)
            sensors['gripper_empty'] = dict(kind='gripper_empty', distance=reading, threshold=empty,
                                            epsilon_raw=reading / empty, epsilon_smooth=reading / empty,
                                            alarm=holding and reading >= empty - .5 / 255, holding=holding,
                                            component=None)
        if 'gripper_position' in sample and 'gripper_position' not in channels:
            # Watched for the release check; shown, but not scored by a GMR.
            channels['gripper_position'] = dict(observed=float(sample['gripper_position']), expected=None,
                                                lower=None, upper=None, standardized_deviation=None)
        # The displayed score is the sensor furthest outside its own limit; the empty-grip rule is not a
        # distance and only raises its alarm.
        worst = max((name for name in sensors if name != 'gripper_empty'), key=lambda name: sensors[name]['epsilon_raw'])
        sources = [name for name, result in sensors.items() if result['alarm']]
        row = dict(sample_index=self.step, timestamp_sec=timestamp, signal_time_sec=timestamp,
                   anomaly_score=sensors[worst]['epsilon_raw'], threshold=1.,
                   above_threshold=bool(sensors[worst]['epsilon_raw'] > 1.),
                   alarm_filtered=bool(sources), alarm_sources=sources, score_sensor=worst, sensors=sensors,
                   raw_distance=sensors[worst]['distance'], adaptive_threshold=sensors[worst]['threshold'],
                   epsilon_raw=sensors[worst]['epsilon_raw'],
                   epsilon_smooth=max(result['epsilon_smooth'] for name, result in sensors.items() if name != 'gripper_empty'),
                   channels=channels,
                   phase=phase, stage=sample['stage'], component=sensors[worst]['component'],
                   signal_mode=self.signal_mode, cached_feature_sec=time.monotonic() - started,
                   classifier_sec=0., verification=None, score_kind=self.score_kind,
                   vector_kind='gmr_sensor_ratio_and_phase', visual_matching={'status': 'disabled'})
        self.step += 1
        return row, np.array([row['anomaly_score'], phase]), None


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
                sampler = self.detector.sampler
                if getattr(self.detector, 'release_check', False) and sampler.release is None and sampler.clock.stage == 3:
                    self.update(gmr_release=self.detector.check_release(None))
                return None
            timestamp, kind, payload = packet
            if self.stop.wait(max(0., self.origin + timestamp - self.clock())):
                return None
            if kind == 'camera':
                self.publish(decode_image_message(*payload), timestamp)
            elif kind == 'stage':
                self.update(gmr_stage=payload)
            elif kind == 'release':
                self.update(gmr_release=self.detector.check_release(payload))
            else:
                self.detector.pending = payload
                self.index += 1
                return self.index - 1, timestamp, None
        return None

    def close(self):
        if hasattr(self.packets, 'close'):
            self.packets.close()
