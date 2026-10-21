"""Adapters for existing DINO scoring and explicitly offline GMR playback."""

import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image


def check_stop(stop):
    if stop.is_set():
        raise InterruptedError('Detector preparation stopped.')


class DinoReplayDetector:
    kind = 'dino'
    score_kind = 'dino_standardized_distance'
    score_source = 'dino_nominal_patch_evidence'
    note = 'DINO scores every sampled image at the existing DINO sampling rate, through the end of the recording. No VLM interval or eight-frame limit applies. If processing is slower than playback, replay falls behind instead of skipping inputs. Recorded-stage restrictions use retrospective bag annotations.'
    knn = True  # Common renderer: distance rather than probability.
    capture_visual = False

    def __init__(self, config, progress, topic=None, extractor=None):
        from .run import _slug
        from vision_dinov2 import run_vision_test as settings
        from vision_dinov2.nominal_cache import load_nominal_cache
        from vision_dinov2.dinov2_features import DINOv2FeatureExtractor
        from vision_dinov2.threshold import validate_threshold_scale
        from vision_dinov2.model_settings import model_settings
        from rosbag_io import RosbagImageSource
        self.config = config
        self.threshold_scale = validate_threshold_scale(config.get('dino_threshold_scale', 1.0))
        if "dino_score_mode" in config:
            progress("dino_score_mode is retired; using unified DINO attention scoring.")
        self.topic = topic or (config.get('camera_topics') or config.get('rynnbrain', {}).get('camera_topics', []))[0]
        self.frame_count = None  # Run through the recording, independently of VLM sampling.
        self.path = Path(config.get('nominal_cache_dir', '/outputs/nominal_cache'))
        if config.get('vision_memory_name'):
            self.path /= config['vision_memory_name']
        self.path /= _slug(self.topic)
        self.signature = settings.nominal_cache_runtime_signature()
        model_name, input_size = model_settings(config, settings.DINO_INPUT_SIZE)
        self.signature.update(model_name=model_name, dino_input_size=input_size,
                              stage_constrained_matching=config.get('stage_constrained_matching', False))
        calibration_bags = config.get('dino_calibration_bags', [])
        self.signature['calibration_sources'] = settings.source_signatures([
            RosbagImageSource(path, self.topic,
                stage_topic=None if config.get('ignore_recorded_stages', False) else config.get('stage_topic', '/recording_stage'),
                stage_count=config.get('stage_count', 3)) for path in calibration_bags])
        self.sample_fps = self.signature['sample_fps']
        progress('Loading saved DINO nominal memory')
        loaded = load_nominal_cache(self.path, self.signature)
        if loaded is None:
            raise ValueError(f'No compatible DINO calibration at {self.path}. See the setting differences in the terminal. Stop replay, run docker compose run --rm --build vision-memory with the same config, then restart online-replay. Compatible extracted features will be reused when rebuild_nominal_cache is false.')
        self.memory, self.grid, self.thresholds, _ = loaded
        saved = json.loads((self.path / 'signature.json').read_text(encoding='utf-8'))
        self.signature['calibration_method'] = saved.get('calibration_method', 'legacy_frame_quantile')
        self.signature['calibration_sources'] = saved.get('calibration_sources', [])
        self.signature['dino_threshold_scale'] = self.threshold_scale
        calibration_label = 'separate nominal recordings' if calibration_bags else 'leave-one-recording-out'
        self.note += (f" Patch differences are standardized by learned nominal variation. "
                      "The score averages the largest 1% of 3x3-pooled responses. "
                      f"Thresholds use per-recording progress quantiles from {calibration_label}. "
                      f"Threshold multiplier {self.threshold_scale:g}. "
                      "Calibration recordings are excluded from evaluation.")
        self.sources = saved.get('sources', [])
        if not self.sources or any(source.get('topic') != self.topic or not source.get('bag_path') for source in self.sources):
            raise ValueError('DINO replay requires a named rosbag memory for the selected camera.')
        if self.signature['stage_constrained_matching'] and any(
                source.get('stage_count') != 3 or source.get('stage_topic') != config.get('stage_topic', '/recording_stage')
                for source in self.sources):
            raise ValueError('DINO memory stage settings do not match the three-stage replay annotations.')
        self.excluded = {Path(item['bag_path']).name for item in self.sources if 'bag_path' in item}
        self.excluded.update(Path(item['bag_path']).name for item in saved.get('calibration_sources', []) if 'bag_path' in item)
        progress('Loading DINO model')
        self.extractor = extractor or DINOv2FeatureExtractor(self.signature['model_name'], self.signature['dino_input_size'])
        self.preview_image = self.matched_image = None
        self.thumbnails = {}

    def reset(self, bag):
        from .visual_matching import read_recorded_stages
        if Path(bag).name in self.excluded:
            raise ValueError('Select a bag outside the DINO memory and threshold calibration recordings.')
        self.step = 0
        self.stages = None
        self.start_sec = 0.
        if self.signature['stage_constrained_matching']:
            if self.config.get('ignore_recorded_stages', False):
                raise ValueError('Stage-constrained DINO replay requires recorded stages.')
            self.stages = read_recorded_stages(bag, self.config.get('stage_topic', '/recording_stage'))
            self.start_sec = self.stages.starts[0]

    def _thumbnail(self, item):
        from rosbag_io import RosbagImageSource, iter_rosbag_image_frames
        video_id, frame_id = int(item['video_id']), int(item['frame_id'])
        key = video_id, frame_id
        source = self.sources[video_id]
        if key not in self.thumbnails:
            frames = iter_rosbag_image_frames(RosbagImageSource(source['bag_path'], source['topic']),
                sample_fps=30., start_sec=max(0., float(item['timestamp_sec']) - 1e-6),
                end_sec=float(item['timestamp_sec']) + 1e-6)
            try:
                packet = next(frames, None)
            finally:
                frames.close()
            if packet is None:
                raise ValueError('Matched nominal frame is unavailable in its source bag.')
            preview = packet[2].copy()
            preview.thumbnail((640, 480))
            self.thumbnails[key] = preview
        return source['bag_path'], self.thumbnails[key]

    def append(self, image, timestamp, verify=False):
        from .online_replay import synchronized_time
        from vision_dinov2.anomaly_detection import compare_frame_to_nominal_memory
        from vision_dinov2.visualization import make_heatmap_overlay
        from vision_dinov2.threshold import score_matched_frame
        started = synchronized_time()
        features, attention, cls, grid = self.extractor.extract(image)
        feature_sec = synchronized_time() - started
        started = time.monotonic()
        stage = self.stages.at(timestamp) if self.stages else None
        score, patches, weighted, weights, best, distance = compare_frame_to_nominal_memory(
            features, attention, cls, self.memory, top_k_cls=self.signature['top_k_cls'],
            attention_power=self.signature['attention_power'], min_weight=self.signature['min_weight'],
            required_stage=stage)
        decision = score_matched_frame(score, float(best['progress']), self.thresholds,
                                      getattr(self, 'threshold_scale', 1.0),
                                      patches=patches, attention_weights=weights)
        score, threshold, base_threshold = decision['score'], decision['threshold'], decision['base_threshold']
        scoring_sec = time.monotonic() - started
        self.preview_image = Image.fromarray(make_heatmap_overlay(image, patches, grid))
        scored_map = decision["heatmap"] if decision["heatmap"] is not None else weighted
        self.scored_preview_image = Image.fromarray(make_heatmap_overlay(image, scored_map, grid,
            scale_max=2 * threshold if decision["heatmap"] is not None else None))
        self.attention_preview_image = Image.fromarray(make_heatmap_overlay(image, weights, grid))
        self.matched_image = None
        match = {'status': 'unavailable', 'stage': stage}
        try:
            bag, self.matched_image = self._thumbnail(best)
            match.update(status='matched', candidate_count=sum(stage is None or x['stage'] == stage for x in self.memory),
                         best=dict(bag=bag, frame_id=int(best['frame_id']), timestamp_sec=float(best['timestamp_sec']),
                                   stage=int(best['stage']), cosine_distance=float(distance)))
        except (OSError, ValueError, KeyError, IndexError) as error:
            match['reason'] = str(error)
        row = dict(sample_index=self.step, timestamp_sec=timestamp, anomaly_score=float(score),
                   threshold=float(threshold), above_threshold=bool(score > threshold),
                   base_threshold=float(base_threshold), threshold_scale=getattr(self, 'threshold_scale', 1.0),
                   cached_feature_sec=feature_sec, classifier_sec=scoring_sec, verification=None,
                   score_kind=self.score_kind, visual_matching=match,
                   vector_kind='dino_cls',
                   epsilon=float(score / max(threshold, 1e-8)))
        row.update({key: value for key, value in decision.items()
                    if key not in {'score', 'threshold', 'base_threshold', 'epsilon', 'heatmap'}})
        self.step += 1
        return row, cls, None


def wrench_topic(bag):
    from rosbag_io import list_bag_topics
    topics = list_bag_topics(bag)
    supported = topics.loc[topics.message_type.isin(['geometry_msgs/msg/Wrench', 'geometry_msgs/msg/WrenchStamped']), 'topic'].unique().tolist()
    if '/tcp_force' in supported:
        return '/tcp_force'
    if len(supported) == 1:
        return supported[0]
    if not supported:
        raise ValueError(f'GMR needs force/torque Wrench data; no supported topic exists in {bag}. Pose and joint data cannot substitute for this model.')
    raise ValueError(f'Multiple wrench topics in {bag}: {supported}. Select a common force/torque topic before GMR preparation.')


def synchronized_wrench(bag, topic, settings):
    from rosbag_io import read_numeric_topic
    from time_series_gmr_scripts.ts_data import sync_and_preprocess_ts_dataframe
    raw = read_numeric_topic(bag, topic)
    processed, info, _ = sync_and_preprocess_ts_dataframe(raw,
        resample_length=settings.RESAMPLE_LENGTH, baseline_fraction=settings.BASELINE_FRACTION,
        threshold_std_factor=settings.THRESHOLD_STD_FACTOR, min_force_change=settings.MIN_FORCE_CHANGE,
        padding_seconds=settings.PADDING_SECONDS)
    # Legacy preprocessing resets time at the trimmed segment. Restore bag time for video alignment.
    trimmed = raw if info['used_full_trajectory'] else raw.loc[raw.time.between(info['start_time'], info['end_time'])]
    processed['time'] += float(trimmed.time.iloc[0])
    return processed


GMR_SIGNALS = {'pose': 'TCP pose', 'joints': 'Joint positions', 'gripper': 'Gripper position',
               'pose_gripper': 'TCP pose + gripper position', 'force': 'Force / torque'}


def gmr_signal_topics(bag, mode):
    """Select typed topics explicitly, rather than treating arbitrary TS as force."""
    from rosbag_io import list_bag_topics
    if mode == 'force':
        return {'force': wrench_topic(bag)}
    if mode not in GMR_SIGNALS:
        raise ValueError('Select a supported GMR signal in the GUI.')
    topics = list_bag_topics(bag)
    kinds = ['pose', 'gripper'] if mode == 'pose_gripper' else [mode]
    specifications = {
        'pose': ('/tcp_pose_broadcaster/pose', ['geometry_msgs/msg/PoseStamped']),
        'joints': ('/joint_states', ['sensor_msgs/msg/JointState']),
        'gripper': ('/Robotiq2FGripperRobotInputMsg',
                    ['robotiq_hand_e_gripper_interfaces/msg/Robotiq2FGripperRobotInput']),
    }
    selected = {}
    for kind in kinds:
        preferred, types = specifications[kind]
        candidates = topics.loc[topics.message_type.isin(types) & (topics.message_count > 0), 'topic'].unique().tolist()
        if preferred in candidates:
            selected[kind] = preferred
        elif len(candidates) == 1:
            selected[kind] = candidates[0]
        else:
            raise ValueError(f'{bag}: {kind} needs one unambiguous populated topic; found {candidates}. '
                             'Select another TS signal; missing channels are not substituted.')
    return selected


def prepare_gmr_signals(bag, topics, settings, config):
    if set(topics) == {'force'}:
        return synchronized_wrench(bag, topics['force'], settings)
    from rosbag_io import read_numeric_topic
    from .visual_matching import read_recorded_stages
    from time_series_gmr_scripts.ts_data import preprocess_replay_signals
    if config.get('ignore_recorded_stages', False) or config.get('stage_count', 3) != 3:
        raise ValueError('Pose/joint/gripper GMR requires the recorded three-stage annotations.')
    stages = read_recorded_stages(bag, config.get('stage_topic', '/recording_stage'))
    return preprocess_replay_signals({kind: read_numeric_topic(bag, topic) for kind, topic in topics.items()},
                                     stages, settings.RESAMPLE_LENGTH)


class GmrReplayDetector:
    kind = 'gmr'
    score_kind = 'gmr_epsilon'
    score_source = 'offline_gmr_smoothed_epsilon'
    note = 'GMR OFFLINE PLAYBACK: every prepared TS sample is replayed at its timestamp. Whole-recording alignment, resampling and centered smoothing use future data. Epsilon is distance divided by the adaptive threshold, then smoothed; its threshold is 1. The alarm uses the existing majority-vote filter. Lookup time is not online inference latency.'
    knn = True
    capture_visual = False

    def __init__(self, config, progress):
        self.config = config
        self.frame_count = None
        self.nominals = list(dict.fromkeys(config.get('nominal_bags', [])))
        self.excluded = {Path(bag).name for bag in self.nominals}
        self.path = Path(config['output_dir']) / 'online_replay' / 'gmr_offline'
        self.signature = {'protocol': 'existing_offline_gmr_v1'}
        self.model = None
        self.topic = None
        self.training_identity = None
        self.signal_mode = 'force'  # GUI explicitly chooses a signal; legacy direct callers keep force.
        self.preview_image = self.matched_image = None

    def reset(self, bag):
        if Path(bag).name in self.excluded:
            raise ValueError('Select a bag outside GMR nominal training data.')
        self.bag = bag
        self.step = 0

    def prepare_replay(self, progress, stop):
        import pandas as pd
        from time_series_gmr_scripts import run_time_series_test as settings
        from time_series_gmr_scripts.anomaly_detection import GMRAnomalyDetector
        from time_series_gmr_scripts.threshold import calibrate_gmr_leave_one_run_out, apply_adaptive_gmr_threshold
        from .visual_matching import _source_identity
        if len(self.nominals) < 2:
            raise ValueError('GMR calibration requires at least two nominal_bags.')
        check_stop(stop)
        topics = gmr_signal_topics(self.nominals[0], self.signal_mode)
        for bag in [*self.nominals[1:], self.bag]:
            check_stop(stop)
            if gmr_signal_topics(bag, self.signal_mode) != topics:
                raise ValueError(f'{bag}: TS topics differ from nominal training topics {topics}.')
        topic = next(iter(topics.values())) if len(topics) == 1 else ', '.join(topics.values())
        progress(f'GMR signal: {GMR_SIGNALS[self.signal_mode]} ({topic})')
        processed = prepare_gmr_signals(self.bag, topics, settings, self.config)
        output_columns = sorted(c for c in processed if c not in {'time', 'phase'})
        if self.signal_mode == 'force':
            output_columns = settings.OUTPUT_COLUMNS
        pose_frame = processed.attrs.get('pose_frame')
        parameters = {name: value for name, value in vars(settings).items() if name.isupper() and isinstance(value, (int, float, str, list))}
        # Paths/legacy experiment selection are supplied by the current GUI config instead.
        parameters.pop('NOMINAL_BAG_PATHS', None)
        parameters.pop('TOPIC_NAME', None)
        parameters['OUTPUT_COLUMNS'] = output_columns
        if self.signal_mode != 'force':
            for name in ('BASELINE_FRACTION', 'THRESHOLD_STD_FACTOR', 'MIN_FORCE_CHANGE', 'PADDING_SECONDS'):
                parameters.pop(name, None)
        protocol = 'existing_offline_gmr_v1' if self.signal_mode == 'force' else 'stage_aligned_ts_gmr_v1'
        identity = dict(protocol=protocol, topics=topics, signal_mode=self.signal_mode,
                        output_columns=output_columns, pose_frame=pose_frame,
                        stage_topic=self.config.get('stage_topic', '/recording_stage'), parameters=parameters,
                        sources=[dict(bag=str(Path(b).resolve()), files=_source_identity(Path(b))) for b in [*self.nominals, str(self.bag)]])
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        training_identity = json.dumps({**identity, 'sources': identity['sources'][:-1]}, sort_keys=True)
        self.path.mkdir(parents=True, exist_ok=True)
        csv_path = self.path / f'{key}.csv'
        if csv_path.is_file():
            progress('Loading saved offline GMR scores')
            self.results = pd.read_csv(csv_path)
        else:
            if self.model is None or self.training_identity != training_identity:
                nominals = []
                for i, bag in enumerate(self.nominals):
                    check_stop(stop)
                    progress(f'GMR offline preparation: nominal {i + 1}/{len(self.nominals)}')
                    nominal = prepare_gmr_signals(bag, topics, settings, self.config)
                    if sorted(c for c in nominal if c not in {'time', 'phase'}) != sorted(output_columns):
                        raise ValueError(f'{bag}: TS channels differ from the selected execution. No channels are silently dropped.')
                    if nominal.attrs.get('pose_frame') != pose_frame:
                        raise ValueError(f'{bag}: TCP pose reference frame differs from the selected execution.')
                    nominals.append(nominal)
                check_stop(stop)
                progress('GMR offline nominal calibration')
                self.thresholds, _ = calibrate_gmr_leave_one_run_out(nominals,
                    input_columns=settings.INPUT_COLUMNS, output_columns=output_columns,
                    n_components=settings.N_COMPONENTS, threshold_margin=settings.COMPONENT_THRESHOLD_MARGIN,
                    random_state=settings.RANDOM_STATE, n_bins=settings.THRESHOLD_BINS,
                    quantile=settings.THRESHOLD_QUANTILE, margin=settings.THRESHOLD_MARGIN,
                    min_points_per_bin=settings.MIN_POINTS_PER_BIN,
                    threshold_floor_quantile=settings.THRESHOLD_FLOOR_QUANTILE,
                    smooth_threshold_window=settings.THRESHOLD_SMOOTH_WINDOW)
                self.model = GMRAnomalyDetector(settings.INPUT_COLUMNS, output_columns,
                    n_components=settings.N_COMPONENTS, threshold_margin=settings.COMPONENT_THRESHOLD_MARGIN,
                    random_state=settings.RANDOM_STATE)
                self.model.fit(nominals)
                self.topic = topic
                self.training_identity = training_identity
            check_stop(stop)
            progress('GMR offline preparation: synchronizing and scoring the whole test bag')
            self.results = apply_adaptive_gmr_threshold(self.model.score_dataframe(processed), self.thresholds,
                score_smooth_window=settings.SCORE_SMOOTH_WINDOW, majority_window=settings.MAJORITY_WINDOW,
                majority_votes=settings.MAJORITY_VOTES)
            temporary = csv_path.with_suffix('.tmp')
            self.results.to_csv(temporary, index=False)
            temporary.replace(csv_path)
            csv_path.with_suffix('.json').write_text(json.dumps(identity, indent=2), encoding='utf-8')
        check_stop(stop)
        required = ['time', 'epsilon_smooth', 'mahalanobis', 'phase', 'alarm_filtered',
                    'adaptive_threshold_mahalanobis', 'epsilon_adaptive']
        if (not set(required).issubset(self.results.columns) or len(self.results) < 2
                or not np.isfinite(self.results[required].to_numpy(dtype=float)).all()
                or (np.diff(self.results.time) <= 0).any()):
            raise ValueError('Invalid cached GMR results; remove the reported GMR CSV and prepare it again.')
        self.start_sec = float(self.results.time.iloc[0])
        self.end_sec = float(self.results.time.iloc[-1])
        self.signature = dict(protocol=protocol, topics=topics, signal_mode=self.signal_mode,
                              output_columns=output_columns, pose_frame=pose_frame, result_csv=str(csv_path))
        self.signal_description = GMR_SIGNALS[self.signal_mode]
        return {'status': 'ready', 'offline': True, **self.signature,
                'signal_description': self.signal_description}

    def append(self, image, timestamp, verify=False):
        started = time.monotonic()
        index = int(np.searchsorted(self.results.time.to_numpy(), timestamp, side='right') - 1)
        if index < 0 or timestamp > self.end_sec + 1e-6:
            raise ValueError('No GMR sample available at this video timestamp.')
        sample = self.results.iloc[index]
        score = float(sample.epsilon_smooth)
        row = dict(sample_index=self.step, timestamp_sec=timestamp, anomaly_score=score, threshold=1.,
                   above_threshold=bool(score > 1.), alarm_filtered=bool(sample.alarm_filtered),
                   raw_distance=float(sample.mahalanobis),
                   adaptive_threshold=float(sample.adaptive_threshold_mahalanobis),
                   epsilon_raw=float(sample.epsilon_adaptive),
                   signal_time_sec=float(sample.time), cached_feature_sec=time.monotonic() - started,
                   signal_mode=self.signal_mode,
                   classifier_sec=0., verification=None, score_kind=self.score_kind,
                   vector_kind='gmr_mahalanobis_and_phase',
                   visual_matching={'status': 'disabled'})
        self.step += 1
        return row, np.array([sample.mahalanobis, sample.phase]), None
