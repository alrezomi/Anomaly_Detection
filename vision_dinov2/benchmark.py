"""Selected-folder DINO benchmark; shared extraction, independent metric calibration."""
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
import hashlib
import json
import os
import re

import numpy as np
import pandas as pd


def select_bags(data_root, folders, excluded):
    """Scan only explicit, nonoverlapping subfolders. Paths cannot escape the data root."""
    root = Path(data_root).resolve(strict=True)
    if not isinstance(folders, list) or not folders or any(not isinstance(x, str) or not x.strip() for x in folders):
        raise ValueError('Set dino_benchmark.folders to a nonempty list, or repeat --folder. No whole-root default is used.')
    selected = []
    for value in folders:
        path = (root / value).resolve(strict=True)
        if not path.is_dir() or path == root or not path.is_relative_to(root):
            raise ValueError(f'Select a folder strictly inside {root}: {value}')
        if any(path.is_relative_to(other) or other.is_relative_to(path) for other in selected):
            raise ValueError(f'Selected folders overlap: {value}')
        selected.append(path)
    excluded_paths = set()
    for value in excluded:
        path = Path(value).resolve()
        if path.is_file() and (path.parent/'metadata.yaml').is_file():
            path = path.parent
        excluded_paths.add(path)
    rows = []
    for folder in selected:
        bags = [folder] if (folder / 'metadata.yaml').is_file() else sorted({p.parent.resolve() for p in folder.rglob('metadata.yaml')})
        for bag in bags:
            if not bag.is_relative_to(folder):
                raise ValueError(f'Bag resolves outside selected folder: {bag}')
            rows.append(dict(bag_path=str(bag), bag_name=bag.name, folder=str(folder.relative_to(root)),
                             status='excluded_training' if bag in excluded_paths else 'ok'))
    if not any(row['status'] == 'ok' for row in rows):
        raise ValueError('No eligible ROS 2 bags in the selected folders after excluding memory/calibration bags.')
    return rows, [str(path.relative_to(root)) for path in selected]


def annotate_bags(rows, stage_topic, startup_ignore):
    from build_dataset_manifest import infer_bag_record
    from .benchmark_alarm import read_execution_annotation
    result = []
    for row in rows:
        if row['status'] != 'ok':
            result.append(dict(row, ground_truth_label='unknown', recorded_failure_time_sec=None))
            continue
        label = infer_bag_record(Path(row['bag_path']), stage_topic, startup_ignore)
        onset = read_execution_annotation(row['bag_path'], stage_topic)
        result.append(dict(row, ground_truth_label=label['label'], label_read_status=label['read_status'],
            used_stage_markers=label['used_stage_markers'], recorded_failure_time_sec=onset['time_sec'],
            annotation_status=onset['status'], annotation_error=onset.get('error'),
            failure_marker=onset.get('marker'), label_onset_conflict=label['label']=='normal' and onset['time_sec'] is not None,
            **{key: onset.get(key) for key in ['stage1_start_sec', 'stage3_start_sec', 'recorded_stage_messages',
                'stage_message_count', 'ignored_pre_stage1_failure_count']}))
    return result


def slug(value):
    return re.sub(r'[^a-zA-Z0-9]+', '_', value).strip('_').lower()


def _signature(config, topic, metric):
    from . import run_vision_test as settings
    from .model_settings import model_settings
    from rosbag_io import RosbagImageSource
    signature = settings.nominal_cache_signature()
    model, size = model_settings(config, settings.DINO_INPUT_SIZE)
    stage_topic = None if config.get('ignore_recorded_stages', False) else config.get('stage_topic', '/recording_stage')
    def sources(key):
        return [RosbagImageSource(path, topic, stage_topic=stage_topic, stage_count=config.get('stage_count', 3))
                for path in config.get(key, [])]
    nominal, calibration = sources('nominal_bags'), sources('dino_calibration_bags')
    signature.update(model_name=model, dino_input_size=size, dino_metric=metric,
                     sources=settings.source_signatures(nominal),
                     calibration_sources=settings.source_signatures(calibration),
                     stage_constrained_matching=bool(config.get('stage_constrained_matching', False)))
    return signature, nominal, calibration


def _load_calibration(directory, signature, patch_count):
    """Benchmark calibration has its own directory; never duplicate large feature arrays."""
    from .patch_normality import validate_patch_profile
    directory = Path(directory)
    try:
        saved = json.loads((directory / 'signature.json').read_text(encoding='utf-8'))
        if not isinstance(saved, dict):
            print(f'Calibration cache rejected ({directory}): signature is not an object.', flush=True)
            return None
        if saved.get('dino_metric') == 'mahalanobis':
            saved.setdefault('attention_weighting', 'direct')
        from .threshold import ATTENTION_CALIBRATION_METHOD, build_attention_threshold
        convert_margin = (saved.get('calibration_method') == ATTENTION_CALIBRATION_METHOD
                          and signature.get('calibration_method') == ATTENTION_CALIBRATION_METHOD
                          and saved.get('threshold_margin') == 1.1
                          and signature.get('threshold_margin') == 1.)
        if convert_margin:
            saved['threshold_margin'] = 1.
        if saved != signature:
            keys = sorted(key for key in set(saved) | set(signature) if saved.get(key) != signature.get(key))
            print(f'Calibration cache rejected ({directory}); different settings: {", ".join(keys)}.', flush=True)
            for key in keys:
                if key in {'sources', 'calibration_sources'}:
                    print(f'  {key}: saved {len(saved.get(key, []))} sources; requested {len(signature.get(key, []))}; paths/order/settings must match.', flush=True)
                    for old, new in zip(saved.get(key, []), signature.get(key, [])):
                        if old != new:
                            print(f'    first differing source: saved={old!r}; requested={new!r}', flush=True)
                            break
                else:
                    print(f'  {key}: saved={saved.get(key)!r}; requested={signature.get(key)!r}', flush=True)
            return None
        with np.load(directory / 'patch_normality.npz', allow_pickle=False) as arrays:
            profile = {key: arrays[key] for key in arrays.files}
        validate_patch_profile(profile, patch_count)
        if str(profile.get('metric', 'euclidean')) != signature['dino_metric']:
            return None
        table = pd.read_csv(directory / 'adaptive_threshold.csv')
        if convert_margin:
            table, _ = build_attention_threshold(pd.read_csv(directory/'calibration.csv'),
                n_bins=signature.get('threshold_bins', 30),
                min_points_per_bin=signature.get('min_points_per_bin', 2),
                smooth_window=signature.get('threshold_smooth_window', 3),
                threshold_floor_quantile=signature.get('threshold_floor_quantile', .20))
            print(f'Reused benchmark calibration scores without the legacy 1.1 margin: {directory}', flush=True)
        centers = table['stage_center'].to_numpy(float)
        if (not len(centers) or not np.isfinite(centers).all() or (np.diff(centers) <= 0).any()
                or (centers < 0).any() or (centers > 1).any()):
            return None
        for name in ['adaptive_threshold', 'threshold_pooled']:
            if name not in table or not np.isfinite(table[name]).all() or (table[name] < 0).any():
                return None
        table.attrs['patch_profile'] = profile
        table.attrs['score_mode'] = 'attention'
        return table
    except (OSError, ValueError, KeyError, TypeError):
        import sys
        print(f'Calibration cache unavailable ({directory}): {sys.exc_info()[1]}', flush=True)
        return None


def prepare_camera(config, topic, get_extractor, output):
    from . import run_vision_test as settings
    from .anomaly_detection import build_nominal_video_memory
    from .nominal_cache import load_feature_cache, save_feature_cache, save_nominal_calibration, memory_signature
    from .threshold import build_attention_threshold
    from .threshold import calibrate_attention_leave_one_video_out, score_calibration_memory
    from .patch_normality import fit_patch_profile

    started = perf_counter()
    signature, sources, calibration = _signature(config, topic, 'euclidean')
    root = Path(config.get('nominal_cache_dir') or Path(config['output_dir']).parent / 'nominal_cache')
    if config.get('vision_memory_name'):
        root /= config['vision_memory_name']
    root /= slug(topic)
    rebuild = bool(config.get('rebuild_nominal_cache', False))
    print(f'Nominal cache: {root}; rebuild_nominal_cache={rebuild}', flush=True)
    cached = None if rebuild else load_feature_cache(root, signature)
    if cached is not None:
        memory, grid = cached
        feature_status = 'reused'
    else:
        memory, grid = build_nominal_video_memory(get_extractor(), sources,
            sample_fps=settings.SAMPLE_FPS, start_sec=settings.START_SEC, end_sec=settings.END_SEC,
            max_frames_per_video=settings.MAX_NOMINAL_FRAMES_PER_VIDEO)
        save_feature_cache(root, memory, grid, signature)
        feature_status = 'extracted'
    timings = [dict(camera_topic=topic, phase='nominal_features', seconds=perf_counter()-started,
                    status=feature_status, nominal_frames=len(memory))]
    matcher = dict(top_k_cls=signature['top_k_cls'], attention_power=signature['attention_power'],
                   min_weight=signature['min_weight'], stage_constrained_matching=signature['stage_constrained_matching'])
    bounds = dict(n_bins=signature['threshold_bins'], min_points_per_bin=signature['min_points_per_bin'],
                  smooth_window=signature['threshold_smooth_window'],
                  threshold_floor_quantile=signature['threshold_floor_quantile'])
    tables, signatures = {}, {}
    for metric in ('euclidean', 'mahalanobis'):
        started = perf_counter()
        metric_signature = dict(signature, dino_metric=metric)
        signatures[metric] = metric_signature
        directory = root / 'benchmark_calibration' / metric
        table = None if rebuild else _load_calibration(directory, metric_signature, grid*grid)
        status = 'reused'
        if table is None and not rebuild:
            table = _load_calibration(root, metric_signature, grid*grid)
            if table is not None:
                status = 'reused_existing_memory_calibration'
                directory.mkdir(parents=True, exist_ok=True)
                save_nominal_calibration(directory, table, pd.read_csv(root/'calibration.csv'), metric_signature)
        if table is None:
            print(f'{metric}: no compatible calibration was found; fitting and saving {directory}.', flush=True)
            status = 'calibrated'
            if not calibration:
                table, rows = calibrate_attention_leave_one_video_out(memory, metric=metric, **matcher, **bounds)
            else:
                profile = fit_patch_profile(memory, metric=metric, n_bins=bounds['n_bins'], **matcher)
                rows = []
                for run_id, source in enumerate(calibration):
                    query_signature = memory_signature(dict(metric_signature, sources=settings.source_signatures([source])))
                    key = hashlib.sha256(json.dumps(query_signature, sort_keys=True).encode()).hexdigest()[:24]
                    query_directory = root / 'calibration_features' / key
                    query_cache = None if rebuild and metric == 'euclidean' else load_feature_cache(query_directory, query_signature)
                    if query_cache is None:
                        queries, query_grid = build_nominal_video_memory(get_extractor(), [source],
                            sample_fps=settings.SAMPLE_FPS, start_sec=settings.START_SEC, end_sec=settings.END_SEC,
                            max_frames_per_video=settings.MAX_NOMINAL_FRAMES_PER_VIDEO)
                        save_feature_cache(query_directory, queries, query_grid, query_signature)
                    else:
                        queries, query_grid = query_cache
                    if grid != query_grid:
                        raise ValueError('Calibration feature grid differs from nominal memory.')
                    rows.extend(score_calibration_memory(queries, memory, run_id, patch_profile=profile, **matcher))
                    del queries
                table, rows = build_attention_threshold(pd.DataFrame(rows), **bounds)
                table.attrs['patch_profile'] = profile
            directory.mkdir(parents=True, exist_ok=True)
            save_nominal_calibration(directory, table, rows, metric_signature)
        tables[metric] = table
        print(f'{metric}: calibration {status} in {perf_counter()-started:.1f}s; cache: {directory}', flush=True)
        table.to_csv(output / f'{slug(topic)}_{metric}_threshold.csv', index=False)
        timings.append(dict(camera_topic=topic, metric=metric, phase='calibration',
                            seconds=perf_counter()-started, status=status, cache_directory=str(directory)))
    return memory, tables, signature, timings, signatures


def score_bag(config, topic, bag, memory, tables, signature, extractor):
    from .dinov2_features import iter_source_frames
    from .anomaly_detection import compare_frame_to_nominal_memory
    from .threshold import score_matched_frame
    from rosbag_io import RosbagImageSource
    from rynnbrain_vlm.visual_matching import read_recorded_stages
    import torch

    def clock():
        if str(extractor.device).startswith('cuda'):
            torch.cuda.synchronize(extractor.device)
        return perf_counter()

    stages = read_recorded_stages(bag, config.get('stage_topic', '/recording_stage')) if signature['stage_constrained_matching'] else None
    source = RosbagImageSource(bag, topic)
    rows = []
    iterator = iter_source_frames(source, sample_fps=signature['sample_fps'],
        start_sec=stages.starts[0] if stages else signature['start_sec'], end_sec=signature['end_sec'])
    try:
        while True:
            read_started = perf_counter()
            packet = next(iterator, None)
            read_sec = perf_counter()-read_started
            if packet is None:
                break
            frame_id, timestamp, image = packet
            started = clock()
            features, attention, cls, grid = extractor.extract(image)
            feature_sec = clock()-started
            started = perf_counter()
            raw, patches, _, weights, best, _ = compare_frame_to_nominal_memory(features, attention, cls, memory,
                top_k_cls=signature['top_k_cls'], attention_power=signature['attention_power'],
                min_weight=signature['min_weight'], required_stage=stages.at(timestamp) if stages else None)
            match_sec = perf_counter()-started
            shared = dict(bag_path=str(bag), camera_topic=topic, frame_id=frame_id, timestamp_sec=float(timestamp),
                matched_progress=float(best['progress']), matched_video_id=int(best['video_id']),
                matched_frame_id=int(best['frame_id']), grid=int(grid), feature_sec=feature_sec,
                comparison_sec=match_sec, read_decode_sec=read_sec)
            for metric, table in tables.items():
                started = perf_counter()
                decision = score_matched_frame(raw, float(best['progress']), table,
                    patches=patches, attention_weights=weights,
                    query_features=features, reference_features=best['features'])
                metric_sec = perf_counter()-started
                pooled = decision['score_variants']['pooled']
                rows.append(dict(shared, metric=metric, pooled_score=pooled['score'], base_threshold=pooled['threshold'],
                                 metric_score_sec=metric_sec, processing_sec=feature_sec+match_sec+metric_sec))
    finally:
        iterator.close()
    if not rows:
        raise ValueError(f'No sampled images for {topic} in {bag}')
    return rows


def run(config, folders, data_root='/data', output_root=None, labels_csv=None):
    """Build/reuse both metric memories, then score exactly the selected bags."""
    from .threshold import validate_threshold_scale
    from rosbag_io import resolve_bag_path
    from .benchmark_report import write_reports
    from .benchmark_report import REPORT_FORMAT_VERSION
    from .dinov2_features import DINOv2FeatureExtractor
    from .model_settings import model_settings
    from .benchmark_labels import apply_csv_labels, validate_ground_truth, update_labels_csv
    from .benchmark_alarm import ensure_stage_windows, ALARM_POLICY

    config = deepcopy(config)
    config['output_dir'] = config.get('output_dir') or 'outputs'
    scale = validate_threshold_scale(config.get('dino_threshold_scale', 1.))
    nominal = config.get('nominal_bags', [])
    calibration = config.get('dino_calibration_bags', [])
    if not isinstance(nominal, list) or not isinstance(calibration, list):
        raise ValueError('nominal_bags and dino_calibration_bags must be lists.')
    nominal = [str(resolve_bag_path(path)) for path in nominal]
    calibration = [str(resolve_bag_path(path)) for path in calibration]
    if len(set(nominal)) != len(nominal) or len(set(calibration)) != len(calibration) or set(nominal) & set(calibration):
        raise ValueError('Nominal/calibration lists must be unique and disjoint.')
    if len(nominal) < (2 if calibration else 3):
        raise ValueError('Need three nominal bags, or two with separate calibration bags.')
    topics = config.get('camera_topics', [])
    if (not isinstance(topics, list) or not topics or any(not isinstance(x, str) or not x.strip() for x in topics)
            or len(set(topics)) != len(topics) or len({slug(x) for x in topics}) != len(topics)):
        raise ValueError('Set unique camera_topics with distinct cache names.')
    name = config.get('vision_memory_name', '')
    if name and slug(name) != name:
        raise ValueError('vision_memory_name must contain only lowercase letters, numbers and underscores.')
    if config.get('stage_constrained_matching') and config.get('ignore_recorded_stages'):
        raise ValueError('Stage-constrained matching requires recorded stages.')
    if config.get('stage_constrained_matching') and config.get('stage_count', 3) != 3:
        raise ValueError('DINO benchmark stage matching requires the recorded three-stage annotations.')
    config.update(nominal_bags=nominal, dino_calibration_bags=calibration)
    selected, folder_names = select_bags(data_root, folders, nominal+calibration)
    output = Path(output_root or Path(config['output_dir']) / 'dino_benchmark') / datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True, exist_ok=False)
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
    print(f'DINO benchmark report v{REPORT_FORMAT_VERSION}; source {source_hash}; fixed threshold margin=1 (sole gain: dino_threshold_scale).', flush=True)
    started = perf_counter()
    report = dict(status='running', report_format_version=REPORT_FORMAT_VERSION, benchmark_source_sha256=source_hash,
        selected_folders=folder_names, data_root=str(Path(data_root).resolve()),
        camera_topics=topics, configured_scale=scale, score_variant='pooled', bag_rule=ALARM_POLICY,
        stage_topic=config.get('stage_topic', '/recording_stage'),
        label_policy='existing final-three eligible stage messages',
        onset_policy='first failure marker at/after corrected stage 1; pre-stage-1 markers ignored; bag-relative time',
        camera_fusion='OR', retrospective_stages=bool(config.get('stage_constrained_matching')),
        roc_selection='exploratory Youden J on selected benchmark; do not treat tuned results as held-out validation',
        timing_note='Offline throughput, not live playback latency. Feature/matching times are shared by metrics. Phase durations can overlap (model load inside preparation/inference); actual_wall_sec is the total.')
    manifest, timings, frames = [], [], []
    extractor = None
    def get_extractor():
        nonlocal extractor
        if extractor is None:
            t = perf_counter()
            model, size = model_settings(config, 518)
            extractor = DINOv2FeatureExtractor(model, size)
            timings.append(dict(phase='model_load', seconds=perf_counter()-t))
        return extractor
    try:
        startup = float(os.environ.get('STAGE_STARTUP_IGNORE_SEC', '.1'))
        if not np.isfinite(startup) or startup < 0:
            raise ValueError('STAGE_STARTUP_IGNORE_SEC must be finite and nonnegative.')
        manifest = annotate_bags(selected, config.get('stage_topic', '/recording_stage'), startup)
        labels_csv = (labels_csv or (config.get('dino_benchmark') or {}).get('labels_csv')
                      or Path(os.environ.get('ANOMALY_OUTPUT_ROOT') or Path(config['output_dir']).parent)/'dataset_labels.csv')
        labels_csv = update_labels_csv(manifest, labels_csv, data_root)
        manifest = apply_csv_labels(manifest, labels_csv, data_root)
        report.update(labels_csv=str(labels_csv), label_policy='authoritative labels CSV; missing selected entries generated from recorded stages')
        pd.DataFrame(manifest).to_csv(output / 'selection.csv', index=False)
        counts = validate_ground_truth(manifest)
        report['ground_truth_counts'] = counts
        manifest = ensure_stage_windows(manifest, report['stage_topic'])
        pd.DataFrame(manifest).to_csv(output/'selection.csv', index=False)
        eligible = [item for item in manifest if item['status']=='ok']
        onset_count = sum(item.get('recorded_failure_time_sec') is not None for item in eligible)
        print(f'Timing preflight: {onset_count}/{len(eligible)} evaluated bags have a recorded failure onset at/after stage 1. '
              'Without an onset, only first-alarm time is available, not detection delay.', flush=True)
        print(f'Ground-truth preflight: {counts["normal"]} normal, {counts["fail"]} failure, {counts["excluded"]} excluded.', flush=True)
        report['signatures'] = {}
        for topic in topics:
            print(f'DINO benchmark: preparing {topic}', flush=True)
            memory, tables, signature, camera_timings, signatures = prepare_camera(config, topic, get_extractor, output)
            timings.extend(camera_timings)
            report['signatures'][topic] = signatures
            for index, item in enumerate(manifest):
                if item['status'] == 'excluded_training':
                    continue
                t = perf_counter()
                try:
                    rows = score_bag(config, topic, item['bag_path'], memory, tables, signature, get_extractor())
                    frames.extend(rows)
                    partial = output / 'frame_results_partial.csv'
                    pd.DataFrame(rows).to_csv(partial, mode='a', header=not partial.exists(), index=False)
                    print(f"{topic}: {index+1}/{len(manifest)} {item['bag_name']}: {len(rows)//2} images", flush=True)
                    timings.append(dict(phase='bag_inference', camera_topic=topic, bag_path=item['bag_path'],
                                        seconds=perf_counter()-t, frames=len(rows)//2, status='ok'))
                except Exception as error:
                    item['status'] = 'error'
                    item['error'] = (item.get('error', '') + f' {topic}: {type(error).__name__}: {error}').strip()
                    timings.append(dict(phase='bag_inference', camera_topic=topic, bag_path=item['bag_path'],
                                        seconds=perf_counter()-t, status='error', error=str(error)))
                    print(f"DINO benchmark error: {item['bag_path']}: {error}", flush=True)
            del memory, tables
        metrics, roc = write_reports(manifest, pd.DataFrame(frames), output, scale, folder_names)
        report.update(status='completed_with_errors' if any(x['status']=='error' for x in manifest) else 'completed',
                      roc=roc, actual_wall_sec=perf_counter()-started)
        print(metrics[['metric', 'operating_point', 'folder', 'recall', 'false_positive_rate']].to_string(index=False))
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        report['actual_wall_sec'] = perf_counter()-started
        pd.DataFrame(manifest or selected).to_csv(output / 'selection.csv', index=False)
        pd.DataFrame(timings).to_csv(output / 'runtime.csv', index=False)
        (output / 'run.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
        print(f'DINO benchmark outputs: {output}', flush=True)
    return output
