"""Regenerate reports from saved scores, refreshing stage metadata without inference."""
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
import json
import shutil

import numpy as np
import pandas as pd

from .benchmark_labels import apply_csv_labels, validate_ground_truth, update_labels_csv
from .benchmark_report import write_reports, REPORT_FORMAT_VERSION
from .benchmark_alarm import ensure_stage_windows, ALARM_POLICY, read_execution_annotation


def regenerate_reports(source, labels_csv=None, output_root=None, stage_topic=None):
    started = perf_counter()
    source = Path(source).resolve(strict=True)
    state = json.loads((source/'run.json').read_text(encoding='utf-8'))
    if state.get('score_variant') != 'pooled':
        raise ValueError('Recovery requires the saved pooled-score DINO benchmark run.')
    selection = pd.read_csv(source/'selection.csv', keep_default_na=False)
    manifest = selection.to_dict('records')
    for row in manifest:
        value = row.get('recorded_failure_time_sec')
        row['recorded_failure_time_sec'] = float(value) if value not in (None, '') else None
    if labels_csv:
        labels_csv = update_labels_csv(manifest, labels_csv, state['data_root'])
        manifest = apply_csv_labels(manifest, labels_csv, state['data_root'])
    counts = validate_ground_truth(manifest)
    stage_topic = stage_topic or state.get('stage_topic', '/recording_stage')
    for row in manifest:
        if row['status'] != 'ok':
            continue
        # Old onset extraction used a startup cutoff, not the corrected stage-1 start.
        # Refresh metadata only; never recompute image features or detector scores.
        annotation = read_execution_annotation(row['bag_path'], stage_topic)
        row['stage_annotation_refresh_error'] = annotation.get('error')
        if annotation.get('recorded_stage_messages'):
            row['recorded_stage_messages'] = annotation['recorded_stage_messages']
            row['stage_message_count'] = annotation.get('stage_message_count')
        if annotation.get('stage3_start_sec') is not None:
            for key in ['stage1_start_sec', 'stage3_start_sec', 'recorded_stage_messages',
                        'stage_message_count', 'ignored_pre_stage1_failure_count']:
                row[key] = annotation.get(key)
            if row.get('onset_source') in (None, '', 'recorded_stage_topic'):
                row.update(recorded_failure_time_sec=annotation['time_sec'],
                           annotation_status=annotation['status'], annotation_error=annotation.get('error'),
                           failure_marker=annotation.get('marker'), onset_source='recorded_stage_topic')
        row['label_onset_conflict'] = row['ground_truth_label']=='normal' and row.get('recorded_failure_time_sec') is not None
    manifest = ensure_stage_windows(manifest, stage_topic)
    frames = pd.read_csv(source/'frame_results.csv')
    required = {'metric','bag_path','camera_topic','timestamp_sec','pooled_score','base_threshold',
                'feature_sec','comparison_sec','metric_score_sec','processing_sec','read_decode_sec'}
    if not required.issubset(frames):
        raise ValueError(f'Saved frame_results.csv lacks columns: {sorted(required-set(frames))}')
    for column in required-{'metric','bag_path','camera_topic'}:
        if not np.isfinite(frames[column].to_numpy(float)).all():
            raise ValueError(f'Saved frame results contain invalid {column} values.')
    original = pd.read_csv(source/'bag_results.csv')
    original = original[original.operating_point.eq('configured')].set_index(['metric','bag_path'])
    if original.index.duplicated().any():
        raise ValueError('Duplicate configured decisions in saved bag_results.csv.')
    timelines = {key: group for key, group in frames.groupby(['metric','bag_path'])}
    for row in manifest:
        if row['status'] != 'ok':
            continue
        for metric in ['euclidean','mahalanobis']:
            group = timelines.get((metric, row['bag_path']))
            if group is None or set(group.camera_topic) != set(state['camera_topics']):
                raise ValueError(f'Incomplete saved frame results for {row["bag_path"]}, {metric}. Do not use a partial timeline.')
            identity = (metric, row['bag_path'])
            if identity not in original.index or len(group) != int(original.loc[identity, 'frames_scored']):
                raise ValueError(f'Saved frame count differs from original bag result: {row["bag_path"]}, {metric}.')
    output = Path(output_root or source)/('reports_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir(parents=True, exist_ok=False)
    folders = state.get('selected_folders') or list(dict.fromkeys(row['folder'] for row in manifest))
    pd.DataFrame(manifest).to_csv(output/'selection.csv', index=False)
    metrics, roc = write_reports(manifest, frames, output, float(state['configured_scale']), folders)
    if (source/'runtime.csv').is_file():
        shutil.copy2(source/'runtime.csv', output/'runtime.csv')
    report = dict(state, mode='report_only', report_format_version=REPORT_FORMAT_VERSION,
                  source_run=str(source), inference_rerun=False, bag_rule=ALARM_POLICY, stage_topic=stage_topic,
                  onset_policy='first failure marker at/after corrected stage 1; pre-stage-1 markers ignored',
                  ground_truth_counts=counts, labels_csv=str(Path(labels_csv).resolve()) if labels_csv else None,
                  roc=roc, report_generation_sec=perf_counter()-started)
    if labels_csv:
        report.update(original_label_policy=state.get('label_policy'), label_policy='authoritative labels CSV')
    (output/'run.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(metrics[['metric','operating_point','folder','recall','false_positive_rate']].to_string(index=False))
    print(f'Regenerated reports without inference: {output}', flush=True)
    return output
