"""Regenerate DINO reports from saved frames, without ROS reads or model inference."""
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
import json
import shutil

import numpy as np
import pandas as pd

from .benchmark_labels import apply_csv_labels, validate_ground_truth
from .benchmark_report import write_reports


def regenerate_reports(source, labels_csv=None, output_root=None):
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
        manifest = apply_csv_labels(manifest, labels_csv, state['data_root'])
    counts = validate_ground_truth(manifest)
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
    report = dict(state, mode='report_only', source_run=str(source), inference_rerun=False,
                  ground_truth_counts=counts, labels_csv=str(Path(labels_csv).resolve()) if labels_csv else None,
                  roc=roc, report_generation_sec=perf_counter()-started)
    if labels_csv:
        report.update(original_label_policy=state.get('label_policy'), label_policy='authoritative labels CSV')
    (output/'run.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(metrics[['metric','operating_point','folder','recall','false_positive_rate']].to_string(index=False))
    print(f'Regenerated reports without inference: {output}', flush=True)
    return output
