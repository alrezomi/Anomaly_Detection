"""Authoritative benchmark labels and cheap validation before model preparation."""
from collections import Counter
from pathlib import Path
import posixpath
import re

import numpy as np
import pandas as pd


def label_path(value, data_root):
    value = str(value).strip().replace('\\', '/')
    root = str(data_root).replace('\\', '/')
    absolute = value.startswith('/') or re.match(r'^[a-zA-Z]:/', value)
    return posixpath.normpath(value if absolute else root.rstrip('/')+'/'+value)


def apply_csv_labels(manifest, csv_path, data_root):
    """CSV paths are absolute or relative to data root. Never guess from folder names."""
    path = Path(csv_path).resolve(strict=True)
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    if 'label' not in table or not ({'bag_path','bag_name'} & set(table.columns)):
        raise ValueError('Labels CSV needs label and bag_path (or a unique bag_name) columns.')
    aliases = {'normal':'normal', 'nominal':'normal', 'success':'normal',
               'fail':'fail', 'failure':'fail', 'anomaly':'fail', 'unknown':'unknown', '':'unknown'}
    by_path, by_name = {}, {}
    for row in table.to_dict('records'):
        value = row['label'].strip().lower()
        if value not in aliases:
            raise ValueError(f'Unsupported CSV label {row["label"]!r}; use normal, fail, or unknown.')
        record = dict(label=aliases[value], onset=None)
        for key in ['failure_time_sec', 'recorded_failure_time_sec']:
            if row.get(key, '').strip():
                onset = float(row[key])
                if not np.isfinite(onset) or onset < 0 or record['label'] != 'fail':
                    raise ValueError(f'Invalid {key} for {row.get("bag_path", row.get("bag_name"))}.')
                if record['onset'] is not None and record['onset'] != onset:
                    raise ValueError('Conflicting onset time columns in labels CSV.')
                record['onset'] = onset
        if row.get('bag_path','').strip():
            key, mapping = label_path(row['bag_path'], data_root), by_path
        elif row.get('bag_name','').strip():
            key, mapping = row['bag_name'].strip(), by_name
        else:
            raise ValueError('Empty bag identity in labels CSV.')
        if key in mapping:
            raise ValueError(f'Duplicate bag identity in labels CSV: {key}')
        mapping[key] = record
    names = Counter(item['bag_name'] for item in manifest)
    results, missing = [], []
    for item in manifest:
        row = dict(item)
        if item['status'] == 'excluded_training':
            results.append(row)
            continue
        record = by_path.get(label_path(item['bag_path'], data_root))
        if record is None and item['bag_name'] in by_name:
            if names[item['bag_name']] != 1:
                raise ValueError(f'Ambiguous bag_name {item["bag_name"]}; use bag_path in labels CSV.')
            record = by_name[item['bag_name']]
        row['stage_ground_truth_label'] = item.get('stage_ground_truth_label', item.get('ground_truth_label'))
        row['ground_truth_label_source'] = str(path)
        row.setdefault('onset_source', 'recorded_stage_topic')
        if record is None:
            row['ground_truth_label'] = 'unknown'
            if item['status'] == 'ok':
                missing.append(item['bag_path'])
        else:
            row['ground_truth_label'] = record['label']
            row['label_disagrees_with_stage'] = record['label'] != row['stage_ground_truth_label']
            if record['onset'] is not None:
                row['stage_recorded_failure_time_sec'] = item.get('stage_recorded_failure_time_sec', item.get('recorded_failure_time_sec'))
                row.update(recorded_failure_time_sec=record['onset'], annotation_status='available',
                           onset_source=str(path), annotation_error=None)
        onset = row.get('recorded_failure_time_sec')
        row['label_onset_conflict'] = row['ground_truth_label']=='normal' and onset is not None and np.isfinite(onset)
        results.append(row)
    if missing:
        raise ValueError('Labels CSV has no entry for selected bags: '+', '.join(missing[:10])+
                         '. Use full paths or paths relative to the saved data root.')
    return results


def validate_ground_truth(manifest):
    eligible = [row for row in manifest if row['status']=='ok']
    counts = Counter(row.get('ground_truth_label', 'unknown') for row in eligible)
    unknown = [row['bag_path'] for row in eligible if row.get('ground_truth_label') not in {'normal','fail'}]
    if unknown:
        raise ValueError('Ground truth is missing/unknown for selected bags: '+', '.join(unknown[:10])+
                         '. Supply authoritative labels with --labels-csv before running inference.')
    if not counts['normal'] or not counts['fail']:
        raise ValueError(f'ROC preflight failed: {counts["normal"]} normal and {counts["fail"]} failure bags. '
                         'Both classes are required. A Failure_* folder name is not a recorded failure label. '
                         'Supply dataset_labels.csv with --labels-csv, and include unseen normal and failure bags. '
                         'No model preparation or inference has started.')
    return dict(normal=counts['normal'], fail=counts['fail'], excluded=len(manifest)-len(eligible))
