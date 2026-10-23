"""Authoritative benchmark labels and cheap validation before model preparation."""
from collections import Counter
from pathlib import Path
import posixpath
import re
import os
import tempfile

import numpy as np
import pandas as pd


def label_path(value, data_root):
    value = str(value).strip().replace('\\', '/')
    root = str(data_root).replace('\\', '/')
    absolute = value.startswith('/') or re.match(r'^[a-zA-Z]:/', value)
    return posixpath.normpath(value if absolute else root.rstrip('/')+'/'+value)


def update_labels_csv(manifest, csv_path, data_root):
    """Add newly selected bags using recorded labels; preserve existing edits/columns.

    The manifest is already annotated by the benchmark (or saved by report-only).
    No whole-dataset scan or duplicate ROS reads are needed.
    """
    path = Path(csv_path).resolve()
    if path.is_file():
        apply_csv_labels([], path, data_root)  # Reject malformed existing data before writing.
        table = pd.read_csv(path, dtype=str, keep_default_na=False)
    else:
        table = pd.DataFrame(columns=['bag_name', 'bag_path', 'label'])
    columns = list(dict.fromkeys([*table.columns, 'bag_name', 'bag_path', 'label']))
    records = table.reindex(columns=columns, fill_value='').to_dict('records')
    paths = {label_path(row['bag_path'], data_root) for row in records if row['bag_path'].strip()}
    names = {row['bag_name'].strip() for row in records if not row['bag_path'].strip()}
    additions = []
    for row in manifest:
        if row['status'] != 'ok':
            continue
        key = label_path(row['bag_path'], data_root)
        if key in paths or row['bag_name'] in names:
            continue
        record = dict.fromkeys(columns, '')
        record.update(bag_name=row['bag_name'],
                      bag_path=posixpath.relpath(key, label_path('.', data_root)),
                      label=row.get('stage_ground_truth_label', row.get('ground_truth_label', 'unknown')))
        additions.append(record)
        paths.add(key)
    if path.is_file() and not additions:
        apply_csv_labels(manifest, path, data_root)
        print(f'Labels CSV current: {path}; existing labels preserved.', flush=True)
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='',
                dir=path.parent, prefix=path.name+'.', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            pd.DataFrame(records+additions, columns=columns).to_csv(stream, index=False)
        apply_csv_labels(manifest, temporary, data_root)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(f'Labels CSV updated: {path}; added {len(additions)} selected bags; existing labels preserved.', flush=True)
    return path


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
    if not eligible:
        raise ValueError('No eligible labelled bags remain for evaluation.')
    if not counts['normal'] or not counts['fail']:
        print(f'Ground truth has {counts["normal"]} normal and {counts["fail"]} failure bags. '
              'Single-class evaluation is valid; ROC and best-scale selection are unavailable. '
              'Task-success labels do not establish whether a visual change is anomalous.', flush=True)
    return dict(normal=counts['normal'], fail=counts['fail'], excluded=len(manifest)-len(eligible))
