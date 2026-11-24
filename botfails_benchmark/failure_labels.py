"""Build separate, derived failure annotations without using detector predictions.

Run: python -m botfails_benchmark.failure_labels --dataset <BotFails>
The five soldering hand-intervention episodes were adjudicated as failures by the user.
Original event timing is inherited, not claimed to be independently reannotated.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import TASKS


def derive(task, episode, anomaly):
    if task == 'industrial_soldering' and 10 <= int(episode.split('_')[1]) <= 14:
        return anomaly.copy(), 'failure', 'User adjudication 2026-11-12: operator hand in the tool work area requires stopping; the task includes returning the iron to its holder. Retain original event timing as a proxy.'
    if not anomaly.any():
        return anomaly.copy(), 'normal', 'No annotated anomaly; sampled views do not establish a new failure.'
    return anomaly.copy(), 'failure', 'Task-error category (spill, drop, wrong placement, incomplete action or wrong sequence); retain source event timing as a proxy.'


def build(dataset, output):
    rows = []
    for task in TASKS:
        for position, source in enumerate(sorted((dataset / 'labels' / (task + '_anomaly')).glob('*_labels.csv'))):
            episode = source.name.removesuffix('_labels.csv')
            raw = np.loadtxt(source, dtype=int, ndmin=1)
            if not set(np.unique(raw)) <= {0, 1}:
                raise ValueError(f'Invalid anomaly labels: {source}')
            labels, status, reason = derive(task, episode, raw)
            target = output / (task + '_anomaly') / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(target, labels, fmt='%d')
            rows.append(dict(task=task, episode=episode, status=status, reason=reason,
                             review_evidence=f'review/{task}_{position // (5 if task in ("domotic_makingCoffee", "industrial_soldering") else 8) * (5 if task in ("domotic_makingCoffee", "industrial_soldering") else 8)}.jpg',
                             metadata_description=(dataset / 'test' / (task + '_anomaly') / 'meta/tasks.jsonl').read_text().strip(),
                             frames=len(raw), source_positive_frames=int(raw.sum()),
                             failure_frames=int((labels == 1).sum()),
                             onset_policy='inherited anomaly timing' if status == 'failure' else 'not applicable',
                             review='metadata and sampled visual review; not exhaustive video annotation',
                             source_sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
    manifest = pd.DataFrame(rows)
    manifest.to_csv(output / 'manifest.csv', index=False)
    (output / 'provenance.json').write_text(json.dumps(dict(
        kind='derived failure labels v2; user-adjudicated soldering interventions; not the authors\' failure annotations',
        values={'0': 'no established failure', '1': 'task failure', '-1': 'unresolved; exclude episode'},
        timing='Original 30 Hz anomaly timings retained for task-error episodes; approximate failure onset.',
        prediction_independent=True, counts=manifest.status.value_counts().to_dict()), indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or args.dataset.parents[1] / 'failure_labels'
    if output.resolve() == (args.dataset / 'labels').resolve():
        raise ValueError('Failure annotations must not overwrite original anomaly labels.')
    print(build(args.dataset, output).status.value_counts().to_string())
