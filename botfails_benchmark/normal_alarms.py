"""DINO alarms in the normal training runs, scored leave-one-episode-out (no test data).

    python -m botfails_benchmark.normal_alarms <dino run> [threshold scale]
"""
import sys
from pathlib import Path

import pandas as pd

from .data import TASKS
from .report import first_alarm


def main():
    run, scale = Path(sys.argv[1]), float(sys.argv[2]) if len(sys.argv) > 2 else 1.1
    rows = []
    for task in TASKS:
        path = run / task / 'calibration_rows.csv'
        if not path.is_file():
            continue
        frame = pd.read_csv(path).rename(columns={'score_pooled': 'pooled_score', 'threshold_pooled': 'base_threshold'})
        alarms = [first_alarm(part, scale) for _, part in frame.groupby('episode')]
        ratio = frame.pooled_score / frame.base_threshold
        rows.append(dict(task=task, normal_runs=len(alarms), runs_with_alarm=sum(a is not None for a in alarms),
                         frames_above=f'{100 * (ratio > scale).mean():.1f}%'))
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    table.to_csv(run / 'normal_training_alarms.csv', index=False)


if __name__ == '__main__':
    main()
