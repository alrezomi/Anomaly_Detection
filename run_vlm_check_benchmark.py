"""Benchmark the DINO -> VLM check on a saved DINO benchmark run; DINO is not rerun.

Results are written to <dino run>/vlm_check/ or --output. VLM answers are cached
there, so rerunning resumes an interrupted run and regenerates the reports.
"""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('pipeline_config.json'))
    parser.add_argument('--dino-run', type=Path, required=True,
                        help='Saved DINO benchmark run directory (with run.json, bag_results.csv, frame_results.csv).')
    parser.add_argument('--metric', action='append', choices=['euclidean', 'mahalanobis'],
                        help='DINO metric to evaluate; repeat for several. Default: every metric in the saved run.')
    parser.add_argument('--limit', type=int, help='Evaluate only the first N bags (quick check).')
    parser.add_argument('--per-folder', type=int, help='Evaluate only the first N bags of each folder (quick check).')
    parser.add_argument('--output', type=Path, help='Results directory (default: <dino run>/vlm_check).')
    parser.add_argument('--all-checks', action='store_true',
                        help='Check every alarm, also after the first failure answer, to judge each VLM answer '
                             '(alarm_checks.csv). The per-bag results stay those of the live rule.')
    arguments = parser.parse_args()
    from pipeline_settings import load_pipeline_config
    from rynnbrain_vlm.cascade_benchmark import run
    run(load_pipeline_config(arguments.config), arguments.dino_run, arguments.metric, arguments.limit,
        progress=lambda message: print(message, flush=True), output=arguments.output,
        per_folder=arguments.per_folder, all_checks=arguments.all_checks)


if __name__ == '__main__':
    main()
