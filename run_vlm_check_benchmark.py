"""Benchmark the DINO -> VLM check on a saved DINO benchmark run; DINO is not rerun.

Results are written to <dino run>/vlm_check/. VLM answers are cached there, so
rerunning resumes an interrupted run and regenerates the reports.
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
    arguments = parser.parse_args()
    from pipeline_settings import load_pipeline_config
    from rynnbrain_vlm.cascade_benchmark import run
    run(load_pipeline_config(arguments.config), arguments.dino_run, arguments.metric, arguments.limit,
        progress=lambda message: print(message, flush=True))


if __name__ == '__main__':
    main()
