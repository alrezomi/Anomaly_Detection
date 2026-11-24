"""Evaluate our DINO -> VLM cascade on BotFails (FIDeL benchmark), step by step.

  dino     memory + calibration from normal_train, scores of every labelled test frame (2 fps)
  cascade  replay the DINO alarms with the VLM sub-goal check (task phases from the gripper)
  report   FIDeL's metrics (frame AUROC/F1, TPR/TNR/Acc/W.Acc) plus episode-level results

Example (inside the GPU image, evaluation folder mounted at /evaluation):
  python run_botfails_benchmark.py dino --dataset /evaluation/dataset/BotFails \
      --frames /evaluation/frames --output /evaluation/results/dino_euclidean
"""
import argparse
import json
import sys
import time
from pathlib import Path


def log(message):
    print(f'[{time.strftime("%H:%M:%S")}] {message}', flush=True)


def main():
    from botfails_benchmark.data import TASKS
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    dino = sub.add_parser('dino')
    dino.add_argument('--config', type=Path, default=Path('/config/pipeline_config.json'))
    dino.add_argument('--dataset', type=Path, required=True)
    dino.add_argument('--frames', type=Path, required=True)
    dino.add_argument('--output', type=Path, required=True)
    dino.add_argument('--task', action='append', choices=TASKS, help='Repeat; default: all ten tasks.')
    dino.add_argument('--metric', default='euclidean', choices=['euclidean', 'mahalanobis'])
    dino.add_argument('--camera', default='logitech_1')
    dino.add_argument('--max-train', type=int, help='Quick check: first N normal training episodes.')
    dino.add_argument('--max-test', type=int, help='Quick check: first N test episodes.')
    dino.add_argument('--throttle-sec', type=float, default=0., help='Pause after each test frame (shared GPU).')
    dino.add_argument('--cache', type=Path, help='Feature cache directory (float16 per episode).')
    dino.add_argument('--split-file', type=Path, help='JSON with "dev" and "heldout" test episode indices per task.')
    dino.add_argument('--subset', choices=['dev', 'heldout'], help='Score only this part of --split-file.')

    cascade = sub.add_parser('cascade')
    cascade.add_argument('--config', type=Path, default=Path('/config/pipeline_config.json'))
    cascade.add_argument('--dataset', type=Path, required=True)
    cascade.add_argument('--frames', type=Path, required=True)
    cascade.add_argument('--dino-run', type=Path, required=True)
    cascade.add_argument('--tasks-config', type=Path, default=Path('botfails_benchmark/tasks.json'))
    cascade.add_argument('--output', type=Path, required=True)
    cascade.add_argument('--task', action='append', choices=TASKS)
    cascade.add_argument('--camera', default='logitech_1', help='Camera the VLM looks at.')
    cascade.add_argument('--split', default='test', choices=['test', 'normal_train'],
                         help='normal_train replays leave-one-episode-out DINO scores (false alarms only).')
    cascade.add_argument('--per-task', type=int, help='Quick check: first N episodes of each task.')
    cascade.add_argument('--split-file', type=Path, help='JSON with "dev" and "heldout" test episode indices per task.')
    cascade.add_argument('--subset', choices=['dev', 'heldout'], help='Only this part of --split-file.')
    cascade.add_argument('--threshold-scale', type=float, help='DINO threshold scale (default: pipeline_config).')
    cascade.add_argument('--recheck-sec', type=float, help='Ask the VLM again every N s while an alarm persists.')
    cascade.add_argument('--release-delay', type=float, help=argparse.SUPPRESS)  # Reject legacy hybrid commands.
    cascade.add_argument('--confirm-sec', type=float, help='A failing check must fail again this much later.')
    cascade.add_argument('--vlm-thresholds', type=Path, help='{task: {question: threshold}} from normal data.')
    cascade.add_argument('--vlm-cache', type=Path, help='Shared answer cache (default: inside --output).')

    report = sub.add_parser('report')
    report.add_argument('--dataset', type=Path, required=True)
    report.add_argument('--dino-run', type=Path, required=True)
    report.add_argument('--cascade-run', type=Path)
    report.add_argument('--output', type=Path, required=True)
    report.add_argument('--threshold-scale', type=float, default=1.1)
    report.add_argument('--split-file', type=Path, help='JSON with "dev" and "heldout" test episode indices per task.')
    report.add_argument('--subset', choices=['dev', 'heldout'], help='Only this part of --split-file.')
    report.add_argument('--no-timelines', action='store_true', help='Skip the per-episode plots.')
    report.add_argument('--label-type', choices=['failure', 'anomaly'], default='failure',
                        help='End-to-end ground truth (default: derived failure labels). Raw detector diagnostics retain anomaly labels.')
    report.add_argument('--labels-dir', type=Path, help='Failure CSV root; default: <evaluation>/failure_labels. No fallback to anomaly labels.')
    arguments = parser.parse_args()

    if arguments.command == 'dino':
        from pipeline_settings import load_pipeline_config
        from botfails_benchmark.dino import run_task, settings
        config = load_pipeline_config(arguments.config)
        arguments.output.mkdir(parents=True, exist_ok=True)
        (arguments.output / 'run.json').write_text(json.dumps(dict(
            command=sys.argv, settings=settings(config), metric=arguments.metric, camera=arguments.camera,
            dataset=str(arguments.dataset), started=time.strftime('%Y-%m-%d %H:%M:%S')), indent=1))
        extractor = None
        split = json.loads(arguments.split_file.read_text()) if arguments.split_file else None
        for task in arguments.task or TASKS:
            subset = split[arguments.subset][task] if split and arguments.subset else None
            extractor = run_task(config, task, arguments.dataset, arguments.frames, arguments.output,
                                 arguments.metric, arguments.camera, arguments.max_train, arguments.max_test,
                                 arguments.throttle_sec, extractor, log, arguments.cache, subset)
        log('DINO done')
    elif arguments.command == 'cascade':
        from pipeline_settings import load_pipeline_config
        from botfails_benchmark.cascade import run
        run(load_pipeline_config(arguments.config), arguments, log)
    else:
        from botfails_benchmark.report import run
        run(arguments, log)


if __name__ == '__main__':
    main()
