"""Build/reuse both DINO metrics, then benchmark only explicitly selected folders."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('pipeline_config.json'))
    parser.add_argument('--data-root', type=Path, default=Path(os.environ.get('ANOMALY_DATA_ROOT', '/data')))
    parser.add_argument('--folder', action='append', help='Folder under the data root; repeat to select several. Overrides config list.')
    parser.add_argument('--output-dir', type=Path, help='Parent for a new timestamped report directory.')
    parser.add_argument('--labels-csv', type=Path, help='Dataset labels; missing selected bags are added from recorded stages, existing entries preserved. Overrides dino_benchmark.labels_csv.')
    parser.add_argument('--report-only', type=Path, metavar='SAVED_RUN', help='Regenerate reports from saved frame results, without memory building or inference.')
    arguments = parser.parse_args()
    config = (json.loads(arguments.config.read_text(encoding='utf-8'))
              if arguments.config.is_file() or not arguments.report_only else {})
    selection = config.get('dino_benchmark') or {}
    if not isinstance(selection, dict):
        raise ValueError('dino_benchmark must be an object containing a folders list.')
    folders = arguments.folder if arguments.folder is not None else selection.get('folders', [])
    labels_csv = arguments.labels_csv or selection.get('labels_csv')
    if arguments.report_only:
        from vision_dinov2.benchmark_recovery import regenerate_reports
        regenerate_reports(arguments.report_only, labels_csv, arguments.output_dir,
                           stage_topic=config.get('stage_topic'))
        return
    from vision_dinov2.benchmark import run
    run(config, folders, arguments.data_root, arguments.output_dir, labels_csv=labels_csv)


if __name__ == '__main__':
    main()
