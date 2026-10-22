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
    arguments = parser.parse_args()
    config = json.loads(arguments.config.read_text(encoding='utf-8'))
    selection = config.get('dino_benchmark') or {}
    if not isinstance(selection, dict):
        raise ValueError('dino_benchmark must be an object containing a folders list.')
    folders = arguments.folder if arguments.folder is not None else selection.get('folders', [])
    from vision_dinov2.benchmark import run
    run(config, folders, arguments.data_root, arguments.output_dir)


if __name__ == '__main__':
    main()
