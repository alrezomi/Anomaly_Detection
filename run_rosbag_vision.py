"""Run one independent DINOv2 detector per selected rosbag camera topic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from rosbag_io import (
    RosbagImageSource,
    export_model_input_video,
    resolve_bag_path,
)
from vision_dinov2 import run_vision_test


def _slug(topic: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", topic).strip("_").lower()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="JSON pipeline configuration.")
    parser.add_argument(
        "--nominal-bag",
        action="append",
        default=None,
        type=Path,
        help="Nominal ROS 2 bag directory. Repeat for each nominal run.",
    )
    parser.add_argument("--test-bag", type=Path)
    parser.add_argument(
        "--camera-topic",
        action="append",
        default=None,
        help="Image topic. Repeat to process 2, 3, or more viewpoints.",
    )
    parser.add_argument("--preview-only", action="store_true", default=None)
    parser.add_argument("--max-preview-frames", type=int)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Host/container output directory for videos and CSV files.",
    )
    parser.add_argument("--nominal-cache-dir", type=Path)
    parser.add_argument(
        "--rebuild-nominal-cache", action="store_true", default=None
    )
    parser.add_argument("--stage-topic")
    parser.add_argument("--stage-count", type=int)
    parser.add_argument(
        "--ignore-recorded-stages",
        action="store_true",
        default=None,
        help="Always use the original equal-duration progress assignment.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    configuration = {}
    if arguments.config:
        with arguments.config.open("r", encoding="utf-8") as config_file:
            configuration = json.load(config_file)

    nominal_bags = arguments.nominal_bag or [
        Path(value) for value in configuration.get("nominal_bags", [])
    ]
    test_bag_value = arguments.test_bag or configuration.get("test_bag")
    camera_topics = arguments.camera_topic or configuration.get("camera_topics", [])
    output_value = (
        arguments.output_dir
        or configuration.get("output_dir")
        or run_vision_test.OUTPUT_DIRECTORY
    )
    cache_value = (
        arguments.nominal_cache_dir
        or configuration.get("nominal_cache_dir")
        or (Path(output_value).parent / "nominal_cache")
    )
    rebuild_nominal_cache = (
        arguments.rebuild_nominal_cache
        if arguments.rebuild_nominal_cache is not None
        else bool(configuration.get("rebuild_nominal_cache", False))
    )
    preview_only = (
        arguments.preview_only
        if arguments.preview_only is not None
        else bool(configuration.get("preview_only", False))
    )
    max_preview_frames = (
        arguments.max_preview_frames
        if arguments.max_preview_frames is not None
        else configuration.get("max_preview_frames")
    )
    stage_topic_value = (
        arguments.stage_topic
        if arguments.stage_topic is not None
        else configuration.get("stage_topic", "/recording_stage")
    )
    stage_count = (
        arguments.stage_count
        if arguments.stage_count is not None
        else int(configuration.get("stage_count", 3))
    )
    ignore_recorded_stages = (
        arguments.ignore_recorded_stages
        if arguments.ignore_recorded_stages is not None
        else bool(configuration.get("ignore_recorded_stages", False))
    )

    if not test_bag_value:
        raise ValueError("Set test_bag in the config or pass --test-bag.")
    if not camera_topics:
        raise ValueError("Set camera_topics in the config or pass --camera-topic.")

    test_bag = Path(test_bag_value)
    output_directory = Path(output_value).resolve()
    run_vision_test.OUTPUT_DIRECTORY = output_directory
    output_directory.mkdir(parents=True, exist_ok=True)

    paths_to_validate = [test_bag]
    if not preview_only:
        paths_to_validate.extend(nominal_bags)
    invalid_paths: list[str] = []
    for path in paths_to_validate:
        try:
            resolve_bag_path(path)
        except (FileNotFoundError, ValueError) as error:
            invalid_paths.append(f"  - {path}: {error}")
    if invalid_paths:
        raise FileNotFoundError(
            "Invalid ROS bag paths in pipeline_config.json:\n"
            + "\n".join(invalid_paths)
        )

    for topic in camera_topics:
        name = _slug(topic)
        stage_topic = (
            None if ignore_recorded_stages else stage_topic_value
        )
        test_source = RosbagImageSource(
            test_bag,
            topic,
            "test",
            stage_topic=stage_topic,
            stage_count=stage_count,
        )
        preview_path = output_directory / f"{name}_model_input.mp4"
        count = export_model_input_video(
            source=test_source,
            output_path=preview_path,
            sample_fps=run_vision_test.SAMPLE_FPS,
            model_input_size=run_vision_test.DINO_INPUT_SIZE,
            start_sec=run_vision_test.START_SEC,
            end_sec=run_vision_test.END_SEC,
            max_frames=max_preview_frames,
        )
        print(f"Exported {count} exact model-input frames: {preview_path}")
        if preview_only:
            continue

        if len(nominal_bags) < 2:
            raise ValueError(
                "Adaptive calibration needs at least two nominal bags. "
                "Repeat --nominal-bag for two or more normal recordings."
            )

        run_vision_test.NOMINAL_VIDEO_PATHS = [
            RosbagImageSource(
                path,
                topic,
                f"nominal_{index}",
                stage_topic=stage_topic,
                stage_count=stage_count,
            )
            for index, path in enumerate(nominal_bags)
        ]
        run_vision_test.TEST_VIDEO_PATH = test_source
        run_vision_test.THRESHOLD_CSV = output_directory / f"{name}_threshold.csv"
        run_vision_test.CALIBRATION_CSV = output_directory / f"{name}_calibration.csv"
        run_vision_test.TEST_RESULT_CSV = output_directory / f"{name}_results.csv"
        run_vision_test.OUTPUT_VIDEO = output_directory / f"{name}_heatmap.mp4"
        run_vision_test.NOMINAL_CACHE_DIRECTORY = Path(cache_value) / name
        run_vision_test.REBUILD_NOMINAL_CACHE = rebuild_nominal_cache
        print(f"\nRunning independent camera detector: {topic}")
        run_vision_test.main()


if __name__ == "__main__":
    main()
