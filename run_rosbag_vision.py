"""Run one independent DINOv2 detector per selected rosbag camera topic."""

from __future__ import annotations

import argparse
from pathlib import Path
import re

from rosbag_io import RosbagImageSource, export_model_input_video
from vision_dinov2 import run_vision_test


def _slug(topic: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", topic).strip("_").lower()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nominal-bag",
        action="append",
        default=[],
        type=Path,
        help="Nominal ROS 2 bag directory. Repeat for each nominal run.",
    )
    parser.add_argument("--test-bag", required=True, type=Path)
    parser.add_argument(
        "--camera-topic",
        action="append",
        required=True,
        help="Image topic. Repeat to process 2, 3, or more viewpoints.",
    )
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument("--max-preview-frames", type=int)
    parser.add_argument("--stage-topic", default="/recording_stage")
    parser.add_argument("--stage-count", type=int, default=3)
    parser.add_argument(
        "--ignore-recorded-stages",
        action="store_true",
        help="Always use the original equal-duration progress assignment.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    output_directory = run_vision_test.OUTPUT_DIRECTORY
    output_directory.mkdir(parents=True, exist_ok=True)

    for topic in arguments.camera_topic:
        name = _slug(topic)
        stage_topic = (
            None if arguments.ignore_recorded_stages else arguments.stage_topic
        )
        test_source = RosbagImageSource(
            arguments.test_bag,
            topic,
            "test",
            stage_topic=stage_topic,
            stage_count=arguments.stage_count,
        )
        preview_path = output_directory / f"{name}_model_input.mp4"
        count = export_model_input_video(
            source=test_source,
            output_path=preview_path,
            sample_fps=run_vision_test.SAMPLE_FPS,
            model_input_size=run_vision_test.DINO_INPUT_SIZE,
            start_sec=run_vision_test.START_SEC,
            end_sec=run_vision_test.END_SEC,
            max_frames=arguments.max_preview_frames,
        )
        print(f"Exported {count} exact model-input frames: {preview_path}")
        if arguments.preview_only:
            continue

        if len(arguments.nominal_bag) < 2:
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
                stage_count=arguments.stage_count,
            )
            for index, path in enumerate(arguments.nominal_bag)
        ]
        run_vision_test.TEST_VIDEO_PATH = test_source
        run_vision_test.THRESHOLD_CSV = output_directory / f"{name}_threshold.csv"
        run_vision_test.CALIBRATION_CSV = output_directory / f"{name}_calibration.csv"
        run_vision_test.TEST_RESULT_CSV = output_directory / f"{name}_results.csv"
        run_vision_test.OUTPUT_VIDEO = output_directory / f"{name}_heatmap.mp4"
        print(f"\nRunning independent camera detector: {topic}")
        run_vision_test.main()


if __name__ == "__main__":
    main()
