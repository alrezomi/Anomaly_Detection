"""Inspect and extract selected modalities from a ROS 2 bag."""

from __future__ import annotations

import argparse
from pathlib import Path

from rosbag_io import (
    RosbagImageSource,
    export_model_input_video,
    list_bag_topics,
    read_numeric_topic,
    read_stage_events,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    topics = subparsers.add_parser("topics", help="List recorded topics.")
    topics.add_argument("bag", type=Path)

    video = subparsers.add_parser(
        "export-video",
        help="Export sampled, resized model-input frames from a camera topic.",
    )
    video.add_argument("bag", type=Path)
    video.add_argument("--topic", required=True)
    video.add_argument("--output", type=Path, required=True)
    video.add_argument("--fps", type=float, default=2.0)
    video.add_argument("--model-size", type=int, default=518)
    video.add_argument("--max-frames", type=int)

    numeric = subparsers.add_parser(
        "export-topic",
        help="Export a Wrench, Pose, or JointState topic to CSV.",
    )
    numeric.add_argument("bag", type=Path)
    numeric.add_argument("--topic", required=True)
    numeric.add_argument("--output", type=Path, required=True)

    stages = subparsers.add_parser(
        "export-stages",
        help="Export recording-stage events to CSV.",
    )
    stages.add_argument("bag", type=Path)
    stages.add_argument("--topic", default="/recording_stage")
    stages.add_argument("--output", type=Path, required=True)

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    if arguments.command == "topics":
        print(list_bag_topics(arguments.bag).to_string(index=False))
        return

    arguments.output.parent.mkdir(parents=True, exist_ok=True)

    if arguments.command == "export-video":
        count = export_model_input_video(
            source=RosbagImageSource(arguments.bag, arguments.topic),
            output_path=arguments.output,
            sample_fps=arguments.fps,
            model_input_size=arguments.model_size,
            max_frames=arguments.max_frames,
        )
        print(f"Exported {count} frames to {arguments.output.resolve()}")
    elif arguments.command == "export-topic":
        dataframe = read_numeric_topic(arguments.bag, arguments.topic)
        dataframe.to_csv(arguments.output, index=False)
        print(f"Exported {len(dataframe)} rows to {arguments.output.resolve()}")
    elif arguments.command == "export-stages":
        dataframe = read_stage_events(arguments.bag, arguments.topic)
        dataframe.to_csv(arguments.output, index=False)
        print(f"Exported {len(dataframe)} events to {arguments.output.resolve()}")


if __name__ == "__main__":
    main()
