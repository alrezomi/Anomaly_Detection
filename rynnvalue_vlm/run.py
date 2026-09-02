"""Run RynnValue anomaly detection on configured rosbag or raw-video inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from rosbag_io import RosbagImageSource, sample_rosbag_image_frames_uniform
from .analysis import anomaly_decision
from .model import RynnValueModel


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()


def _video_frames(
    path: str | Path,
    count: int,
    start_sec: float = 0.0,
    end_sec: float | None = None,
) -> tuple[list[Image.Image], list[dict[str, Any]]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open raw video: {path}")
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 1:
        capture.release()
        raise ValueError(f"Video contains no readable frames: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 1.0
    first = max(0, min(total - 1, int(round(start_sec * fps))))
    last = total - 1
    if end_sec is not None:
        last = max(first, min(last, int(round(end_sec * fps))))
    indices = np.linspace(first, last, min(count, last - first + 1)).round().astype(int)
    frames: list[Image.Image] = []
    metadata: list[dict[str, Any]] = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        success, bgr = capture.read()
        if success:
            frames.append(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
            metadata.append(
                {
                    "source_video": str(path),
                    "frame_index": int(index),
                    "timestamp_sec": float(index) / fps,
                    "video_fps": fps,
                    "video_frame_count": total,
                }
            )
    capture.release()
    if not frames:
        raise ValueError(f"No selected frames could be decoded from: {path}")
    return frames, metadata


def _topic_frames(
    config: dict[str, Any], vlm: dict[str, Any], topic: str, frame_count: int
) -> tuple[list[Image.Image], list[dict[str, Any]]]:
    start_sec = float(vlm.get("sampling_start_sec", 0.0))
    end_value = vlm.get("sampling_end_sec")
    end_sec = float(end_value) if end_value is not None else None
    source = vlm.get("source", "generated_videos")

    if source == "generated_videos":
        default_path = Path(config["output_dir"]) / f"{_slug(topic)}_raw_original.mp4"
        path = Path(vlm.get("raw_video_paths", {}).get(topic, default_path))
        frames, metadata = _video_frames(path, frame_count, start_sec, end_sec)
    elif source == "rosbag":
        sampled = sample_rosbag_image_frames_uniform(
            RosbagImageSource(Path(config["test_bag"]), topic),
            num_frames=frame_count,
            start_sec=start_sec,
            end_sec=end_sec,
        )
        frames = [frame for _, _, frame in sampled]
        metadata = [
            {
                "source_bag": str(config["test_bag"]),
                "frame_index": int(frame_index),
                "timestamp_sec": float(timestamp_sec),
            }
            for frame_index, timestamp_sec, _ in sampled
        ]
    else:
        raise ValueError(
            "rynnvalue.source must be 'generated_videos' or 'rosbag'"
        )
    for row in metadata:
        row["topic"] = topic
    return frames, metadata


def _save_selected_inputs(
    output_directory: Path,
    topic: str,
    images: list[Image.Image],
    metadata: list[dict[str, Any]],
) -> None:
    directory = output_directory / "selected_inputs" / _slug(topic)
    directory.mkdir(parents=True, exist_ok=True)
    for index, image in enumerate(images):
        image.save(directory / f"{index:03d}.jpg", quality=92)

    thumb_width, thumb_height, label_height = 320, 240, 40
    columns = min(4, max(1, len(images)))
    rows = (len(images) + columns - 1) // columns
    sheet = Image.new(
        "RGB", (columns * thumb_width, rows * (thumb_height + label_height)), "white"
    )
    draw = ImageDraw.Draw(sheet)
    for index, image in enumerate(images):
        column, row = index % columns, index // columns
        preview = image.convert("RGB").copy()
        preview.thumbnail((thumb_width, thumb_height))
        x = column * thumb_width + (thumb_width - preview.width) // 2
        y = row * (thumb_height + label_height)
        sheet.paste(preview, (x, y))
        timestamp = metadata[index].get("timestamp_sec")
        draw.text(
            (column * thumb_width + 5, y + thumb_height + 4),
            f"frame {index + 1}, t={timestamp:.3f}s",
            fill="black",
        )
    sheet.save(directory / "rynnvalue_input_storyboard.jpg", quality=92)


def _plot_values(output_directory: Path, topic: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    figure, axis = plt.subplots(figsize=(10, 5))
    timestamps = [row["timestamp_sec"] for row in rows]
    remaining = [row["remaining_time_seconds"] for row in rows]
    axis.plot(timestamps, remaining, marker="o", linewidth=2)
    regression_x = [
        row["timestamp_sec"] for row in rows if row["temporal_regression"]
    ]
    regression_y = [
        row["remaining_time_seconds"] for row in rows if row["temporal_regression"]
    ]
    if regression_x:
        axis.scatter(regression_x, regression_y, color="red", label="regression")
        axis.legend()
    axis.set_title(f"RynnValue remaining time: {topic}")
    axis.set_xlabel("Trajectory timestamp (s)")
    axis.set_ylabel("Predicted remaining time (s)")
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(
        output_directory / f"{_slug(topic)}_rynnvalue_remaining_time.png", dpi=160
    )
    plt.close(figure)


def evaluate_rynnvalue(
    model: RynnValueModel,
    config: dict[str, Any],
    vlm: dict[str, Any],
    output_directory: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Evaluate all configured cameras with one already-loaded model."""

    instruction = vlm.get(
        "task_instruction", vlm.get("task_description", "Robot manipulation task")
    )
    topics = list(vlm.get("camera_topics", config.get("camera_topics", [])))
    if not topics:
        raise ValueError("rynnvalue.camera_topics must contain at least one topic")
    frame_count = int(vlm.get("num_frames", 8))
    if frame_count < 2:
        raise ValueError("rynnvalue.num_frames must be at least 2")
    regression_tolerance = float(vlm.get("regression_tolerance_sec", 1.0))
    if regression_tolerance < 0:
        raise ValueError("rynnvalue.regression_tolerance_sec cannot be negative")

    output_directory.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    value_rows: list[dict[str, Any]] = []
    raw_records: list[dict[str, Any]] = []
    camera_descriptions = dict(vlm.get("camera_descriptions", {}))

    print(f"RynnValue task: {instruction}")
    for topic in topics:
        print(f"\nRynnValue camera: {topic}")
        images, metadata = _topic_frames(config, vlm, topic, frame_count)
        _save_selected_inputs(output_directory, topic, images, metadata)
        prediction = model.predict(
            instruction=instruction,
            images=images,
            generation=dict(vlm.get("generation", {})),
            robot_description=vlm.get("robot_description"),
            camera_description=camera_descriptions.get(
                topic, vlm.get("camera_description")
            ),
        )
        if len(prediction.remaining_time_seconds) != len(images):
            raise ValueError(
                f"RynnValue returned {len(prediction.remaining_time_seconds)} values "
                f"for {len(images)} images on {topic}"
            )

        decision = anomaly_decision(prediction.match, prediction.success)
        topic_rows: list[dict[str, Any]] = []
        for index, (frame_metadata, remaining) in enumerate(
            zip(metadata, prediction.remaining_time_seconds)
        ):
            previous = (
                prediction.remaining_time_seconds[index - 1] if index > 0 else None
            )
            remaining_change = remaining - previous if previous is not None else None
            temporal_progress = previous - remaining if previous is not None else None
            if len(prediction.relative_time_seconds) == len(images) - 1:
                relative = (
                    prediction.relative_time_seconds[index - 1] if index > 0 else None
                )
            elif len(prediction.relative_time_seconds) == len(images):
                relative = prediction.relative_time_seconds[index]
            else:
                relative = None
            entropy = (
                prediction.entropy[index]
                if len(prediction.entropy) == len(images)
                else None
            )
            row = {
                **frame_metadata,
                "sample_index": index,
                "remaining_time_seconds": remaining,
                "progress_value": -remaining,
                "remaining_time_change_seconds": remaining_change,
                "temporal_progress_seconds": temporal_progress,
                "relative_time_seconds": relative,
                "entropy": entropy,
                "temporal_regression": bool(
                    remaining_change is not None
                    and remaining_change > regression_tolerance
                ),
            }
            topic_rows.append(row)
            value_rows.append(row)

        regressions = sum(bool(row["temporal_regression"]) for row in topic_rows)
        mean_entropy = (
            float(np.mean(prediction.entropy)) if prediction.entropy else None
        )
        summary = {
            "test_bag": config.get("test_bag", ""),
            "topic": topic,
            "decision": decision,
            "anomaly_detected": decision == "failure",
            "match": prediction.match,
            "success": prediction.success,
            "video_description": prediction.video_description,
            "initial_remaining_time_seconds": prediction.remaining_time_seconds[0],
            "final_remaining_time_seconds": prediction.remaining_time_seconds[-1],
            "net_temporal_progress_seconds": (
                prediction.remaining_time_seconds[0]
                - prediction.remaining_time_seconds[-1]
            ),
            "temporal_regression_count": regressions,
            "temporal_regression_detected": regressions > 0,
            "mean_entropy": mean_entropy,
            "ground_truth_label": vlm.get("ground_truth_label", ""),
        }
        summaries.append(summary)
        raw_records.append(
            {
                **summary,
                "task_instruction": instruction,
                "analysis_text": prediction.analysis_text,
                "selected_frames": metadata,
                "remaining_time_seconds": prediction.remaining_time_seconds,
                "relative_time_seconds": prediction.relative_time_seconds,
                "entropy": prediction.entropy,
            }
        )
        _plot_values(output_directory, topic, topic_rows)
        print(
            f"decision={decision}, match={prediction.match}, "
            f"success={prediction.success}, regressions={regressions}"
        )
        print(f"analysis:\n{prediction.analysis_text}")

    pd.DataFrame(summaries).to_csv(
        output_directory / "rynnvalue_results.csv", index=False
    )
    pd.DataFrame(value_rows).to_csv(
        output_directory / "rynnvalue_values.csv", index=False
    )
    (output_directory / "rynnvalue_analysis.json").write_text(
        json.dumps(
            {"task_instruction": instruction, "results": raw_records}, indent=2
        ),
        encoding="utf-8",
    )
    print(f"RynnValue results saved to: {output_directory}")
    return summaries, value_rows, raw_records


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    config = json.loads(arguments.config.read_text(encoding="utf-8"))
    vlm = dict(config.get("rynnvalue", {}))
    if not vlm:
        raise ValueError("Add a 'rynnvalue' section to pipeline_config.json")
    output_directory = Path(
        vlm.get("output_dir", Path(config["output_dir"]) / "rynnvalue")
    )
    model = RynnValueModel(dict(vlm.get("model", {})))
    evaluate_rynnvalue(model, config, vlm, output_directory)


if __name__ == "__main__":
    main()
