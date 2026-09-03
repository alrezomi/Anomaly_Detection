"""Run RynnValue on videos already produced by the vision pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from .model import RynnValueModel


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()


def _sample_video(
    path: Path,
    count: int,
    start_sec: float,
    end_sec: float | None,
) -> tuple[list[Image.Image], list[float]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(
            f"RynnValue input is missing: {path}. Run vision-test first."
        )
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 1.0
    if total < 1:
        capture.release()
        raise ValueError(f"Video contains no readable frames: {path}")

    first = max(0, min(total - 1, int(round(start_sec * fps))))
    last = total - 1
    if end_sec is not None:
        last = max(first, min(last, int(round(end_sec * fps))))
    indices = (
        np.linspace(first, last, min(count, last - first + 1))
        .round()
        .astype(int)
    )

    images: list[Image.Image] = []
    timestamps: list[float] = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        success, bgr = capture.read()
        if success:
            images.append(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
            timestamps.append(float(index) / fps)
    capture.release()
    if not images:
        raise ValueError(f"No selected frames could be decoded from: {path}")
    return images, timestamps


def _pair_images(
    raw_images: list[Image.Image], heatmaps: list[Image.Image]
) -> list[Image.Image]:
    if len(raw_images) != len(heatmaps):
        raise ValueError("Raw and heatmap frame counts do not match.")
    paired: list[Image.Image] = []
    for raw_image, heatmap in zip(raw_images, heatmaps):
        raw = raw_image.convert("RGB")
        heatmap = heatmap.convert("RGB")
        heatmap_width = max(1, round(heatmap.width * raw.height / heatmap.height))
        heatmap = heatmap.resize(
            (heatmap_width, raw.height), Image.Resampling.BICUBIC
        )
        combined = Image.new("RGB", (raw.width + heatmap.width, raw.height))
        combined.paste(raw, (0, 0))
        combined.paste(heatmap, (raw.width, 0))
        paired.append(combined)
    return paired


def _decision(match: str | None, success: str | None) -> str:
    match_value = (match or "").lower()
    success_value = (success or "").lower()
    if match_value == "no" or success_value == "no":
        return "failure"
    if match_value == "yes" and success_value == "yes":
        return "success"
    return "uncertain"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()
    config = json.loads(arguments.config.read_text(encoding="utf-8"))
    settings = config.get("rynnvalue", {})
    instruction = settings.get("task_description")
    if not instruction:
        raise ValueError("rynnvalue.task_description is missing from the config.")
    robot_description = settings.get("robot_description")
    camera_description = settings.get("camera_description")
    if not robot_description and not camera_description:
        raise ValueError(
            "RynnValue-8B requires rynnvalue.robot_description and/or "
            "rynnvalue.camera_description."
        )

    topics = list(config["camera_topics"])
    if not topics:
        raise ValueError("camera_topics must contain at least one topic.")
    input_modes = list(settings.get("input_modes", ["raw"]))
    if not input_modes:
        raise ValueError("rynnvalue.input_modes must contain at least one mode.")
    unsupported = set(input_modes) - {"raw", "heatmap", "raw_heatmap"}
    if unsupported:
        raise ValueError(f"Unsupported RynnValue input modes: {sorted(unsupported)}")
    frame_count = int(settings.get("num_frames", 8))
    if frame_count < 1:
        raise ValueError("rynnvalue.num_frames must be at least 1.")
    start_sec = float(settings.get("sampling_start_sec", 0.0))
    end_value = settings.get("sampling_end_sec")
    end_sec = float(end_value) if end_value is not None else None
    output_root = Path(config["output_dir"])
    result_directory = output_root / "rynnvalue"
    result_directory.mkdir(parents=True, exist_ok=True)

    model = RynnValueModel(dict(settings.get("model", {})))
    summaries: list[dict[str, Any]] = []
    values: list[dict[str, Any]] = []
    for topic in topics:
        raw_path = output_root / f"{_slug(topic)}_raw_original.mp4"
        heatmap_path = output_root / f"{_slug(topic)}_heatmap.mp4"
        for input_mode in input_modes:
            if input_mode == "raw":
                images, timestamps = _sample_video(
                    raw_path, frame_count, start_sec, end_sec
                )
            elif input_mode == "heatmap":
                images, timestamps = _sample_video(
                    heatmap_path, frame_count, start_sec, end_sec
                )
            else:
                raw_images, timestamps = _sample_video(
                    raw_path, frame_count, start_sec, end_sec
                )
                heatmaps, _ = _sample_video(
                    heatmap_path, frame_count, start_sec, end_sec
                )
                images = _pair_images(raw_images, heatmaps)

            prediction = model.predict(
                instruction,
                images,
                robot_description,
                camera_description,
            )
            decision = _decision(prediction["match"], prediction["success"])
            summaries.append(
                {
                    "test_bag": config["test_bag"],
                    "topic": topic,
                    "input_mode": input_mode,
                    "decision": decision,
                    "match": prediction["match"],
                    "success": prediction["success"],
                    "video_description": prediction["description"],
                    "analysis_text": prediction["analysis_text"],
                    "initial_remaining_time_seconds": prediction["remaining"][0],
                    "final_remaining_time_seconds": prediction["remaining"][-1],
                }
            )
            for index, (timestamp, remaining) in enumerate(
                zip(timestamps, prediction["remaining"])
            ):
                values.append(
                    {
                        "topic": topic,
                        "input_mode": input_mode,
                        "sample_index": index,
                        "timestamp_sec": timestamp,
                        "remaining_time_seconds": remaining,
                        "relative_time_seconds": (
                            prediction["relative"][index - 1]
                            if index > 0
                            else None
                        ),
                        "entropy": prediction["entropy"][index],
                    }
                )
            print(
                f"{topic} ({input_mode}): decision={decision}, "
                f"match={prediction['match']}, success={prediction['success']}"
            )

    pd.DataFrame(summaries).to_csv(
        result_directory / "rynnvalue_results.csv", index=False
    )
    pd.DataFrame(values).to_csv(
        result_directory / "rynnvalue_values.csv", index=False
    )
    print(f"RynnValue results saved to: {result_directory}")


if __name__ == "__main__":
    main()
