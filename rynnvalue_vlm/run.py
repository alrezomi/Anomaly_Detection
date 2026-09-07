"""Run RynnValue on videos already produced by the vision pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import textwrap
from typing import Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from rosbag_io import RosbagImageSource, sample_rosbag_image_frames_uniform
from .model import RynnValueModel


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()


def _sample_video(
    path: Path,
    count: int,
    start_sec: float,
    end_sec: float | None,
) -> tuple[list[Image.Image], list[dict[str, Any]]]:
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
    frame_metadata: list[dict[str, Any]] = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        success, bgr = capture.read()
        if success:
            images.append(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
            frame_metadata.append(
                {
                    "source_video": str(path),
                    "source_frame_index": int(index),
                    "timestamp_sec": float(index) / fps,
                    "video_fps": fps,
                    "video_frame_count": total,
                }
            )
    capture.release()
    if not images:
        raise ValueError(f"No selected frames could be decoded from: {path}")
    return images, frame_metadata


def _sample_bag(
    source: RosbagImageSource,
    count: int,
    start_sec: float,
    end_sec: float | None,
) -> tuple[list[Image.Image], list[dict[str, Any]]]:
    sampled = sample_rosbag_image_frames_uniform(
        source, num_frames=count, start_sec=start_sec, end_sec=end_sec
    )
    images = [image for _, _, image in sampled]
    metadata = [
        {
            "source_video": str(source.bag_path),
            "source_frame_index": frame_index,
            "timestamp_sec": timestamp_sec,
            "video_fps": None,
            "video_frame_count": None,
            "source_kind": "rosbag",
        }
        for frame_index, timestamp_sec, _ in sampled
    ]
    return images, metadata


def _sample_multi_view_bag(
    sources: list[RosbagImageSource],
    count: int,
    start_sec: float,
    end_sec: float | None,
) -> tuple[list[Image.Image], list[dict[str, Any]]]:
    samples = [
        _sample_bag(source, count, start_sec, end_sec) for source in sources
    ]
    sample_count = min(len(images) for images, _ in samples)
    images = [
        _combine_images([sample[0][index] for sample in samples])
        for index in range(sample_count)
    ]
    metadata = []
    for index in range(sample_count):
        camera_metadata = [sample[1][index] for sample in samples]
        metadata.append(
            {
                "sample_index": index,
                "timestamp_sec": camera_metadata[0]["timestamp_sec"],
                "raw_source_video": json.dumps(
                    [item["source_video"] for item in camera_metadata]
                ),
                "raw_frame_index": json.dumps(
                    [item["source_frame_index"] for item in camera_metadata]
                ),
                "raw_timestamp_sec": camera_metadata[0]["timestamp_sec"],
                "raw_video_fps": None,
                "heatmap_source_video": None,
                "heatmap_frame_index": None,
                "heatmap_timestamp_sec": None,
                "heatmap_video_fps": None,
                "input_mode": "multi_view_raw",
                "camera_topics": json.dumps([source.topic for source in sources]),
                "camera_timestamps_sec": json.dumps(
                    [item["timestamp_sec"] for item in camera_metadata]
                ),
            }
        )
    return images, metadata


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


def _combine_images(images: list[Image.Image]) -> Image.Image:
    """Combine one synchronized observation from each camera horizontally."""
    if not images:
        raise ValueError("At least one image is required.")
    height = min(image.height for image in images)
    resized = []
    for image in images:
        rgb = image.convert("RGB")
        width = max(1, round(rgb.width * height / rgb.height))
        resized.append(rgb.resize((width, height), Image.Resampling.BICUBIC))
    canvas = Image.new(
        "RGB", (sum(image.width for image in resized), height), "black"
    )
    x = 0
    for image in resized:
        canvas.paste(image, (x, 0))
        x += image.width
    return canvas


def _decision(match: str | None, success: str | None) -> str:
    match_value = (match or "").lower()
    success_value = (success or "").lower()
    if match_value == "no" or success_value == "no":
        return "failure"
    if match_value == "yes" and success_value == "yes":
        return "success"
    return "uncertain"


def _input_metadata(
    input_mode: str,
    raw_metadata: list[dict[str, Any]] | None,
    heatmap_metadata: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    raw_metadata = raw_metadata or []
    heatmap_metadata = heatmap_metadata or []
    count = len(raw_metadata) if raw_metadata else len(heatmap_metadata)
    if (
        raw_metadata
        and heatmap_metadata
        and len(raw_metadata) != len(heatmap_metadata)
    ):
        raise ValueError("Raw and heatmap frame metadata counts do not match.")

    records: list[dict[str, Any]] = []
    for index in range(count):
        raw = raw_metadata[index] if raw_metadata else None
        heatmap = heatmap_metadata[index] if heatmap_metadata else None
        reference = raw or heatmap
        if reference is None:
            continue
        records.append(
            {
                "sample_index": index,
                "timestamp_sec": reference["timestamp_sec"],
                "raw_source_video": raw["source_video"] if raw else None,
                "raw_frame_index": raw["source_frame_index"] if raw else None,
                "raw_timestamp_sec": raw["timestamp_sec"] if raw else None,
                "raw_video_fps": raw["video_fps"] if raw else None,
                "heatmap_source_video": (
                    heatmap["source_video"] if heatmap else None
                ),
                "heatmap_frame_index": (
                    heatmap["source_frame_index"] if heatmap else None
                ),
                "heatmap_timestamp_sec": (
                    heatmap["timestamp_sec"] if heatmap else None
                ),
                "heatmap_video_fps": heatmap["video_fps"] if heatmap else None,
                "input_mode": input_mode,
            }
        )
    return records


def _frame_label(metadata: dict[str, Any]) -> str:
    parts = [f"sample {metadata['sample_index'] + 1}"]
    if metadata["raw_frame_index"] is not None:
        parts.append(
            f"raw frame {metadata['raw_frame_index']} "
            f"@ {metadata['raw_timestamp_sec']:.2f}s"
        )
    if metadata["heatmap_frame_index"] is not None:
        parts.append(
            f"heatmap frame {metadata['heatmap_frame_index']} "
            f"@ {metadata['heatmap_timestamp_sec']:.2f}s"
        )
    return " | ".join(parts)


def _save_storyboard(
    path: Path,
    images: list[Image.Image],
    frame_metadata: list[dict[str, Any]],
    header_lines: list[str],
) -> None:
    """Save one compact visual record of the exact ordered model inputs."""
    columns = min(4, len(images))
    rows = (len(images) + columns - 1) // columns
    cell_width = 370
    image_height = 245
    caption_height = 58
    margin = 20
    line_height = 17
    wrapped_header = [
        line
        for value in header_lines
        for line in textwrap.wrap(value, width=150) or [""]
    ]
    header_height = (len(wrapped_header) + 1) * line_height + margin
    canvas = Image.new(
        "RGB",
        (
            columns * cell_width + (columns + 1) * margin,
            header_height + rows * (image_height + caption_height + margin),
        ),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    y = margin
    for line in wrapped_header:
        draw.text((margin, y), line, fill="black")
        y += line_height

    for index, (image, metadata) in enumerate(zip(images, frame_metadata)):
        row, column = divmod(index, columns)
        x = margin + column * (cell_width + margin)
        y = header_height + row * (image_height + caption_height + margin)
        preview = image.copy()
        preview.thumbnail((cell_width, image_height), Image.Resampling.LANCZOS)
        image_x = x + (cell_width - preview.width) // 2
        image_y = y + (image_height - preview.height) // 2
        canvas.paste(preview, (image_x, image_y))
        caption = _frame_label(metadata)
        caption_lines = textwrap.wrap(caption, width=54)
        for caption_index, line in enumerate(caption_lines[:3]):
            draw.text(
                (x, y + image_height + 5 + caption_index * line_height),
                line,
                fill="black",
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=92)


def _print_run_settings(
    model_id: str,
    config: dict[str, Any],
    settings: dict[str, Any],
    topics: list[str],
    input_modes: list[str],
    result_directory: Path,
) -> None:
    print("\nRynnValue run", flush=True)
    print(f"  Model: {model_id}", flush=True)
    print(f"  Test bag: {config['test_bag']}", flush=True)
    print(f"  Task instruction: {settings['task_description']}", flush=True)
    print(
        f"  Robot description: "
        f"{settings.get('robot_description') or '(not set)'}",
        flush=True,
    )
    print(
        f"  Camera description: "
        f"{settings.get('camera_description') or '(not set)'}",
        flush=True,
    )
    print(f"  Camera topics: {', '.join(topics)}", flush=True)
    print(f"  Input modes: {', '.join(input_modes)}", flush=True)
    print(f"  Samples per sequence: {settings.get('num_frames', 8)}", flush=True)
    print(
        "  Native questions: remaining time per observation; relative time "
        "between observations; whole-sequence description/Match/Success",
        flush=True,
    )
    print(
        "  Decision scope: one complete sampled sequence per camera or joint view",
        flush=True,
    )
    print(
        "  Wrapper decision rule: failure if Match=No or Success=No",
        flush=True,
    )
    print(f"  Output: {result_directory}", flush=True)


def _print_case_input(
    topic: str,
    input_mode: str,
    metadata: list[dict[str, Any]],
    storyboard_path: Path,
) -> None:
    print(f"\nEvaluating sequence: topic={topic}, mode={input_mode}", flush=True)
    sources = []
    for key in ("raw_source_video", "heatmap_source_video"):
        source = metadata[0].get(key) if metadata else None
        if source:
            sources.append(source)
    print(f"  Source video(s): {', '.join(sources)}", flush=True)
    print("  Ordered model observations:", flush=True)
    for item in metadata:
        print(f"    {_frame_label(item)}", flush=True)
    print(f"  Input storyboard: {storyboard_path}", flush=True)


def _print_case_result(
    prediction: dict[str, Any],
    decision: str,
    metadata: list[dict[str, Any]],
) -> None:
    print("  Whole-sequence analysis:", flush=True)
    print(f"    Video description: {prediction['description']}", flush=True)
    print(f"    Match with task instruction: {prediction['match']}", flush=True)
    print(
        f"    Successful task completion observed: {prediction['success']}",
        flush=True,
    )
    print(f"    Wrapper decision: {decision}", flush=True)
    print("  Per-observation temporal values:", flush=True)
    for index, (item, remaining, entropy) in enumerate(
        zip(metadata, prediction["remaining"], prediction["entropy"])
    ):
        relative = prediction["relative"][index - 1] if index > 0 else None
        relative_text = "n/a" if relative is None else f"{relative:.3f}s"
        print(
            f"    sample {index + 1}: t={item['timestamp_sec']:.2f}s, "
            f"remaining={remaining:.3f}s, "
            f"relative_from_previous={relative_text}, entropy={entropy:.3f}",
            flush=True,
        )


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
    unsupported = set(input_modes) - {
        "raw", "heatmap", "raw_heatmap", "multi_view_raw"
    }
    if unsupported:
        raise ValueError(f"Unsupported RynnValue input modes: {sorted(unsupported)}")
    if "multi_view_raw" in input_modes and len(topics) < 2:
        raise ValueError("multi_view_raw requires at least two camera topics.")
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
    _print_run_settings(
        model.model_id,
        config,
        settings,
        topics,
        input_modes,
        result_directory,
    )
    summaries: list[dict[str, Any]] = []
    values: list[dict[str, Any]] = []
    evaluation_topics = list(topics)
    if "multi_view_raw" in input_modes:
        evaluation_topics.append("__multi_view__")

    for topic in evaluation_topics:
        multi_view = topic == "__multi_view__"
        raw_path = output_root / f"{_slug(topic)}_raw_original.mp4"
        heatmap_path = output_root / f"{_slug(topic)}_heatmap.mp4"
        raw_sample = (
            _sample_multi_view_bag(
                [
                    RosbagImageSource(
                        config["test_bag"], camera_topic, "test", stage_topic=None
                    )
                    for camera_topic in topics
                ],
                frame_count,
                start_sec,
                end_sec,
            )
            if multi_view
            else (
                _sample_bag(
                    RosbagImageSource(
                        config["test_bag"], topic, "test", stage_topic=None
                    ),
                    frame_count,
                    start_sec,
                    end_sec,
                )
                if {"raw", "raw_heatmap"} & set(input_modes)
                else None
            )
        )
        heatmap_sample = (
            _sample_video(heatmap_path, frame_count, start_sec, end_sec)
            if not multi_view and {"heatmap", "raw_heatmap"} & set(input_modes)
            else None
        )
        for input_mode in input_modes:
            if multi_view != (input_mode == "multi_view_raw"):
                continue
            if input_mode == "raw":
                assert raw_sample is not None
                images, raw_metadata = raw_sample
                heatmap_metadata = None
            elif input_mode == "multi_view_raw":
                assert raw_sample is not None
                images, frame_metadata = raw_sample
                raw_metadata = None
                heatmap_metadata = None
            elif input_mode == "heatmap":
                assert heatmap_sample is not None
                images, heatmap_metadata = heatmap_sample
                raw_metadata = None
            else:
                assert raw_sample is not None and heatmap_sample is not None
                raw_images, raw_metadata = raw_sample
                heatmaps, heatmap_metadata = heatmap_sample
                images = _pair_images(raw_images, heatmaps)

            if input_mode != "multi_view_raw":
                frame_metadata = _input_metadata(
                    input_mode, raw_metadata, heatmap_metadata
                )
            model_images = model.prepare_images(images)
            storyboard_path = (
                result_directory
                / "storyboards"
                / f"{_slug(topic)}_{input_mode}_vlm_input.jpg"
            )
            _save_storyboard(
                storyboard_path,
                model_images,
                frame_metadata,
                [
                    f"RynnValue model: {model.model_id}",
                    f"Task instruction: {instruction}",
                    f"Robot description: {robot_description or '(not set)'}",
                    f"Camera description: {camera_description or '(not set)'}",
                    f"Test bag: {config['test_bag']}",
                    f"Camera topic(s): {', '.join(topics) if multi_view else topic}",
                    f"Input mode: {input_mode}",
                    "Scope: all observations below are passed together in "
                    "chronological order; Match and Success describe the whole "
                    "sampled sequence.",
                ],
            )
            _print_case_input(topic, input_mode, frame_metadata, storyboard_path)
            prediction = model.predict(
                instruction,
                model_images,
                robot_description,
                camera_description,
            )
            decision = _decision(prediction["match"], prediction["success"])
            source_record = frame_metadata[0]
            summaries.append(
                {
                    "test_bag": config["test_bag"],
                    "topic": ",".join(topics) if multi_view else topic,
                    "input_mode": input_mode,
                    "decision_scope": "whole_sampled_sequence",
                    "decision": decision,
                    "match": prediction["match"],
                    "success": prediction["success"],
                    "video_description": prediction["description"],
                    "analysis_text": prediction["analysis_text"],
                    "initial_remaining_time_seconds": prediction["remaining"][0],
                    "final_remaining_time_seconds": prediction["remaining"][-1],
                    "sample_count": len(model_images),
                    "model_id": model.model_id,
                    "task_description": instruction,
                    "robot_description": robot_description,
                    "camera_description": camera_description,
                    "sampling_start_sec": start_sec,
                    "sampling_end_sec": end_sec,
                    "raw_source_video": source_record["raw_source_video"],
                    "heatmap_source_video": source_record["heatmap_source_video"],
                    "input_storyboard": str(storyboard_path),
                    "decision_rule": "failure if Match=No or Success=No",
                }
            )
            for index, (metadata, model_image, remaining) in enumerate(
                zip(frame_metadata, model_images, prediction["remaining"])
            ):
                values.append(
                    {
                        "test_bag": config["test_bag"],
                        "topic": topic,
                        "input_mode": input_mode,
                        "sample_index": index,
                        "timestamp_sec": metadata["timestamp_sec"],
                        "raw_source_video": metadata["raw_source_video"],
                        "raw_frame_index": metadata["raw_frame_index"],
                        "raw_timestamp_sec": metadata["raw_timestamp_sec"],
                        "raw_video_fps": metadata["raw_video_fps"],
                        "heatmap_source_video": metadata["heatmap_source_video"],
                        "heatmap_frame_index": metadata["heatmap_frame_index"],
                        "heatmap_timestamp_sec": metadata["heatmap_timestamp_sec"],
                        "heatmap_video_fps": metadata["heatmap_video_fps"],
                        "model_input_width": model_image.width,
                        "model_input_height": model_image.height,
                        "remaining_time_seconds": remaining,
                        "relative_time_seconds": (
                            prediction["relative"][index - 1]
                            if index > 0
                            else None
                        ),
                        "entropy": prediction["entropy"][index],
                    }
                )
            _print_case_result(prediction, decision, frame_metadata)

    pd.DataFrame(summaries).to_csv(
        result_directory / "rynnvalue_results.csv", index=False
    )
    pd.DataFrame(values).to_csv(
        result_directory / "rynnvalue_values.csv", index=False
    )
    print(f"\nRynnValue results saved to: {result_directory}", flush=True)


if __name__ == "__main__":
    main()
