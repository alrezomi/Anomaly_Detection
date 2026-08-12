"""Run RynnBrain semantic anomaly evaluation from rosbag/heatmap inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from rosbag_io import RosbagImageSource, sample_rosbag_image_frames_uniform
from .model import RynnBrainModel
from .prompts import DESCRIPTION_PROMPT, evaluation_prompt


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()


def _video_frames(path: str | Path, count: int) -> list[Image.Image]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open heatmap video: {path}")
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, max(total - 1, 0), min(count, total)).round().astype(int)
    frames: list[Image.Image] = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        success, bgr = capture.read()
        if success:
            frames.append(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    capture.release()
    return frames


def _raw_inputs(
    bag: Path, topics: list[str], frame_count: int
) -> tuple[list[tuple[str, Image.Image]], list[dict[str, Any]]]:
    inputs: list[tuple[str, Image.Image]] = []
    metadata: list[dict[str, Any]] = []
    sampled_by_topic = {
        topic: sample_rosbag_image_frames_uniform(
            RosbagImageSource(bag, topic), num_frames=frame_count
        )
        for topic in topics
    }
    for time_index in range(frame_count):
        for topic in topics:
            frames = sampled_by_topic[topic]
            if time_index >= len(frames):
                continue
            frame_id, timestamp, image = frames[time_index]
            label = f"Time {time_index + 1}, camera {_slug(topic)}:"
            inputs.append((label, image))
            metadata.append(
                {"topic": topic, "frame_id": frame_id, "timestamp_sec": timestamp}
            )
    return inputs, metadata


def _heatmap_inputs(
    paths: dict[str, str], topics: list[str], frame_count: int
) -> list[tuple[str, Image.Image]]:
    missing_topics = [topic for topic in topics if topic not in paths]
    if missing_topics:
        raise ValueError(
            "heatmap_video_paths is missing entries for: " + ", ".join(missing_topics)
        )
    frames_by_topic = {
        topic: _video_frames(paths[topic], frame_count) for topic in topics
    }
    output: list[tuple[str, Image.Image]] = []
    for time_index in range(frame_count):
        for topic in topics:
            frames = frames_by_topic[topic]
            if time_index < len(frames):
                output.append(
                    (f"Time {time_index + 1}, heatmap {_slug(topic)}:", frames[time_index])
                )
    return output


def _parse_response(response: str) -> tuple[str, str]:
    decision_match = re.search(
        r"Decision\s*:\s*(success|failure|uncertain)", response, re.IGNORECASE
    )
    confidence_match = re.search(
        r"Confidence\s*:\s*(high|medium|low)", response, re.IGNORECASE
    )
    return (
        decision_match.group(1).lower() if decision_match else "not_parsed",
        confidence_match.group(1).lower() if confidence_match else "not_parsed",
    )


def _save_inputs(
    output_directory: Path,
    mode: str,
    images: list[tuple[str, Image.Image]],
) -> None:
    directory = output_directory / "selected_inputs" / mode
    directory.mkdir(parents=True, exist_ok=True)
    for index, (_, image) in enumerate(images):
        image.save(directory / f"{index:03d}.jpg", quality=92)

    # One human-readable overview containing the exact images and ordering sent
    # to the model. Individual full-resolution inputs remain beside it.
    thumb_width, thumb_height = 320, 240
    label_height = 46
    columns = min(4, max(1, len(images)))
    rows = (len(images) + columns - 1) // columns
    sheet = Image.new(
        "RGB", (columns * thumb_width, rows * (thumb_height + label_height)), "white"
    )
    draw = ImageDraw.Draw(sheet)
    for index, (label, image) in enumerate(images):
        column, row = index % columns, index // columns
        preview = image.convert("RGB").copy()
        preview.thumbnail((thumb_width, thumb_height))
        x = column * thumb_width + (thumb_width - preview.width) // 2
        y = row * (thumb_height + label_height)
        sheet.paste(preview, (x, y))
        draw.text((column * thumb_width + 5, y + thumb_height + 4), label[:55], fill="black")
    sheet.save(directory / "vlm_input_storyboard.jpg", quality=92)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    config = json.loads(arguments.config.read_text(encoding="utf-8"))
    vlm = dict(config.get("rynnbrain", {}))
    if not vlm:
        raise ValueError("Add a 'rynnbrain' section to pipeline_config.json")
    vision_output_directory = Path(config["output_dir"])
    output_directory = Path(
        vlm.get("output_dir", vision_output_directory / "rynnbrain")
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    topics = list(vlm.get("camera_topics", config["camera_topics"]))
    memory_topics = list(vlm.get("memory_camera_topics", config["camera_topics"]))
    frame_count = int(vlm.get("num_frames", 4))
    if not topics:
        raise ValueError("camera_topics must contain at least one rosbag image topic")
    if frame_count < 1:
        raise ValueError("num_frames must be at least 1")
    generation = dict(vlm.get("generation", {}))
    model = RynnBrainModel(vlm["model"])

    configured_description = str(vlm.get("task_description", "")).strip()
    memory_path = Path(vlm.get("task_memory_path", "/outputs/rynnbrain_memories/task.json"))
    rebuild_memory = bool(vlm.get("rebuild_task_memory", False))
    if configured_description:
        nominal_description = configured_description
        print("Using task_description from pipeline_config.json")
    elif memory_path.is_file() and not rebuild_memory:
        memory = json.loads(memory_path.read_text(encoding="utf-8"))
        nominal_description = memory["nominal_description"]
        print(f"Loaded nominal task description: {memory_path}")
    else:
        descriptions = []
        reference_bags = list(vlm.get("reference_bags", config["nominal_bags"][:1]))
        if not reference_bags:
            raise ValueError(
                "Set rynnbrain.task_description or provide at least one reference bag"
            )
        for bag_value in reference_bags:
            inputs, _ = _raw_inputs(Path(bag_value), memory_topics, frame_count)
            descriptions.append(model.generate(inputs, DESCRIPTION_PROMPT, generation))
        if len(descriptions) == 1:
            nominal_description = descriptions[0]
        else:
            nominal_description = model.text(
                "Create one concise canonical nominal robot-task description from these descriptions. Keep only behavior consistently supported across demonstrations; do not discuss success or failure.\n\n"
                + "\n\n---\n\n".join(descriptions),
                generation,
            )
        memory_path.parent.mkdir(parents=True, exist_ok=True)
        memory_path.write_text(
            json.dumps(
                {
                    "nominal_description": nominal_description,
                    "source_descriptions": descriptions,
                    "reference_bags": reference_bags,
                    "camera_topics": memory_topics,
                    "num_frames": frame_count,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Saved nominal task description: {memory_path}")

    source = vlm.get("source", "generated_videos")
    raw_video_paths = {
        topic: str(vision_output_directory / f"{_slug(topic)}_raw_original.mp4")
        for topic in topics
    }
    # Backward compatibility for experiments created before original-resolution
    # exports were added.
    for topic, original_path in list(raw_video_paths.items()):
        if not Path(original_path).is_file():
            raw_video_paths[topic] = str(
                vision_output_directory / f"{_slug(topic)}_model_input.mp4"
            )
    heatmap_paths = {
        topic: str(vision_output_directory / f"{_slug(topic)}_heatmap.mp4")
        for topic in topics
    }
    raw_video_paths.update(vlm.get("raw_video_paths", {}))
    heatmap_paths.update(vlm.get("heatmap_video_paths", {}))
    if source == "generated_videos":
        raw_inputs = _heatmap_inputs(raw_video_paths, topics, frame_count)
        raw_inputs = [
            (label.replace("heatmap", "raw frame"), image)
            for label, image in raw_inputs
        ]
        frame_metadata = [
            {"source_video": raw_video_paths[topic]}
            for _ in range(frame_count)
            for topic in topics
        ][: len(raw_inputs)]
    elif source == "rosbag":
        raw_inputs, frame_metadata = _raw_inputs(
            Path(config["test_bag"]), topics, frame_count
        )
    else:
        raise ValueError("rynnbrain.source must be 'generated_videos' or 'rosbag'")
    rows = []
    raw_records = []
    for mode in vlm.get("input_modes", ["raw"]):
        if mode == "raw":
            inputs = raw_inputs
        elif mode == "heatmap":
            inputs = _heatmap_inputs(heatmap_paths, topics, frame_count)
        elif mode == "raw_heatmap":
            heatmaps = _heatmap_inputs(heatmap_paths, topics, frame_count)
            inputs = [item for pair in zip(raw_inputs, heatmaps) for item in pair]
        else:
            raise ValueError(f"Unsupported input mode: {mode}")
        _save_inputs(output_directory, mode, inputs)
        response = model.generate(
            inputs, evaluation_prompt(nominal_description, mode), generation
        )
        decision, confidence = _parse_response(response)
        rows.append(
            {
                "test_bag": config["test_bag"],
                "input_mode": mode,
                "decision": decision,
                "confidence": confidence,
                "ground_truth_label": vlm.get("ground_truth_label", ""),
                "response": response,
            }
        )
        raw_records.append({"input_mode": mode, "response": response})
        print(f"{mode}: decision={decision}, confidence={confidence}")

    pd.DataFrame(rows).to_csv(output_directory / "rynnbrain_results.csv", index=False)
    (output_directory / "rynnbrain_responses.json").write_text(
        json.dumps(
            {
                "nominal_description": nominal_description,
                "selected_raw_frames": frame_metadata,
                "results": raw_records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Results saved to: {output_directory}")


if __name__ == "__main__":
    main()
