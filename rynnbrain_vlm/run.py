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


def _video_frames(
    path: str | Path,
    count: int,
    start_sec: float = 0.0,
    end_sec: float | None = None,
) -> tuple[list[Image.Image], list[dict[str, Any]]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open heatmap video: {path}")
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 1:
        capture.release()
        raise ValueError(f"Video contains no readable frames: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 1.0
    first = max(0, min(total - 1, int(round(start_sec * fps))))
    last = total - 1
    if end_sec is not None:
        last = max(first, min(last, int(round(end_sec * fps))))
    available = max(0, last - first + 1)
    indices = np.linspace(first, last, min(count, available)).round().astype(int)
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
    return frames, metadata


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
    paths: dict[str, str], topics: list[str], frame_count: int,
    start_sec: float = 0.0, end_sec: float | None = None,
) -> tuple[list[tuple[str, Image.Image]], list[dict[str, Any]]]:
    missing_topics = [topic for topic in topics if topic not in paths]
    if missing_topics:
        raise ValueError(
            "heatmap_video_paths is missing entries for: " + ", ".join(missing_topics)
        )
    sampled_by_topic = {
        topic: _video_frames(paths[topic], frame_count, start_sec, end_sec)
        for topic in topics
    }
    output: list[tuple[str, Image.Image]] = []
    metadata: list[dict[str, Any]] = []
    for time_index in range(frame_count):
        for topic in topics:
            frames, topic_metadata = sampled_by_topic[topic]
            if time_index < len(frames):
                output.append(
                    (f"Time {time_index + 1}, heatmap {_slug(topic)}:", frames[time_index])
                )
                metadata.append({"topic": topic, **topic_metadata[time_index]})
    return output, metadata


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


def _is_meaningful_description(response: str) -> bool:
    """Reject headings/empty generations before they can become task memory."""
    visible = re.sub(r"<think>.*?</think>", " ", response, flags=re.DOTALL | re.IGNORECASE)
    visible = re.sub(r"</?answer>|####", " ", visible, flags=re.IGNORECASE)
    visible = re.sub(r"^\s*answer\s*:\s*", "", visible, flags=re.IGNORECASE)
    words = re.findall(r"[A-Za-z]{2,}", visible)
    return len(visible.strip()) >= 40 and len(words) >= 8


def _description_generation(generation: dict[str, Any]) -> dict[str, Any]:
    """Use conservative decoding for structured descriptions."""
    output = dict(generation)
    output["max_new_tokens"] = max(400, int(output.get("max_new_tokens", 300)))
    output["do_sample"] = False
    output["repetition_penalty"] = 1.0
    output["no_repeat_ngram_size"] = 0
    return output


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
    parser.add_argument("--mode", choices=("memory", "test"), required=True)
    return parser.parse_args()


def _print_exchange(title: str, prompt: str, response: str) -> None:
    bar = "=" * 90
    print(f"\n{bar}\n{title} - PROMPT\n{bar}\n{prompt}")
    print(f"\n{bar}\n{title} - MODEL RESPONSE\n{bar}\n{response}\n{bar}")


def _print_inputs(title: str, images: list[tuple[str, Image.Image]]) -> None:
    print(f"\n{title} - IMAGES SENT TO MODEL ({len(images)}):")
    for index, (label, image) in enumerate(images):
        print(f"  [{index}] {label} size={image.size}")


def _common_config(arguments: argparse.Namespace) -> tuple[dict, dict, int, dict]:
    config = json.loads(arguments.config.read_text(encoding="utf-8"))
    vlm = dict(config.get("rynnbrain", {}))
    if not vlm:
        raise ValueError("Add a 'rynnbrain' section to pipeline_config.json")
    frame_count = int(vlm.get("num_frames", 4))
    if frame_count < 1:
        raise ValueError("rynnbrain.num_frames must be at least 1")
    generation = dict(vlm.get("generation", {}))
    return config, vlm, frame_count, generation


def build_memory(arguments: argparse.Namespace) -> None:
    config, vlm, frame_count, generation = _common_config(arguments)
    reference_bags = list(vlm.get("reference_bags", []))
    memory_topics = list(vlm.get("memory_camera_topics", []))
    if not reference_bags:
        raise ValueError("rynnbrain-memory requires rynnbrain.reference_bags")
    if not memory_topics:
        raise ValueError("rynnbrain-memory requires rynnbrain.memory_camera_topics")
    model = RynnBrainModel(vlm["model"])
    memory_path = Path(vlm["task_memory_path"])
    audit_directory = memory_path.parent / f"{memory_path.stem}_inputs"
    descriptions: list[str] = []
    records: list[dict[str, Any]] = []
    for bag_index, bag_value in enumerate(reference_bags):
        inputs, metadata = _raw_inputs(Path(bag_value), memory_topics, frame_count)
        mode_name = f"reference_{bag_index:02d}"
        _save_inputs(audit_directory, mode_name, inputs)
        _print_inputs(f"NOMINAL DESCRIPTION ({bag_value})", inputs)
        description_generation = _description_generation(generation)
        response = model.generate(inputs, DESCRIPTION_PROMPT, description_generation)
        _print_exchange(f"NOMINAL DESCRIPTION ({bag_value})", DESCRIPTION_PROMPT, response)
        if not _is_meaningful_description(response):
            retry_prompt = (
                DESCRIPTION_PROMPT
                + "\n\nYour previous answer was empty. Inspect the images again and provide every requested field with concrete visible details."
            )
            print("\nNominal description was empty or incomplete; retrying once with a stronger prompt.")
            response = model.generate(inputs, retry_prompt, description_generation)
            _print_exchange(f"NOMINAL DESCRIPTION RETRY ({bag_value})", retry_prompt, response)
        if not _is_meaningful_description(response):
            raise RuntimeError(
                "RynnBrain returned an empty/incomplete nominal description twice. "
                "Task memory was not written, so an invalid reference cannot be used for testing."
            )
        descriptions.append(response)
        records.append({"bag": bag_value, "prompt": DESCRIPTION_PROMPT, "response": response, "selected_frames": metadata})
    if len(descriptions) == 1:
        nominal_description = descriptions[0]
        consolidation_prompt = None
    else:
        consolidation_prompt = (
            "Create one concise canonical nominal robot-task description from these descriptions. "
            "Keep only behavior consistently supported across demonstrations; do not discuss success or failure.\n\n"
            + "\n\n---\n\n".join(descriptions)
        )
        nominal_description = model.text(consolidation_prompt, _description_generation(generation))
        _print_exchange("NOMINAL CONSOLIDATION", consolidation_prompt, nominal_description)
        if not _is_meaningful_description(nominal_description):
            raise RuntimeError("RynnBrain returned an invalid consolidated description; task memory was not written.")
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    memory_path.write_text(json.dumps({
        "nominal_description": nominal_description,
        "description_prompt": DESCRIPTION_PROMPT,
        "reference_calls": records,
        "consolidation_prompt": consolidation_prompt,
        "reference_bags": reference_bags,
        "camera_topics": memory_topics,
        "num_frames": frame_count,
    }, indent=2), encoding="utf-8")
    print(f"\nSaved RynnBrain task memory: {memory_path}")
    print("Memory build finished. No test bag was evaluated.")


def run_test(arguments: argparse.Namespace) -> None:
    config, vlm, frame_count, generation = _common_config(arguments)
    memory_path = Path(vlm["task_memory_path"])
    if not memory_path.is_file():
        raise FileNotFoundError(f"Task memory not found: {memory_path}. Run rynnbrain-memory first.")
    memory = json.loads(memory_path.read_text(encoding="utf-8"))
    nominal_description = str(memory.get("nominal_description", ""))
    if not _is_meaningful_description(nominal_description):
        raise ValueError(
            f"Task memory contains an empty/incomplete nominal description: {memory_path}. "
            "Run rynnbrain-memory again before testing."
        )
    print(f"Loaded RynnBrain task memory: {memory_path}")
    print(f"\nNOMINAL DESCRIPTION USED FOR TESTING:\n{nominal_description}")

    vision_output_directory = Path(config["output_dir"])
    output_directory = Path(vlm.get("output_dir", vision_output_directory / "rynnbrain"))
    output_directory.mkdir(parents=True, exist_ok=True)
    topics = list(vlm.get("camera_topics", config["camera_topics"]))
    if not topics:
        raise ValueError("rynnbrain.camera_topics must contain at least one topic")
    sampling_start_sec = float(vlm.get("sampling_start_sec", 0.0))
    end_value = vlm.get("sampling_end_sec")
    sampling_end_sec = float(end_value) if end_value is not None else None

    source = vlm.get("source", "generated_videos")
    input_modes = list(vlm.get("input_modes", ["raw"]))
    raw_video_paths = {
        topic: str(vision_output_directory / f"{_slug(topic)}_raw_original.mp4")
        for topic in topics
    }
    heatmap_paths = {
        topic: str(vision_output_directory / f"{_slug(topic)}_heatmap.mp4")
        for topic in topics
    }
    raw_video_paths.update(vlm.get("raw_video_paths", {}))
    heatmap_paths.update(vlm.get("heatmap_video_paths", {}))
    if source == "generated_videos":
        raw_inputs = []
        frame_metadata = []
        if any(mode in {"raw", "raw_heatmap"} for mode in input_modes):
            missing_raw = [topic for topic in topics if not Path(raw_video_paths[topic]).is_file()]
            if missing_raw:
                expected = "\n  - ".join(raw_video_paths[topic] for topic in missing_raw)
                raise FileNotFoundError(
                    "Original-resolution raw video(s) are missing; refusing to mix them with "
                    f"square DINO model-input videos. Expected:\n  - {expected}\n"
                    "Run vision-test again with original-video export enabled, or explicitly set "
                    "rynnbrain.raw_video_paths."
                )
            raw_inputs, frame_metadata = _heatmap_inputs(
                raw_video_paths, topics, frame_count, sampling_start_sec, sampling_end_sec
            )
            raw_inputs = [
                (label.replace("heatmap", "raw frame"), image)
                for label, image in raw_inputs
            ]
    elif source == "rosbag":
        raw_inputs, frame_metadata = _raw_inputs(
            Path(config["test_bag"]), topics, frame_count
        )
    else:
        raise ValueError("rynnbrain.source must be 'generated_videos' or 'rosbag'")
    model = RynnBrainModel(vlm["model"])
    rows = []
    raw_records = []
    for mode in input_modes:
        if mode == "raw":
            inputs = raw_inputs
        elif mode == "heatmap":
            inputs, _ = _heatmap_inputs(
                heatmap_paths, topics, frame_count, sampling_start_sec, sampling_end_sec
            )
        elif mode == "raw_heatmap":
            heatmaps, _ = _heatmap_inputs(
                heatmap_paths, topics, frame_count, sampling_start_sec, sampling_end_sec
            )
            inputs = [item for pair in zip(raw_inputs, heatmaps) for item in pair]
        else:
            raise ValueError(f"Unsupported input mode: {mode}")
        _save_inputs(output_directory, mode, inputs)
        prompt = evaluation_prompt(nominal_description, mode)
        _print_inputs(f"TEST EVALUATION ({mode})", inputs)
        response = model.generate(inputs, prompt, generation)
        _print_exchange(f"TEST EVALUATION ({mode})", prompt, response)
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
        raw_records.append({"input_mode": mode, "prompt": prompt, "response": response})
        print(f"{mode}: decision={decision}, confidence={confidence}")

    pd.DataFrame(rows).to_csv(output_directory / "rynnbrain_results.csv", index=False)
    pd.DataFrame(frame_metadata).to_csv(
        output_directory / "selected_vlm_frames.csv", index=False
    )
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


def main() -> None:
    arguments = parse_arguments()
    if arguments.mode == "memory":
        build_memory(arguments)
    else:
        run_test(arguments)


if __name__ == "__main__":
    main()
