"""Run RynnBrain semantic anomaly evaluation from rosbag/heatmap inputs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any
from uuid import uuid4

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from rosbag_io import RosbagImageSource, sample_rosbag_image_frames_uniform
from .cop_classifier import classifier_method, classifier_model_paths, load_classifier
from .cop_analysis import save_cop_vector
from .model import COP_REPRESENTATION_ID, RynnBrainModel
from .prompts import task_context_prompt, evaluation_prompt_multiturn, visual_evidence_prompt


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


def _print_exchange(title: str, prompt: str, response: str) -> None:
    bar = "=" * 90
    print(f"\n{bar}\n{title} - PROMPT\n{bar}\n{prompt}")
    print(f"\n{bar}\n{title} - MODEL RESPONSE\n{bar}\n{response}\n{bar}")


def _print_inputs(title: str, images: list[tuple[str, Image.Image]]) -> None:
    print(f"\n{title} - IMAGES SENT TO MODEL ({len(images)}):")
    for index, (label, image) in enumerate(images):
        if image is None:
            print(f"  [{index}] {label} (no image for this mode)")
        else:
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


def cop_comparison_signature(model, config, vlm, mode, nominal_response, frame_count, generation):
    """Shared cache identity for classifier preparation and evaluation."""
    task = vlm.get("task_description", "Robot manipulation task")
    topics = list(vlm.get("camera_topics", config.get("camera_topics", [])))
    signature = {
        "representation_id": COP_REPRESENTATION_ID,
        "model_id": model.model_id,
        "model_revision": getattr(model.model.config, "_commit_hash", None),
        "model_config": {key: value for key, value in vlm.get("model", {}).items()
                         if key != "lora_adapter_path" or value},
        "task_description": task,
        "nominal_response": nominal_response,
        "source": vlm.get("source", "generated_videos"),
        "reference_bags": [str(value) for value in vlm.get("reference_bags", [])],
        "test_camera_topics": topics,
        "memory_camera_topics": list(vlm.get("memory_camera_topics", topics)),
        "input_mode": mode,
        "num_frames": frame_count,
        "sampling_start_sec": float(vlm.get("sampling_start_sec", 0.0)),
        "sampling_end_sec": float(vlm["sampling_end_sec"]) if vlm.get("sampling_end_sec") is not None else None,
        "enable_thinking": bool(generation.get("enable_thinking", False)),
        "hidden_size": getattr(getattr(model.model.config, "text_config", None), "hidden_size", None),
        "turn1_prompt": task_context_prompt(task),
        "turn2_prompt": evaluation_prompt_multiturn(task, mode),
    }
    if getattr(model, "adapter_identity", None):
        signature["lora_adapter_sha256"] = model.adapter_identity
    return signature


def evaluate_multiturn(
    model: RynnBrainModel,
    config: dict[str, Any],
    vlm: dict[str, Any],
    frame_count: int,
    generation: dict[str, Any],
    output_directory: Path,
    *,
    training_vectors_only: bool = False,
    reference_response: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], str]:
    """Evaluate one test bag/video set with an already-loaded RynnBrain model.

    Split out from run_test_multiturn so a benchmark driver can load the model
    once and call this per test bag instead of reloading it every time.
    Returns (rows, frame_metadata, raw_records, task_description).
    """
    task_description = vlm.get("task_description", "Robot manipulation task")
    provenance = {
        "evaluation_id": str(uuid4()),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_role": "classifier_training" if training_vectors_only else "evaluation",
        "model_id": model.model_id,
        "lora_adapter_path": getattr(model, "lora_adapter_path", None),
        "lora_adapter_sha256": getattr(model, "adapter_identity", None),
        "nominal_response_source": "base_model",
        "nominal_response_reused": reference_response is not None,
        "response_source": "lora_adapted_model" if getattr(model, "adapter_identity", None) else "base_model",
    }

    if not training_vectors_only and Path(config["test_bag"]).name in getattr(model, "adapter_training_bags", set()):
        raise ValueError("This bag was used to train the loaded LoRA adapter; select a held-out evaluation bag.")

    print("[MULTI-TURN MODE] Nominal reference followed by execution evaluation")
    print(f"Task: {task_description}\n")

    output_directory.mkdir(parents=True, exist_ok=True)
    vision_output_directory = Path(config["output_dir"])
    topics = list(vlm.get("camera_topics", config["camera_topics"]))
    if not topics:
        raise ValueError("rynnbrain.camera_topics must contain at least one topic")
    sampling_start_sec = float(vlm.get("sampling_start_sec", 0.0))
    end_value = vlm.get("sampling_end_sec")
    sampling_end_sec = float(end_value) if end_value is not None else None
    cop_config = dict(vlm.get("cop_vectors", {}))
    capture_cop_vectors = training_vectors_only or bool(cop_config.get("enabled", True))

    source = vlm.get("source", "generated_videos")
    input_modes = list(vlm.get("input_modes", ["raw"]))
    classifier_config = dict(vlm.get("cop_classifier", {}))
    classifier_enabled = not training_vectors_only and bool(classifier_config.get("enabled", False))
    classifiers: dict[str, Any] = {}
    if classifier_enabled:
        if not capture_cop_vectors:
            raise ValueError(
                "rynnbrain.cop_classifier requires rynnbrain.cop_vectors.enabled=true."
            )
        model_paths = classifier_model_paths(classifier_config)
        if not isinstance(model_paths, dict):
            raise ValueError("rynnbrain.cop_classifier.model_paths must be an object.")
        missing_models = [mode for mode in input_modes if not model_paths.get(mode)]
        if missing_models:
            raise ValueError(
                "Missing classifier model path(s) for input mode(s): "
                + ", ".join(missing_models)
            )
        classifiers = {
            mode: load_classifier(Path(str(model_paths[mode]))) for mode in input_modes
        }
        from .cop_knn import CoPKNNDetector
        if any(isinstance(classifier, CoPKNNDetector) != (classifier_method(classifier_config) == "knn") for classifier in classifiers.values()):
            raise ValueError("Saved classifier type does not match cop_classifier.method; train the selected method first.")
        incompatible_modes = [
            mode for mode, classifier in classifiers.items()
            if classifier.input_mode != mode
        ]
        if incompatible_modes:
            raise ValueError(
                "Classifier file input mode does not match configured mode(s): "
                + ", ".join(incompatible_modes)
            )
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

    reference_bags = list(vlm.get("reference_bags", []))
    if not reference_bags:
        raise ValueError(
            "rynnbrain.reference_bags must list at least one nominal demonstration bag "
            "for multi-turn evaluation (it supplies the turn-1 'nominal demonstration' images)."
        )
    memory_topics = list(vlm.get("memory_camera_topics", topics))
    nominal_images: list[tuple[str, Image.Image]] = []
    for bag_value in reference_bags:
        bag_images, _ = _raw_inputs(Path(bag_value), memory_topics, frame_count)
        nominal_images.extend(bag_images)
    _save_inputs(output_directory, "nominal", nominal_images)

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

        # Turn 1: Model sees nominal demonstration and learns the task
        turn1_text = task_context_prompt(task_description)
        
        # Turn 2: Model evaluates test case against observed nominal
        turn2_text = evaluation_prompt_multiturn(task_description, mode)
        
        # Create multi-turn conversation
        turns = [
            {
                "role": "user",
                "images": nominal_images,
                "text": turn1_text
            },
            {
                "role": "user",
                "images": inputs,
                "text": turn2_text
            }
        ]
        
        _print_inputs(f"MULTITURN NOMINAL ({mode})", nominal_images)
        _print_inputs(f"MULTITURN TEST ({mode})", inputs)
        
        # Use multi-turn generation. When requested, capture the final normalized
        # decoder state from the turn-2 prompt prefill without an extra forward pass.
        cop_vector = None
        if capture_cop_vectors:
            options = {"reference_response": reference_response} if reference_response is not None else {}
            generated = model.generate_multiturn_with_cop_vector(turns, generation, **options)
            nominal_response = generated.nominal_response
            response = generated.evaluation_response
            cop_vector = generated.cop_vector
        else:
            nominal_response, response = model.generate_multiturn(turns, generation)
        
        # Display prompts and response
        prompt_display = f"[Turn 1] Nominal demonstration:\n{turn1_text}\n\n[Turn 2] Test evaluation:\n{turn2_text}"
        _print_exchange(
            f"MULTITURN TURN 1 ({mode})", turn1_text, nominal_response
        )
        _print_exchange(f"MULTITURN TURN 2 ({mode})", turn2_text, response)
        
        decision, confidence = _parse_response(response)
        cop_vector_path: str | None = None
        cop_metadata_path: str | None = None
        classifier_failure_probability: float | None = None
        classifier_decision: str | None = None
        classifier_threshold: float | None = None
        classifier_model_path: str | None = None
        classifier_error: str | None = None
        classifier_anomaly_score: float | None = None
        score_kind = "knn_distance" if classifier_enabled and classifier_method(classifier_config) == "knn" else "failure_probability"
        if cop_vector is not None:
            text_config = getattr(model.model.config, "text_config", None)
            model_revision = getattr(model.model.config, "_commit_hash", None)
            comparison_signature = cop_comparison_signature(
                model, config, vlm, mode, nominal_response, frame_count, generation
            )
            if classifier_enabled:
                classifier = classifiers[mode]
                classifier_threshold = classifier.threshold
                classifier_model_path = str(
                    Path(str(model_paths[mode])).resolve()
                )
                try:
                    if classifier.representation_id != COP_REPRESENTATION_ID:
                        raise ValueError("Classifier representation does not match the captured CoP vector.")
                    if score_kind == "knn_distance":
                        classifier_anomaly_score = classifier.predict_anomaly_score(
                            cop_vector.numpy(), input_mode=mode, comparison_signature=comparison_signature,
                        )
                        classifier_decision = classifier.predict_label(classifier_anomaly_score)
                    else:
                        classifier_failure_probability = classifier.predict_failure_probability(
                            cop_vector.numpy(), input_mode=mode, comparison_signature=comparison_signature,
                        )
                        classifier_decision = classifier.predict_label(classifier_failure_probability)
                except ValueError as error:
                    # Incompatible scoring must not discard a completed VLM answer.
                    # Leave the probability empty and report the reason explicitly.
                    classifier_failure_probability = None
                    classifier_anomaly_score = None
                    classifier_error = str(error)
                    print(f"Classifier scoring failed ({mode}); keeping VLM response and vector: {error}")
            vector_path, metadata_path = save_cop_vector(
                output_directory,
                mode,
                cop_vector.numpy(),
                {
                    **provenance,
                    "representation_id": COP_REPRESENTATION_ID,
                    "representation_description": (
                        "Post-final-normalization language-decoder state at the "
                        "last non-padding token of the complete turn-2 prompt, "
                        "captured before answer generation."
                    ),
                    "model_id": model.model_id,
                    "model_revision": model_revision,
                    "bag_name": Path(config["test_bag"]).name,
                    "test_bag": str(config["test_bag"]),
                    "ground_truth_label": vlm.get("ground_truth_label", "unknown"),
                    "decision": decision,
                    "confidence": confidence,
                    "classifier_failure_probability": classifier_failure_probability,
                    "classifier_decision": classifier_decision,
                    "classifier_threshold": classifier_threshold,
                    "classifier_model_path": classifier_model_path,
                    "classifier_error": classifier_error,
                    "classifier_anomaly_score": classifier_anomaly_score,
                    "classifier_score_kind": score_kind if classifier_enabled else None,
                    "comparison_signature": comparison_signature,
                },
            )
            cop_vector_path = str(vector_path)
            cop_metadata_path = str(metadata_path)
            print(
                f"Saved {mode} CoP vector ({cop_vector.numel()} values): "
                f"{vector_path}"
            )
        visual_evidence = None
        evidence_error = None
        evidence_prompt = None
        evidence_source = None
        if not training_vectors_only and generation.get("explain_decision", False):
            evidence_prompt = visual_evidence_prompt(task_description, mode)
            evidence_source = "base_model_separate_pass"
            try:
                visual_evidence = model.generate_visual_evidence(
                    turns, nominal_response, evidence_prompt, generation
                )
                _print_exchange(
                    f"VISUAL EVIDENCE ({mode}) - BASE MODEL, SEPARATE PASS",
                    evidence_prompt, visual_evidence,
                )
            except RuntimeError as error:
                # An optional extra generation must not discard the decision
                # and vector already produced (e.g. if this pass runs out of VRAM).
                evidence_error = str(error)
                print(f"Visual evidence unavailable; keeping the evaluation: {error}")
        rows.append(
            {
                "test_bag": config["test_bag"],
                "input_mode": mode,
                "decision": decision,
                "confidence": confidence,
                "ground_truth_label": vlm.get("ground_truth_label", ""),
                "nominal_response": nominal_response,
                "response": response,
                "visual_evidence": visual_evidence,
                "visual_evidence_source": evidence_source,
                "visual_evidence_error": evidence_error,
                "evaluation_method": "multiturn",
                "cop_representation_id": (
                    COP_REPRESENTATION_ID if cop_vector is not None else None
                ),
                "cop_vector_dimension": (
                    int(cop_vector.numel()) if cop_vector is not None else None
                ),
                "cop_vector_path": cop_vector_path,
                "cop_metadata_path": cop_metadata_path,
                "classifier_failure_probability": classifier_failure_probability,
                "classifier_failure_percent": (
                    classifier_failure_probability * 100.0
                    if classifier_failure_probability is not None
                    else None
                ),
                "classifier_decision": classifier_decision,
                "classifier_threshold": classifier_threshold,
                "classifier_model_path": classifier_model_path,
                "classifier_error": classifier_error,
                "classifier_anomaly_score": classifier_anomaly_score,
                "classifier_score_kind": score_kind if classifier_enabled else None,
                **provenance,
            }
        )
        raw_records.append({
            **provenance,
            "input_mode": mode,
            "evaluation_method": "multiturn",
            "turns": [
                {
                    "role": "user",
                    "prompt": turn1_text,
                    "images": [
                        {"label": label, "size": list(image.size)}
                        for label, image in nominal_images
                    ],
                    "response": nominal_response,
                    "response_source": "base_model",
                },
                {
                    "role": "user",
                    "prompt": turn2_text,
                    "response": response,
                    "response_source": provenance["response_source"],
                    "images": [
                        {"label": label, "size": list(image.size)}
                        for label, image in inputs
                    ],
                },
            ],
            "prompt": prompt_display,
            "nominal_response": nominal_response,
            "response": response,
            "visual_evidence": visual_evidence,
            "visual_evidence_source": evidence_source,
            "visual_evidence_prompt": evidence_prompt,
            "visual_evidence_error": evidence_error,
            "cop_representation_id": (
                COP_REPRESENTATION_ID if cop_vector is not None else None
            ),
            "cop_vector_path": cop_vector_path,
            "cop_metadata_path": cop_metadata_path,
            "classifier_failure_probability": classifier_failure_probability,
            "classifier_failure_percent": (
                classifier_failure_probability * 100.0
                if classifier_failure_probability is not None
                else None
            ),
            "classifier_decision": classifier_decision,
            "classifier_threshold": classifier_threshold,
            "classifier_model_path": classifier_model_path,
            "classifier_error": classifier_error,
            "classifier_anomaly_score": classifier_anomaly_score,
            "classifier_score_kind": score_kind if classifier_enabled else None,
        })
        print(f"{mode} (multiturn): decision={decision}, confidence={confidence}")
        if classifier_failure_probability is not None:
            print(
                f"{mode} (frozen-vector logistic classifier): "
                f"failure={classifier_failure_probability * 100.0:.2f}%, "
                f"decision={classifier_decision}"
            )
        if classifier_anomaly_score is not None:
            print(f"{mode} (nominal kNN): anomaly distance={classifier_anomaly_score:.6g}, "
                  f"threshold={classifier_threshold:.6g}, decision={classifier_decision} (not a probability)")

    return rows, frame_metadata, raw_records, task_description


def write_multiturn_outputs(
    output_directory: Path,
    rows: list[dict[str, Any]],
    frame_metadata: list[dict[str, Any]],
    raw_records: list[dict[str, Any]],
    task_description: str,
    *,
    merge_modes: bool = False,
) -> None:
    """Persist the same CSV/JSON files run_test_multiturn has always written."""
    if merge_modes:
        # Classifier preparation extracts one mode at a time in the same folder.
        # Keep the other modes' responses when updating this mode.
        modes = {row["input_mode"] for row in rows}
        try:
            previous = pd.read_csv(output_directory / "rynnbrain_results_multiturn.csv", keep_default_na=False).to_dict("records")
            rows = [row for row in previous if row.get("input_mode") not in modes] + rows
        except (OSError, ValueError):
            pass
        try:
            previous = json.loads((output_directory / "rynnbrain_responses_multiturn.json").read_text(encoding="utf-8"))
            raw_records = [row for row in previous["results"] if row.get("input_mode") not in modes] + raw_records
        except (OSError, ValueError, KeyError, TypeError):
            pass
    pd.DataFrame(rows).to_csv(output_directory / "rynnbrain_results_multiturn.csv", index=False)
    pd.DataFrame(frame_metadata).to_csv(
        output_directory / "selected_vlm_frames.csv", index=False
    )
    (output_directory / "rynnbrain_responses_multiturn.json").write_text(
        json.dumps(
            {
                "task_description": task_description,
                "evaluation_method": "multiturn (visual memory - no saved description)",
                "selected_raw_frames": frame_metadata,
                "results": raw_records
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Results saved to: {output_directory}")


def multiturn_outputs_match(output_directory: Path, metadata: dict[str, Any]) -> bool:
    """Only reuse a vector when its original full response was also persisted."""
    evaluation_id = metadata.get("evaluation_id")
    if not evaluation_id or not (output_directory / "selected_vlm_frames.csv").is_file():
        return False
    try:
        saved = json.loads((output_directory / "rynnbrain_responses_multiturn.json").read_text(encoding="utf-8"))
        records = pd.read_csv(output_directory / "rynnbrain_results_multiturn.csv", keep_default_na=False).to_dict("records")
        mode = metadata["input_mode"]
        responses = [row for row in saved["results"] if row.get("input_mode") == mode and row.get("evaluation_id") == evaluation_id]
        rows = [row for row in records if row.get("input_mode") == mode and row.get("evaluation_id") == evaluation_id]
        return (len(responses) == len(rows) == 1
                and isinstance(responses[0].get("response"), str)
                and rows[0].get("response") == responses[0]["response"]
                and rows[0].get("nominal_response") == responses[0].get("nominal_response"))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def run_test_multiturn(arguments: argparse.Namespace) -> None:
    """
    Run test evaluation using multi-turn conversation with visual memory.
    The model sees nominal frames first, then evaluates test frames against them.
    No need for saved nominal description - model uses visual understanding.
    """
    config, vlm, frame_count, generation = _common_config(arguments)
    vision_output_directory = Path(config["output_dir"])
    output_directory = Path(vlm.get("output_dir", vision_output_directory / "rynnbrain_multiturn"))
    model = RynnBrainModel(vlm["model"])
    rows, frame_metadata, raw_records, task_description = evaluate_multiturn(
        model, config, vlm, frame_count, generation, output_directory
    )
    write_multiturn_outputs(output_directory, rows, frame_metadata, raw_records, task_description)


def main() -> None:
    arguments = parse_arguments()
    run_test_multiturn(arguments)


if __name__ == "__main__":
    main()
