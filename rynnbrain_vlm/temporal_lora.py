"""Optional timestamp and prefix supervision shared by LoRA training/inference."""

import json
import math
from pathlib import Path
import re

from .failure_timing import read_failure_annotation

PROTOCOL = "timestamped_multiview_prefix_decision_onset_v1"
VLM_TIME_FIELDS = ("vlm_failure_onset_sec", "vlm_failure_onset_status",
                   "vlm_failure_time_error_sec", "vlm_failure_time_absolute_error_sec")


def training_profile(config, settings):
    enabled = settings.get("temporal_supervision", False)
    if not isinstance(enabled, bool):
        raise ValueError("lora.training.temporal_supervision must be true or false.")
    if not enabled:
        return {}
    vlm = config["rynnbrain"]
    if vlm.get("source") != "rosbag" or settings["input_mode"] != "raw":
        raise ValueError("Temporal LoRA requires source=rosbag and training.input_mode=raw.")
    topics = vlm.get("camera_topics", config.get("camera_topics", []))
    memory = vlm.get("memory_camera_topics", topics)
    for name, values in (("camera_topics", topics), ("memory_camera_topics", memory)):
        if (not isinstance(values, list) or not values or any(not isinstance(value, str) or not value for value in values)
                or len(set(values)) != len(values)):
            raise ValueError(f"Temporal LoRA requires nonempty, unique {name}.")
    if memory != topics:
        raise ValueError("Temporal LoRA requires the same ordered camera_topics and memory_camera_topics.")
    return {"protocol": PROTOCOL, "camera_topics": list(topics), "memory_camera_topics": list(memory),
            "num_frames": int(vlm.get("num_frames", 4)), "input_mode": "raw", "source": "rosbag",
            "timestamp_decimals": 3}


def adapter_profile(adapter_path):
    manifest = json.loads((Path(adapter_path) / "training_manifest.json").read_text(encoding="utf-8"))
    profile = manifest.get("temporal_profile", {})
    if profile and profile.get("protocol") != PROTOCOL:
        raise ValueError("Unsupported temporal LoRA protocol; update the code before loading this adapter.")
    return profile


def validate_profile(profile, config, vlm, frame_count, input_modes):
    if not profile:
        return
    topics = list(vlm.get("camera_topics", config.get("camera_topics", [])))
    if (vlm.get("source") != "rosbag" or list(input_modes) != ["raw"]
            or topics != profile["camera_topics"]
            or list(vlm.get("memory_camera_topics", topics)) != profile["memory_camera_topics"]
            or frame_count != profile["num_frames"]):
        raise ValueError("Temporal adapter inputs differ from training. Use source=rosbag, input_modes=['raw'], "
                         "and the same camera_topics, memory_camera_topics and num_frames as its training_manifest.json.")


def timestamped_images(images, metadata, topics, frame_count):
    """Keep actual per-camera timestamps; never claim asynchronous views are simultaneous."""
    if len(images) != len(metadata) or len(images) != len(topics) * frame_count:
        raise ValueError("Every selected camera must provide num_frames images for temporal LoRA.")
    result, seen = [], set()
    for (_, image), frame in zip(images, metadata):
        topic, step, time = frame.get("topic"), frame.get("sample_index"), float(frame["timestamp_sec"])
        if (topic not in topics or type(step) is not int or not 0 <= step < frame_count
                or (topic, step) in seen or not math.isfinite(time) or time < 0):
            raise ValueError("Invalid or duplicate temporal camera/frame timestamp.")
        seen.add((topic, step))
        result.append((f"Frame {step + 1}, camera {topic}, time {time:.3f} s from this bag start:", image))
    for topic in topics:
        times = [float(row["timestamp_sec"]) for row in sorted(metadata, key=lambda row: row["sample_index"]) if row["topic"] == topic]
        if any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError("Each camera's sampled timestamps must strictly increase.")
    return result


def temporal_annotation(row, config, startup_ignore_sec):
    annotation = read_failure_annotation(row["bag_path"], config.get("stage_topic", "/recording_stage"), startup_ignore_sec)
    if annotation["status"] == "unavailable":
        raise ValueError(f"Cannot read temporal labels for {row['bag_name']}: {annotation.get('error')}")
    if row["label"] == "fail" and annotation["status"] != "available":
        raise ValueError(f"Failure bag {row['bag_name']} has no eligible error-button marker.")
    if row["label"] == "normal" and annotation["status"] == "available":
        raise ValueError(f"Nominal bag {row['bag_name']} contains an error-button marker; correct the training label.")
    return annotation


def prefix_samples(row, images, metadata, profile, annotation):
    from .cop_timeline import selected_frame_prefixes

    timestamped_images(images, metadata, profile["camera_topics"], profile["num_frames"])
    snapshots = selected_frame_prefixes(metadata, len(images), profile["num_frames"])
    onset = annotation.get("time_sec")
    first_time = min(float(frame["timestamp_sec"]) for frame in metadata)
    if onset is not None and not first_time <= onset <= snapshots[-1]["timestamp_sec"]:
        raise ValueError(f"Error marker for {row['bag_name']} lies outside the sampled camera window.")
    samples = []
    for snapshot in snapshots:
        failed = onset is not None and snapshot["timestamp_sec"] >= onset
        # The input carries frame times only; the annotation occurs exclusively in the answer.
        samples.append({**row, "bag_label": row["label"], "label": "fail" if failed else "normal",
                        "prefix_index": snapshot["sample_index"], "cutoff_sec": snapshot["timestamp_sec"],
                        "images": [images[index] for index in snapshot["image_indices"]],
                        "target_text": temporal_answer("fail" if failed else "normal", onset if failed else None)})
    return samples


def temporal_answer(label, onset):
    if label not in {"normal", "fail"}:
        raise ValueError("Temporal label must be normal or fail.")
    if (label == "fail" and (onset is None or not math.isfinite(onset) or onset < 0)) or (label == "normal" and onset is not None):
        raise ValueError("Temporal answer must pair failure with a finite onset, or success with none.")
    return f"Decision: {'failure' if label == 'fail' else 'success'}\nFailure onset (s): {onset:.3f}" if label == "fail" else "Decision: success\nFailure onset (s): none"


def evaluate_vlm_onset(response, decision, metadata, annotation, ground_truth):
    """Evaluate the model's generated time separately from classifier threshold alerts."""
    result = dict.fromkeys(VLM_TIME_FIELDS)
    result["vlm_failure_onset_status"] = "not_parsed"
    matches = re.findall(r"^\s*Failure onset \(s\):\s*(.*?)\s*$", response, flags=re.MULTILINE | re.IGNORECASE)
    if len(matches) != 1:
        return result
    value = matches[0].strip().lower()
    if value in {"none", "unknown"}:
        result["vlm_failure_onset_status"] = "not_detected" if decision == "success" else "onset_unavailable"
        if decision == "success" and ground_truth in {"normal", "nominal"}:
            result["vlm_failure_onset_status"] = "normal_no_failure"
        return result
    try:
        time = float(value)
    except ValueError:
        return result
    if not math.isfinite(time) or time < 0 or not metadata or time > max(float(frame["timestamp_sec"]) for frame in metadata) + .0005:
        return {**result, "vlm_failure_onset_status": "invalid_time"}
    if decision != "failure":
        return {**result, "vlm_failure_onset_status": "decision_time_conflict"}
    result["vlm_failure_onset_sec"] = time
    if ground_truth in {"normal", "nominal"}:
        return {**result, "vlm_failure_onset_status": "false_alert"}
    if annotation.get("status") != "available":
        return {**result, "vlm_failure_onset_status": "annotation_unavailable"}
    recorded = annotation["time_sec"]
    if not min(float(frame["timestamp_sec"]) for frame in metadata) <= recorded <= max(float(frame["timestamp_sec"]) for frame in metadata):
        return {**result, "vlm_failure_onset_status": "annotation_outside_window"}
    error = time - recorded
    return {**result, "vlm_failure_onset_status": "compared", "vlm_failure_time_error_sec": error,
            "vlm_failure_time_absolute_error_sec": abs(error)}
