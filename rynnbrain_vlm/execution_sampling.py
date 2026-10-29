"""One causal recording-time sampling policy for interval-trained executions."""

import math
from pathlib import Path


INTERVAL_PROTOCOL = "bag_interval_full_history_v1"


def execution_sampling(vlm):
    interval = vlm.get("execution_interval_sec")
    if interval is None:
        return None
    if type(interval) not in (int, float) or not math.isfinite(interval) or interval <= 0:
        raise ValueError("rynnbrain.execution_interval_sec must be finite and positive.")
    start = vlm.get("sampling_start_sec", 0.)
    end = vlm.get("sampling_end_sec")
    if type(start) not in (int, float) or not math.isfinite(start) or start < 0:
        raise ValueError("sampling_start_sec must be finite and nonnegative.")
    if end is not None and (type(end) not in (int, float) or not math.isfinite(end) or end < start):
        raise ValueError("sampling_end_sec must be finite and at least sampling_start_sec.")
    return dict(protocol=INTERVAL_PROTOCOL, interval_sec=float(interval), start_sec=float(start),
                end_sec=float(end) if end is not None else None, context_frames=None)


def interval_execution_inputs(bag, topics, policy, *, frames=None):
    """Select the first available image at/after each due time, retaining real times.

    The next due time follows the actual selected frame, matching cached replay.
    Reference demonstrations retain their separate num_frames sampling.
    """
    from rosbag_io import RosbagImageSource, iter_rosbag_image_frames
    if len(topics) != 1:
        raise ValueError("Interval execution sampling currently requires one camera topic.")
    topic = topics[0]
    due = policy["start_sec"]
    source = frames if frames is not None else iter_rosbag_image_frames(
        RosbagImageSource(Path(bag), topic), sample_fps=15., start_sec=due, end_sec=policy["end_sec"])
    images, metadata = [], []
    try:
        for frame_id, timestamp, image in source:
            if not math.isfinite(timestamp) or timestamp < 0:
                raise ValueError("Invalid execution timestamp.")
            if timestamp < due:
                continue
            if policy["end_sec"] is not None and timestamp > policy["end_sec"]:
                break
            step = len(images)
            images.append((f"Frame {step + 1}, camera {topic}, time {timestamp:.3f} s from this bag start:", image))
            metadata.append(dict(topic=topic, frame_id=int(frame_id), timestamp_sec=float(timestamp), sample_index=step))
            due = float(timestamp) + policy["interval_sec"]
    finally:
        if hasattr(source, "close"):
            source.close()
    if not images:
        raise ValueError(f"No eligible interval observations in {bag}.")
    return images, metadata


def execution_step_count(metadata, fallback, sampling=None):
    return len(metadata) if sampling and sampling.get("protocol") == INTERVAL_PROTOCOL else fallback


def validate_interval_replay(vlm):
    policy = execution_sampling(vlm)
    if not policy:
        return
    options = vlm.get("online_replay", {})
    if (options.get("interval_mode", "constant") != "constant"
            or options.get("interval_sec", policy["interval_sec"]) != policy["interval_sec"]
            or options.get("max_observations", vlm.get("num_frames", 8)) is not None):
        raise ValueError("Interval benchmark must match training: online_replay.interval_mode='constant', "
                         "interval_sec=execution_interval_sec, and max_observations=null.")
