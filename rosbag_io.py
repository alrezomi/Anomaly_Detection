"""Cross-platform ROS 2 bag inspection and modality readers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterator, Sequence

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore


ROS2_TYPESTORE = get_typestore(Stores.ROS2_HUMBLE)


@dataclass(frozen=True)
class RosbagImageSource:
    """One camera topic inside one ROS 2 bag."""

    bag_path: Path
    topic: str
    label: str | None = None
    stage_topic: str | None = "/recording_stage"
    stage_count: int = 3

    def __post_init__(self) -> None:
        object.__setattr__(self, "bag_path", Path(self.bag_path).resolve())

    @property
    def name(self) -> str:
        return self.label or f"{self.bag_path.name}:{self.topic}"

    def __str__(self) -> str:
        return self.name


def resolve_bag_path(bag_path: str | Path) -> Path:
    """Resolve a ROS 2 bag directory, MCAP file, or SQLite DB3 file."""
    path = Path(bag_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"ROS bag path was not found: {path}")

    if path.is_dir():
        metadata_path = path / "metadata.yaml"
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"ROS bag directory has no metadata.yaml: {path}"
            )
        return path

    if path.suffix.lower() in {".mcap", ".db3"}:
        # AnyReader accepts standalone MCAP files. SQLite ROS bags need their
        # containing folder so metadata.yaml can describe the storage.
        if path.suffix.lower() == ".db3":
            metadata_path = path.parent / "metadata.yaml"
            if not metadata_path.is_file():
                raise FileNotFoundError(
                    f"SQLite bag requires metadata.yaml beside {path.name}."
                )
            return path.parent
        return path

    raise ValueError(f"Unsupported ROS bag path: {path}")


def open_bag(bag_path: str | Path) -> AnyReader:
    """Create an AnyReader configured for the standard ROS 2 messages."""
    return AnyReader(
        [resolve_bag_path(bag_path)],
        default_typestore=ROS2_TYPESTORE,
    )


def list_bag_topics(bag_path: str | Path) -> pd.DataFrame:
    """Return topic names, message types, and message counts."""
    rows: list[dict[str, str | int]] = []
    with open_bag(bag_path) as reader:
        for connection in reader.connections:
            rows.append(
                {
                    "topic": connection.topic,
                    "message_type": connection.msgtype,
                    "message_count": int(connection.msgcount),
                }
            )
    return pd.DataFrame(rows).sort_values("topic").reset_index(drop=True)


def _connection_for_topic(reader: AnyReader, topic: str):
    connections = [
        connection
        for connection in reader.connections
        if connection.topic == topic
    ]
    if not connections:
        available = sorted({c.topic for c in reader.connections})
        raise ValueError(
            f"Topic {topic!r} was not found. Available topics: {available}"
        )
    return connections


def decode_image_message(message, message_type: str) -> Image.Image:
    """Decode sensor_msgs/Image or sensor_msgs/CompressedImage as RGB PIL."""
    if message_type == "sensor_msgs/msg/CompressedImage":
        encoded = np.asarray(message.data, dtype=np.uint8)
        bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("OpenCV could not decode CompressedImage data.")
        return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    if message_type != "sensor_msgs/msg/Image":
        raise TypeError(
            f"Topic type must be sensor_msgs/msg/Image or CompressedImage, "
            f"not {message_type!r}."
        )

    encoding = str(message.encoding).lower()
    layouts = {
        "rgb8": (3, None),
        "bgr8": (3, cv2.COLOR_BGR2RGB),
        "rgba8": (4, cv2.COLOR_RGBA2RGB),
        "bgra8": (4, cv2.COLOR_BGRA2RGB),
        "mono8": (1, cv2.COLOR_GRAY2RGB),
    }
    if encoding not in layouts:
        raise ValueError(
            f"Unsupported image encoding {message.encoding!r}. "
            f"Supported encodings: {sorted(layouts)}"
        )

    channels, conversion = layouts[encoding]
    height = int(message.height)
    width = int(message.width)
    step = int(message.step)
    row_width = width * channels
    raw = np.asarray(message.data, dtype=np.uint8)

    expected = height * step
    if raw.size < expected:
        raise ValueError(
            f"Image contains {raw.size} bytes; expected at least {expected}."
        )

    rows = raw[:expected].reshape(height, step)
    pixels = rows[:, :row_width]
    if channels == 1:
        image = pixels.reshape(height, width)
    else:
        image = pixels.reshape(height, width, channels)

    if conversion is not None:
        image = cv2.cvtColor(image, conversion)
    return Image.fromarray(image).convert("RGB")


def iter_rosbag_image_frames(
    source: RosbagImageSource,
    sample_fps: float = 2.0,
    start_sec: float | None = None,
    end_sec: float | None = None,
    max_frames: int | None = None,
) -> Iterator[tuple[int, float, Image.Image]]:
    """Yield timestamp-sampled RGB frames from one image topic."""
    if sample_fps <= 0:
        raise ValueError("sample_fps must be greater than zero.")

    start_sec = 0.0 if start_sec is None else float(start_sec)
    if start_sec < 0:
        raise ValueError("start_sec cannot be negative.")
    if end_sec is not None and end_sec < start_sec:
        raise ValueError("end_sec must not be earlier than start_sec.")

    sample_period = 1.0 / sample_fps
    next_sample_sec = start_sec
    emitted = 0

    with open_bag(source.bag_path) as reader:
        connections = _connection_for_topic(reader, source.topic)
        message_types = {connection.msgtype for connection in connections}
        allowed = {"sensor_msgs/msg/Image", "sensor_msgs/msg/CompressedImage"}
        if not message_types.issubset(allowed):
            raise TypeError(
                f"Topic {source.topic!r} is not an image topic: "
                f"{sorted(message_types)}"
            )

        bag_start_ns = int(reader.start_time)
        for connection, timestamp_ns, raw_data in reader.messages(
            connections=connections
        ):
            timestamp_sec = (int(timestamp_ns) - bag_start_ns) / 1e9
            if timestamp_sec < start_sec:
                continue
            if end_sec is not None and timestamp_sec > end_sec:
                break
            if timestamp_sec + 1e-9 < next_sample_sec:
                continue

            message = reader.deserialize(raw_data, connection.msgtype)
            frame = decode_image_message(message, connection.msgtype)
            yield emitted, timestamp_sec, frame
            emitted += 1

            while next_sample_sec <= timestamp_sec:
                next_sample_sec += sample_period
            if max_frames is not None and emitted >= max_frames:
                break


def export_model_input_video(
    source: RosbagImageSource,
    output_path: str | Path,
    sample_fps: float = 2.0,
    model_input_size: int = 518,
    start_sec: float | None = None,
    end_sec: float | None = None,
    max_frames: int | None = None,
) -> int:
    """Export exactly the sampled, square-resized frames seen by DINOv2."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        sample_fps,
        (model_input_size, model_input_size),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {output_path}")

    count = 0
    try:
        for _, _, frame in iter_rosbag_image_frames(
            source=source,
            sample_fps=sample_fps,
            start_sec=start_sec,
            end_sec=end_sec,
            max_frames=max_frames,
        ):
            resized = frame.resize(
                (model_input_size, model_input_size),
                Image.Resampling.BICUBIC,
            )
            writer.write(cv2.cvtColor(np.asarray(resized), cv2.COLOR_RGB2BGR))
            count += 1
    finally:
        writer.release()

    if count == 0:
        raise ValueError(f"No frames were exported from {source}.")
    return count


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()


def _numeric_message_values(message, message_type: str) -> dict[str, float]:
    if message_type in {"geometry_msgs/msg/WrenchStamped", "geometry_msgs/msg/Wrench"}:
        wrench = message.wrench if hasattr(message, "wrench") else message
        return {
            "force_x": float(wrench.force.x),
            "force_y": float(wrench.force.y),
            "force_z": float(wrench.force.z),
            "torque_x": float(wrench.torque.x),
            "torque_y": float(wrench.torque.y),
            "torque_z": float(wrench.torque.z),
        }

    if message_type in {"geometry_msgs/msg/PoseStamped", "geometry_msgs/msg/Pose"}:
        pose = message.pose if hasattr(message, "pose") else message
        return {
            "position_x": float(pose.position.x),
            "position_y": float(pose.position.y),
            "position_z": float(pose.position.z),
            "orientation_x": float(pose.orientation.x),
            "orientation_y": float(pose.orientation.y),
            "orientation_z": float(pose.orientation.z),
            "orientation_w": float(pose.orientation.w),
        }

    if message_type == "sensor_msgs/msg/JointState":
        values: dict[str, float] = {}
        for field_name in ("position", "velocity", "effort"):
            field_values = getattr(message, field_name)
            for index, value in enumerate(field_values):
                joint_name = (
                    str(message.name[index])
                    if index < len(message.name)
                    else f"joint_{index}"
                )
                values[f"{_safe_name(joint_name)}_{field_name}"] = float(value)
        return values

    raise TypeError(
        f"Unsupported numeric time-series message type: {message_type}"
    )


def read_numeric_topic(
    bag_path: str | Path,
    topic: str,
) -> pd.DataFrame:
    """Read one Wrench, Pose, or JointState topic into a numeric DataFrame."""
    rows: list[dict[str, float | int]] = []
    with open_bag(bag_path) as reader:
        connections = _connection_for_topic(reader, topic)
        bag_start_ns = int(reader.start_time)
        for connection, timestamp_ns, raw_data in reader.messages(
            connections=connections
        ):
            message = reader.deserialize(raw_data, connection.msgtype)
            values = _numeric_message_values(message, connection.msgtype)
            rows.append(
                {
                    "time_ns": int(timestamp_ns),
                    "time": (int(timestamp_ns) - bag_start_ns) / 1e9,
                    **values,
                }
            )

    if not rows:
        raise ValueError(f"No messages were read from topic {topic!r}.")
    return pd.DataFrame(rows).sort_values("time_ns").reset_index(drop=True)


def read_selected_numeric_topics(
    bag_path: str | Path,
    topics: Sequence[str],
) -> dict[str, pd.DataFrame]:
    """Read any selected numeric topics without assuming they share a rate."""
    if not topics:
        raise ValueError("At least one numeric topic must be selected.")
    return {topic: read_numeric_topic(bag_path, topic) for topic in topics}


def read_stage_events(
    bag_path: str | Path,
    topic: str = "/recording_stage",
) -> pd.DataFrame:
    """Read std_msgs/String stage markers with bag-relative timestamps."""
    rows: list[dict[str, str | float | int]] = []
    with open_bag(bag_path) as reader:
        connections = _connection_for_topic(reader, topic)
        bag_start_ns = int(reader.start_time)
        for connection, timestamp_ns, raw_data in reader.messages(
            connections=connections
        ):
            if connection.msgtype != "std_msgs/msg/String":
                raise TypeError(
                    f"Stage topic must be std_msgs/msg/String, not "
                    f"{connection.msgtype}."
                )
            message = reader.deserialize(raw_data, connection.msgtype)
            rows.append(
                {
                    "time_ns": int(timestamp_ns),
                    "time": (int(timestamp_ns) - bag_start_ns) / 1e9,
                    "stage": str(message.data),
                }
            )
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class StageTimeline:
    """Validated stage boundaries or an equal-duration fallback."""

    starts: tuple[float, ...]
    end_sec: float
    source: str

    @property
    def stage_count(self) -> int:
        return len(self.starts)

    def locate(self, timestamp_sec: float) -> tuple[int, float]:
        """Return the 1-based stage and continuous normalized progress."""
        time_value = min(max(float(timestamp_sec), 0.0), self.end_sec)
        stage_index = int(np.searchsorted(self.starts, time_value, side="right") - 1)
        stage_index = min(max(stage_index, 0), self.stage_count - 1)
        stage_start = self.starts[stage_index]
        stage_end = (
            self.starts[stage_index + 1]
            if stage_index + 1 < self.stage_count
            else self.end_sec
        )
        fraction = (
            (time_value - stage_start) / (stage_end - stage_start)
            if stage_end > stage_start
            else 0.0
        )
        progress = (stage_index + min(max(fraction, 0.0), 1.0)) / self.stage_count
        return stage_index + 1, min(max(progress, 0.0), 1.0)


def resolve_stage_timeline(
    bag_path: str | Path,
    end_sec: float,
    topic: str = "/recording_stage",
    stage_count: int = 3,
) -> StageTimeline:
    """Use a complete ordered marker sequence, otherwise equal intervals.

    Text such as ``Error`` is ignored. The sequence is built backwards from
    the last stage, choosing the latest preceding marker for every stage. This
    rejects the rapid 1/2/3 initialization messages seen in some recordings.
    """
    if stage_count < 1:
        raise ValueError("stage_count must be at least one.")
    if end_sec <= 0:
        raise ValueError("end_sec must be greater than zero.")

    fallback = StageTimeline(
        starts=tuple(index * end_sec / stage_count for index in range(stage_count)),
        end_sec=float(end_sec),
        source="equal_intervals",
    )

    try:
        events = read_stage_events(bag_path, topic)
    except (FileNotFoundError, TypeError, ValueError):
        return fallback

    numeric_events: list[tuple[int, float]] = []
    for row in events.itertuples(index=False):
        try:
            stage = int(str(row.stage).strip())
        except ValueError:
            continue
        if 1 <= stage <= stage_count:
            numeric_events.append((stage, float(row.time)))

    selected: list[float] = []
    before = float("inf")
    minimum_gap = max(0.05, end_sec * 0.001)
    for expected_stage in range(stage_count, 0, -1):
        candidates = [
            time_value
            for stage, time_value in numeric_events
            if stage == expected_stage and time_value < before - minimum_gap
        ]
        if not candidates:
            return fallback
        chosen = max(candidates)
        selected.append(chosen)
        before = chosen

    starts = tuple(reversed(selected))
    if starts[0] > end_sec or starts[-1] >= end_sec:
        return fallback
    return StageTimeline(starts=starts, end_sec=float(end_sec), source="recorded")
