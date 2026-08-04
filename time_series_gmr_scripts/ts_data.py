"""ROS2 MCAP reading, force-based synchronization, and time-series preprocessing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SIGNAL_COLUMNS = [
    "force_x",
    "force_y",
    "force_z",
    "torque_x",
    "torque_y",
    "torque_z",
]

try:
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore

    HAS_ROSBAGS = True
except ImportError:
    HAS_ROSBAGS = False


def read_wrench_mcap_with_rosbags(
    bag_path: str | Path,
    topic: str = "/tcp_force",
) -> pd.DataFrame:
    """Read ``geometry_msgs/msg/WrenchStamped`` data with ``rosbags``."""
    if not HAS_ROSBAGS:
        raise ImportError(
            "The 'rosbags' package is not installed. Install it with: "
            "python -m pip install rosbags"
        )

    path = Path(bag_path)
    if not path.exists():
        raise FileNotFoundError(f"Bag path was not found: {path}")

    typestore = get_typestore(Stores.ROS2_HUMBLE)
    rows: list[dict[str, float | int]] = []

    with AnyReader([path], default_typestore=typestore) as reader:
        connections = [
            connection
            for connection in reader.connections
            if connection.topic == topic
        ]

        if not connections:
            available_topics = sorted(
                {connection.topic for connection in reader.connections}
            )
            raise ValueError(
                f"Topic '{topic}' was not found in {path}.\n"
                f"Available topics: {available_topics}"
            )

        for connection, timestamp, raw_data in reader.messages(
            connections=connections
        ):
            message = reader.deserialize(raw_data, connection.msgtype)
            rows.append(
                {
                    "time_ns": int(timestamp),
                    "force_x": float(message.wrench.force.x),
                    "force_y": float(message.wrench.force.y),
                    "force_z": float(message.wrench.force.z),
                    "torque_x": float(message.wrench.torque.x),
                    "torque_y": float(message.wrench.torque.y),
                    "torque_z": float(message.wrench.torque.z),
                }
            )

    return _rows_to_wrench_dataframe(rows=rows, source=path, topic=topic)


def read_wrench_mcap_with_rosbag2_py(
    bag_path: str | Path,
    topic: str = "/tcp_force",
) -> pd.DataFrame:
    """Read the same topic with ROS2's ``rosbag2_py`` as a fallback."""
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
    except ImportError as error:
        raise ImportError(
            "rosbag2_py is unavailable. Use rosbags on Windows, or source "
            "your ROS2 environment before running this script on Ubuntu."
        ) from error

    path = Path(bag_path)
    if not path.exists():
        raise FileNotFoundError(f"Bag path was not found: {path}")

    bag_uri = str(path.parent if path.suffix.lower() == ".mcap" else path)
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag_uri, storage_id="mcap"),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        ),
    )

    topic_types = reader.get_all_topics_and_types()
    type_map = {topic_info.name: topic_info.type for topic_info in topic_types}

    if topic not in type_map:
        raise ValueError(
            f"Topic '{topic}' was not found in {bag_uri}. "
            f"Available topics: {sorted(type_map)}"
        )

    message_type = get_message(type_map[topic])
    rows: list[dict[str, float | int]] = []

    while reader.has_next():
        read_topic, data, timestamp = reader.read_next()
        if read_topic != topic:
            continue

        message = deserialize_message(data, message_type)
        rows.append(
            {
                "time_ns": int(timestamp),
                "force_x": float(message.wrench.force.x),
                "force_y": float(message.wrench.force.y),
                "force_z": float(message.wrench.force.z),
                "torque_x": float(message.wrench.torque.x),
                "torque_y": float(message.wrench.torque.y),
                "torque_z": float(message.wrench.torque.z),
            }
        )

    return _rows_to_wrench_dataframe(
        rows=rows,
        source=Path(bag_uri),
        topic=topic,
    )


def _rows_to_wrench_dataframe(
    rows: list[dict[str, float | int]],
    source: Path,
    topic: str,
) -> pd.DataFrame:
    """Convert extracted message rows into the common wrench DataFrame."""
    if not rows:
        raise ValueError(f"No messages were read from '{topic}' in {source}.")

    dataframe = pd.DataFrame(rows)
    dataframe = dataframe.sort_values("time_ns").reset_index(drop=True)
    dataframe["time"] = (
        dataframe["time_ns"] - dataframe["time_ns"].iloc[0]
    ) / 1e9

    return dataframe[["time", *SIGNAL_COLUMNS]]


def read_wrench_bag(
    bag_path: str | Path,
    topic: str = "/tcp_force",
) -> pd.DataFrame:
    """Read one bag, preferring the cross-platform ``rosbags`` package."""
    try:
        return read_wrench_mcap_with_rosbags(bag_path=bag_path, topic=topic)
    except Exception as rosbags_error:
        print("rosbags reader failed:")
        print(rosbags_error)
        print("Trying rosbag2_py...")

        try:
            return read_wrench_mcap_with_rosbag2_py(
                bag_path=bag_path,
                topic=topic,
            )
        except Exception as rosbag2_error:
            raise RuntimeError(
                "The MCAP bag could not be read with either reader.\n"
                f"rosbags error: {rosbags_error}\n"
                f"rosbag2_py error: {rosbag2_error}"
            ) from rosbag2_error


def preprocess_ts_dataframe(
    dataframe: pd.DataFrame,
    resample_length: int = 500,
) -> pd.DataFrame:
    """Clean, interpolate, resample, and add normalized task phase."""
    if resample_length < 2:
        raise ValueError("resample_length must be at least 2.")

    missing_columns = [
        column for column in SIGNAL_COLUMNS if column not in dataframe.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing signal columns: {missing_columns}")

    if len(dataframe) < 2:
        raise ValueError("At least two samples are required for resampling.")

    cleaned = dataframe.copy()

    for column in SIGNAL_COLUMNS:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
        cleaned[column] = (
            cleaned[column].interpolate().bfill().ffill()
        )

    if cleaned[SIGNAL_COLUMNS].isna().any().any():
        raise ValueError("Signal data still contain missing values after cleaning.")

    old_phase = np.linspace(0.0, 1.0, len(cleaned))
    new_phase = np.linspace(0.0, 1.0, resample_length)

    resampled: dict[str, np.ndarray] = {
        column: np.interp(new_phase, old_phase, cleaned[column].to_numpy())
        for column in SIGNAL_COLUMNS
    }

    if "time" in cleaned.columns:
        time_values = pd.to_numeric(cleaned["time"], errors="coerce")
        time_values = time_values.interpolate().bfill().ffill().to_numpy()
        resampled["time"] = np.interp(new_phase, old_phase, time_values)
    else:
        resampled["time"] = np.arange(resample_length, dtype=float)

    output = pd.DataFrame(resampled)
    output["phase"] = new_phase
    return output


def add_force_magnitude(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Add the Euclidean force magnitude to a wrench DataFrame."""
    required = {"force_x", "force_y", "force_z"}
    if not required.issubset(dataframe.columns):
        raise ValueError(
            "The DataFrame must contain force_x, force_y, and force_z."
        )

    output = dataframe.copy()
    output["force_mag"] = np.sqrt(
        output["force_x"] ** 2
        + output["force_y"] ** 2
        + output["force_z"] ** 2
    )
    return output


def find_active_segment_by_force(
    raw_dataframe: pd.DataFrame,
    baseline_fraction: float = 0.10,
    threshold_std_factor: float = 4.0,
    min_force_change: float = 0.5,
    padding_seconds: float = 0.5,
) -> tuple[float, float, pd.DataFrame, dict[str, Any]]:
    """Estimate the active manipulation interval from force magnitude."""
    if raw_dataframe.empty:
        raise ValueError("raw_dataframe is empty.")
    if not 0.0 < baseline_fraction <= 1.0:
        raise ValueError("baseline_fraction must be in the interval (0, 1].")

    debug_dataframe = add_force_magnitude(raw_dataframe)

    if "time" not in debug_dataframe.columns:
        debug_dataframe["time"] = np.arange(len(debug_dataframe), dtype=float)

    debug_dataframe = debug_dataframe.sort_values("time").reset_index(drop=True)

    sample_count = len(debug_dataframe)
    baseline_count = min(
        sample_count,
        max(20, int(sample_count * baseline_fraction)),
    )
    baseline_values = debug_dataframe["force_mag"].iloc[:baseline_count].to_numpy()

    upper_quartile = float(np.quantile(baseline_values, 0.75))
    clean_baseline = baseline_values[baseline_values <= upper_quartile]
    if len(clean_baseline) < min(10, len(baseline_values)):
        clean_baseline = baseline_values

    baseline_median = float(np.median(clean_baseline))
    median_absolute_deviation = float(
        np.median(np.abs(clean_baseline - baseline_median))
    )
    baseline_std = 1.4826 * median_absolute_deviation

    if baseline_std < 1e-6:
        baseline_std = float(np.std(clean_baseline))

    threshold = baseline_median + max(
        threshold_std_factor * baseline_std,
        min_force_change,
    )

    debug_dataframe["force_active"] = (
        debug_dataframe["force_mag"] > threshold
    )
    active_indices = np.flatnonzero(
        debug_dataframe["force_active"].to_numpy()
    )

    full_start = float(debug_dataframe["time"].iloc[0])
    full_end = float(debug_dataframe["time"].iloc[-1])

    if len(active_indices) == 0:
        start_time = full_start
        end_time = full_end
        active_found = False
    else:
        start_time = max(
            full_start,
            float(debug_dataframe["time"].iloc[active_indices[0]])
            - padding_seconds,
        )
        end_time = min(
            full_end,
            float(debug_dataframe["time"].iloc[active_indices[-1]])
            + padding_seconds,
        )
        active_found = True

    sync_info: dict[str, Any] = {
        "baseline_median": baseline_median,
        "baseline_std": baseline_std,
        "threshold": float(threshold),
        "active_found": active_found,
        "start_time": float(start_time),
        "end_time": float(end_time),
    }

    return start_time, end_time, debug_dataframe, sync_info


def sync_and_preprocess_ts_dataframe(
    raw_dataframe: pd.DataFrame,
    resample_length: int = 500,
    baseline_fraction: float = 0.10,
    threshold_std_factor: float = 4.0,
    min_force_change: float = 0.5,
    padding_seconds: float = 0.5,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    """Synchronize one execution, then clean and resample it."""
    (
        start_time,
        end_time,
        debug_dataframe,
        sync_info,
    ) = find_active_segment_by_force(
        raw_dataframe=raw_dataframe,
        baseline_fraction=baseline_fraction,
        threshold_std_factor=threshold_std_factor,
        min_force_change=min_force_change,
        padding_seconds=padding_seconds,
    )

    trimmed = raw_dataframe.loc[
        (raw_dataframe["time"] >= start_time)
        & (raw_dataframe["time"] <= end_time)
    ].copy()

    if len(trimmed) < 20:
        print(
            "Warning: the synchronized segment is too short. "
            "Using the full execution instead."
        )
        trimmed = raw_dataframe.copy()
        sync_info["used_full_trajectory"] = True
    else:
        sync_info["used_full_trajectory"] = False

    if "time" in trimmed.columns:
        trimmed["time"] = trimmed["time"] - float(trimmed["time"].iloc[0])

    processed = preprocess_ts_dataframe(
        dataframe=trimmed,
        resample_length=resample_length,
    )

    return processed, sync_info, debug_dataframe
