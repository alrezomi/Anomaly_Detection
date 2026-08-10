"""Label ROS 2 bags automatically from their recorded stage messages."""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import pandas as pd

from rosbag_io import list_bag_topics, read_stage_events


FAIL_PATTERN = re.compile(
    r"(?:^|[^a-z])(error|anomaly|fail|failure|failed|nok|not[_ -]?ok|fehler)(?:$|[^a-z])",
    re.IGNORECASE,
)


def discover_bags(data_root: Path, recursive: bool) -> list[Path]:
    pattern = "**/metadata.yaml" if recursive else "*/metadata.yaml"
    return sorted({path.parent.resolve() for path in data_root.glob(pattern)})


def _format_events(events: pd.DataFrame) -> str:
    return " | ".join(
        f"{float(row.time):.6f}s:{str(row.stage).strip()}"
        for row in events.itertuples(index=False)
    )


def infer_bag_record(
    bag_path: Path,
    stage_topic: str,
    startup_ignore_sec: float,
) -> dict[str, str | int | float]:
    record: dict[str, str | int | float] = {
        "bag_name": bag_path.name,
        "bag_path": str(bag_path),
        "read_status": "ok",
        "message_count": 0,
        "startup_ignore_sec": startup_ignore_sec,
        "ignored_startup_markers": "",
        "used_stage_markers": "",
        "failure_markers": "",
        "label": "unknown",
    }
    try:
        topics = list_bag_topics(bag_path)
        record["message_count"] = int(topics["message_count"].sum())
        if stage_topic not in set(topics["topic"]):
            record["used_stage_markers"] = "<stage topic missing>"
            return record

        events = read_stage_events(bag_path, stage_topic)
        ignored_events = events[events["time"] < startup_ignore_sec]
        used_events = events[events["time"] >= startup_ignore_sec]
        markers = [str(value).strip() for value in used_events["stage"]]
        failure_mask = used_events["stage"].astype(str).map(
            lambda value: FAIL_PATTERN.search(value) is not None
        )
        failure_events = used_events[failure_mask]
        failures = [str(value).strip() for value in failure_events["stage"]]
        record["ignored_startup_markers"] = _format_events(ignored_events)
        record["used_stage_markers"] = _format_events(used_events)
        record["failure_markers"] = _format_events(failure_events)
        if markers:
            record["label"] = "fail" if failures else "normal"
    except Exception as error:  # Keep damaged bags visible in the manifest.
        record["read_status"] = f"error: {type(error).__name__}: {error}"
    return record


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("dataset_labels.csv"))
    parser.add_argument("--details-output", type=Path)
    parser.add_argument("--stage-topic", default="/recording_stage")
    parser.add_argument("--startup-ignore-sec", type=float, default=0.1)
    parser.add_argument("--recursive", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    if arguments.startup_ignore_sec < 0:
        raise ValueError("--startup-ignore-sec cannot be negative.")
    bags = discover_bags(arguments.data_root.resolve(), arguments.recursive)
    if not bags:
        raise FileNotFoundError(
            f"No ROS 2 bag metadata.yaml files found under {arguments.data_root}."
        )

    records = [
        infer_bag_record(
            path,
            arguments.stage_topic,
            arguments.startup_ignore_sec,
        )
        for path in bags
    ]
    data_root = arguments.data_root.resolve()
    for record in records:
        record["bag_path"] = str(
            Path(str(record["bag_path"])).relative_to(data_root)
        )
    details = pd.DataFrame(records)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    labels = details[["bag_name", "bag_path", "label"]]
    labels.to_csv(arguments.output, index=False)

    details_output = arguments.details_output or arguments.output.with_name(
        f"{arguments.output.stem}_details.csv"
    )
    details_output.parent.mkdir(parents=True, exist_ok=True)
    details.to_csv(details_output, index=False)

    print(labels.to_string(index=False))
    print("\nLabel counts:")
    print(labels["label"].value_counts(dropna=False).to_string())
    print(f"\nClean labels: {arguments.output.resolve()}")
    print(f"Timing details: {details_output.resolve()}")
    print("Nominal/test selection remains manual and is not stored here.")


if __name__ == "__main__":
    main()
