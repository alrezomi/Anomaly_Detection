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


def infer_bag_record(bag_path: Path, stage_topic: str) -> dict[str, str | int]:
    record: dict[str, str | int] = {
        "bag_name": bag_path.name,
        "bag_path": str(bag_path),
        "read_status": "ok",
        "message_count": 0,
        "stage_markers": "",
        "failure_markers": "",
        "label": "unknown",
    }
    try:
        topics = list_bag_topics(bag_path)
        record["message_count"] = int(topics["message_count"].sum())
        if stage_topic not in set(topics["topic"]):
            record["stage_markers"] = "<stage topic missing>"
            return record

        events = read_stage_events(bag_path, stage_topic)
        markers = [str(value).strip() for value in events["stage"]]
        failures = [value for value in markers if FAIL_PATTERN.search(value)]
        record["stage_markers"] = " | ".join(markers)
        record["failure_markers"] = " | ".join(failures)
        if markers:
            record["label"] = "fail" if failures else "normal"
    except Exception as error:  # Keep damaged bags visible in the manifest.
        record["read_status"] = f"error: {type(error).__name__}: {error}"
    return record


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("dataset_manifest.csv"))
    parser.add_argument("--stage-topic", default="/recording_stage")
    parser.add_argument("--recursive", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    bags = discover_bags(arguments.data_root.resolve(), arguments.recursive)
    if not bags:
        raise FileNotFoundError(
            f"No ROS 2 bag metadata.yaml files found under {arguments.data_root}."
        )

    records = [infer_bag_record(path, arguments.stage_topic) for path in bags]
    data_root = arguments.data_root.resolve()
    for record in records:
        record["bag_path"] = str(
            Path(str(record["bag_path"])).relative_to(data_root)
        )
    dataframe = pd.DataFrame(records)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(arguments.output, index=False)

    print(dataframe[["bag_name", "read_status", "label"]].to_string(index=False))
    print(f"\nManifest written to: {arguments.output.resolve()}")
    print("Nominal/test selection remains manual and is not stored here.")


if __name__ == "__main__":
    main()
