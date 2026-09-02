"""Benchmark DINO + RynnValue across every non-nominal demonstration."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from build_dataset_manifest import discover_bags, infer_bag_record
from rynnvalue_vlm.model import RynnValueModel
from rynnvalue_vlm.run import evaluate_rynnvalue


REPO_ROOT = Path(__file__).resolve().parent
GROUND_TRUTH_TO_DECISION = {"normal": "success", "fail": "failure"}


def _slug(value: str) -> str:
    import re

    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()


def _excluded_bag_names(config: dict[str, Any]) -> set[str]:
    """Use the same held-out set as the existing RynnBrain benchmark."""

    names = {Path(bag).name for bag in config.get("nominal_bags", [])}
    rynnbrain = config.get("rynnbrain", {})
    names |= {Path(bag).name for bag in rynnbrain.get("reference_bags", [])}
    return names


def _run_dino_test(
    config: dict[str, Any], bag_path: Path, bag_output_dir: Path
) -> bool:
    bag_output_dir.mkdir(parents=True, exist_ok=True)
    bag_config = dict(config)
    bag_config["test_bag"] = str(bag_path)
    bag_config["output_dir"] = str(bag_output_dir)
    config_path = bag_output_dir / "pipeline_config.json"
    config_path.write_text(json.dumps(bag_config, indent=2), encoding="utf-8")
    try:
        subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "run_rosbag_vision.py"),
                "--config",
                str(config_path),
                "--mode",
                "test",
            ],
            cwd=REPO_ROOT,
            check=True,
        )
        return True
    except subprocess.CalledProcessError as error:
        print(f"  [DINO FAILED] {bag_path.name}: {error}")
        return False


def _dino_summary(
    camera_topics: list[str], bag_output_dir: Path
) -> dict[str, Any]:
    total_alarms = 0
    total_frames = 0
    alarm_rates: list[float] = []
    max_epsilons: list[float] = []
    for topic in camera_topics:
        results_path = bag_output_dir / f"{_slug(topic)}_results.csv"
        if not results_path.is_file():
            continue
        frame = pd.read_csv(results_path)
        if frame.empty:
            continue
        alarms = int(frame["alarm"].sum())
        total_alarms += alarms
        total_frames += len(frame)
        alarm_rates.append(alarms / len(frame))
        max_epsilons.append(float(frame["epsilon"].max()))
    return {
        "dino_alarm_frames": total_alarms,
        "dino_total_frames": total_frames,
        "dino_alarm_rate": (
            sum(alarm_rates) / len(alarm_rates) if alarm_rates else None
        ),
        "dino_max_epsilon": max(max_epsilons) if max_epsilons else None,
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("pipeline_config.json"))
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--benchmark-dir", type=Path, default=None)
    parser.add_argument("--no-recursive", action="store_true")
    parser.add_argument("--startup-ignore-sec", type=float, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--skip-dino", action="store_true", help="Reuse already-generated raw videos."
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    config = json.loads(arguments.config.read_text(encoding="utf-8"))
    vlm = dict(config.get("rynnvalue", {}))
    if not vlm:
        raise ValueError("Add a 'rynnvalue' section to pipeline_config.json")

    data_root = arguments.data_root or Path(
        os.environ.get("ANOMALY_DATA_ROOT", "/data")
    )
    startup_ignore_sec = (
        arguments.startup_ignore_sec
        if arguments.startup_ignore_sec is not None
        else float(os.environ.get("STAGE_STARTUP_IGNORE_SEC", 0.1))
    )
    base_output = Path(
        vlm.get("output_dir", Path(config.get("output_dir", "outputs")) / "rynnvalue")
    )
    benchmark_root = arguments.benchmark_dir or Path(f"{base_output}_benchmark")
    benchmark_root.mkdir(parents=True, exist_ok=True)

    all_bags = discover_bags(data_root, recursive=not arguments.no_recursive)
    excluded_names = _excluded_bag_names(config)
    records = [
        infer_bag_record(
            bag_path, config.get("stage_topic", "/recording_stage"), startup_ignore_sec
        )
        for bag_path in all_bags
        if bag_path.name not in excluded_names
    ]
    if arguments.limit is not None:
        records = records[: arguments.limit]
    pd.DataFrame(records).to_csv(
        benchmark_root / "benchmark_bag_selection.csv", index=False
    )
    print(
        f"Found {len(all_bags)} bag(s); benchmarking {len(records)} after "
        f"excluding {len(excluded_names)} nominal/reference bag name(s)."
    )

    print("Loading RynnValue model once for the whole benchmark...")
    model = RynnValueModel(dict(vlm.get("model", {})))
    camera_topics = list(config.get("camera_topics", []))
    master_rows: list[dict[str, Any]] = []

    for index, record in enumerate(records, start=1):
        bag_path = Path(record["bag_path"])
        if not bag_path.is_absolute():
            bag_path = (data_root / bag_path).resolve()
        bag_name = record["bag_name"]
        ground_truth = record["label"]
        bag_output_dir = benchmark_root / bag_name
        print(f"\n[{index}/{len(records)}] {bag_name} (ground_truth={ground_truth})")

        dino_ok = True
        if not arguments.skip_dino:
            dino_ok = _run_dino_test(config, bag_path, bag_output_dir)
        dino = _dino_summary(camera_topics, bag_output_dir) if dino_ok else {}

        summaries: list[dict[str, Any]] = []
        if dino_ok:
            bag_config = dict(config)
            bag_config["test_bag"] = str(bag_path)
            bag_config["output_dir"] = str(bag_output_dir)
            bag_vlm = dict(vlm)
            bag_vlm["ground_truth_label"] = ground_truth
            bag_vlm["output_dir"] = str(bag_output_dir / "rynnvalue")
            try:
                summaries, _, _ = evaluate_rynnvalue(
                    model,
                    bag_config,
                    bag_vlm,
                    Path(bag_vlm["output_dir"]),
                )
            except Exception as error:
                print(f"  [RYNNVALUE FAILED] {bag_name}: {error}")

        if not summaries:
            master_rows.append(
                {
                    "bag_name": bag_name,
                    "bag_path": str(bag_path),
                    "ground_truth_label": ground_truth,
                    "topic": None,
                    "decision": "failed",
                    "decision_correct": None,
                    **dino,
                }
            )
            continue

        expected = GROUND_TRUTH_TO_DECISION.get(ground_truth)
        for summary in summaries:
            decision = summary["decision"]
            master_rows.append(
                {
                    "bag_name": bag_name,
                    "bag_path": str(bag_path),
                    **summary,
                    "decision_correct": (
                        decision == expected if expected is not None else None
                    ),
                    **dino,
                }
            )

    summary_frame = pd.DataFrame(master_rows)
    summary_path = benchmark_root / "benchmark_summary.csv"
    summary_frame.to_csv(summary_path, index=False)
    print(f"\nRynnValue benchmark summary: {summary_path}")
    if not summary_frame.empty and "decision_correct" in summary_frame:
        scored = summary_frame.dropna(subset=["decision_correct"])
        if not scored.empty:
            print("\nAccuracy by camera topic:")
            print(scored.groupby("topic")["decision_correct"].mean().to_string())
            print(
                f"\nOverall accuracy: {scored['decision_correct'].mean():.3f} "
                f"({len(scored)} scored rows)"
            )


if __name__ == "__main__":
    main()
