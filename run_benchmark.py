"""Benchmark the DINO + RynnBrain pipeline against every non-nominal demonstration.

For every ROS 2 bag under the data root that was not used to build the DINO
nominal memory or the RynnBrain task memory, this:

1. Runs DINO test mode (`run_rosbag_vision.py --mode test`) against that bag,
   reusing the existing nominal cache and task memory unchanged.
2. Runs the RynnBrain multi-turn evaluation against the videos DINO produced,
   reusing one already-loaded VLM across every bag instead of reloading it
   per bag.
3. Writes each bag's full output (videos, CSVs, VLM responses) into its own
   folder, and appends one row per (bag, input_mode) to a single benchmark
   summary table.

Ground truth for scoring comes from explicit CLI labels when supplied, or from
the same recorded-stage-marker heuristic as build_dataset_manifest.py
(normal/fail/unknown) otherwise.
"""

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
from rynnbrain_vlm.cop_analysis import analyze_saved_vectors
from rynnbrain_vlm.model import RynnBrainModel
from rynnbrain_vlm.run import evaluate_multiturn, write_multiturn_outputs

REPO_ROOT = Path(__file__).resolve().parent

# Ground truth from build_dataset_manifest.py vs. the VLM's own decision vocabulary.
GROUND_TRUTH_TO_DECISION = {"normal": "success", "fail": "failure"}
ABSTAIN_DECISIONS = {"uncertain", "not_parsed"}


def _slug(value: str) -> str:
    import re

    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("pipeline_config.json"))
    parser.add_argument(
        "--data-root", type=Path, default=None,
        help="Defaults to $ANOMALY_DATA_ROOT, then /data.",
    )
    parser.add_argument(
        "--benchmark-dir", type=Path, default=None,
        help="Defaults to '<rynnbrain.output_dir>_benchmark'.",
    )
    parser.add_argument("--no-recursive", action="store_true", help="Only scan immediate subdirectories for bags.")
    parser.add_argument(
        "--startup-ignore-sec", type=float, default=None,
        help="Defaults to $STAGE_STARTUP_IGNORE_SEC, then 0.1.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only benchmark the first N candidate bags.")
    parser.add_argument(
        "--bag",
        action="append",
        default=[],
        metavar="BAG_NAME",
        help=(
            "Benchmark only this eligible bag name. Repeat for several held-out "
            "normal/fail bags. Reference bags remain excluded."
        ),
    )
    parser.add_argument(
        "--normal-bag",
        action="append",
        default=[],
        metavar="BAG_NAME",
        help=(
            "Benchmark this bag and set its ground truth to normal, overriding "
            "the recorded-stage label. Repeat for multiple normal bags."
        ),
    )
    parser.add_argument(
        "--failure-bag",
        action="append",
        default=[],
        metavar="BAG_NAME",
        help=(
            "Benchmark this bag and set its ground truth to fail, overriding "
            "the recorded-stage label. Repeat for multiple failure bags."
        ),
    )
    parser.add_argument(
        "--include-nominal-bags",
        action="store_true",
        help=(
            "Allow DINO nominal-memory bags as exploratory evaluation samples. "
            "RynnBrain reference bags remain excluded to prevent direct leakage."
        ),
    )
    parser.add_argument("--skip-dino", action="store_true", help="Reuse already-generated per-bag videos/CSVs.")
    parser.add_argument("--skip-vlm", action="store_true", help="Only run the DINO stage.")
    return parser.parse_args()


def _excluded_bag_names(
    config: dict[str, Any], include_nominal_bags: bool = False
) -> set[str]:
    rynnbrain = config.get("rynnbrain", {})
    names = {Path(bag).name for bag in rynnbrain.get("reference_bags", [])}
    if not include_nominal_bags:
        names |= {Path(bag).name for bag in config.get("nominal_bags", [])}
    return names


def _run_dino_test(config: dict[str, Any], bag_path: Path, bag_output_dir: Path) -> bool:
    """Run DINO test mode for one bag via subprocess; returns success."""
    bag_output_dir.mkdir(parents=True, exist_ok=True)
    bag_config = dict(config)
    bag_config["test_bag"] = str(bag_path)
    bag_config["output_dir"] = str(bag_output_dir)
    config_path = bag_output_dir / "pipeline_config.json"
    config_path.write_text(json.dumps(bag_config, indent=2), encoding="utf-8")
    try:
        subprocess.run(
            [sys.executable, str(REPO_ROOT / "run_rosbag_vision.py"), "--config", str(config_path), "--mode", "test"],
            cwd=REPO_ROOT, check=True,
        )
        return True
    except subprocess.CalledProcessError as error:
        print(f"  [DINO FAILED] {bag_path.name}: {error}")
        return False


def _dino_summary(camera_topics: list[str], bag_output_dir: Path) -> dict[str, Any]:
    """Aggregate each camera's {topic}_results.csv into one row of summary columns."""
    alarm_rates: list[float] = []
    max_epsilons: list[float] = []
    total_alarms = 0
    total_frames = 0
    for topic in camera_topics:
        results_path = bag_output_dir / f"{_slug(topic)}_results.csv"
        if not results_path.is_file():
            continue
        topic_df = pd.read_csv(results_path)
        if topic_df.empty:
            continue
        alarms = int(topic_df["alarm"].sum())
        frames = len(topic_df)
        total_alarms += alarms
        total_frames += frames
        alarm_rates.append(alarms / frames)
        max_epsilons.append(float(topic_df["epsilon"].max()))
    return {
        "dino_alarm_frames": total_alarms,
        "dino_total_frames": total_frames,
        "dino_alarm_rate": sum(alarm_rates) / len(alarm_rates) if alarm_rates else None,
        "dino_max_epsilon": max(max_epsilons) if max_epsilons else None,
    }


def _decision_correct(ground_truth: str, decision: str) -> bool | None:
    expected = GROUND_TRUTH_TO_DECISION.get(ground_truth)
    if expected is None:
        return None
    if decision in ABSTAIN_DECISIONS:
        return False
    return decision == expected


def _select_named_records(
    records: list[dict[str, Any]],
    requested_names: list[str],
    excluded_names: set[str],
    manual_labels: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Select eligible records by unique bag name while preserving CLI order."""
    if not requested_names:
        return records
    if len(requested_names) != len(set(requested_names)):
        raise ValueError(
            "Each bag name may appear only once across --bag, --normal-bag, "
            "and --failure-bag."
        )

    excluded = sorted(set(requested_names) & excluded_names)
    if excluded:
        raise ValueError(
            "The following selected bags are configured as nominal/reference memory "
            "and cannot be evaluation samples: " + ", ".join(excluded)
        )

    by_name: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_name.setdefault(str(record["bag_name"]), []).append(record)
    missing = [name for name in requested_names if name not in by_name]
    if missing:
        raise ValueError("Eligible bag(s) not found: " + ", ".join(missing))
    ambiguous = [name for name in requested_names if len(by_name[name]) > 1]
    if ambiguous:
        raise ValueError(
            "Bag name is not unique under the data root: " + ", ".join(ambiguous)
        )
    selected: list[dict[str, Any]] = []
    overrides = manual_labels or {}
    for name in requested_names:
        record = dict(by_name[name][0])
        record["recorded_label"] = record.get("label", "unknown")
        if name in overrides:
            record["label"] = overrides[name]
            record["label_source"] = "manual_cli"
        else:
            record["label_source"] = "recorded_stage"
        selected.append(record)
    return selected


def main() -> None:
    arguments = parse_arguments()
    config = json.loads(arguments.config.read_text(encoding="utf-8"))
    vlm = dict(config.get("rynnbrain", {}))
    if not vlm and not arguments.skip_vlm:
        raise ValueError("Add a 'rynnbrain' section to pipeline_config.json, or pass --skip-vlm.")
    camera_topics = list(config.get("camera_topics", []))
    frame_count = int(vlm.get("num_frames", 4))
    generation = dict(vlm.get("generation", {}))

    data_root = arguments.data_root or Path(os.environ.get("ANOMALY_DATA_ROOT", "/data"))
    stage_topic = config.get("stage_topic", "/recording_stage")
    startup_ignore_sec = (
        arguments.startup_ignore_sec
        if arguments.startup_ignore_sec is not None
        else float(os.environ.get("STAGE_STARTUP_IGNORE_SEC", 0.1))
    )

    base_vlm_output = Path(vlm.get("output_dir", Path(config.get("output_dir", "outputs")) / "rynnbrain"))
    benchmark_root = arguments.benchmark_dir or Path(f"{base_vlm_output}_benchmark")
    benchmark_root.mkdir(parents=True, exist_ok=True)

    print(f"Scanning bags under: {data_root}")
    all_bags = discover_bags(data_root, recursive=not arguments.no_recursive)
    excluded_names = _excluded_bag_names(config, arguments.include_nominal_bags)
    records = [
        infer_bag_record(bag_path, stage_topic, startup_ignore_sec)
        for bag_path in all_bags
        if bag_path.name not in excluded_names
    ]
    requested_names = (
        list(arguments.bag)
        + list(arguments.normal_bag)
        + list(arguments.failure_bag)
    )
    manual_labels = {
        **{name: "normal" for name in arguments.normal_bag},
        **{name: "fail" for name in arguments.failure_bag},
    }
    records = _select_named_records(
        records,
        requested_names,
        excluded_names,
        manual_labels,
    )
    selection_df = pd.DataFrame(records)
    selection_df.to_csv(benchmark_root / "benchmark_bag_selection.csv", index=False)
    print(f"Found {len(all_bags)} bag(s), excluding {len(excluded_names)} nominal/reference bag(s).")
    print(f"Benchmarking {len(records)} demonstration(s).")
    selection_columns = ["bag_name", "label"]
    if records and "label_source" in selection_df:
        selection_columns.extend(["label_source", "recorded_label"])
    print(selection_df[selection_columns].to_string(index=False) if records else "  (none)")

    candidates = records[: arguments.limit] if arguments.limit else records

    model: RynnBrainModel | None = None
    if not arguments.skip_vlm:
        print("\nLoading RynnBrain model once for the whole benchmark...")
        model = RynnBrainModel(vlm["model"])

    master_rows: list[dict[str, Any]] = []
    for index, record in enumerate(candidates, start=1):
        bag_path = Path(record["bag_path"])
        # bag_path was stored relative to data_root by infer_bag_record's caller convention.
        if not bag_path.is_absolute():
            bag_path = (data_root / bag_path).resolve()
        bag_name = record["bag_name"]
        ground_truth = record["label"]
        bag_output_dir = benchmark_root / bag_name
        print(f"\n[{index}/{len(candidates)}] {bag_name} (ground_truth={ground_truth})")

        dino_ok = True
        if not arguments.skip_dino:
            dino_ok = _run_dino_test(config, bag_path, bag_output_dir)
        dino_summary = _dino_summary(camera_topics, bag_output_dir) if dino_ok else {}

        rows: list[dict[str, Any]] = []
        if model is not None and dino_ok:
            vlm_bag = dict(vlm)
            vlm_bag["output_dir"] = str(bag_output_dir / "rynnbrain_multiturn")
            vlm_bag["ground_truth_label"] = ground_truth
            config_bag = dict(config)
            config_bag["test_bag"] = str(bag_path)
            config_bag["output_dir"] = str(bag_output_dir)
            try:
                rows, frame_metadata, raw_records, task_description = evaluate_multiturn(
                    model, config_bag, vlm_bag, frame_count, generation, Path(vlm_bag["output_dir"])
                )
                write_multiturn_outputs(
                    Path(vlm_bag["output_dir"]), rows, frame_metadata, raw_records, task_description
                )
            except Exception as error:
                print(f"  [VLM FAILED] {bag_name}: {error}")
                rows = []

        if not rows:
            master_rows.append({
                "bag_name": bag_name,
                "bag_path": str(bag_path),
                "ground_truth_label": ground_truth,
                "input_mode": None,
                "decision": "not_run" if arguments.skip_vlm else "failed",
                "confidence": None,
                "decision_correct": None,
                "response": None,
                "cop_vector_path": None,
                "cop_metadata_path": None,
                **dino_summary,
            })
            continue

        for row in rows:
            master_rows.append({
                "bag_name": bag_name,
                "bag_path": str(bag_path),
                "ground_truth_label": ground_truth,
                "input_mode": row["input_mode"],
                "decision": row["decision"],
                "confidence": row["confidence"],
                "decision_correct": _decision_correct(ground_truth, row["decision"]),
                "response": row["response"],
                "cop_vector_path": row.get("cop_vector_path"),
                "cop_metadata_path": row.get("cop_metadata_path"),
                **dino_summary,
            })

    summary_df = pd.DataFrame(master_rows)
    summary_path = benchmark_root / "benchmark_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    pca_summary = None
    if (
        not arguments.skip_vlm
        and bool(dict(vlm.get("cop_vectors", {})).get("enabled", True))
    ):
        current_metadata_paths = [
            Path(str(row["cop_metadata_path"]))
            for row in master_rows
            if row.get("cop_metadata_path")
        ]
        pca_summary = analyze_saved_vectors(
            benchmark_root,
            metadata_paths=current_metadata_paths,
        )

    print("\n" + "=" * 90)
    print("BENCHMARK FINISHED")
    print("=" * 90)
    print(f"Per-demonstration folders and combined table under: {benchmark_root}")
    print(f"Summary table: {summary_path}")
    if not summary_df.empty and "input_mode" in summary_df:
        scored = summary_df.dropna(subset=["decision_correct"])
        if not scored.empty:
            print("\nAccuracy by input mode (unknown-ground-truth bags excluded):")
            accuracy = scored.groupby("input_mode")["decision_correct"].mean()
            print(accuracy.to_string())
            print(f"\nOverall accuracy: {scored['decision_correct'].mean():.3f} ({len(scored)} scored rows)")
    if pca_summary is not None:
        created_plots = [
            mode["plot_file"]
            for mode in pca_summary["modes"]
            if mode.get("status") == "created"
        ]
        if created_plots:
            print("\nCoP PCA plots:")
            for path in created_plots:
                print(f"  {path}")
        else:
            print(
                "\nCoP PCA did not create a plot. Check cop_pca_summary.json "
                "for missing classes or incompatible vectors."
            )


if __name__ == "__main__":
    main()
