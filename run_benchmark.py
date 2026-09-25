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
from rynnbrain_vlm.benchmark_roc import failure_category, write_roc_report, write_probability_roc_report, write_anomaly_roc_report
from rynnbrain_vlm.cop_analysis import analyze_saved_vectors, write_failure_probability_report, write_anomaly_score_report
from rynnbrain_vlm.model import RynnBrainModel
from rynnbrain_vlm.cop_classifier import classifier_method, classifier_model_paths
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
    parser.add_argument("--prepare-classifier", action="store_true", help="Enable classifier preparation/scoring for this run even if disabled in config; otherwise preparation is automatic when cop_classifier.enabled=true.")
    return parser.parse_args()


def _excluded_bag_names(
    config: dict[str, Any], include_nominal_bags: bool = False
) -> set[str]:
    rynnbrain = config.get("rynnbrain", {})
    names = {Path(bag).name for bag in rynnbrain.get("reference_bags", [])}
    adapter_path = rynnbrain.get("model", {}).get("lora_adapter_path")
    if adapter_path:
        from rynnbrain_vlm.lora import adapter_training_bag_names

        names |= adapter_training_bag_names(adapter_path)
    if not include_nominal_bags:
        names |= {Path(bag).name for bag in config.get("nominal_bags", [])}
    classifier = rynnbrain.get("cop_classifier", {})
    if classifier.get("enabled", False):
        training = classifier.get("training", {})
        keys = ("normal_bags",) if classifier_method(classifier) == "knn" else ("bags", "normal_bags", "failure_bags")
        for key in keys:
            names |= {str(bag).replace("\\", "/").rstrip("/").rsplit("/", 1)[-1] for bag in training.get(key, [])}
        for path in classifier_model_paths(classifier).values():
            metadata = Path(path).with_suffix(".json")
            if metadata.is_file():
                names |= set(json.loads(metadata.read_text(encoding="utf-8")).get("training_bags", []))
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


def _build_clean_report(master_rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Keep only the fields needed to inspect each model decision."""
    columns = [
        "bag_name",
        "label",
        "model_decision",
        "correct",
        "failure_probability",
    ]
    knn = any(row.get("classifier_score_kind") == "knn_distance" for row in master_rows)
    if knn:
        columns += ["anomaly_score", "anomaly_threshold", "classifier_decision"]
    return pd.DataFrame(
        [
            {
                "bag_name": row.get("bag_name"),
                "label": row.get("ground_truth_label"),
                "model_decision": row.get("decision"),
                "correct": row.get("decision_correct"),
                "failure_probability": row.get("classifier_failure_probability"),
                **({"anomaly_score": row.get("classifier_anomaly_score"),
                    "anomaly_threshold": row.get("classifier_threshold") if row.get("classifier_score_kind") == "knn_distance" else None,
                    "classifier_decision": row.get("classifier_decision")} if knn else {}),
            }
            for row in master_rows
        ],
        columns=columns,
    )


def _report_statistics(master_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Return JSON-serializable correctness counts for the benchmark rows."""
    dataframe = pd.DataFrame(master_rows)
    if dataframe.empty:
        return {
            "total_rows": 0,
            "scored_rows": 0,
            "correct_rows": 0,
            "incorrect_rows": 0,
            "unscored_rows": 0,
            "accuracy": None,
            "accuracy_percent": None,
            "by_label": {},
            "by_decision": {},
            "by_input_mode": {},
        }

    def counts(group: pd.DataFrame) -> dict[str, Any]:
        group_scored = group[group["decision_correct"].notna()]
        correct = int(group_scored["decision_correct"].sum())
        scored_count = len(group_scored)
        return {
            "total_rows": int(len(group)),
            "scored_rows": scored_count,
            "correct_rows": correct,
            "incorrect_rows": scored_count - correct,
            "unscored_rows": int(len(group) - scored_count),
            "accuracy": correct / scored_count if scored_count else None,
            "accuracy_percent": 100.0 * correct / scored_count if scored_count else None,
        }

    def grouped_counts(column: str) -> dict[str, dict[str, Any]]:
        return {
            str(value): counts(group)
            for value, group in dataframe.groupby(column, dropna=False, sort=True)
        }

    overall = counts(dataframe)
    return {
        **overall,
        "by_label": grouped_counts("ground_truth_label"),
        "by_decision": grouped_counts("decision"),
        "by_input_mode": grouped_counts("input_mode"),
    }


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
            "The following selected bags are nominal/reference memory, LoRA training or classifier training bags "
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

    model: RynnBrainModel | None = None
    reference_response = None
    if arguments.prepare_classifier and arguments.skip_vlm:
        raise ValueError("--prepare-classifier requires VLM evaluation; remove --skip-vlm.")
    prepare_classifier = not arguments.skip_vlm and (
        arguments.prepare_classifier or bool(vlm.get("cop_classifier", {}).get("enabled", False))
    )
    if prepare_classifier:
        from rynnbrain_vlm.cop_classifier import (
            configured_training_settings, prepare_training_vectors, train_from_saved_vectors,
        )
        # Enable scoring only in this run; do not rewrite the user's config.
        vlm["cop_classifier"] = {**vlm.get("cop_classifier", {}), "enabled": True}
        vlm["cop_vectors"] = {**vlm.get("cop_vectors", {}), "enabled": True}
        config["rynnbrain"] = vlm
        settings_by_mode = [configured_training_settings(arguments.config, mode)
                            for mode in vlm.get("input_modes", ["raw"])]
        for settings in settings_by_mode:
            configured_path = classifier_model_paths(vlm["cop_classifier"]).get(settings["input_mode"])
            if not configured_path or Path(configured_path).resolve() != settings["output_file"].resolve():
                raise ValueError("Classifier training output_file must match model_paths for benchmark scoring.")
        print("Loading RynnBrain once for classifier preparation and benchmark evaluation...")
        model = RynnBrainModel(vlm["model"])
        for settings in settings_by_mode:
            paths, reference_response = prepare_training_vectors(config, settings, model=model, data_root=data_root)
            classifier_metadata = train_from_saved_vectors(**settings, metadata_paths=paths)
            print(f"Prepared {settings['method']} classifier: {classifier_metadata['model_file']}")

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
    print(f"Found {len(all_bags)} bag(s), excluding {len(excluded_names)} nominal/reference/LoRA/classifier-training bag(s).")
    print(f"Benchmarking {len(records)} demonstration(s).")
    selection_columns = ["bag_name", "label"]
    if records and "label_source" in selection_df:
        selection_columns.extend(["label_source", "recorded_label"])
    print(selection_df[selection_columns].to_string(index=False) if records else "  (none)")

    candidates = records[: arguments.limit] if arguments.limit else records

    if not arguments.skip_vlm and model is None:
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
        evaluation_error = None if dino_ok else "DINO processing failed; VLM evaluation was not run."
        if model is not None and dino_ok:
            vlm_bag = dict(vlm)
            vlm_bag["output_dir"] = str(bag_output_dir / "rynnbrain_multiturn")
            vlm_bag["ground_truth_label"] = ground_truth
            config_bag = dict(config)
            config_bag["test_bag"] = str(bag_path)
            config_bag["output_dir"] = str(bag_output_dir)
            try:
                rows, frame_metadata, raw_records, task_description = evaluate_multiturn(
                    model, config_bag, vlm_bag, frame_count, generation, Path(vlm_bag["output_dir"]),
                    reference_response=reference_response,
                )
                write_multiturn_outputs(
                    Path(vlm_bag["output_dir"]), rows, frame_metadata, raw_records, task_description
                )
            except Exception as error:
                print(f"  [VLM FAILED] {bag_name}: {error}")
                evaluation_error = str(error)
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
                "evaluation_error": evaluation_error,
                "cop_vector_path": None,
                "cop_metadata_path": None,
                "classifier_failure_probability": None,
                "classifier_failure_percent": None,
                "classifier_decision": None,
                "classifier_decision_correct": None,
                "classifier_threshold": None,
                "classifier_model_path": None,
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
                "evaluation_error": None,
                "evaluation_id": row.get("evaluation_id"),
                "generated_at_utc": row.get("generated_at_utc"),
                "model_id": row.get("model_id"),
                "lora_adapter_path": row.get("lora_adapter_path"),
                "lora_adapter_sha256": row.get("lora_adapter_sha256"),
                "response_source": row.get("response_source"),
                "cop_vector_path": row.get("cop_vector_path"),
                "cop_metadata_path": row.get("cop_metadata_path"),
                "classifier_failure_probability": row.get(
                    "classifier_failure_probability"
                ),
                "classifier_failure_percent": row.get("classifier_failure_percent"),
                "classifier_decision": row.get("classifier_decision"),
                "classifier_decision_correct": (
                    _decision_correct(ground_truth, row["classifier_decision"])
                    if row.get("classifier_decision")
                    else None
                ),
                "classifier_threshold": row.get("classifier_threshold"),
                "classifier_model_path": row.get("classifier_model_path"),
                "classifier_error": row.get("classifier_error"),
                "classifier_anomaly_score": row.get("classifier_anomaly_score"),
                "classifier_score_kind": row.get("classifier_score_kind"),
                **dino_summary,
            })

    for row in master_rows:
        row["failure_category"] = failure_category(row["bag_path"], row["ground_truth_label"])
    summary_df = pd.DataFrame(master_rows)
    summary_path = benchmark_root / "benchmark_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    clean_report = _build_clean_report(master_rows)
    clean_report_path = benchmark_root / "benchmark_clean.csv"
    clean_report.to_csv(clean_report_path, index=False)
    statistics = _report_statistics(master_rows)
    statistics["failure_probability"] = write_failure_probability_report(
        master_rows, benchmark_root, input_modes=vlm.get("input_modes", ["raw"])
    )
    if any(row.get("classifier_decision") is not None for row in master_rows):
        statistics["classifier"] = _report_statistics([
            {**{key: row.get(key) for key in ("bag_name", "ground_truth_label", "input_mode")},
             "decision": row.get("classifier_decision"),
             "decision_correct": row.get("classifier_decision_correct")}
            for row in master_rows
        ])
    roc_report = write_roc_report(master_rows, benchmark_root)
    statistics["roc"] = roc_report
    statistics["probability_roc"] = write_probability_roc_report(
        master_rows, benchmark_root, input_modes=vlm.get("input_modes", ["raw"])
    )
    if classifier_method(vlm.get("cop_classifier", {})) == "knn" or (benchmark_root / "benchmark_anomaly_roc").exists():
        statistics["anomaly_score"] = write_anomaly_score_report(master_rows, benchmark_root, input_modes=vlm.get("input_modes", ["raw"]))
        statistics["anomaly_roc"] = write_anomaly_roc_report(master_rows, benchmark_root, input_modes=vlm.get("input_modes", ["raw"]))
    statistics_path = benchmark_root / "benchmark_statistics.json"
    statistics_path.write_text(
        json.dumps(statistics, indent=2) + "\n", encoding="utf-8"
    )

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
    print(f"Clean report: {clean_report_path}")
    print(f"Statistics: {statistics_path}")
    if "anomaly_score" in statistics:
        print(f"Nominal kNN distance boxplots: {benchmark_root}")
        print(f"Anomaly-distance ROC/AUROC: {benchmark_root / 'benchmark_anomaly_roc'}")
    for mode in statistics["failure_probability"]["modes"]:
        if mode["plot_file"]:
            print(f"Failure probability boxplot ({mode['input_mode']}): {mode['plot_file']}")
    print(f"VLM decision ROC and per-category metrics: {benchmark_root / 'benchmark_roc'}")
    print("  Binary decision AUROC equals balanced accuracy on decided bags; see abstention counts and coverage.")
    for metric in roc_report["metrics"]:
        value = f"AUROC={metric['auroc']:.3f}" if metric["status"] == "created" else metric["reason"]
        print(f"  {metric['input_mode']} / {metric['failure_category']}: {value}")
    print(f"CoP probability ROC/AUROC: {benchmark_root / 'benchmark_probability_roc'}")
    for metric in statistics["probability_roc"]["metrics"]:
        value = f"AUROC={metric['auroc']:.3f}" if metric["status"] == "created" else metric["reason"]
        print(f"  {metric['input_mode']} / {metric['failure_category']}: {value}")
    if statistics["scored_rows"]:
        print(
            "\nOverall accuracy: "
            f"{statistics['accuracy']:.3f} "
            f"({statistics['correct_rows']}/{statistics['scored_rows']} scored rows)"
        )
        print("Accuracy by input mode (unknown-ground-truth bags excluded):")
        for mode, mode_stats in statistics["by_input_mode"].items():
            if mode_stats["scored_rows"]:
                print(f"  {mode}: {mode_stats['accuracy']:.3f}")
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
