"""Benchmark the DINO + RynnBrain pipeline on selected folders or all eligible bags.

For every selected ROS 2 bag that was not used to build the DINO
nominal memory or the RynnBrain task memory, this:

1. Runs DINO test mode when videos/heatmaps are needed or --with-dino is
   requested. Raw rosbag-only VLM evaluation automatically skips DINO.
2. Runs RynnBrain multi-turn evaluation using the configured images, reusing
   one already-loaded VLM across every bag instead of reloading it per bag.
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
from pipeline_settings import load_pipeline_config

from build_dataset_manifest import discover_bags, infer_bag_record
from rynnbrain_vlm.benchmark_roc import failure_category, write_roc_report, write_probability_roc_report, write_anomaly_roc_report
from rynnbrain_vlm.cop_analysis import analyze_saved_vectors, write_failure_probability_report, write_anomaly_score_report
from rynnbrain_vlm.model import RynnBrainModel
from rynnbrain_vlm.cop_classifier import classifier_method, classifier_model_paths
from rynnbrain_vlm.run import evaluate_multiturn, write_multiturn_outputs
from rynnbrain_vlm.failure_timing import TIMING_FIELDS, read_failure_annotation, evaluate_failure_timing, write_failure_timing_report
from rynnbrain_vlm.temporal_lora import VLM_TIME_FIELDS
from rynnbrain_vlm.runtime import RUNTIME_FIELDS, clock, record_run
from rynnbrain_vlm.benchmark_protocol import configure_decision_mode, timeline_benchmark_row

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
    parser.add_argument('--folder', action='append', help='Only scan this folder under the data root; repeat. Overrides vlm_benchmark.folders.')
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
    dino_options = parser.add_mutually_exclusive_group()
    dino_options.add_argument("--skip-dino", action="store_true", help="Skip DINO; video/heatmap input modes require existing files.")
    dino_options.add_argument("--with-dino", action="store_true", help="Also run DINO when the VLM uses raw rosbag images (otherwise skipped automatically).")
    parser.add_argument("--skip-vlm", action="store_true", help="Only run the DINO stage.")
    parser.add_argument("--prepare-classifier", action="store_true", help="Train/prepare and enable the classifier for this run, overriding train_on_benchmark=false.")
    return parser.parse_args()


def _raw_rosbag_inputs(vlm):
    return vlm.get("source") == "rosbag" and list(vlm.get("input_modes", ["raw"])) == ["raw"]


def _discover_benchmark_bags(data_root, folders, recursive=True):
    """An omitted/empty selection preserves the existing whole-root discovery."""
    root = Path(data_root).resolve()
    if folders is None or folders == []:
        return discover_bags(root, recursive=recursive), []
    if not isinstance(folders, list) or any(not isinstance(value, str) or not value.strip() for value in folders):
        raise ValueError('vlm_benchmark.folders must be a list of nonempty folder paths.')
    selected = []
    for value in folders:
        folder = (root/value).resolve(strict=True)
        if not folder.is_dir() or folder == root or not folder.is_relative_to(root):
            raise ValueError(f'Select a folder strictly inside {root}: {value}')
        if any(folder.is_relative_to(other) or other.is_relative_to(folder) for other in selected):
            raise ValueError(f'Selected VLM benchmark folders overlap: {value}')
        selected.append(folder)
    bags = []
    for folder in selected:
        found = [folder] if (folder/'metadata.yaml').is_file() else discover_bags(folder, recursive=recursive)
        if any(not path.resolve().is_relative_to(folder) for path in found):
            raise ValueError(f'A discovered bag resolves outside selected folder: {folder}')
        bags.extend(found)
    return sorted(set(bags)), [str(folder.relative_to(root)).replace('\\', '/') for folder in selected]


def _benchmark_folder(path, data_root, folders):
    root, bag = Path(data_root).resolve(), Path(path).resolve()
    for value in folders:
        if bag.is_relative_to(root/value):
            return value
    return str(bag.parent.relative_to(root)).replace('\\', '/')


def _should_run_dino(arguments, vlm):
    if arguments.skip_dino:
        return False
    return bool(getattr(arguments, "with_dino", False) or arguments.skip_vlm or not _raw_rosbag_inputs(vlm))


def _load_reused_classifiers(vlm):
    """Load existing artifacts only; never create missing models or vectors."""
    from rynnbrain_vlm.cop_classifier import load_classifier
    from rynnbrain_vlm.cop_knn import CoPKNNDetector
    config = vlm["cop_classifier"]
    paths = classifier_model_paths(config)
    classifiers = {}
    for mode in vlm.get("input_modes", ["raw"]):
        if not paths.get(mode):
            raise ValueError(f"Set cop_classifier.model_paths.{mode} to the existing classifier file.")
        try:
            classifier = load_classifier(Path(paths[mode]))
        except FileNotFoundError as error:
            raise FileNotFoundError(
                f"Saved classifier is unavailable for {mode}: {paths[mode]}. "
                f"Set cop_classifier.model_paths.{mode} to the trained file (keep its .json sidecar). "
                "train_on_benchmark=false will not train a replacement."
            ) from error
        if classifier.input_mode != mode or isinstance(classifier, CoPKNNDetector) != (classifier_method(config) == "knn"):
            raise ValueError(f"Saved classifier for {mode} does not match the configured method/input mode.")
        if not isinstance(classifier.comparison_signature.get("nominal_response"), str):
            raise ValueError("Saved classifier has no nominal reference text; train a compatible classifier first.")
        classifiers[mode] = classifier
        print(f"Reusing saved {classifier_method(config)} classifier: {paths[mode]}")
    references = {classifier.comparison_signature["nominal_response"] for classifier in classifiers.values()}
    if len(references) != 1:
        raise ValueError("Saved classifiers must use the same nominal reference response for one benchmark.")
    return classifiers, references.pop()


def _validate_reused_classifiers(classifiers, model, config, vlm, reference_response, frame_count, generation):
    from rynnbrain_vlm.run import cop_comparison_signature
    from rynnbrain_vlm.model import COP_REPRESENTATION_ID
    for mode, classifier in classifiers.items():
        expected = cop_comparison_signature(model, config, vlm, mode, reference_response, frame_count, generation)
        if classifier.representation_id != COP_REPRESENTATION_ID or classifier.comparison_signature != expected:
            raise ValueError(
                f"Saved {mode} classifier is incompatible with this model/LoRA, cameras, prompts or sampling settings. "
                "Select its matching settings or explicitly train a new classifier; reuse will not retrain it."
            )


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
        command = [sys.executable, "-u", str(REPO_ROOT / "run_rosbag_vision.py"), "--config", str(config_path), "--mode", "test"]
        with subprocess.Popen(command, cwd=REPO_ROOT, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace") as process:
            for line in process.stdout:
                print(line, end="", flush=True)
            if process.wait():
                raise subprocess.CalledProcessError(process.returncode, command)
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
    if expected is None or decision == "not_generated":
        return None
    if decision in ABSTAIN_DECISIONS:
        return False
    return decision == expected


def _align_benchmark_timing(rows, bag_path, stage_topic, vlm, ground_truth):
    """Correct only evaluation annotations; reuse existing classifier timelines."""
    from rynnbrain_vlm.visual_matching import read_corrected_failure_annotation
    from rynnbrain_vlm.failure_timing import attach_failure_timing
    annotation = read_corrected_failure_annotation(bag_path, stage_topic)
    method = classifier_method(vlm.get('cop_classifier', {}))
    for row in rows:
        timeline = {key: row.get(f'{method}_timeline_{key}') for key in ['status', 'csv', 'plot', 'error']}
        row.update(attach_failure_timing(annotation, timeline,
            source=vlm.get('source', 'generated_videos'), mode=row['input_mode'],
            ground_truth=ground_truth, method=method, bag_name=Path(bag_path).name))
    return annotation


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
    timing = any("failure_timing_status" in row for row in master_rows)
    if timing:
        columns += ["input_mode", *TIMING_FIELDS]
    temporal = any("vlm_failure_onset_status" in row for row in master_rows)
    if temporal:
        columns += list(VLM_TIME_FIELDS)
    runtime = any("inference_mode" in row for row in master_rows)
    if runtime:
        columns += ["inference_mode", *RUNTIME_FIELDS, "runtime_total_bag_sec"]
        if any('runtime_frame_count' in row for row in master_rows):
            columns += ['runtime_frame_count', 'runtime_frame_sec_mean', 'runtime_feature_frame_sec_mean', 'runtime_classifier_frame_sec_mean']
        if "classifier_decision" not in columns:
            columns += ["classifier_decision"]
        columns += ["classifier_correct"]
    timeline = any(row.get('benchmark_decision_mode') == 'timeline' for row in master_rows)
    if timeline:
        columns += ['benchmark_decision_mode', 'timeline_first_alarm_time_sec',
                    'whole_demo_classifier_decision', 'whole_demo_classifier_failure_probability',
                    'timeline_error']
        if 'classifier_decision' not in columns:
            columns += ['classifier_decision', 'classifier_correct']
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
                **({"input_mode": row.get("input_mode"), **{key: row.get(key) for key in TIMING_FIELDS}} if timing else {}),
                **({key: row.get(key) for key in VLM_TIME_FIELDS} if temporal else {}),
                **({"inference_mode": row.get("inference_mode"),
                    **{key: row.get(key) for key in (*RUNTIME_FIELDS, "runtime_total_bag_sec")},
                    **{key: row.get(key) for key in ('runtime_frame_count', 'runtime_frame_sec_mean',
                        'runtime_feature_frame_sec_mean', 'runtime_classifier_frame_sec_mean')},
                    "classifier_decision": row.get("classifier_decision"),
                    "classifier_correct": row.get("classifier_decision_correct")} if runtime else {}),
                **({'benchmark_decision_mode': row.get('benchmark_decision_mode'),
                    'timeline_first_alarm_time_sec': row.get('timeline_first_alarm_time_sec'),
                    'whole_demo_classifier_decision': row.get('whole_demo_classifier_decision'),
                    'whole_demo_classifier_failure_probability': row.get('whole_demo_classifier_failure_probability'),
                    'timeline_error': row.get('timeline_error'),
                    'classifier_decision': row.get('classifier_decision'),
                    'classifier_correct': row.get('classifier_decision_correct')} if timeline else {}),
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


def _write_tables(master_rows, benchmark_root):
    """Checkpoint completed bags without rerunning any model or plots."""
    for row in master_rows:
        row["failure_category"] = failure_category(row["bag_path"], row["ground_truth_label"])
    for name, data in (("benchmark_summary.csv", pd.DataFrame(master_rows)),
                       ("benchmark_clean.csv", _build_clean_report(master_rows))):
        path = benchmark_root / name
        temporary = path.with_suffix(".csv.tmp")
        data.to_csv(temporary, index=False)
        temporary.replace(path)


def _run_benchmark(arguments, config, update) -> dict[str, Any] | None:
    vlm = dict(config.get("rynnbrain", {}))
    if not vlm and not arguments.skip_vlm:
        raise ValueError("Add a 'rynnbrain' section to pipeline_config.json, or pass --skip-vlm.")
    camera_topics = list(config.get("camera_topics", []))
    frame_count = int(vlm.get("num_frames", 4))
    generation = dict(vlm.get("generation", {}))
    run_dino = _should_run_dino(arguments, vlm)
    print("DINO: enabled" if run_dino else "DINO: skipped; raw rosbag VLM inputs do not need DINO. Heatmap/video modes use existing files when explicitly skipped.")

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
    selection = config.get('vlm_benchmark') or {}
    if not isinstance(selection, dict):
        raise ValueError('vlm_benchmark must be an object containing a folders list.')
    from rynnbrain_vlm.benchmark_replay import threshold_search_enabled, validate_threshold_search
    threshold_search = threshold_search_enabled(selection)
    if arguments.prepare_classifier:
        vlm['cop_classifier'] = {**vlm.get('cop_classifier', {}), 'enabled': True}
    decision_mode = configure_decision_mode(selection, vlm, skip_vlm=arguments.skip_vlm)
    config['rynnbrain'] = vlm
    if threshold_search:
        validate_threshold_search(config, decision_mode, run_dino, arguments.skip_vlm)
    benchmark_root = benchmark_root / decision_mode
    benchmark_root.mkdir(parents=True, exist_ok=True)
    update(decision_mode=decision_mode, benchmark_output_dir=str(benchmark_root))
    print(f'VLM benchmark decision mode: {decision_mode}')
    if decision_mode == 'timeline' and not threshold_search:
        print('Offline prefix evaluation: any score >= its threshold is failure; no real-time waiting. '
              'Whole-demo classifier results are retained in whole_demo_classifier_* columns.')
        (benchmark_root/'benchmark_protocol.json').write_text(json.dumps({
            'decision_mode': decision_mode,
            'alarm_rule': 'Any sampled prefix score >= its own classifier threshold, including the final prefix.',
            'sampling': 'Existing offline timeline sampling; only current and preceding execution images at each step. Not elapsed-time replay sampling.',
            'stage_cutoff': None, 'consecutive_alarms': 1,
            'score': 'Maximum prefix probability (logistic) or maximum prefix score/threshold ratio (kNN). Not a calibrated bag-level probability.',
            'original_result': 'whole_demo_classifier_* columns; per-bag rynnbrain_results_multiturn.csv remains the whole-demo result.',
        }, indent=2)+'\n', encoding='utf-8')
    folders = getattr(arguments, 'folder', None)
    folders = folders if folders is not None else selection.get('folders', [])
    all_bags, folder_names = _discover_benchmark_bags(data_root, folders, not arguments.no_recursive)
    print(f'VLM benchmark folders: {folder_names or "all folders under data root"}')
    update(selected_folders=folder_names, folder_selection='explicit' if folder_names else 'all')

    model: RynnBrainModel | None = None
    reference_response = None
    reused_classifiers = {}
    if arguments.prepare_classifier and arguments.skip_vlm:
        raise ValueError("--prepare-classifier requires VLM evaluation; remove --skip-vlm.")
    classifier_config = vlm.get("cop_classifier", {})
    train_on_benchmark = classifier_config.get("train_on_benchmark", True)
    if not isinstance(train_on_benchmark, bool):
        raise ValueError("cop_classifier.train_on_benchmark must be true or false.")
    score_classifier = not arguments.skip_vlm and (arguments.prepare_classifier or bool(classifier_config.get("enabled", False)))
    prepare_classifier = score_classifier and (arguments.prepare_classifier or train_on_benchmark)
    if score_classifier:
        vlm["cop_classifier"] = {**classifier_config, "enabled": True}
        vlm["cop_vectors"] = {**vlm.get("cop_vectors", {}), "enabled": True}
        config["rynnbrain"] = vlm
    if score_classifier and not prepare_classifier:
        update(phase="classifier_loading", classifier_mode="reuse", classifier_preparation_sec=0.0)
        started = clock()
        reused_classifiers, reference_response = _load_reused_classifiers(vlm)
        update(classifier_load_sec=clock() - started)
    if prepare_classifier:
        update(phase="classifier_preparation", classifier_mode="train")
        preparation_started = clock()
        from rynnbrain_vlm.cop_classifier import (
            configured_training_settings, prepare_training_vectors, train_from_saved_vectors,
        )
        settings_by_mode = [configured_training_settings(arguments.config, mode)
                            for mode in vlm.get("input_modes", ["raw"])]
        for settings in settings_by_mode:
            if settings.get("onset_options") is not None:
                settings["onset_options"]["startup_ignore_sec"] = startup_ignore_sec
            configured_path = classifier_model_paths(vlm["cop_classifier"]).get(settings["input_mode"])
            if not configured_path or Path(configured_path).resolve() != settings["output_file"].resolve():
                raise ValueError("Classifier training output_file must match model_paths for benchmark scoring.")
        if getattr(arguments, "lora_training_config", None):
            from rynnbrain_vlm.benchmark_training import run_lora_training
            run_lora_training(arguments.lora_training_config, data_root,
                              vlm["model"]["lora_adapter_path"], update, startup_ignore_sec)
            update(phase="classifier_preparation")
        print("Loading RynnBrain once for classifier preparation and benchmark evaluation...")
        started = clock()
        model = RynnBrainModel(vlm["model"])
        update(model_load_sec=clock() - started)
        for settings in settings_by_mode:
            print(f"Classifier-training artifacts (separate from evaluated bags): {settings['input_directory']}")
            paths, reference_response = prepare_training_vectors(config, settings, model=model, data_root=data_root)
            classifier_metadata = train_from_saved_vectors(**settings, metadata_paths=paths)
            print(f"Prepared {settings['method']} classifier: {classifier_metadata['model_file']}")
        update(classifier_preparation_sec=clock() - preparation_started)

    update(phase="bag_selection")
    print(f"Using bags discovered under: {data_root}")
    excluded_names = _excluded_bag_names(config, arguments.include_nominal_bags)
    records = [
        infer_bag_record(bag_path, stage_topic, startup_ignore_sec)
        for bag_path in all_bags
        if bag_path.name not in excluded_names
    ]
    for record in records:
        path = Path(record['bag_path'])
        record['folder'] = _benchmark_folder(path if path.is_absolute() else data_root/path, data_root, folder_names)
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
    excluded_count = sum(path.name in excluded_names for path in all_bags)
    print(f"Found {len(all_bags)} bag(s), excluding {excluded_count} nominal/reference/LoRA/classifier-training bag(s) from this selection.")
    print(f"Benchmarking {len(records)} demonstration(s).")
    selection_columns = ["bag_name", "label"]
    if records and "label_source" in selection_df:
        selection_columns.extend(["label_source", "recorded_label"])
    print(selection_df[selection_columns].to_string(index=False) if records else "  (none)")

    candidates = records[: arguments.limit] if arguments.limit else records
    duplicates = pd.Series([record['bag_name'] for record in candidates], dtype=str).duplicated(keep=False)
    if duplicates.any():
        raise ValueError('Selected bag names must be unique to avoid overwriting per-bag outputs. '
                         'Narrow vlm_benchmark.folders or use --bag to select unique names.')
    if threshold_search and not {'normal', 'fail'}.issubset({record['label'] for record in candidates}):
        raise ValueError('threshold_search requires selected held-out bags labeled both normal and fail.')

    if not arguments.skip_vlm and model is None:
        print("\nLoading RynnBrain model once for the whole benchmark...")
        update(phase="model_loading")
        started = clock()
        model = RynnBrainModel(vlm["model"])
        update(model_load_sec=clock() - started)

    if reused_classifiers and not threshold_search:
        _validate_reused_classifiers(reused_classifiers, model, config, vlm, reference_response, frame_count, generation)
        print("Saved classifier settings verified; reusing its nominal reference text. No classifier training performed.")

    if threshold_search:
        from rynnbrain_vlm.online_replay import OnlineDetector
        from rynnbrain_vlm.benchmark_replay import run_threshold_search
        detector = OnlineDetector(config, progress=print, model=model)
        selected_records = [{**record, 'bag_path': str(Path(record['bag_path']) if Path(record['bag_path']).is_absolute()
                            else (data_root / record['bag_path']).resolve())} for record in candidates]
        master_rows, best_root, effective = run_threshold_search(config, selected_records, benchmark_root,
                                                               detector, update, startup_ignore_sec)
        _write_benchmark_reports(arguments, effective, effective['rynnbrain'], master_rows, best_root,
                                 folder_names, selected_records, decision_mode, update)
        return effective

    master_rows: list[dict[str, Any]] = []
    _write_tables(master_rows, benchmark_root)
    for index, record in enumerate(candidates, start=1):
        bag_started = clock()
        bag_path = Path(record["bag_path"])
        # bag_path was stored relative to data_root by infer_bag_record's caller convention.
        if not bag_path.is_absolute():
            bag_path = (data_root / bag_path).resolve()
        bag_name = record["bag_name"]
        ground_truth = record["label"]
        bag_output_dir = benchmark_root / bag_name
        print(f"\n[{index}/{len(candidates)}] {bag_name} (ground_truth={ground_truth})")
        update(phase="evaluation", current_bag=bag_name, selected_bags=len(candidates))

        dino_ok = True
        if run_dino:
            dino_ok = _run_dino_test(config, bag_path, bag_output_dir)
        # Keep the explicit --skip-dino option's legacy saved-summary behavior.
        # Automatic skipping must not present stale DINO metrics as newly run.
        dino_summary = (_dino_summary(camera_topics, bag_output_dir)
                        if dino_ok and (run_dino or arguments.skip_dino) else {})
        dino_summary.update(dino_status="succeeded" if run_dino and dino_ok else "failed" if run_dino else "skipped")

        rows: list[dict[str, Any]] = []
        inputs_ready = dino_ok or _raw_rosbag_inputs(vlm)
        evaluation_error = None if inputs_ready else "DINO processing failed; VLM evaluation was not run."
        if model is not None and inputs_ready:
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
                    stage_startup_ignore_sec=startup_ignore_sec,
                )
                _align_benchmark_timing(rows, bag_path, stage_topic, vlm_bag, ground_truth)
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
                "folder": record['folder'],
                "ground_truth_label": ground_truth,
                "input_mode": None,
                'benchmark_decision_mode': decision_mode,
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
                "runtime_total_bag_sec": clock() - bag_started,
                **evaluate_failure_timing(
                    read_failure_annotation(bag_path, stage_topic, startup_ignore_sec), {},
                    source=vlm.get("source", "generated_videos"), mode="raw", ground_truth=ground_truth,
                    method=classifier_method(vlm.get("cop_classifier", {})),
                ),
                **dino_summary,
            })
            _write_tables(master_rows, benchmark_root)
            update(completed_bags=index)
            continue

        for row in rows:
            result = {
                "bag_name": bag_name,
                "bag_path": str(bag_path),
                "folder": record['folder'],
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
                "inference_mode": row.get("inference_mode", "generation"),
                **{key: row.get(key) for key in RUNTIME_FIELDS},
                "runtime_total_bag_sec": clock() - bag_started,
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
                **{key: row.get(key) for key in TIMING_FIELDS},
                **{key: row[key] for key in VLM_TIME_FIELDS if key in row},
                **{key: row.get(key) for key in ("knn_timeline_status", "knn_timeline_csv", "knn_timeline_plot", "knn_timeline_error")},
                **{key: row.get(key) for key in ("logistic_timeline_status", "logistic_timeline_csv", "logistic_timeline_plot", "logistic_timeline_error",
                                                "logistic_last_below_before_alert_sec", "logistic_first_above_threshold_sec")},
                **{key: row.get(key) for key in ("knn_neighbors_status", "knn_neighbors_csv", "knn_neighbors_plot", "knn_neighbors_error",
                                                "knn_last_below_before_alert_sec", "knn_first_above_threshold_sec")},
                **dino_summary,
                'benchmark_decision_mode': decision_mode,
            }
            if decision_mode == 'timeline':
                result = timeline_benchmark_row(result, classifier_method(vlm['cop_classifier']))
                print(f"  Timeline ({result['input_mode']}): {result.get('classifier_decision') or 'unavailable'}; "
                      f"first alarm={result.get('timeline_first_alarm_time_sec')} s; "
                      f"whole-demo={result.get('whole_demo_classifier_decision')}")
                if result.get('timeline_error'):
                    print(f"  Timeline warning: {result['timeline_error']}")
            master_rows.append(result)
        _write_tables(master_rows, benchmark_root)
        update(completed_bags=index)

    _write_benchmark_reports(arguments, config, vlm, master_rows, benchmark_root,
                             folder_names, candidates, decision_mode, update)


def _write_benchmark_reports(arguments, config, vlm, master_rows, benchmark_root,
                             folder_names, candidates, decision_mode, update):
    update(phase="reports", current_bag=None,
           mean_runtime_sec={key: float(pd.Series([row.get(key) for row in master_rows], dtype=float).dropna().mean())
                             for key in RUNTIME_FIELDS if any(row.get(key) is not None for row in master_rows)})
    _write_tables(master_rows, benchmark_root)
    summary_path = benchmark_root / "benchmark_summary.csv"
    clean_report_path = benchmark_root / "benchmark_clean.csv"
    statistics = _report_statistics(master_rows)
    statistics['decision_mode'] = decision_mode
    if any('runtime_frame_count' in row for row in master_rows):
        from rynnbrain_vlm.benchmark_replay import write_frame_runtime_report
        statistics['frame_runtime'] = write_frame_runtime_report(master_rows, benchmark_root)
        print(f"Average processing per observation: {statistics['frame_runtime']['average_processing_sec_per_frame']:.4f} s")
    from rynnbrain_vlm.benchmark_metrics import write_metrics_report
    statistics['folder_metrics'] = write_metrics_report(master_rows, benchmark_root,
        folder_names or list(dict.fromkeys(record['folder'] for record in candidates)),
        input_modes=vlm.get('input_modes', ['raw']))
    statistics["failure_timing"] = write_failure_timing_report(
        master_rows, benchmark_root, input_modes=vlm.get("input_modes", ["raw"]),
        method=classifier_method(vlm.get("cop_classifier", {})),
    )
    statistics["failure_probability"] = write_failure_probability_report(
        master_rows, benchmark_root, input_modes=vlm.get("input_modes", ["raw"])
    )
    if decision_mode == 'timeline' or any(row.get("classifier_decision") is not None for row in master_rows):
        statistics["classifier"] = _report_statistics([
            {**{key: row.get(key) for key in ("bag_name", "ground_truth_label", "input_mode")},
             "decision": row.get("classifier_decision"),
             "decision_correct": row.get("classifier_decision_correct")}
            for row in master_rows
        ])
    if decision_mode == 'timeline':
        statistics['generated_response'] = _report_statistics(master_rows)
        statistics.update(statistics['classifier'])
        statistics['decision_source'] = 'classifier_timeline'
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
    from rynnbrain_vlm.report_artifacts import write_report_index
    report_index = write_report_index(master_rows, statistics, pca_summary, benchmark_root,
                                     classifier_method(vlm.get('cop_classifier', {})))
    print(f'Report index (all graphs and availability): {report_index}')
    print("BENCHMARK FINISHED")
    print("=" * 90)
    print(f"Per-demonstration folders and combined table under: {benchmark_root}")
    print(f"Summary table: {summary_path}")
    print(f"Clean report: {clean_report_path}")
    print(f"Statistics: {statistics_path}")
    if decision_mode == 'timeline':
        if any('threshold_selection_peak_score' in row for row in master_rows):
            print('Sequential replay metrics use the chosen threshold. ROC ranks complete uncensored scan trajectories; '
                  'per-bag outputs stop at the first alarm when configured. Whole-demo inference was not run.')
        else:
            print('Classifier metrics use any sampled timeline alarm; whole-demo results remain in whole_demo_classifier_* columns.')
    print(f"Recorded vs. predicted failure timing: {benchmark_root / 'benchmark_failure_timing'}")
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
    print(f"Threshold sweep (60-90%, step 1%): {benchmark_root / 'benchmark_threshold_sweep'}")
    for comparison in statistics["probability_roc"].get("threshold_sweep", {}).get("comparisons", []):
        if comparison["failure_category"] == "all_failures" and comparison["status"] == "created":
            thresholds = ', '.join(f"{100 * value:.0f}%" for value in comparison["best_accuracy_thresholds"])
            print(f"  {comparison['input_mode']} / {comparison['benchmark_decision_mode']}: "
                  f"best observed accuracy={comparison['best_accuracy']:.1%}; tied threshold(s): {thresholds}")
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


def main() -> None:
    arguments = parse_arguments()
    config = load_pipeline_config(arguments.config)
    from rynnbrain_vlm.benchmark_training import (
        train_lora_enabled, prepare_pipeline_run, save_reusable_config, resolve_trained_artifacts,
    )
    automatic_training = train_lora_enabled(config)
    artifact_messages = []
    if not automatic_training and not arguments.skip_vlm:
        resolved = resolve_trained_artifacts(config, artifact_messages.append)
        if resolved is not config:
            config = resolved
    base = config.get("rynnbrain", {}).get("output_dir", str(Path(config.get("output_dir", "outputs")) / "rynnbrain"))
    directory = arguments.benchmark_dir or Path(f"{base}_benchmark")
    directory = directory / (config.get('vlm_benchmark') or {}).get('decision_mode', 'whole_demo')
    if automatic_training:
        arguments, config, directory = prepare_pipeline_run(arguments, config)
    with record_run(directory, "benchmark") as update:
        if artifact_messages:
            from rynnbrain_vlm.runtime import write_json
            arguments.config = directory / "benchmark_input_config.json"
            write_json(arguments.config, config)
            update(effective_config=str(arguments.config))
            for message in artifact_messages:
                print(message, flush=True)
        update(arguments=vars(arguments) | {key: str(value) for key, value in vars(arguments).items() if isinstance(value, Path)})
        if automatic_training:
            update(automatic_lora_training=True, effective_config=str(arguments.config))
            print(f"Automatic LoRA -> classifier -> benchmark run: {arguments.benchmark_dir}", flush=True)
            print(f"Effective configuration: {arguments.config}", flush=True)
        finished_config = _run_benchmark(arguments, config, update)
        if automatic_training:
            save_reusable_config(arguments.config, finished_config or config)
            print(f"Reusable trained configuration (no retraining): {arguments.config}", flush=True)


if __name__ == "__main__":
    main()
