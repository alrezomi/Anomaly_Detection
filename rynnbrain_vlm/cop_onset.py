"""Timestamp-supervised logistic classifier for visual execution prefixes.

Annotations are labels, never input features. The complete-execution classifier
remains separate, since final outcome and failure-already-occurred differ.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .failure_timing import read_failure_annotation


ONSET_PROTOCOL = "recorded_failure_logistic_prefixes_v1"


def onset_enabled(classifier_config):
    options = classifier_config.get("logistic_onset", {})
    if not isinstance(options, dict) or not isinstance(options.get("enabled", False), bool):
        raise ValueError("cop_classifier.logistic_onset must be an object with a boolean enabled field.")
    return classifier_config.get("method", "logistic") == "logistic" and options.get("enabled", False)


def onset_training_options(config, mode):
    vlm = config.get("rynnbrain", {})
    if not onset_enabled(vlm.get("cop_classifier", {})):
        return None
    if vlm.get("source") != "rosbag" or mode != "raw":
        raise ValueError("Timestamp-supervised logistic training requires source=rosbag and input_mode=raw.")
    return {"topic": config.get("stage_topic", "/recording_stage"),
            "startup_ignore_sec": float(os.environ.get("STAGE_STARTUP_IGNORE_SEC", .1))}


def onset_model_path(complete_path):
    path = Path(complete_path)
    return path.with_name(path.stem + "_onset.npz")


def select_annotated_records(records, targets, options):
    """Exclude unreadable/conflicting annotations rather than guessing onset labels."""
    selected, annotations, excluded = [], {}, []
    for record, target in zip(records, targets):
        name = record.metadata["bag_name"]
        annotation = read_failure_annotation(record.metadata["test_bag"], **options)
        reason = None
        if annotation["status"] == "unavailable":
            reason = "unavailable_stage_annotations"
        elif target and annotation["status"] != "available":
            reason = "failure_bag_without_error_marker"
        elif not target and annotation["status"] == "available":
            reason = "nominal_bag_with_error_marker"
        if reason:
            excluded.append({"bag_name": name, "reason": reason, "annotation": annotation})
            print(f"Skipping onset training for {name}: {reason}")
        else:
            selected.append(record)
            annotations[name] = {**annotation, "bag_target": int(target)}
    return selected, annotations, excluded


def _grouped_predictions(matrix, labels, groups, bag_targets, *, c_value, class_weight):
    """Stratify by bag outcome; no bag's prefixes occur in both fold partitions."""
    from .cop_classifier import fit_logistic_classifier, _normalize_rows, _sigmoid

    fold_count = min(5, min(list(bag_targets.values()).count(target) for target in (0, 1)))
    if fold_count < 2:
        raise ValueError("Onset training requires at least two usable nominal and two timestamp-annotated failure bags.")
    folds = [[] for _ in range(fold_count)]
    rng = np.random.default_rng(0)
    for target in (0, 1):
        names = sorted(name for name, value in bag_targets.items() if value == target)
        rng.shuffle(names)
        for index, name in enumerate(names):
            folds[index % fold_count].append(name)
    probabilities = np.full(len(labels), np.nan)
    assignments = np.full(len(labels), -1, dtype=int)
    for index, names in enumerate(folds):
        validation = np.isin(groups, names)
        weights, intercept, mean, _ = fit_logistic_classifier(
            matrix[~validation], labels[~validation], c_value=c_value, class_weight=class_weight)
        normalized, _ = _normalize_rows(matrix[validation])
        probabilities[validation] = _sigmoid((normalized - mean) @ weights + intercept)
        assignments[validation] = index
    return probabilities, assignments, folds


def train_onset_classifier(records, targets, complete_path, complete_metadata, options):
    from .cop_classifier import (CoPLogisticClassifier, fit_logistic_classifier, save_classifier,
                                 load_classifier, _classification_metrics)
    from .cop_timeline import _load_prefix_cache, _prefix_cache_paths, timeline_sampling, _model_fingerprint

    if complete_metadata["input_mode"] != "raw" or complete_metadata["comparison_signature"].get("source") != "rosbag":
        raise ValueError("Timestamp-supervised training needs raw rosbag vectors on the stage annotation clock.")
    path = onset_model_path(complete_path)
    # Never leave a previously trained onset model looking like this attempt succeeded.
    path.unlink(missing_ok=True)
    path.with_suffix(".json").unlink(missing_ok=True)
    selected, annotations, excluded = select_annotated_records(records, targets, options)
    sampling = timeline_sampling({}, complete_metadata["comparison_signature"]["num_frames"])
    matrix, labels, rows, bag_targets = [], [], [], {}
    for record in selected:
        name = record.metadata["bag_name"]
        annotation = annotations[name]
        try:
            vectors = _load_prefix_cache(record, sampling, require_timestamps=True)
        except (OSError, ValueError, KeyError) as error:
            raise ValueError(f"Missing/stale timestamped prefixes for {name}; run cop-classifier-train without --saved-vectors-only.") from error
        cache = json.loads(_prefix_cache_paths(record)[1].read_text(encoding="utf-8"))
        times = np.asarray([row["timestamp_sec"] for row in cache["snapshots"]])
        onset = annotation["time_sec"]
        if onset is not None and onset > times[-1]:
            excluded.append({"bag_name": name, "reason": "error_after_last_sample", "annotation": annotation})
            print(f"Skipping onset training for {name}: error_after_last_sample")
            continue
        bag_targets[name] = annotation["bag_target"]
        for step, (vector, time) in enumerate(zip(vectors, times)):
            target = int(onset is not None and time >= onset)
            matrix.append(vector)
            labels.append(target)
            rows.append({"bag_name": name, "sample_index": step, "timestamp_sec": float(time),
                         "recorded_failure_time_sec": onset, "failure_has_occurred": target,
                         "bag_path": record.metadata["test_bag"],
                         "full_evaluation_id": record.metadata["evaluation_id"]})
    if min(list(bag_targets.values()).count(value) for value in (0, 1)) < 2:
        reasons = ", ".join(f"{row['bag_name']}: {row['reason']}" for row in excluded)
        raise ValueError("Onset training needs at least two usable nominal and two annotated failure bags. " + reasons)
    matrix, labels = np.stack(matrix), np.asarray(labels)
    groups = np.asarray([row["bag_name"] for row in rows])
    c_value = complete_metadata["fit_diagnostics"]["regularization_c"]
    class_weight, threshold = complete_metadata["class_weight"], complete_metadata["threshold"]
    probabilities, assignments, folds = _grouped_predictions(
        matrix, labels, groups, bag_targets, c_value=c_value, class_weight=class_weight)
    weights, intercept, mean, diagnostics = fit_logistic_classifier(
        matrix, labels, c_value=c_value, threshold=threshold, class_weight=class_weight)
    predictions = pd.DataFrame(rows)
    predictions["validation_fold"] = assignments
    predictions["out_of_fold_failure_probability"] = probabilities
    predictions_path = path.with_name(path.stem + "_training_predictions.csv")
    predictions.to_csv(predictions_path, index=False)
    metadata = {"schema_version": 1, "classifier_type": "timestamp_supervised_logistic_regression",
                "protocol": ONSET_PROTOCOL, "sampling": sampling,
                "complete_model_sha256": _model_fingerprint(Path(complete_path)),
                "input_mode": complete_metadata["input_mode"],
                "representation_id": complete_metadata["representation_id"],
                "comparison_signature": complete_metadata["comparison_signature"],
                "vector_dimension": matrix.shape[1], "threshold": threshold,
                "training_bags": list(bag_targets), "training_bag_count": len(bag_targets),
                "training_prefix_count": len(labels), "training_options": options,
                "training_annotations": {name: annotations[name] for name in bag_targets},
                "excluded_bags": excluded, "fit_diagnostics": diagnostics,
                "cross_validation": {"method": "stratified_by_bag_outcome_grouped_prefixes", "fold_count": len(folds),
                    "validation_bags_by_fold": folds, "metrics": _classification_metrics(labels, probabilities, threshold),
                    "predictions_csv": str(predictions_path),
                    "note": "Only this classifier is refit per fold; the frozen VLM/LoRA is not retrained. Evaluate onset on separate held-out bags."},
                "target_definition": "0 before the first eligible error marker, 1 at/after it; nominal prefixes are 0.",
                "note": "Probability-like score for failure already having occurred, not guaranteed calibration. "
                        "Labels assume no failure before the button press and remain positive after it, including recovery. "
                        "Button delay and sparse visual sampling limit onset precision.",
                "model_file": str(path), "metadata_file": str(path.with_suffix('.json'))}
    save_classifier(path, CoPLogisticClassifier(weights, intercept, mean, threshold,
        metadata["input_mode"], metadata["representation_id"], metadata["comparison_signature"], metadata))
    load_classifier.cache_clear()
    print(f"Saved timestamp-supervised logistic timeline model: {path}")
    print(f"Onset training: {len(bag_targets)} bags, {len(labels)} prefixes; {len(excluded)} excluded bags.")
    return metadata


def load_onset_classifier(complete_path, complete_classifier, *, bag_name):
    from .cop_classifier import load_classifier
    from .cop_timeline import _model_fingerprint, timeline_sampling

    path = onset_model_path(complete_path)
    if not path.is_file() or not path.with_suffix('.json').is_file():
        raise ValueError("Timestamp-supervised logistic model is missing; run cop-classifier-train or benchmark with logistic_onset.enabled=true.")
    # These tiny artifacts may have been retrained in the same process.
    load_classifier.cache_clear()
    classifier = load_classifier(path)
    metadata = classifier.metadata
    if (metadata.get("protocol") != ONSET_PROTOCOL
            or metadata.get("complete_model_sha256") != _model_fingerprint(Path(complete_path))
            or classifier.comparison_signature != complete_classifier.comparison_signature
            or metadata.get("sampling") != timeline_sampling({}, complete_classifier.comparison_signature["num_frames"])):
        raise ValueError("Timestamp-supervised logistic model is stale; rerun cop-classifier-train.")
    if bag_name in metadata["training_bags"]:
        raise ValueError("This bag trained the onset classifier; select a held-out evaluation bag.")
    return classifier, path
