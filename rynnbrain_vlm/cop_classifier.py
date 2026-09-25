"""Train and apply nominal kNN or supervised logistic CoP classifiers."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .cop_analysis import SavedCoPVector, discover_saved_vectors


CLASSIFIER_SCHEMA_VERSION = 1
LABEL_TO_TARGET = {"normal": 0, "nominal": 0, "fail": 1, "failure": 1}


def classifier_method(config: dict[str, Any]) -> str:
    method = str(config.get("method", "logistic"))
    if method not in {"logistic", "knn"}:
        raise ValueError("cop_classifier.method must be 'logistic' or 'knn'.")
    return method


def _method_model_path(value: Any, method: str) -> Path:
    path = Path(str(value))
    if method == "knn" and not path.stem.endswith("_knn"):
        # Keep the existing logistic file available when switching methods.
        path = path.with_name(path.stem.removesuffix("_logistic") + "_knn" + path.suffix)
    return path


def classifier_model_paths(config: dict[str, Any]) -> dict[str, str]:
    method = classifier_method(config)
    paths = config.get("model_paths", {})
    if not isinstance(paths, dict):
        raise ValueError("cop_classifier.model_paths must be an object.")
    return {mode: str(_method_model_path(value, method)) for mode, value in paths.items() if value}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    output = np.empty_like(values)
    positive = values >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


def _normalize_rows(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 2:
        raise ValueError("Classifier input must be a 2-D matrix with at least two features.")
    if not np.isfinite(values).all():
        raise ValueError("Classifier input contains non-finite values.")
    norms = np.linalg.norm(values, axis=1)
    if np.any(norms <= np.finfo(np.float64).eps):
        raise ValueError("Classifier cannot normalize an all-zero vector.")
    return values / norms[:, None], norms


@dataclass(frozen=True)
class CoPLogisticClassifier:
    weights: np.ndarray
    intercept: float
    feature_mean: np.ndarray
    threshold: float
    input_mode: str
    representation_id: str
    comparison_signature: dict[str, Any]
    metadata: dict[str, Any]

    def predict_failure_probability(
        self,
        vector: Any,
        *,
        input_mode: str | None = None,
        comparison_signature: dict[str, Any] | None = None,
    ) -> float:
        """Return the logistic estimate that one frozen vector is a failure."""
        values = np.asarray(vector, dtype=np.float64)
        if values.ndim != 1 or values.size != self.weights.size:
            raise ValueError(
                f"Classifier expects one {self.weights.size}-value vector; got {values.shape}."
            )
        if not np.isfinite(values).all():
            raise ValueError("Classifier vector contains non-finite values.")
        norm = float(np.linalg.norm(values))
        if norm <= np.finfo(np.float64).eps:
            raise ValueError("Classifier cannot score an all-zero vector.")
        if input_mode is not None and input_mode != self.input_mode:
            raise ValueError(
                f"Classifier was trained for input mode {self.input_mode!r}, "
                f"not {input_mode!r}."
            )
        if comparison_signature is not None and _canonical_json(
            comparison_signature
        ) != _canonical_json(self.comparison_signature):
            raise ValueError(
                "Classifier and vector use different RynnBrain representation settings. "
                "Use the same model, prompt, reference bags, cameras, frames, and input mode."
            )
        centered = values / norm - self.feature_mean
        logit = float(centered @ self.weights + self.intercept)
        return float(_sigmoid(np.asarray([logit]))[0])

    def predict_label(self, probability: float) -> str:
        return "failure" if probability >= self.threshold else "success"


def fit_logistic_classifier(
    vectors: Any,
    targets: Any,
    *,
    c_value: float = 1.0,
    threshold: float = 0.5,
    class_weight: str = "balanced",
    max_iterations: int = 200,
    tolerance: float = 1e-8,
) -> tuple[np.ndarray, float, np.ndarray, dict[str, Any]]:
    """Fit L2-regularized binary logistic regression without changing RynnBrain."""
    if c_value <= 0:
        raise ValueError("c_value must be greater than zero.")
    if not 0 < threshold < 1:
        raise ValueError("threshold must be strictly between zero and one.")
    if class_weight not in {"balanced", "none"}:
        raise ValueError("class_weight must be 'balanced' or 'none'.")
    normalized, _ = _normalize_rows(np.asarray(vectors))
    labels = np.asarray(targets, dtype=np.float64)
    if labels.ndim != 1 or labels.size != normalized.shape[0]:
        raise ValueError("targets must contain one value per vector.")
    if set(np.unique(labels)) != {0.0, 1.0}:
        raise ValueError("Training requires both normal (0) and failure (1) targets.")

    class_counts = {target: int(np.sum(labels == target)) for target in (0.0, 1.0)}
    if class_weight == "balanced":
        class_weights = {
            target: labels.size / (2.0 * count) for target, count in class_counts.items()
        }
        sample_weights = np.asarray([class_weights[target] for target in labels])
    else:
        class_weights = {0.0: 1.0, 1.0: 1.0}
        sample_weights = np.ones(labels.size, dtype=np.float64)
    total_weight = float(sample_weights.sum())

    feature_mean = normalized.mean(axis=0)
    centered = normalized - feature_mean
    _, singular_values, right_vectors = np.linalg.svd(centered, full_matrices=False)
    if not singular_values.size or singular_values[0] <= np.finfo(np.float64).eps:
        raise ValueError("Classifier cannot learn from identical vectors.")
    rank_tolerance = max(centered.shape) * np.finfo(np.float64).eps * singular_values[0]
    rank = int(np.sum(singular_values > rank_tolerance))
    basis = right_vectors[:rank].T
    reduced = centered @ basis
    design = np.column_stack([reduced, np.ones(reduced.shape[0])])

    parameters = np.zeros(rank + 1, dtype=np.float64)
    prevalence = float(
        np.clip(np.average(labels, weights=sample_weights), 1e-6, 1 - 1e-6)
    )
    parameters[-1] = np.log(prevalence / (1.0 - prevalence))
    regularization = 1.0 / (c_value * total_weight)
    regularizer = np.zeros(rank + 1, dtype=np.float64)
    regularizer[:-1] = regularization

    def objective(candidate: np.ndarray) -> float:
        logits = design @ candidate
        losses = np.logaddexp(0.0, logits) - labels * logits
        loss = float(sample_weights @ losses / total_weight)
        penalty = 0.5 * regularization * float(candidate[:-1] @ candidate[:-1])
        return float(loss + penalty)

    converged = False
    gradient_norm = float("inf")
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        logits = design @ parameters
        probabilities = _sigmoid(logits)
        gradient = design.T @ (sample_weights * (probabilities - labels)) / total_weight
        gradient += regularizer * parameters
        gradient_norm = float(np.max(np.abs(gradient)))
        if gradient_norm <= tolerance:
            converged = True
            break
        curvature = sample_weights * probabilities * (1.0 - probabilities)
        hessian = design.T @ (curvature[:, None] * design) / total_weight
        hessian.flat[:: hessian.shape[0] + 1] += regularizer + 1e-10
        try:
            step_direction = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step_direction = np.linalg.lstsq(hessian, gradient, rcond=None)[0]

        current_objective = objective(parameters)
        step_size = 1.0
        while step_size >= 1e-8:
            candidate = parameters - step_size * step_direction
            if objective(candidate) < current_objective:
                parameters = candidate
                break
            step_size *= 0.5
        else:
            break

    weights = basis @ parameters[:-1]
    diagnostics = {
        "converged": converged,
        "iterations": iterations,
        "max_gradient": gradient_norm,
        "objective": objective(parameters),
        "effective_rank": rank,
        "regularization_c": float(c_value),
        "regularization_strength": float(regularization),
        "class_weight": class_weight,
        "class_weights": {
            "normal": float(class_weights[0.0]),
            "fail": float(class_weights[1.0]),
        },
        "threshold": float(threshold),
    }
    return weights, float(parameters[-1]), feature_mean, diagnostics


def _classification_metrics(
    targets: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, float]:
    predictions = (probabilities >= threshold).astype(np.int64)
    normal_mask = targets == 0
    failure_mask = targets == 1
    normal_recall = float(np.mean(predictions[normal_mask] == 0))
    failure_recall = float(np.mean(predictions[failure_mask] == 1))
    clipped = np.clip(probabilities, 1e-12, 1 - 1e-12)
    log_loss = -float(np.mean(targets * np.log(clipped) + (1 - targets) * np.log(1 - clipped)))

    order = np.argsort(probabilities, kind="stable")
    ranks = np.empty(probabilities.size, dtype=np.float64)
    start = 0
    while start < order.size:
        end = start + 1
        while end < order.size and probabilities[order[end]] == probabilities[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    positive_count = int(failure_mask.sum())
    negative_count = int(normal_mask.sum())
    auc = (
        float(ranks[failure_mask].sum()) - positive_count * (positive_count + 1) / 2
    ) / (positive_count * negative_count)
    return {
        "accuracy": float(np.mean(predictions == targets)),
        "balanced_accuracy": (normal_recall + failure_recall) / 2.0,
        "normal_recall": normal_recall,
        "failure_recall": failure_recall,
        "log_loss": log_loss,
        "roc_auc": float(auc),
    }


def _cross_validated_probabilities(
    vectors: np.ndarray,
    targets: np.ndarray,
    c_value: float,
    class_weight: str,
) -> tuple[np.ndarray, int]:
    class_counts = [int(np.sum(targets == target)) for target in (0, 1)]
    fold_count = min(5, min(class_counts))
    if fold_count < 2:
        raise ValueError("Cross-validation requires at least two bags in each class.")
    rng = np.random.default_rng(0)
    folds: list[list[int]] = [[] for _ in range(fold_count)]
    for target in (0, 1):
        indices = np.flatnonzero(targets == target)
        rng.shuffle(indices)
        for offset, index in enumerate(indices):
            folds[offset % fold_count].append(int(index))

    probabilities = np.full(targets.size, np.nan, dtype=np.float64)
    all_indices = np.arange(targets.size)
    for validation_indices in folds:
        validation = np.asarray(sorted(validation_indices), dtype=np.int64)
        training = np.setdiff1d(all_indices, validation, assume_unique=True)
        weights, intercept, mean, _ = fit_logistic_classifier(
            vectors[training],
            targets[training],
            c_value=c_value,
            class_weight=class_weight,
        )
        normalized, _ = _normalize_rows(vectors[validation])
        probabilities[validation] = _sigmoid((normalized - mean) @ weights + intercept)
    return probabilities, fold_count


def save_classifier(
    output_file: Path,
    classifier: CoPLogisticClassifier,
) -> tuple[Path, Path]:
    model_path = output_file.resolve()
    if model_path.suffix.lower() != ".npz":
        raise ValueError("Classifier output file must use the .npz extension.")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = model_path.with_suffix(".json")
    np.savez_compressed(
        model_path,
        weights=np.asarray(classifier.weights, dtype=np.float32),
        intercept=np.asarray(classifier.intercept, dtype=np.float64),
        feature_mean=np.asarray(classifier.feature_mean, dtype=np.float32),
        threshold=np.asarray(classifier.threshold, dtype=np.float64),
    )
    metadata_path.write_text(
        json.dumps(classifier.metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return model_path, metadata_path


@lru_cache(maxsize=16)
def load_classifier(model_file: Path):
    model_path = model_file.resolve()
    metadata_path = model_path.with_suffix(".json")
    if not model_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            f"Classifier requires both {model_path} and {metadata_path}."
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != CLASSIFIER_SCHEMA_VERSION:
        raise ValueError(f"Unsupported classifier schema in {metadata_path}.")
    from .cop_knn import KNN_TYPE, load_knn
    if metadata.get("classifier_type") == KNN_TYPE:
        return load_knn(model_path, metadata)
    with np.load(model_path, allow_pickle=False) as archive:
        weights = np.asarray(archive["weights"], dtype=np.float64)
        feature_mean = np.asarray(archive["feature_mean"], dtype=np.float64)
        intercept = float(archive["intercept"])
        threshold = float(archive["threshold"])
    if weights.ndim != 1 or feature_mean.shape != weights.shape:
        raise ValueError(f"Invalid classifier arrays in {model_path}.")
    if (
        not np.isfinite(weights).all()
        or not np.isfinite(feature_mean).all()
        or not np.isfinite(intercept)
        or not np.isfinite(threshold)
    ):
        raise ValueError(f"Classifier contains non-finite values: {model_path}.")
    if not 0 < threshold < 1:
        raise ValueError(f"Classifier threshold must be between zero and one: {model_path}.")
    if int(metadata.get("vector_dimension", -1)) != weights.size:
        raise ValueError(f"Classifier metadata dimension does not match {model_path}.")
    return CoPLogisticClassifier(
        weights=weights,
        intercept=intercept,
        feature_mean=feature_mean,
        threshold=threshold,
        input_mode=str(metadata["input_mode"]),
        representation_id=str(metadata["representation_id"]),
        comparison_signature=dict(metadata["comparison_signature"]),
        metadata=metadata,
    )


def _normalize_bag_selector(value: Any) -> str:
    return str(value).strip().replace("\\", "/").rstrip("/")


def _training_record_keys(record: SavedCoPVector) -> set[str]:
    """Return accepted short, /data, and data-relative identifiers for a vector."""
    bag_name = _normalize_bag_selector(record.metadata.get("bag_name", ""))
    test_bag = _normalize_bag_selector(record.metadata.get("test_bag", ""))
    keys = {value for value in (bag_name, test_bag) if value}
    keys.update(_normalize_bag_selector(value) for value in record.metadata.get("selection_aliases", []))
    if test_bag.startswith("/data/"):
        keys.add(test_bag[len("/data/"):])
    return keys


def _select_training_records(
    records: list[SavedCoPVector], input_mode: str, bag_names: list[str]
) -> list[SavedCoPVector]:
    mode_records = [
        record
        for record in records
        if str(record.metadata.get("input_mode")) == input_mode
    ]
    if bag_names:
        selectors = [_normalize_bag_selector(value) for value in bag_names]
        if len(selectors) != len(set(selectors)):
            raise ValueError("Each training bag name may be supplied only once.")
        by_name: dict[str, list[SavedCoPVector]] = {}
        for record in mode_records:
            for key in _training_record_keys(record):
                by_name.setdefault(key, []).append(record)
        missing = [name for name in selectors if name not in by_name]
        if missing:
            raise ValueError("Labeled vector(s) not found for: " + ", ".join(missing))
        ambiguous = [name for name in selectors if len(by_name[name]) > 1]
        if ambiguous:
            raise ValueError(
                "Multiple training vectors found for bag(s): " + ", ".join(ambiguous)
            )
        usable = [by_name[name][0] for name in selectors]
    else:
        usable = [
            record
            for record in mode_records
            if str(record.metadata.get("ground_truth_label", "")).strip().lower()
            in LABEL_TO_TARGET
        ]
    names = [str(record.metadata.get("bag_name")) for record in usable]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError("Duplicate training bag vectors found for: " + ", ".join(duplicates))
    return usable


def train_from_saved_vectors(
    input_directory: Path,
    output_file: Path,
    *,
    input_mode: str = "raw",
    bag_names: list[str] | None = None,
    label_overrides: dict[str, str] | None = None,
    c_value: float = 1.0,
    threshold: float = 0.5,
    class_weight: str = "balanced",
    metadata_paths: list[Path] | None = None,
    method: str = "logistic",
    knn_options: dict[str, Any] | None = None,
    include_timeline: bool = False,
) -> dict[str, Any]:
    classifier_method({"method": method})
    if method == "knn" and not bag_names:
        raise ValueError("Nominal kNN requires an explicit normal_bags selection; never train on all benchmark vectors.")
    records = _select_training_records(
        discover_saved_vectors(input_directory, metadata_paths), input_mode, bag_names or []
    )
    if not records:
        raise ValueError(f"No labeled {input_mode!r} vectors were found.")
    overrides = {
        _normalize_bag_selector(key): value
        for key, value in (label_overrides or {}).items()
    }
    selected_keys = set().union(*(_training_record_keys(record) for record in records))
    unused_overrides = sorted(set(overrides) - selected_keys)
    if unused_overrides:
        raise ValueError(
            "Configured training labels have no selected vector for: "
            + ", ".join(unused_overrides)
        )
    training_labels: list[str] = []
    for record in records:
        override_labels = {
            str(overrides[key]).strip().lower()
            for key in _training_record_keys(record)
            if key in overrides
        }
        if len(override_labels) > 1:
            raise ValueError(
                f"Conflicting configured labels for {record.metadata_path}."
            )
        label = (
            next(iter(override_labels))
            if override_labels
            else str(record.metadata.get("ground_truth_label", "")).strip().lower()
        )
        training_labels.append(label)
    invalid_labels = sorted({label for label in training_labels if label not in LABEL_TO_TARGET})
    if invalid_labels:
        raise ValueError(
            "Training vectors need explicit normal/fail labels; invalid label(s): "
            + ", ".join(invalid_labels)
        )
    targets = np.asarray(
        [LABEL_TO_TARGET[label] for label in training_labels],
        dtype=np.int64,
    )
    counts = {"normal": int(np.sum(targets == 0)), "fail": int(np.sum(targets == 1))}
    if method == "knn" and counts["fail"]:
        raise ValueError("Nominal kNN cannot use failure training bags.")
    if method == "logistic" and min(counts.values()) < 2:
        raise ValueError("Training requires at least two normal and two failure bags.")
    dimensions = {record.vector.size for record in records}
    if len(dimensions) != 1:
        raise ValueError(f"Training vectors have inconsistent dimensions: {sorted(dimensions)}")
    signatures = {
        _canonical_json(record.metadata.get("comparison_signature")) for record in records
    }
    if len(signatures) != 1 or "null" in signatures:
        raise ValueError("Training vectors use different or missing comparison settings.")
    representation_ids = {str(record.metadata.get("representation_id") or "") for record in records}
    if len(representation_ids) != 1 or not next(iter(representation_ids)):
        raise ValueError("Training vectors use different or missing representation definitions.")

    matrix = np.stack([record.vector for record in records])
    if method == "knn":
        from .cop_knn import train_knn
        metadata = train_knn(records, output_file, **(knn_options or {}))
        if include_timeline:
            from .cop_timeline import train_prefix_detector
            train_prefix_detector(records, output_file, metadata)
        load_classifier.cache_clear()
        return metadata
    cv_probabilities, fold_count = _cross_validated_probabilities(
        matrix, targets, c_value, class_weight
    )
    cv_metrics = _classification_metrics(targets, cv_probabilities, threshold)
    weights, intercept, feature_mean, diagnostics = fit_logistic_classifier(
        matrix,
        targets,
        c_value=c_value,
        threshold=threshold,
        class_weight=class_weight,
    )
    signature = dict(records[0].metadata["comparison_signature"])
    model_path = output_file.resolve()
    predictions_path = model_path.with_name(f"{model_path.stem}_training_predictions.csv")
    rows = [
        {
            "bag_name": str(record.metadata.get("bag_name")),
            "bag_path": str(record.metadata.get("test_bag", "")),
            "ground_truth_label": "fail" if target else "normal",
            "out_of_fold_failure_probability": float(probability),
            "out_of_fold_decision": "failure" if probability >= threshold else "success",
            "vector_path": str(record.vector_path),
            "metadata_path": str(record.metadata_path),
        }
        for record, target, probability in zip(records, targets, cv_probabilities)
    ]
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(predictions_path, index=False)
    metadata: dict[str, Any] = {
        "schema_version": CLASSIFIER_SCHEMA_VERSION,
        "classifier_type": "l2_regularized_binary_logistic_regression",
        "feature_preprocessing": "row_l2_normalize_then_subtract_training_feature_mean",
        "input_mode": input_mode,
        "representation_id": next(iter(representation_ids)),
        "comparison_signature": signature,
        "vector_dimension": int(matrix.shape[1]),
        "training_bag_count": len(records),
        "training_class_counts": counts,
        "class_weight": class_weight,
        "training_bags": [row["bag_name"] for row in rows],
        "training_labels": {
            row["bag_name"]: row["ground_truth_label"] for row in rows
        },
        "failure_probability_note": (
            "Logistic estimate from the frozen hidden vector; calibration depends on "
            "training-set size and representativeness. Balanced class weights use an "
            "equal-class training prior rather than the observed real-world failure rate."
            if class_weight == "balanced"
            else "Logistic estimate from the frozen hidden vector; calibration depends "
            "on training-set size, class prevalence, and representativeness."
        ),
        "cross_validation": {
            "method": "deterministic_stratified_k_fold",
            "fold_count": fold_count,
            "metrics": cv_metrics,
            "predictions_csv": str(predictions_path),
        },
        "fit_diagnostics": diagnostics,
        "threshold": float(threshold),
        "model_file": str(model_path),
        "metadata_file": str(model_path.with_suffix(".json")),
    }
    classifier = CoPLogisticClassifier(
        weights=weights,
        intercept=intercept,
        feature_mean=feature_mean,
        threshold=threshold,
        input_mode=input_mode,
        representation_id=next(iter(representation_ids)),
        comparison_signature=signature,
        metadata=metadata,
    )
    save_classifier(model_path, classifier)
    load_classifier.cache_clear()
    return metadata


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        help="Read defaults from rynnbrain.cop_classifier.training in this JSON file.",
    )
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-file", type=Path)
    parser.add_argument("--input-mode")
    parser.add_argument("--bag", action="append", default=None, metavar="BAG_NAME")
    parser.add_argument("--normal-bag", action="append", default=None, metavar="BAG_NAME")
    parser.add_argument("--failure-bag", action="append", default=None, metavar="BAG_NAME")
    parser.add_argument("--regularization-c", type=float)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--class-weight", choices=("balanced", "none"))
    parser.add_argument("--saved-vectors-only", action="store_true", help="Train from saved vectors without loading the VLM or extracting bags.")
    parser.add_argument("--refresh-vectors", action="store_true", help="Re-extract selected training bags, even if their settings match (e.g. bag contents changed).")
    parser.add_argument("--data-root", type=Path, default=Path("/data"))
    return parser.parse_args()


def _resolved_training_settings(arguments: argparse.Namespace) -> dict[str, Any]:
    classifier_config: dict[str, Any] = {}
    training_config: dict[str, Any] = {}
    if arguments.config is not None:
        config = json.loads(arguments.config.read_text(encoding="utf-8"))
        classifier_config = dict(config.get("rynnbrain", {}).get("cop_classifier", {}))
        training_config = dict(classifier_config.get("training", {}))

    input_mode = arguments.input_mode or training_config.get("input_mode", "raw")
    input_directory = arguments.input_dir or training_config.get("input_dir")
    method = classifier_method(classifier_config)
    model_paths = classifier_model_paths(classifier_config)
    configured_model = model_paths.get(input_mode) if isinstance(model_paths, dict) else None
    output_file = (
        arguments.output_file
        or training_config.get("output_file")
        or configured_model
    )
    if input_directory is None:
        raise ValueError(
            "Set --input-dir or rynnbrain.cop_classifier.training.input_dir."
        )
    if output_file is None:
        raise ValueError(
            "Set --output-file, training.output_file, or model_paths for the input mode."
        )
    output_file = _method_model_path(output_file, method)
    cli_selection_supplied = any(
        value is not None
        for value in (arguments.bag, arguments.normal_bag, arguments.failure_bag)
    )
    if cli_selection_supplied:
        generic_bags = list(arguments.bag or [])
        normal_bags = list(arguments.normal_bag or [])
        failure_bags = list(arguments.failure_bag or [])
    else:
        generic_bags = list(training_config.get("bags", []))
        normal_bags = list(training_config.get("normal_bags", []))
        failure_bags = list(training_config.get("failure_bags", []))
    if method == "knn":
        if cli_selection_supplied and (generic_bags or failure_bags):
            raise ValueError("Nominal kNN accepts only --normal-bag selections.")
        generic_bags, failure_bags = [], []
        if not normal_bags:
            raise ValueError("Nominal kNN requires an explicit training.normal_bags selection.")
        from .cop_knn import validate_knn_settings
        validate_knn_settings(len(normal_bags), **classifier_config.get("knn", {}))
    selected_bags = generic_bags + normal_bags + failure_bags
    if (
        not selected_bags
        and arguments.config is not None
        and not bool(training_config.get("allow_all_labeled", False))
    ):
        raise ValueError(
            "No classifier training bags are configured. Fill training.normal_bags "
            "and training.failure_bags, or explicitly set training.allow_all_labeled=true."
        )
    label_overrides = {
        **{name: "normal" for name in normal_bags},
        **{name: "fail" for name in failure_bags},
    }

    return {
        "method": method,
        "knn_options": dict(classifier_config.get("knn", {})),
        "include_timeline": method == "knn" and bool(classifier_config.get("plot_timeline", True)),
        "input_directory": Path(str(input_directory)),
        "output_file": Path(str(output_file)),
        "input_mode": str(input_mode),
        "bag_names": selected_bags,
        "label_overrides": label_overrides,
        "c_value": float(
            arguments.regularization_c
            if arguments.regularization_c is not None
            else training_config.get("regularization_c", 1.0)
        ),
        "threshold": float(
            arguments.threshold
            if arguments.threshold is not None
            else training_config.get("threshold", 0.5)
        ),
        "class_weight": str(
            arguments.class_weight
            or training_config.get("class_weight", "balanced")
        ),
    }


def configured_training_settings(config_path: Path, input_mode: str | None = None) -> dict[str, Any]:
    """Use the same config resolution for standalone training and benchmarks."""
    return _resolved_training_settings(argparse.Namespace(
        config=config_path, input_mode=input_mode, input_dir=None, output_file=None,
        bag=None, normal_bag=None, failure_bag=None,
        regularization_c=None, threshold=None, class_weight=None,
    ))


def prepare_training_vectors(
    config: dict[str, Any], settings: dict[str, Any], *, model=None,
    data_root: Path = Path("/data"), refresh: bool = False,
) -> tuple[list[Path], str]:
    """Reuse compatible selected vectors; extract only missing/stale training bags.

    Never scores a benchmark or changes its summary. A reference generation
    verifies the full signature and is shared with any new extractions.
    """
    from .model import RynnBrainModel
    from .prompts import task_context_prompt
    from .run import (
        _raw_inputs, cop_comparison_signature, evaluate_multiturn,
        multiturn_outputs_match, write_multiturn_outputs,
    )

    vlm = dict(config["rynnbrain"])
    generation = dict(vlm.get("generation", {}))
    if vlm.get("raw_video_paths") or vlm.get("heatmap_video_paths"):
        raise ValueError("Training vector extraction uses per-bag inputs; remove global video-path overrides or use --saved-vectors-only.")
    if generation.get("do_sample", False):
        raise ValueError("Classifier vector preparation requires generation.do_sample=false for a consistent reference.")
    selectors = [_normalize_bag_selector(value) for value in settings["bag_names"]]
    names = [value.rsplit("/", 1)[-1] for value in selectors]
    if not selectors or len(set(names)) != len(names):
        raise ValueError("Vector preparation needs explicit training bags with unique bag names; use --saved-vectors-only for legacy all-vector training.")
    reference_bags = list(vlm.get("reference_bags", []))
    if not reference_bags or set(names) & {Path(value).name for value in reference_bags}:
        raise ValueError("Select reference_bags separately from classifier training bags.")
    root, mode = Path(settings["input_directory"]), settings["input_mode"]
    overrides = {_normalize_bag_selector(key): value for key, value in settings["label_overrides"].items()}
    candidates = []
    for path in sorted(root.glob("**/cop_vectors/*.json")):
        metadata = json.loads(path.read_text(encoding="utf-8"))
        if metadata.get("input_mode") == mode:
            candidates.append(SavedCoPVector(path.parent / Path(metadata.get("vector_file", "")).name,
                                             path, np.empty(0), metadata))
    selected = []
    labels = []
    for selector, name in zip(selectors, names):
        matches = [record for record in candidates if selector in _training_record_keys(record)]
        if len(matches) > 1:
            raise ValueError(f"Multiple training vectors found for {selector}; keep one vector per bag and mode in training.input_dir.")
        record = matches[0] if matches else None
        keys = _training_record_keys(record) if record else {selector}
        explicit = {overrides[key] for key in keys if key in overrides}
        if len(explicit) > 1:
            raise ValueError(f"Conflicting training labels for {selector}.")
        label = next(iter(explicit)) if explicit else (record.metadata.get("ground_truth_label") if record else None)
        if label not in LABEL_TO_TARGET:
            raise ValueError(f"Set an explicit normal_bags/failure_bags label for {selector} before extracting its vector.")
        labels.append(LABEL_TO_TARGET[label])
        selected.append((selector, name, record, label))
    if settings.get("method", "logistic") == "knn":
        if any(labels) or len(labels) < 2:
            raise ValueError("Nominal kNN preparation requires only nominal training bags (at least two).")
        from .cop_knn import validate_knn_settings
        validate_knn_settings(len(labels), **settings.get("knn_options", {}))
    elif min(labels.count(0), labels.count(1)) < 2:
        raise ValueError("Training requires at least two normal and two failure bags.")

    if model is None:
        model = RynnBrainModel(vlm["model"])
    count = int(vlm.get("num_frames", 4))
    topics = vlm.get("memory_camera_topics", vlm.get("camera_topics", config.get("camera_topics", [])))
    nominal_images = []
    for bag in reference_bags:
        images, _ = _raw_inputs(Path(bag), list(topics), count)
        nominal_images.extend(images)
    if not nominal_images:
        raise ValueError("No nominal reference images were loaded.")
    reference_response = model.generate_nominal({
        "role": "user", "images": nominal_images,
        "text": task_context_prompt(vlm.get("task_description", "Robot manipulation task")),
    }, generation)
    expected = cop_comparison_signature(model, config, vlm, mode, reference_response, count, generation)
    paths = []
    for selector, name, record, label in selected:
        if not refresh and record is not None and record.metadata.get("comparison_signature") == expected:
            try:
                loaded = discover_saved_vectors(root, [record.metadata_path])[0]
                if loaded.vector.size != expected["hidden_size"] or loaded.metadata.get("representation_id") != expected["representation_id"]:
                    raise ValueError("Vector representation changed")
                if not multiturn_outputs_match(record.metadata_path.parent.parent, loaded.metadata):
                    raise ValueError("Full response files are missing or do not match this vector")
                paths.append(record.metadata_path)
                print(f"Reusing classifier training vector: {name} ({mode})")
                continue
            except (ValueError, OSError, EOFError):
                pass  # Regenerate a missing/corrupt vector or incomplete response files.
        bag_path = Path(selector)
        if not bag_path.is_absolute():
            bag_path = data_root / bag_path
        if not (bag_path / "metadata.yaml").is_file() and "/" not in selector:
            matches = [path.parent for path in data_root.glob("**/metadata.yaml") if path.parent.name == name]
            if len(matches) != 1:
                raise ValueError(f"Cannot uniquely locate ROS bag {selector}; use its full path in the training list.")
            bag_path = matches[0]
        if not (bag_path / "metadata.yaml").is_file():
            raise FileNotFoundError(f"Training bag not found: {bag_path}")
        destination = record.metadata_path.parent.parent if record else root / name / "rynnbrain_multiturn"
        bag_config = {**config, "test_bag": str(bag_path), "output_dir": str(root / name)}
        bag_vlm = {**vlm, "input_modes": [mode], "ground_truth_label": label}
        print(f"Extracting classifier training vector: {name} ({mode})")
        rows, frames, responses, task = evaluate_multiturn(
            model, bag_config, bag_vlm, count, generation, destination,
            training_vectors_only=True, reference_response=reference_response,
        )
        write_multiturn_outputs(destination, rows, frames, responses, task, merge_modes=True)
        path = Path(rows[0]["cop_metadata_path"])
        loaded = discover_saved_vectors(root, [path])[0]
        if loaded.metadata.get("comparison_signature") != expected:
            raise ValueError(f"Extracted vector settings changed unexpectedly for {name}.")
        if selector not in _training_record_keys(loaded):
            loaded.metadata["selection_aliases"] = [selector]
            path.write_text(json.dumps(loaded.metadata, indent=2) + "\n", encoding="utf-8")
        paths.append(path)
    if settings.get("include_timeline", False):
        from .cop_timeline import prepare_nominal_prefixes
        prepare_nominal_prefixes(model, config, vlm, discover_saved_vectors(root, paths),
                                 nominal_images, reference_response, generation, refresh=refresh)
    return paths, reference_response


def main() -> None:
    arguments = parse_arguments()
    settings = _resolved_training_settings(arguments)
    if arguments.refresh_vectors and arguments.saved_vectors_only:
        raise ValueError("--refresh-vectors cannot be combined with --saved-vectors-only.")
    if arguments.refresh_vectors and arguments.config is None:
        raise ValueError("--refresh-vectors needs --config to extract using the current model settings.")
    if arguments.config is not None and not arguments.saved_vectors_only:
        config = json.loads(arguments.config.read_text(encoding="utf-8"))
        paths, _ = prepare_training_vectors(
            config, settings, data_root=arguments.data_root, refresh=arguments.refresh_vectors
        )
        settings["metadata_paths"] = paths
    metadata = train_from_saved_vectors(**settings)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
