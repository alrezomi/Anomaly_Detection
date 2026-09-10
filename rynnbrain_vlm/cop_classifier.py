"""Train and apply a logistic classifier to frozen RynnBrain-CoP vectors."""

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
    max_iterations: int = 200,
    tolerance: float = 1e-8,
) -> tuple[np.ndarray, float, np.ndarray, dict[str, Any]]:
    """Fit L2-regularized binary logistic regression without changing RynnBrain."""
    if c_value <= 0:
        raise ValueError("c_value must be greater than zero.")
    if not 0 < threshold < 1:
        raise ValueError("threshold must be strictly between zero and one.")
    normalized, _ = _normalize_rows(np.asarray(vectors))
    labels = np.asarray(targets, dtype=np.float64)
    if labels.ndim != 1 or labels.size != normalized.shape[0]:
        raise ValueError("targets must contain one value per vector.")
    if set(np.unique(labels)) != {0.0, 1.0}:
        raise ValueError("Training requires both normal (0) and failure (1) targets.")

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
    prevalence = float(np.clip(labels.mean(), 1e-6, 1 - 1e-6))
    parameters[-1] = np.log(prevalence / (1.0 - prevalence))
    regularization = 1.0 / (c_value * labels.size)
    regularizer = np.zeros(rank + 1, dtype=np.float64)
    regularizer[:-1] = regularization

    def objective(candidate: np.ndarray) -> float:
        logits = design @ candidate
        loss = np.mean(np.logaddexp(0.0, logits) - labels * logits)
        penalty = 0.5 * regularization * float(candidate[:-1] @ candidate[:-1])
        return float(loss + penalty)

    converged = False
    gradient_norm = float("inf")
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        logits = design @ parameters
        probabilities = _sigmoid(logits)
        gradient = design.T @ (probabilities - labels) / labels.size
        gradient += regularizer * parameters
        gradient_norm = float(np.max(np.abs(gradient)))
        if gradient_norm <= tolerance:
            converged = True
            break
        curvature = probabilities * (1.0 - probabilities)
        hessian = design.T @ (curvature[:, None] * design) / labels.size
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
            vectors[training], targets[training], c_value=c_value
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
def load_classifier(model_file: Path) -> CoPLogisticClassifier:
    model_path = model_file.resolve()
    metadata_path = model_path.with_suffix(".json")
    if not model_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            f"Classifier requires both {model_path} and {metadata_path}."
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != CLASSIFIER_SCHEMA_VERSION:
        raise ValueError(f"Unsupported classifier schema in {metadata_path}.")
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


def _select_training_records(
    records: list[SavedCoPVector], input_mode: str, bag_names: list[str]
) -> list[SavedCoPVector]:
    usable = [
        record
        for record in records
        if str(record.metadata.get("input_mode")) == input_mode
        and str(record.metadata.get("ground_truth_label", "")).lower() in LABEL_TO_TARGET
    ]
    if bag_names:
        if len(bag_names) != len(set(bag_names)):
            raise ValueError("Each training --bag name may be supplied only once.")
        by_name: dict[str, list[SavedCoPVector]] = {}
        for record in usable:
            by_name.setdefault(str(record.metadata.get("bag_name")), []).append(record)
        missing = [name for name in bag_names if name not in by_name]
        if missing:
            raise ValueError("Labeled vector(s) not found for: " + ", ".join(missing))
        ambiguous = [name for name in bag_names if len(by_name[name]) > 1]
        if ambiguous:
            raise ValueError(
                "Multiple training vectors found for bag(s): " + ", ".join(ambiguous)
            )
        usable = [by_name[name][0] for name in bag_names]
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
    c_value: float = 1.0,
    threshold: float = 0.5,
) -> dict[str, Any]:
    records = _select_training_records(
        discover_saved_vectors(input_directory), input_mode, bag_names or []
    )
    if not records:
        raise ValueError(f"No labeled {input_mode!r} vectors were found.")
    targets = np.asarray(
        [LABEL_TO_TARGET[str(record.metadata["ground_truth_label"]).lower()] for record in records],
        dtype=np.int64,
    )
    counts = {"normal": int(np.sum(targets == 0)), "fail": int(np.sum(targets == 1))}
    if min(counts.values()) < 2:
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
    cv_probabilities, fold_count = _cross_validated_probabilities(matrix, targets, c_value)
    cv_metrics = _classification_metrics(targets, cv_probabilities, threshold)
    weights, intercept, feature_mean, diagnostics = fit_logistic_classifier(
        matrix, targets, c_value=c_value, threshold=threshold
    )
    signature = dict(records[0].metadata["comparison_signature"])
    model_path = output_file.resolve()
    predictions_path = model_path.with_name(f"{model_path.stem}_training_predictions.csv")
    rows = [
        {
            "bag_name": str(record.metadata.get("bag_name")),
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
        "training_bags": [row["bag_name"] for row in rows],
        "failure_probability_note": (
            "Logistic estimate from the frozen hidden vector; calibration depends on "
            "training-set size, balance, and representativeness."
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
    return metadata


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--input-mode", default="raw")
    parser.add_argument("--bag", action="append", default=[], metavar="BAG_NAME")
    parser.add_argument("--regularization-c", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    metadata = train_from_saved_vectors(
        arguments.input_dir,
        arguments.output_file,
        input_mode=arguments.input_mode,
        bag_names=arguments.bag,
        c_value=arguments.regularization_c,
        threshold=arguments.threshold,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
