"""Nominal-only nearest-neighbour detection on full, L2-normalized CoP vectors."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


KNN_TYPE = "nominal_knn_distance"


def _unit_rows(vectors: Any) -> np.ndarray:
    values = np.asarray(vectors, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2 or not np.isfinite(values).all():
        raise ValueError("kNN expects finite vectors with at least two features.")
    norms = np.linalg.norm(values, axis=1)
    if np.any(norms <= np.finfo(float).eps):
        raise ValueError("kNN cannot normalize an all-zero vector.")
    return values / norms[:, None]


def _distance(query: np.ndarray, memory: np.ndarray, k: int) -> float:
    distances = np.linalg.norm(memory - query, axis=1)
    return float(np.partition(distances, k - 1)[:k].mean())


def validate_knn_settings(count: int, *, n_neighbors: int = 3, threshold_quantile: float = 0.95) -> None:
    if isinstance(n_neighbors, bool) or not isinstance(n_neighbors, (int, np.integer)) or not 1 <= n_neighbors < count:
        raise ValueError("kNN n_neighbors must be an integer >= 1 and smaller than the number of nominal training bags.")
    if isinstance(threshold_quantile, bool) or not isinstance(threshold_quantile, (int, float)) or not np.isfinite(threshold_quantile) or not 0 < threshold_quantile < 1:
        raise ValueError("kNN threshold_quantile must be strictly between zero and one.")


@dataclass(frozen=True)
class CoPKNNDetector:
    nominal_vectors: np.ndarray
    n_neighbors: int
    threshold: float
    input_mode: str
    representation_id: str
    comparison_signature: dict[str, Any]
    metadata: dict[str, Any]

    def predict_anomaly_score(self, vector: Any, *, input_mode=None, comparison_signature=None) -> float:
        values = np.asarray(vector, dtype=np.float64)
        if values.ndim != 1 or values.size != self.nominal_vectors.shape[1]:
            raise ValueError(f"kNN expects one {self.nominal_vectors.shape[1]}-value vector; got {values.shape}.")
        if input_mode is not None and input_mode != self.input_mode:
            raise ValueError(f"kNN was trained for input mode {self.input_mode!r}, not {input_mode!r}.")
        if comparison_signature is not None and json.dumps(comparison_signature, sort_keys=True) != json.dumps(self.comparison_signature, sort_keys=True):
            raise ValueError("Classifier and vector use different RynnBrain representation settings; use matching model, adapter, prompts and frames.")
        query = _unit_rows(values[None, :])[0]
        return _distance(query, self.nominal_vectors, self.n_neighbors)

    def predict_label(self, score: float) -> str:
        return "failure" if score >= self.threshold else "success"


def train_knn(records, output_file: Path, *, n_neighbors: int = 3, threshold_quantile: float = 0.95) -> dict[str, Any]:
    """Fit a nominal memory and estimate a cutoff without using failure labels."""
    validate_knn_settings(len(records), n_neighbors=n_neighbors, threshold_quantile=threshold_quantile)
    n_neighbors = int(n_neighbors)
    memory = _unit_rows(np.stack([record.vector for record in records]))
    loo = np.asarray([_distance(row, np.delete(memory, index, axis=0), n_neighbors)
                      for index, row in enumerate(memory)])
    cutoff = float(np.quantile(loo, threshold_quantile, method="higher"))
    # Use the shared >= decision rule, but do not flag scores equal to the
    # nominal quantile (notably identical nominal vectors with distance zero).
    threshold = float(np.nextafter(cutoff, np.inf))
    path = output_file.resolve()
    if path.suffix.lower() != ".npz":
        raise ValueError("kNN output file must use the .npz extension.")
    if path.exists():
        sidecar = path.with_suffix(".json")
        if not sidecar.exists() or json.loads(sidecar.read_text(encoding="utf-8")).get("classifier_type") != KNN_TYPE:
            raise ValueError("Refusing to overwrite a different classifier with kNN; use a separate model file.")
    path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path = path.with_name(f"{path.stem}_training_predictions.csv")
    names = [str(record.metadata["bag_name"]) for record in records]
    metadata = {
        "schema_version": 1, "classifier_type": KNN_TYPE,
        "feature_preprocessing": "row_l2_normalize",
        "score_definition": "mean Euclidean distance to k nearest nominal vectors in full feature space",
        "score_note": "Anomaly distance, not a calibrated failure probability. Larger means farther from nominal examples.",
        "input_mode": records[0].metadata["input_mode"],
        "representation_id": records[0].metadata["representation_id"],
        "comparison_signature": records[0].metadata["comparison_signature"],
        "vector_dimension": int(memory.shape[1]), "n_neighbors": n_neighbors,
        "training_bag_count": len(names), "training_class_counts": {"normal": len(names), "fail": 0},
        "training_bags": names, "training_labels": dict.fromkeys(names, "normal"),
        "threshold": threshold,
        "threshold_estimation": {"method": "leave_one_bag_out_nominal_distance_quantile",
                                 "quantile": float(threshold_quantile), "nominal_quantile": cutoff,
                                 "note": "Training diagnostic cutoff; not a guaranteed false-alarm rate on unseen bags.",
                                 "predictions_csv": str(predictions_path)},
        "model_file": str(path), "metadata_file": str(path.with_suffix(".json")),
    }
    np.savez_compressed(path, nominal_vectors=memory, n_neighbors=np.asarray(n_neighbors), threshold=np.asarray(threshold))
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    pd.DataFrame([{"bag_name": name, "ground_truth_label": "normal", "leave_one_out_anomaly_score": float(score),
                   "decision_at_nominal_cutoff": "failure" if score >= threshold else "success"}
                  for name, score in zip(names, loo)]).to_csv(predictions_path, index=False)
    return metadata


def load_knn(path: Path, metadata: dict[str, Any]) -> CoPKNNDetector:
    with np.load(path, allow_pickle=False) as archive:
        memory = np.asarray(archive["nominal_vectors"], dtype=np.float64)
        k_value = float(archive["n_neighbors"])
        threshold = float(archive["threshold"])
    if memory.ndim != 2 or memory.shape[1] < 2 or memory.shape != (metadata.get("training_bag_count"), metadata.get("vector_dimension")):
        raise ValueError("Invalid kNN memory shape.")
    if not np.isfinite(memory).all() or not np.allclose(np.linalg.norm(memory, axis=1), 1.0, atol=1e-8):
        raise ValueError("kNN memory must contain finite L2-normalized vectors.")
    if not np.isfinite(k_value) or not k_value.is_integer() or not 1 <= k_value < len(memory):
        raise ValueError("Invalid kNN neighbour count.")
    if not np.isfinite(threshold) or threshold <= 0 or threshold != metadata.get("threshold") or k_value != metadata.get("n_neighbors"):
        raise ValueError("Invalid or inconsistent kNN threshold/neighbour metadata.")
    names = metadata.get("training_bags", [])
    if len(names) != len(memory) or len(set(names)) != len(names) or metadata.get("training_labels") != dict.fromkeys(names, "normal"):
        raise ValueError("kNN memory must contain only nominal training bags.")
    return CoPKNNDetector(memory, int(k_value), threshold, str(metadata["input_mode"]),
                          str(metadata["representation_id"]), dict(metadata["comparison_signature"]), metadata)
