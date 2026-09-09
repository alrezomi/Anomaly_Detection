"""Save fixed-size RynnBrain-CoP vectors and visualize them with PCA."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd


VECTOR_SCHEMA_VERSION = 1
PCA_LABELS = ("normal", "fail")


@dataclass(frozen=True)
class SavedCoPVector:
    vector_path: Path
    metadata_path: Path
    vector: np.ndarray
    metadata: dict[str, Any]


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()


def _canonical_label(value: Any) -> str:
    label = str(value or "unknown").strip().lower()
    if label in {"normal", "nominal"}:
        return "normal"
    if label in {"fail", "failure"}:
        return "fail"
    return "unknown"


def _validate_vector(vector: Any, source: str) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32)
    if array.ndim != 1:
        raise ValueError(f"{source} must contain one 1-D vector; got shape {array.shape}.")
    if array.size < 2:
        raise ValueError(f"{source} must contain at least two features.")
    if not np.isfinite(array).all():
        raise ValueError(f"{source} contains non-finite values.")
    return array


def save_cop_vector(
    output_directory: Path,
    input_mode: str,
    vector: Any,
    metadata: dict[str, Any],
) -> tuple[Path, Path]:
    """Save one float32 vector and a human-readable metadata sidecar."""
    array = _validate_vector(vector, "Captured CoP vector")
    vector_directory = output_directory / "cop_vectors"
    vector_directory.mkdir(parents=True, exist_ok=True)
    stem = _slug(input_mode)
    vector_path = vector_directory / f"{stem}.npy"
    metadata_path = vector_directory / f"{stem}.json"

    np.save(vector_path, array, allow_pickle=False)
    record = {
        **metadata,
        "schema_version": VECTOR_SCHEMA_VERSION,
        "input_mode": input_mode,
        "vector_file": vector_path.name,
        "vector_dimension": int(array.size),
        "vector_dtype": str(array.dtype),
    }
    metadata_path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return vector_path, metadata_path


def discover_saved_vectors(
    input_directory: Path,
    metadata_paths: list[Path] | None = None,
) -> list[SavedCoPVector]:
    """Load every vector sidecar below a single run or benchmark directory."""
    root = input_directory.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"CoP-vector input directory does not exist: {root}")

    records: list[SavedCoPVector] = []
    candidates = (
        sorted(path.resolve() for path in metadata_paths)
        if metadata_paths is not None
        else sorted(root.glob("**/cop_vectors/*.json"))
    )
    for metadata_path in candidates:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("schema_version") != VECTOR_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported CoP-vector schema in {metadata_path}: "
                f"{metadata.get('schema_version')!r}."
            )
        vector_name = Path(str(metadata.get("vector_file", ""))).name
        if not vector_name:
            raise ValueError(f"Missing vector_file in {metadata_path}.")
        vector_path = metadata_path.parent / vector_name
        if not vector_path.is_file():
            raise FileNotFoundError(
                f"Metadata {metadata_path} points to missing vector {vector_path}."
            )
        vector = _validate_vector(
            np.load(vector_path, allow_pickle=False), str(vector_path)
        )
        expected_dimension = int(metadata.get("vector_dimension", vector.size))
        if vector.size != expected_dimension:
            raise ValueError(
                f"Vector width mismatch for {vector_path}: metadata says "
                f"{expected_dimension}, file contains {vector.size}."
            )
        records.append(SavedCoPVector(vector_path, metadata_path, vector, metadata))
    return records


def pca_2d(
    vectors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """L2-normalize, center, and project a matrix to two principal components."""
    matrix = np.asarray(vectors, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 2:
        raise ValueError("PCA needs at least two vectors with at least two features.")
    if not np.isfinite(matrix).all():
        raise ValueError("PCA input contains non-finite values.")

    norms = np.linalg.norm(matrix, axis=1)
    if np.any(norms <= np.finfo(matrix.dtype).eps):
        raise ValueError("PCA cannot normalize an all-zero vector.")
    normalized = matrix / norms[:, None]
    feature_mean = normalized.mean(axis=0)
    centered = normalized - feature_mean[None, :]
    _, singular_values, components = np.linalg.svd(centered, full_matrices=False)
    components = components[:2].copy()

    # SVD signs are arbitrary. Fix them so repeated analyses have stable axes.
    for component in components:
        pivot = int(np.argmax(np.abs(component)))
        if component[pivot] < 0:
            component *= -1

    coordinates = centered @ components.T
    variance = singular_values**2 / (matrix.shape[0] - 1)
    total_variance = float(variance.sum())
    if total_variance <= np.finfo(matrix.dtype).eps:
        raise ValueError("PCA cannot separate identical vectors.")
    explained_ratio = np.zeros(2, dtype=np.float64)
    explained_ratio[: min(2, variance.size)] = (
        variance[:2] / total_variance
    )
    return coordinates, components, explained_ratio, feature_mean


def _comparison_signature(record: SavedCoPVector) -> str:
    signature = record.metadata.get("comparison_signature")
    if not isinstance(signature, dict):
        raise ValueError(
            f"Missing comparison_signature in {record.metadata_path}; refusing "
            "to combine vectors whose inference settings cannot be verified."
        )
    return json.dumps(signature, sort_keys=True, separators=(",", ":"))


def _write_pca_plot(
    coordinates: np.ndarray,
    rows: list[dict[str, Any]],
    explained_ratio: np.ndarray,
    input_mode: str,
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    styles = {
        "normal": {"label": "Nominal", "color": "#2f78c4", "marker": "o"},
        "fail": {"label": "Failure", "color": "#d1495b", "marker": "X"},
    }
    figure, axis = plt.subplots(figsize=(9, 7))
    for label in PCA_LABELS:
        indices = [index for index, row in enumerate(rows) if row["label"] == label]
        style = styles[label]
        axis.scatter(
            coordinates[indices, 0],
            coordinates[indices, 1],
            s=75,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.7,
            label=style["label"],
            color=style["color"],
            marker=style["marker"],
        )
        for index in indices:
            axis.annotate(
                rows[index]["bag_name"],
                (coordinates[index, 0], coordinates[index, 1]),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=8,
                alpha=0.8,
            )

    axis.set_xlabel(f"PC1 ({explained_ratio[0] * 100:.1f}% variance)")
    axis.set_ylabel(f"PC2 ({explained_ratio[1] * 100:.1f}% variance)")
    axis.set_title(f"RynnBrain-CoP last-layer vectors — {input_mode}")
    axis.axhline(0, color="#999999", linewidth=0.6, alpha=0.5)
    axis.axvline(0, color="#999999", linewidth=0.6, alpha=0.5)
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def analyze_saved_vectors(
    input_directory: Path,
    output_directory: Path | None = None,
    metadata_paths: list[Path] | None = None,
) -> dict[str, Any]:
    """Fit one PCA per input mode across saved nominal and failure vectors."""
    records = discover_saved_vectors(input_directory, metadata_paths)
    output = (output_directory or input_directory / "cop_pca").resolve()
    output.mkdir(parents=True, exist_ok=True)
    for pattern in (
        "cop_vectors_*.npz",
        "cop_pca_*.csv",
        "cop_pca_*.png",
        "cop_pca_model_*.npz",
        "cop_pca_summary.json",
    ):
        for old_artifact in output.glob(pattern):
            if old_artifact.is_file():
                old_artifact.unlink()

    summary: dict[str, Any] = {
        "input_directory": str(input_directory.resolve()),
        "output_directory": str(output),
        "selection": "all_saved_vectors" if metadata_paths is None else "explicit_vectors",
        "preprocessing": "float32 vectors, L2 normalization, feature centering",
        "saved_vector_count": len(records),
        "modes": [],
    }
    modes = sorted({str(record.metadata.get("input_mode", "")) for record in records})
    for input_mode in modes:
        mode_records = [
            record
            for record in records
            if str(record.metadata.get("input_mode", "")) == input_mode
        ]
        usable = [
            record
            for record in mode_records
            if _canonical_label(record.metadata.get("ground_truth_label")) in PCA_LABELS
        ]
        mode_summary: dict[str, Any] = {
            "input_mode": input_mode,
            "saved_count": len(mode_records),
            "usable_count": len(usable),
            "excluded_unknown_count": len(mode_records) - len(usable),
        }
        if not usable:
            mode_summary.update(status="skipped", reason="no normal/fail vectors")
            summary["modes"].append(mode_summary)
            continue

        dimensions = {record.vector.size for record in usable}
        if len(dimensions) != 1:
            mode_summary.update(
                status="skipped",
                reason=f"inconsistent vector dimensions: {sorted(dimensions)}",
            )
            summary["modes"].append(mode_summary)
            continue

        signatures = {_comparison_signature(record) for record in usable}
        if len(signatures) != 1:
            mode_summary.update(
                status="skipped",
                reason="vectors were produced with different comparison settings",
            )
            summary["modes"].append(mode_summary)
            continue

        rows = [
            {
                "bag_name": str(record.metadata.get("bag_name", "")),
                "bag_path": str(record.metadata.get("test_bag", "")),
                "label": _canonical_label(record.metadata.get("ground_truth_label")),
                "input_mode": input_mode,
                "vector_path": str(record.vector_path),
                "metadata_path": str(record.metadata_path),
            }
            for record in usable
        ]
        labels = {row["label"] for row in rows}
        if labels != set(PCA_LABELS):
            missing = sorted(set(PCA_LABELS) - labels)
            mode_summary.update(
                status="skipped",
                reason=f"missing ground-truth class(es): {', '.join(missing)}",
            )
            summary["modes"].append(mode_summary)
            continue

        matrix = np.stack([record.vector for record in usable])
        stem = _slug(input_mode)
        archive_path = output / f"cop_vectors_{stem}.npz"
        l2_norms = np.linalg.norm(matrix.astype(np.float64), axis=1)
        np.savez_compressed(
            archive_path,
            vectors=matrix.astype(np.float32, copy=False),
            l2_norms=l2_norms,
            labels=np.asarray([row["label"] for row in rows]),
            bag_names=np.asarray([row["bag_name"] for row in rows]),
            vector_paths=np.asarray([row["vector_path"] for row in rows]),
        )

        coordinates, components, explained_ratio, feature_mean = pca_2d(matrix)
        for index, row in enumerate(rows):
            row["pc1"] = float(coordinates[index, 0])
            row["pc2"] = float(coordinates[index, 1])
        coordinates_path = output / f"cop_pca_{stem}.csv"
        pd.DataFrame(rows).to_csv(coordinates_path, index=False)
        pca_model_path = output / f"cop_pca_model_{stem}.npz"
        np.savez_compressed(
            pca_model_path,
            components=components.astype(np.float32),
            feature_mean=feature_mean.astype(np.float32),
            explained_variance_ratio=explained_ratio.astype(np.float32),
            preprocessing=np.asarray("l2_normalize_then_center"),
        )
        plot_path = output / f"cop_pca_{stem}.png"
        _write_pca_plot(coordinates, rows, explained_ratio, input_mode, plot_path)

        mode_summary.update(
            status="created",
            vector_dimension=int(matrix.shape[1]),
            normal_count=sum(row["label"] == "normal" for row in rows),
            failure_count=sum(row["label"] == "fail" for row in rows),
            explained_variance_ratio=[float(value) for value in explained_ratio],
            vectors_archive=str(archive_path),
            coordinates_csv=str(coordinates_path),
            pca_model_file=str(pca_model_path),
            plot_file=str(plot_path),
        )
        summary["modes"].append(mode_summary)

    summary_path = output / "cop_pca_summary.json"
    summary["summary_file"] = str(summary_path)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    summary = analyze_saved_vectors(arguments.input_dir, arguments.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
