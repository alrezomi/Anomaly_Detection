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

from .benchmark_roc import failure_category


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


def write_failure_probability_report(
    rows: list[dict[str, Any]], output_directory: Path, *,
    input_modes: list[str] | None = None,
) -> dict[str, Any]:
    """Summarize held-out classifier probabilities by ground truth, per mode."""
    output_directory.mkdir(parents=True, exist_ok=True)
    modes = sorted(set(input_modes or []) | {str(row["input_mode"]) for row in rows if row.get("input_mode")})
    summary_rows = []
    mode_reports = []
    columns = ["input_mode", "ground_truth_label", "count", "mean_percent", "median_percent",
               "q1_percent", "q3_percent", "min_percent", "max_percent"]
    for mode in modes:
        groups: dict[str, list[float]] = {"normal": [], "fail": []}
        excluded = {"unknown_label": 0, "missing_probability": 0, "invalid_probability": 0}
        for row in rows:
            if row.get("input_mode") != mode:
                continue
            label = _canonical_label(row.get("ground_truth_label"))
            if label not in groups:
                excluded["unknown_label"] += 1
                continue
            value = row.get("classifier_failure_probability")
            if value is None or value == "":
                excluded["missing_probability"] += 1
                continue
            try:
                probability = float(value)
            except (TypeError, ValueError):
                probability = float("nan")
            if not np.isfinite(probability) or not 0 <= probability <= 1:
                excluded["invalid_probability"] += 1
                continue
            groups[label].append(100.0 * probability)
        summaries = []
        for label, values in groups.items():
            summary = {column: None for column in columns}
            summary.update(input_mode=mode, ground_truth_label=label, count=len(values))
            if values:
                q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
                summary.update(mean_percent=float(np.mean(values)), median_percent=float(median),
                               q1_percent=float(q1), q3_percent=float(q3),
                               min_percent=min(values), max_percent=max(values))
            summaries.append(summary)
        summary_rows.extend(summaries)
        plot_path = output_directory / f"benchmark_failure_probability_{_slug(mode)}.png"
        valid_count = sum(len(values) for values in groups.values())
        if valid_count:
            import matplotlib
            matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt

            figure, axis = plt.subplots(figsize=(7, 5))
            try:
                for position, (label, color) in enumerate((("normal", "#4878cf"), ("fail", "#d65f5f")), start=1):
                    values = groups[label]
                    if values:
                        boxes = axis.boxplot([values], positions=[position], widths=0.42, patch_artist=True,
                                             showfliers=False, medianprops={"color": "#202020", "linewidth": 2})
                        boxes["boxes"][0].set(facecolor=color, alpha=0.3)
                        jitter = np.random.default_rng(42).uniform(-0.09, 0.09, len(values))
                        axis.scatter(position + jitter, values, s=22, color=color, alpha=0.7, zorder=3)
                    else:
                        axis.text(position, 50, "No scored bags", ha="center", color="#666666")
                axis.set_xticks([1, 2], [f"Successful\n(n={len(groups['normal'])})", f"Failed\n(n={len(groups['fail'])})"])
                axis.set(xlim=(0.5, 2.5), ylim=(-3, 103), ylabel="Failure probability (%)",
                         xlabel="Ground-truth execution outcome", title=f"CoP failure probability — {mode}")
                axis.set_yticks(range(0, 101, 20))
                axis.grid(axis="y", alpha=0.2)
                axis.spines[["top", "right"]].set_visible(False)
                figure.text(0.5, 0.015, "Box: middle 50% · Line: median · Whiskers: 1.5×IQR · Dots: individual bags",
                            ha="center", fontsize=8, color="#555555")
                figure.tight_layout(rect=(0, 0.04, 1, 1))
                figure.savefig(plot_path, dpi=180)
            finally:
                plt.close(figure)
        else:
            # Do not leave a previous run's plot masquerading as this result.
            plot_path.unlink(missing_ok=True)
        mode_reports.append({"input_mode": mode, "status": "created" if valid_count else "skipped",
                             "reason": None if valid_count else "No valid probabilities with known ground truth.",
                             "plot_file": str(plot_path) if valid_count else None,
                             "scored_rows": valid_count, "excluded_rows": excluded, "groups": summaries})
    csv_path = output_directory / "benchmark_failure_probability_summary.csv"
    pd.DataFrame(summary_rows, columns=columns).to_csv(csv_path, index=False)
    return {"grouping": "ground_truth_label", "units": "percent",
            "note": "Classifier estimates; groups are actual outcomes, not predicted decisions. Input modes are kept separate.",
            "unassigned_mode_rows": sum(not bool(row.get("input_mode")) for row in rows),
            "summary_csv": str(csv_path), "modes": mode_reports}


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


def _pca_category_styles(categories: set[str]) -> dict[str, dict[str, Any]]:
    """Use one marker per category, shared by all input modes in an analysis."""
    markers = ["^", "s", "D", "v", "P", "X", "*", "<", ">", "p", "h", "H"]
    colors = ["#e1812c", "#3a923a", "#c03d3e", "#9372b2", "#845b53",
              "#d684bd", "#797979", "#a9a52b", "#35a0ad"]
    styles: dict[str, dict[str, Any]] = {
        "Nominal": {"marker": "o", "color": "#2f78c4"},
    }
    # Natural ordering keeps Failure_2 before Failure_10.
    ordered = sorted(categories - {"Nominal"}, key=lambda value: [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", value)
    ])
    for index, category in enumerate(ordered):
        styles[category] = {
            "marker": markers[index] if index < len(markers) else (index - len(markers) + 7, 0, 0),
            "color": colors[index % len(colors)],
        }
    return {category: style for category, style in styles.items() if category in categories}


def _pca_category(record: SavedCoPVector) -> str:
    label = _canonical_label(record.metadata.get("ground_truth_label"))
    return failure_category(str(record.metadata.get("test_bag", "")), label) or "Nominal"


def _write_pca_plot(
    coordinates: np.ndarray,
    rows: list[dict[str, Any]],
    explained_ratio: np.ndarray,
    input_mode: str,
    output_path: Path,
    category_styles: dict[str, dict[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(11, 7))
    for category, style in category_styles.items():
        indices = [index for index, row in enumerate(rows) if row["category"] == category]
        if not indices:
            continue
        axis.scatter(
            coordinates[indices, 0],
            coordinates[indices, 1],
            s=75,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.7,
            label=f"{category.replace('_', ' ')} (n={len(indices)})",
            color=style["color"],
            marker=style["marker"],
        )

    axis.set_xlabel(f"PC1 ({explained_ratio[0] * 100:.1f}% variance)")
    axis.set_ylabel(f"PC2 ({explained_ratio[1] * 100:.1f}% variance)")
    axis.set_title(f"RynnBrain-CoP last-layer vectors — {input_mode}")
    axis.axhline(0, color="#999999", linewidth=0.6, alpha=0.5)
    axis.axvline(0, color="#999999", linewidth=0.6, alpha=0.5)
    axis.grid(alpha=0.2)
    axis.set_axisbelow(True)
    axis.legend(title="Category", loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=9)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def analyze_saved_vectors(
    input_directory: Path,
    output_directory: Path | None = None,
    metadata_paths: list[Path] | None = None,
) -> dict[str, Any]:
    """Fit one PCA per input mode across saved nominal and failure vectors."""
    records = discover_saved_vectors(input_directory, metadata_paths)
    category_styles = _pca_category_styles({
        _pca_category(record) for record in records
        if _canonical_label(record.metadata.get("ground_truth_label")) in PCA_LABELS
    })
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
                "category": _pca_category(record),
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
        try:
            coordinates, components, explained_ratio, feature_mean = pca_2d(matrix)
        except ValueError as error:
            mode_summary.update(status="skipped", reason=str(error))
            summary["modes"].append(mode_summary)
            continue

        stem = _slug(input_mode)
        archive_path = output / f"cop_vectors_{stem}.npz"
        l2_norms = np.linalg.norm(matrix.astype(np.float64), axis=1)
        np.savez_compressed(
            archive_path,
            vectors=matrix.astype(np.float32, copy=False),
            l2_norms=l2_norms,
            labels=np.asarray([row["label"] for row in rows]),
            categories=np.asarray([row["category"] for row in rows]),
            bag_names=np.asarray([row["bag_name"] for row in rows]),
            vector_paths=np.asarray([row["vector_path"] for row in rows]),
        )

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
        _write_pca_plot(coordinates, rows, explained_ratio, input_mode, plot_path, category_styles)

        mode_summary.update(
            status="created",
            vector_dimension=int(matrix.shape[1]),
            normal_count=sum(row["label"] == "normal" for row in rows),
            failure_count=sum(row["label"] == "fail" for row in rows),
            categories={
                category: {**style, "count": sum(row["category"] == category for row in rows)}
                for category, style in category_styles.items()
                if any(row["category"] == category for row in rows)
            },
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
