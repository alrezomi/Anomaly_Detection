"""Per-execution views of the nominal neighbours actually used by kNN."""

from pathlib import Path
import re

import numpy as np
import pandas as pd

from .cop_knn import _unit_rows


def neighbor_paths(directory: Path, mode: str):
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", mode).strip("_").lower()
    return directory / f"knn_neighbors_{slug}.csv", directory / f"knn_neighbors_{slug}.png"


def write_knn_neighbors(classifier, vector, *, comparison_signature, directory, mode, bag_name, provenance):
    """Distances remain in full feature space; PCA is a labelled visual aid only."""
    vector = np.asarray(vector.numpy() if hasattr(vector, "numpy") else vector, dtype=float)
    score = classifier.predict_anomaly_score(vector, input_mode=mode, comparison_signature=comparison_signature)
    query = _unit_rows(vector[None])[0]
    memory = classifier.nominal_vectors
    distances = np.linalg.norm(memory - query, axis=1)
    order = np.argsort(distances, kind="stable")
    names = classifier.metadata["training_bags"]
    csv_path, plot_path = neighbor_paths(directory, mode)
    rows = pd.DataFrame([{
        "test_bag": bag_name, "nominal_bag": names[index], "rank": rank + 1,
        "distance": float(distances[index]), "used_for_score": rank < classifier.n_neighbors,
        "anomaly_score": score, "threshold": classifier.threshold,
        "evaluation_id": provenance.get("evaluation_id"),
        "lora_adapter_sha256": provenance.get("lora_adapter_sha256"),
    } for rank, index in enumerate(order)])
    rows.to_csv(csv_path, index=False)

    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    # Fit the projection only to nominal training examples, then project the test.
    center = memory.mean(axis=0)
    _, singular, axes = np.linalg.svd(memory - center, full_matrices=False)
    projected = (memory - center) @ axes[:2].T
    test = (query - center) @ axes[:2].T
    variance = singular[:2] ** 2 / max(float(np.sum(singular ** 2)), np.finfo(float).eps)
    shown = rows.head(30)
    figure, (left, right) = plt.subplots(1, 2, figsize=(13, max(6, 2.8 + 0.22 * len(shown))),
                                        gridspec_kw={"width_ratios": [1, 1.15]})
    try:
        figure.subplots_adjust(left=0.07, right=0.97, top=0.80, bottom=0.17, wspace=0.48)
        figure.suptitle("Test execution and its nominal neighbours", x=0.07, y=0.97,
                       ha="left", fontsize=18, weight="bold", color="#172b40")
        figure.text(0.07, 0.91, f"{bag_name}  /  {mode}  /  k = {classifier.n_neighbors}    "
                    f"Mean neighbour distance: {score:.4f}    Threshold: {classifier.threshold:.4f}",
                    color="#607080", fontsize=10)
        left.scatter(*projected.T, s=42, color="#9dafc0", label="Nominal training bags", zorder=3)
        nearest = order[:classifier.n_neighbors]
        for rank, index in enumerate(nearest):
            left.plot([test[0], projected[index, 0]], [test[1], projected[index, 1]],
                      color="#2f78c4", alpha=0.45, linewidth=1)
            left.annotate(str(rank + 1), projected[index], xytext=(5, 5), textcoords="offset points", fontsize=9)
        left.scatter(*projected[nearest].T, s=65, color="#2f78c4", label="k nearest (full space)", zorder=4)
        left.scatter(*test, s=190, marker="*", color="#c44850", edgecolors="white", linewidth=0.7,
                     label="Test execution", zorder=5)
        left.set(xlabel=f"PC1 ({variance[0]:.1%} nominal variance)",
                 ylabel=f"PC2 ({variance[1]:.1%} nominal variance)", title="PCA projection · visual aid only")
        left.legend(loc="best", fontsize=8, frameon=False)
        positions = np.arange(len(shown))
        colors = np.where(shown.used_for_score, "#2f78c4", "#bbc7d1")
        right.barh(positions, shown.distance, color=colors, height=0.65)
        right.set_yticks(positions, [f"{row.rank}. {row.nominal_bag}" for row in shown.itertuples()])
        right.invert_yaxis()
        right.set(xlabel="Euclidean distance between L2-normalized vectors",
                  title="Actual full-feature distances (nearest first)")
        right.tick_params(axis="y", labelsize=8)
        for axis in (left, right):
            axis.spines[["top", "right"]].set_visible(False)
            axis.grid(axis="x" if axis is right else "both", color="#e7ecf0", linewidth=0.7)
            axis.set_axisbelow(True)
        figure.text(0.07, 0.075, "Blue bars are the neighbours used for the score. The threshold applies to their mean distance, not each bar.\n"
                    "PCA can hide separation; neighbour selection and scoring always use all features. "
                    + ("Showing the nearest 30; the CSV contains all nominal bags." if len(rows) > 30 else ""),
                    fontsize=9, color="#607080")
        figure.savefig(plot_path, dpi=180, bbox_inches="tight", facecolor="white")
    finally:
        plt.close(figure)
    return {"status": "created", "csv": str(csv_path), "plot": str(plot_path), "error": None}
