"""ROC reports for generated decisions and CoP classifier probabilities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import textwrap
from typing import Any

import numpy as np
import pandas as pd


DECISION_SCORES = {"success": 0, "failure": 1}
METRIC_COLUMNS = [
    "input_mode", "failure_category", "normal_count", "failure_count",
    "normal_abstention_count", "failure_abstention_count",
    "uncertain_count", "unparsed_count", "other_excluded_decision_count",
    "excluded_decision_count", "unknown_label_count", "decision_coverage",
    "true_positive", "false_positive", "true_negative", "false_negative",
    "accuracy", "failure_recall_decided", "normal_specificity_decided",
    "false_positive_rate_decided", "balanced_accuracy_decided",
    "auroc", "status", "reason",
]


def failure_category(bag_path: str, label: str) -> str | None:
    """Read the existing Failure_<number>_<description> folder without renaming it."""
    if label != "fail":
        return None
    # Accept saved Linux paths on Windows, and vice versa. Ignore the bag itself.
    parents = str(bag_path).replace("\\", "/").rstrip("/").split("/")[:-1]
    for part in reversed(parents):
        if re.fullmatch(r"Failure_\d+(?:_.*)?", part, flags=re.IGNORECASE):
            return part
    return "uncategorized_failure"


def roc_curve(targets: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Sweep distinct scores, including ties together; failure is the positive class."""
    targets = np.asarray(targets)
    scores = np.asarray(scores, dtype=float)
    if targets.ndim != 1 or scores.shape != targets.shape or not targets.size:
        raise ValueError("ROC requires equally sized, nonempty 1-D targets and scores.")
    if not np.isin(targets, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError("ROC requires binary targets and finite scores.")
    positives = int(targets.sum())
    negatives = targets.size - positives
    if not positives or not negatives:
        raise ValueError("ROC requires both normal and failure samples.")
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_scores)), scores.size - 1]
    true_positives = np.cumsum(targets[order])[ends]
    false_positives = ends + 1 - true_positives
    tpr = np.r_[0.0, true_positives / positives]
    fpr = np.r_[0.0, false_positives / negatives]
    thresholds = np.r_[np.inf, sorted_scores[ends]]
    # Trapezoidal area without requiring a newer NumPy or another dependency.
    auroc = float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2.0))
    return fpr, tpr, thresholds, auroc


def write_roc_report(master_rows: list[dict[str, Any]], benchmark_root: Path) -> dict[str, Any]:
    """Compare all failures and each failure category against the same normal bags.

    Other failure categories are excluded from a category's curve, rather than
    treated as negatives. Modes are never pooled. ROC uses only parsed binary
    decisions (success=0, failure=1), giving one operating point. Abstentions
    are excluded from ROC and counted as incorrect in overall accuracy.
    """
    dataframe = pd.DataFrame(master_rows).reindex(columns=[
        "bag_name", "bag_path", "ground_truth_label", "input_mode", "decision",
    ])
    dataframe["failure_category"] = [
        failure_category(path, label)
        for path, label in zip(dataframe["bag_path"], dataframe["ground_truth_label"])
    ]
    dataframe["decision_score"] = dataframe["decision"].map(DECISION_SCORES)
    output_dir = Path(benchmark_root) / "benchmark_roc"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics: list[dict[str, Any]] = []
    points: list[dict[str, Any]] = []
    plot_files: list[str] = []
    for mode, group in dataframe.dropna(subset=["input_mode"]).groupby("input_mode", sort=True):
        categories = sorted(group.loc[group["ground_truth_label"] == "fail", "failure_category"].unique())
        curves = []
        for category in [None, *categories]:
            selected = group[group["ground_truth_label"].isin(["normal", "fail"])]
            if category is not None:
                selected = selected[
                    (selected["ground_truth_label"] == "normal")
                    | (selected["failure_category"] == category)
                ]
            valid = selected[selected["decision_score"].notna()]
            normal_count = int((valid["ground_truth_label"] == "normal").sum())
            failure_count = int((valid["ground_truth_label"] == "fail").sum())
            abstained = selected[selected["decision_score"].isna()]
            tp = int(((valid["ground_truth_label"] == "fail") & (valid["decision"] == "failure")).sum())
            fp = int(((valid["ground_truth_label"] == "normal") & (valid["decision"] == "failure")).sum())
            tn, fn = normal_count - fp, failure_count - tp
            recall = tp / failure_count if failure_count else None
            specificity = tn / normal_count if normal_count else None
            uncertain_count = int((abstained["decision"] == "uncertain").sum())
            unparsed_count = int((abstained["decision"] == "not_parsed").sum())
            result = {
                "input_mode": str(mode),
                "failure_category": category or "all_failures",
                "normal_count": normal_count,
                "failure_count": failure_count,
                "normal_abstention_count": int((abstained["ground_truth_label"] == "normal").sum()),
                "failure_abstention_count": int((abstained["ground_truth_label"] == "fail").sum()),
                "uncertain_count": uncertain_count,
                "unparsed_count": unparsed_count,
                "other_excluded_decision_count": int(len(abstained) - uncertain_count - unparsed_count),
                "excluded_decision_count": int(len(abstained)),
                "unknown_label_count": int((~group["ground_truth_label"].isin(["normal", "fail"])).sum()),
                "decision_coverage": len(valid) / len(selected) if len(selected) else None,
                "true_positive": tp,
                "false_positive": fp,
                "true_negative": tn,
                "false_negative": fn,
                "accuracy": (tp + tn) / len(selected) if len(selected) else None,
                "failure_recall_decided": recall,
                "normal_specificity_decided": specificity,
                "false_positive_rate_decided": fp / normal_count if normal_count else None,
                "balanced_accuracy_decided": (
                    (recall + specificity) / 2.0
                    if recall is not None and specificity is not None else None
                ),
                "auroc": None,
                "status": "skipped",
                "reason": "",
            }
            metrics.append(result)
            if valid.empty:
                result["reason"] = "No parsed VLM success/failure decisions for labeled bags."
                continue
            if not normal_count or not failure_count:
                result["reason"] = "ROC requires both normal and failure bags with parsed VLM decisions."
                continue
            fpr, tpr, thresholds, auroc = roc_curve(
                (valid["ground_truth_label"] == "fail").to_numpy(dtype=int),
                valid["decision_score"].to_numpy(dtype=float),
            )
            result.update(auroc=auroc, status="created")
            curves.append((result["failure_category"], fpr, tpr, auroc, fp / normal_count, recall))
            points.extend({
                "input_mode": str(mode),
                "failure_category": result["failure_category"],
                "threshold": float(threshold),
                "false_positive_rate": float(false_positive_rate),
                "true_positive_rate": float(true_positive_rate),
            } for threshold, false_positive_rate, true_positive_rate in zip(thresholds, fpr, tpr))

        # Remove this mode's old plot if a rerun cannot produce a valid curve.
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", str(mode)).strip("_").lower()
        plot_path = output_dir / f"roc_{slug}.png"
        if not curves:
            plot_path.unlink(missing_ok=True)
            continue
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(8, 6))
        for category, fpr, tpr, auroc, operating_fpr, operating_tpr in curves:
            line, = axis.plot(fpr, tpr, linestyle=":", label=f"{category} (binary AUROC={auroc:.3f})")
            axis.plot(operating_fpr, operating_tpr, "o", color=line.get_color())
        axis.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Chance")
        axis.set(
            xlabel="False positive rate (normal bags)",
            ylabel="True positive rate (failure bags)",
            title=f"VLM decision ROC - {mode}\nBinary outputs; AUROC = balanced accuracy on decided bags",
            xlim=(0, 1), ylim=(0, 1.02),
        )
        axis.grid(alpha=0.2)
        axis.legend(loc="lower right", fontsize="small")
        figure.tight_layout()
        figure.savefig(plot_path, dpi=160)
        plt.close(figure)
        plot_files.append(str(plot_path))

    pd.DataFrame(metrics, columns=METRIC_COLUMNS).to_csv(output_dir / "auroc.csv", index=False)
    pd.DataFrame(points, columns=[
        "input_mode", "failure_category", "threshold", "false_positive_rate", "true_positive_rate",
    ]).to_csv(output_dir / "roc_points.csv", index=False)
    report = {
        "score_column": "decision",
        "decision_encoding": DECISION_SCORES,
        "roc_interpretation": (
            "Binary generated decisions give one operating point, not a confidence sweep. "
            "AUROC equals balanced accuracy on bags with parsed success/failure decisions."
        ),
        "abstention_policy": (
            "Uncertain, unparsed and missing decisions are excluded from ROC and confusion counts; "
            "accuracy counts them as incorrect. Coverage and abstention counts are reported. "
            "Unknown ground-truth labels are excluded from all decision metrics."
        ),
        "comparison": "Each failure category versus normal; other failures excluded.",
        "rows_without_input_mode": int(dataframe["input_mode"].isna().sum()),
        "metrics": metrics,
        "plot_files": plot_files,
    }
    (output_dir / "roc_summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


def write_probability_roc_report(
    master_rows: list[dict[str, Any]], benchmark_root: Path, *,
    input_modes: list[str] | None = None,
) -> dict[str, Any]:
    """Sweep continuous classifier scores, independent of the VLM's decision."""
    score_column = "classifier_failure_probability"
    dataframe = pd.DataFrame(master_rows).reindex(columns=[
        "bag_name", "bag_path", "ground_truth_label", "input_mode", score_column,
    ])
    dataframe["failure_category"] = [failure_category(path, label)
                                     for path, label in zip(dataframe["bag_path"], dataframe["ground_truth_label"])]
    dataframe["missing_score"] = dataframe[score_column].isna() | dataframe[score_column].astype(str).str.strip().eq("")
    dataframe["score"] = pd.to_numeric(dataframe[score_column], errors="coerce")
    dataframe["valid_score"] = np.isfinite(dataframe["score"]) & dataframe["score"].between(0, 1)
    output_dir = Path(benchmark_root) / "benchmark_probability_roc"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics, points, plot_files = [], [], []
    modes = sorted(set(input_modes or []) | set(dataframe["input_mode"].dropna().astype(str)))
    for mode in modes:
        group = dataframe[dataframe["input_mode"] == mode]
        categories = sorted(group.loc[group["ground_truth_label"] == "fail", "failure_category"].unique())
        curves = []
        for category in [None, *categories]:
            selected = group[group["ground_truth_label"].isin(["normal", "fail"])]
            if category is not None:
                selected = selected[(selected["ground_truth_label"] == "normal") | (selected["failure_category"] == category)]
            valid = selected[selected["valid_score"]]
            normal_count = int((valid["ground_truth_label"] == "normal").sum())
            failure_count = int((valid["ground_truth_label"] == "fail").sum())
            result = {
                "input_mode": mode, "failure_category": category or "all_failures",
                "normal_count": normal_count, "failure_count": failure_count,
                "missing_probability_count": int(selected["missing_score"].sum()),
                "invalid_probability_count": int((~selected["valid_score"] & ~selected["missing_score"]).sum()),
                "excluded_probability_count": int((~selected["valid_score"]).sum()),
                "unknown_label_count": int((~group["ground_truth_label"].isin(["normal", "fail"])).sum()),
                "probability_coverage": len(valid) / len(selected) if len(selected) else None,
                "distinct_score_count": int(valid["score"].nunique()),
                "auroc": None, "status": "skipped", "reason": "",
            }
            metrics.append(result)
            if not normal_count or not failure_count:
                result["reason"] = "ROC requires both normal and failure bags with finite probabilities in [0, 1]."
                continue
            fpr, tpr, thresholds, auroc = roc_curve(
                (valid["ground_truth_label"] == "fail").to_numpy(dtype=int),
                valid["score"].to_numpy(dtype=float),
            )
            result.update(auroc=auroc, status="created")
            curves.append((result["failure_category"], fpr, tpr, auroc))
            points.extend({
                "input_mode": mode, "failure_category": result["failure_category"],
                "threshold": float(threshold), "false_positive_rate": float(x), "true_positive_rate": float(y),
            } for threshold, x, y in zip(thresholds, fpr, tpr))
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", mode).strip("_").lower()
        roc_path, auc_path = output_dir / f"roc_{slug}.png", output_dir / f"auroc_{slug}.png"
        if curves:
            _plot_probability_curves(curves, mode, roc_path, auc_path)
            plot_files.extend([str(roc_path), str(auc_path)])
        else:
            roc_path.unlink(missing_ok=True)
            auc_path.unlink(missing_ok=True)
    pd.DataFrame(metrics, columns=[
        "input_mode", "failure_category", "normal_count", "failure_count", "missing_probability_count",
        "invalid_probability_count", "excluded_probability_count", "unknown_label_count", "probability_coverage",
        "distinct_score_count", "auroc", "status", "reason",
    ]).to_csv(output_dir / "auroc.csv", index=False)
    pd.DataFrame(points, columns=[
        "input_mode", "failure_category", "threshold", "false_positive_rate", "true_positive_rate",
    ]).to_csv(output_dir / "roc_points.csv", index=False)
    report = {
        "score_column": score_column, "positive_class": "fail",
        "threshold_rule": "Predict failure when probability >= threshold; scores and thresholds use [0, 1], with inf as the no-positive endpoint.",
        "roc_interpretation": "ROC sweeps distinct CoP failure probabilities, grouping ties. AUROC measures failure-vs-normal ranking, not calibration or VLM decision accuracy.",
        "exclusion_policy": "Missing/nonfinite/out-of-range scores and unknown ground truth are excluded and counted. VLM uncertain/unparsed decisions do not exclude valid classifier scores.",
        "comparison": "Each failure category versus normal; other failures excluded. Input modes are separate.",
        "rows_without_input_mode": int(dataframe["input_mode"].isna().sum()),
        "metrics": metrics, "plot_files": plot_files,
    }
    (output_dir / "roc_summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def _plot_probability_curves(curves, mode: str, roc_path: Path, auc_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    colors = ["#222222"] + [plt.get_cmap("tab10")(index % 10) for index in range(len(curves) - 1)]
    labels = [textwrap.fill(category.replace("_", " "), width=45) for category, *_ in curves]
    figure, axis = plt.subplots(figsize=(7, 6))
    try:
        for (category, fpr, tpr, auroc), color, label in zip(curves, colors, labels):
            axis.plot(fpr, tpr, color=color, linewidth=2.4 if category == "all_failures" else 1.5,
                      label=f"{label}\nAUROC = {auroc:.3f}")
        axis.plot([0, 1], [0, 1], "--", color="#999999", label="Chance (AUROC = 0.5)")
        axis.set(xlabel="False positive rate (normal bags)", ylabel="True positive rate (failure bags)",
                 title=f"CoP probability ROC — {mode}", xlim=(0, 1), ylim=(0, 1.02))
        axis.grid(alpha=0.2)
        axis.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
        figure.savefig(roc_path, dpi=180, bbox_inches="tight")
    finally:
        plt.close(figure)
    figure, axis = plt.subplots(figsize=(8, max(3.5, 0.75 * len(curves))))
    try:
        values = [curve[3] for curve in curves]
        bars = axis.barh(range(len(curves)), values, color=colors, height=0.6)
        axis.set_yticks(range(len(curves)), labels)
        axis.invert_yaxis()
        axis.set(xlim=(0, 1.12), xlabel="AUROC", title=f"CoP probability AUROC — {mode}")
        axis.set_xticks(np.linspace(0, 1, 6))
        axis.axvline(0.5, color="#777777", linestyle="--", linewidth=1, label="Chance (0.5)")
        for bar, value in zip(bars, values):
            axis.text(value + 0.015, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center")
        axis.grid(axis="x", alpha=0.2)
        axis.set_axisbelow(True)
        axis.legend(loc="lower right", fontsize=8)
        figure.savefig(auc_path, dpi=180, bbox_inches="tight")
    finally:
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True, help="Existing benchmark_summary.csv")
    arguments = parser.parse_args()
    dataframe = pd.read_csv(arguments.summary)
    required = {"bag_path", "ground_truth_label", "input_mode", "decision"}
    missing = required - set(dataframe.columns)
    if missing:
        raise ValueError("Benchmark summary is missing columns: " + ", ".join(sorted(missing)))
    write_roc_report(dataframe.to_dict("records"), arguments.summary.parent)
    write_probability_roc_report(dataframe.to_dict("records"), arguments.summary.parent)
    from .cop_analysis import write_failure_probability_report
    write_failure_probability_report(dataframe.to_dict("records"), arguments.summary.parent)
    print(f"ROC report: {arguments.summary.parent / 'benchmark_roc'}")
    print(f"Probability ROC report: {arguments.summary.parent / 'benchmark_probability_roc'}")
    print(f"Failure-probability boxplots: {arguments.summary.parent}")


if __name__ == "__main__":
    main()
