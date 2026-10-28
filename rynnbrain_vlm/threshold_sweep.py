"""Compare fixed probability thresholds using already evaluated benchmark bags."""

import json
from pathlib import Path
import re

import numpy as np
import pandas as pd


THRESHOLDS = tuple(percent / 100 for percent in range(60, 91))
COLUMNS = ["input_mode", "benchmark_decision_mode", "failure_category", "threshold", "threshold_percent",
           "normal_count", "failure_count", "true_positive", "false_positive", "true_negative", "false_negative",
           "accuracy", "balanced_accuracy", "recall", "false_positive_rate", "precision", "f1", "is_best_accuracy"]


def _threshold_metrics(targets, scores, threshold):
    positive = scores >= threshold
    tp = int(np.sum(positive & targets))
    fp = int(np.sum(positive & ~targets))
    tn = int(np.sum(~positive & ~targets))
    fn = int(np.sum(~positive & targets))
    recall = tp / (tp + fn) if tp + fn else None
    fpr = fp / (fp + tn) if fp + tn else None
    return dict(true_positive=tp, false_positive=fp, true_negative=tn, false_negative=fn,
                accuracy=(tp + tn) / len(scores) if len(scores) else None,
                balanced_accuracy=(recall + 1 - fpr) / 2 if recall is not None and fpr is not None else None,
                recall=recall, false_positive_rate=fpr,
                precision=tp / (tp + fp) if tp + fp else None,
                f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None)


def write_threshold_sweep_report(master_rows, benchmark_root, *, input_modes=None):
    from .benchmark_roc import failure_category

    root = Path(benchmark_root) / "benchmark_threshold_sweep"
    root.mkdir(parents=True, exist_ok=True)
    data = pd.DataFrame(master_rows).reindex(columns=["bag_path", "input_mode", "ground_truth_label",
        "classifier_failure_probability", "benchmark_decision_mode"])
    data["benchmark_decision_mode"] = data["benchmark_decision_mode"].fillna("whole_demo")
    data["score"] = pd.to_numeric(data["classifier_failure_probability"], errors="coerce")
    data["valid_score"] = np.isfinite(data["score"]) & data["score"].between(0, 1)
    data["failure_category"] = [failure_category(path, label) for path, label in
                                zip(data["bag_path"], data["ground_truth_label"])]
    metrics, comparisons, plots = [], [], []
    modes = sorted(set(input_modes or []) | set(data["input_mode"].dropna().astype(str)))
    for mode in modes:
        mode_rows = data[data["input_mode"] == mode]
        protocols = sorted(mode_rows["benchmark_decision_mode"].unique()) or ["whole_demo"]
        for protocol in protocols:
            group = mode_rows[mode_rows["benchmark_decision_mode"] == protocol]
            categories = sorted(group.loc[group["ground_truth_label"] == "fail", "failure_category"].unique())
            curves = []
            for category in ["all_failures", *categories]:
                selected = group[group["ground_truth_label"].isin(["normal", "fail"])]
                if category != "all_failures":
                    selected = selected[(selected["ground_truth_label"] == "normal") | (selected["failure_category"] == category)]
                valid = selected[selected["valid_score"]]
                targets = (valid["ground_truth_label"] == "fail").to_numpy(dtype=bool)
                scores = valid["score"].to_numpy(dtype=float)
                normal, failure = int(np.sum(~targets)), int(np.sum(targets))
                rows = [dict(input_mode=mode, benchmark_decision_mode=protocol, failure_category=category,
                             threshold=threshold, threshold_percent=int(round(threshold * 100)),
                             normal_count=normal, failure_count=failure,
                             **_threshold_metrics(targets, scores, threshold), is_best_accuracy=False)
                        for threshold in THRESHOLDS]
                usable = bool(normal and failure)
                best = max(row["accuracy"] for row in rows) if usable else None
                balanced = max(row["balanced_accuracy"] for row in rows) if usable else None
                best_thresholds = [row["threshold"] for row in rows if usable and row["accuracy"] == best]
                for row in rows:
                    row["is_best_accuracy"] = bool(usable and row["accuracy"] == best)
                metrics.extend(rows)
                summary = dict(input_mode=mode, benchmark_decision_mode=protocol, failure_category=category,
                    normal_count=normal, failure_count=failure, scored_bags=len(valid),
                    excluded_score_count=int((~selected["valid_score"]).sum()),
                    unknown_label_count=int((~group["ground_truth_label"].isin(["normal", "fail"])).sum()),
                    score_coverage=len(valid) / len(selected) if len(selected) else None,
                    status="created" if usable else "skipped",
                    reason=None if usable else "Threshold selection requires both normal and failure bags with valid scores.",
                    best_accuracy=best, best_accuracy_thresholds=best_thresholds,
                    best_balanced_accuracy=balanced,
                    best_balanced_accuracy_thresholds=[row["threshold"] for row in rows
                        if usable and row["balanced_accuracy"] == balanced])
                comparisons.append(summary)
                if usable:
                    curves.append((category, rows))
            slug = re.sub(r"[^a-zA-Z0-9]+", "_", f"{mode}_{protocol}").strip("_").lower()
            path = root / f"threshold_metrics_{slug}.png"
            if curves:
                _plot_sweep(curves, mode, protocol, path)
            else:
                from .report_artifacts import unavailable_plot
                unavailable_plot(path, f"Threshold sweep / {mode} / {protocol}",
                                 "Both normal and failure bags with valid scores are required to select a threshold.")
            plots.append(str(path))
    csv_path = root / "threshold_metrics.csv"
    pd.DataFrame(metrics, columns=COLUMNS).to_csv(csv_path, index=False)
    report = dict(score_column="classifier_failure_probability", thresholds=list(THRESHOLDS),
        threshold_rule="Predict failure when score >= threshold.",
        score_definition="Whole-demo score in whole_demo mode; maximum observed prefix score in timeline mode.",
        interpretation="Empirical accuracy on these scored bags, not independent test performance. "
            "Select thresholds on validation bags and evaluate the chosen threshold on separate test bags.",
        sampling_note="Reuses saved scores and observation sequences. Does not resimulate adaptive replay, "
            "change frame sampling, retrain models, or modify the configured threshold.",
        tie_policy="Report all thresholds tied for best accuracy within 0.60 to 0.90 inclusive.",
        units="Thresholds and metrics use fractions in [0, 1]; plots use percent.",
        comparisons=comparisons, csv=str(csv_path), plot_files=plots)
    (root / "threshold_summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def _plot_sweep(curves, mode, protocol, path):
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    fields = [("accuracy", "Accuracy"), ("balanced_accuracy", "Balanced accuracy"),
              ("recall", "Failure recall"), ("false_positive_rate", "False-positive rate")]
    colors = plt.get_cmap("tab10")
    for index, (category, rows) in enumerate(curves):
        overall = category == "all_failures"
        label = f"All failures (normal={rows[0]['normal_count']}, failure={rows[0]['failure_count']})" if overall else category
        for axis, (field, title) in zip(axes.flat, fields):
            axis.plot([row["threshold_percent"] for row in rows], [100 * row[field] for row in rows],
                      label=label, color="#172b40" if overall else colors((index - 1) % 10),
                      linewidth=2.5 if overall else 1.2, alpha=1 if overall else .7)
            if overall and field == "accuracy":
                best = [row for row in rows if row["is_best_accuracy"]]
                axis.scatter([row["threshold_percent"] for row in best], [100 * row["accuracy"] for row in best],
                             s=18, color="#c27524", zorder=4, label="Best accuracy (including ties)")
            axis.set(title=title, ylim=(-2, 102), xlim=(60, 90), ylabel="Percent")
            axis.grid(alpha=.2)
    for axis in axes[-1]:
        axis.set_xlabel("Failure-score threshold (%)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.suptitle(f"Threshold sweep / {mode} / {protocol}")
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .94), ncol=2, fontsize=8)
    figure.text(.5, .015, "Best thresholds and all ties are listed in threshold_summary.json. Validate on separate bags.",
                ha="center", fontsize=9)
    legend_rows = (len(handles) + 1) // 2
    figure.tight_layout(rect=(0, .04, 1, max(.58, .91 - .035 * legend_rows)))
    figure.savefig(path, dpi=150)
    plt.close(figure)
