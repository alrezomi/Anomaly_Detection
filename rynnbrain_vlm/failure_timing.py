"""Evaluate classifier alert times against recorded error-button annotations."""

from collections import Counter
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd


TIMING_FIELDS = (
    "recorded_failure_time_sec", "predicted_failure_time_sec",
    "failure_time_error_sec", "failure_time_absolute_error_sec",
    "failure_timing_status", "failure_timing_error", "failure_timing_method",
)


def read_failure_annotation(bag_path, topic="/recording_stage", startup_ignore_sec=0.1):
    """First failure marker after startup, independent of the final-three label heuristic."""
    result = {"status": "unavailable", "time_sec": None, "marker": None, "marker_count": 0,
              "topic": topic, "startup_ignore_sec": startup_ignore_sec, "error": None}
    try:
        from build_dataset_manifest import FAIL_PATTERN
        from rosbag_io import read_stage_events

        if not np.isfinite(startup_ignore_sec) or startup_ignore_sec < 0:
            raise ValueError("Stage startup-ignore interval must be finite and nonnegative.")
        events = read_stage_events(bag_path, topic)
        if events.empty:
            return {**result, "status": "missing_marker"}
        times = pd.to_numeric(events["time"], errors="coerce")
        if not np.isfinite(times).all() or (times < 0).any():
            raise ValueError("Stage markers contain invalid bag-relative timestamps.")
        failures = events.loc[(times >= startup_ignore_sec) & events["stage"].astype(str).map(
            lambda value: FAIL_PATTERN.search(value) is not None)].copy()
        if failures.empty:
            return {**result, "status": "missing_marker"}
        failures["time"] = times.loc[failures.index]
        failures = failures.sort_values("time", kind="stable")
        first = failures.iloc[0]
        return {**result, "status": "available", "time_sec": float(first["time"]),
                "marker": str(first["stage"]), "marker_count": len(failures),
                "time_ns": int(first["time_ns"]) if "time_ns" in first else None}
    except Exception as error:
        return {**result, "error": f"{type(error).__name__}: {error}"}


def evaluate_failure_timing(annotation, timeline, *, source, mode, ground_truth, method):
    """Score observed threshold crossings, with no interpolation or label leakage."""
    result = dict.fromkeys(TIMING_FIELDS)
    result.update(recorded_failure_time_sec=annotation.get("time_sec"),
                  failure_timing_method=method, failure_timing_status="timeline_unavailable")
    # MP4 presentation timestamps are not automatically synchronized to ROS bag time.
    # In mixed modes some inputs also use video time. Never silently subtract clocks.
    if source != "rosbag" or mode != "raw":
        return {**result, "failure_timing_status": "unaligned_timebase",
                "failure_timing_error": "Timing evaluation requires raw rosbag frames on the stage topic's bag-relative clock."}
    if timeline.get("status") not in {"created", "partial"} or not timeline.get("csv"):
        return {**result, "failure_timing_error": timeline.get("error")}
    try:
        data = pd.read_csv(timeline["csv"])
        times = pd.to_numeric(data.timestamp_sec, errors="coerce").to_numpy(dtype=float)
        scores = pd.to_numeric(data.anomaly_score, errors="coerce").to_numpy(dtype=float)
        thresholds = pd.to_numeric(data.threshold, errors="coerce").to_numpy(dtype=float)
        if not len(times) or not np.isfinite(times).all() or (times < 0).any() or (np.diff(times) <= 0).any():
            raise ValueError("Timeline timestamps must be finite, nonnegative, and strictly increasing.")
        valid = np.isfinite(scores) & np.isfinite(thresholds)
        alerts = np.flatnonzero(valid & (scores >= thresholds))
        predicted = float(times[alerts[0]]) if len(alerts) else None
        result["predicted_failure_time_sec"] = predicted
        if annotation["status"] == "unavailable":
            return {**result, "failure_timing_status": "annotation_unavailable",
                    "failure_timing_error": annotation.get("error")}
        if annotation["status"] != "available":
            status = "missing_annotation"
            if ground_truth in {"normal", "nominal"}:
                status = "false_alert" if predicted is not None else ("normal_no_alert" if valid.all() else "partial_timeline")
            return {**result, "failure_timing_status": status}
        if ground_truth in {"normal", "nominal"}:
            return {**result, "failure_timing_status": "label_conflict",
                    "failure_timing_error": "A recorded failure marker conflicts with the bag's nominal evaluation label."}
        recorded = float(annotation["time_sec"])
        if recorded < times[0] or recorded > times[-1]:
            return {**result, "failure_timing_status": "annotation_outside_window"}
        # An unavailable prefix could hide an earlier alert. Keep its observed
        # time for inspection but exclude incomplete timelines from timing means.
        if not valid.all() or timeline["status"] == "partial":
            return {**result, "failure_timing_status": "partial_timeline"}
        if predicted is None:
            return {**result, "failure_timing_status": "not_detected"}
        delta = predicted - recorded
        return {**result, "failure_timing_status": "compared", "failure_time_error_sec": delta,
                "failure_time_absolute_error_sec": abs(delta)}
    except Exception as error:
        return {**result, "failure_timing_status": "invalid_timeline",
                "failure_timing_error": f"{type(error).__name__}: {error}"}


def attach_failure_timing(annotation, timeline, *, source, mode, ground_truth, method, bag_name):
    """Add the recorded event to the existing timeline plot; never change predictions."""
    result = evaluate_failure_timing(annotation, timeline, source=source, mode=mode,
                                     ground_truth=ground_truth, method=method)
    if (source == "rosbag" and mode == "raw" and timeline.get("plot")
            and annotation.get("status") == "available"):
        try:
            from .cop_timeline import _plot_timeline
            _plot_timeline(pd.read_csv(timeline["csv"]), Path(timeline["plot"]), bag_name, mode,
                           probability=method == "logistic", failure_timing=result)
        except Exception as error:
            # Timing numbers remain valid if the optional plot overlay fails.
            result["failure_timing_error"] = f"Timing overlay unavailable: {error}"
    return result


def write_failure_timing_report(rows, output_directory: Path, *, input_modes, method):
    """One timing graph per mode/method; means include only comparable complete runs."""
    directory = output_directory / "benchmark_failure_timing"
    directory.mkdir(parents=True, exist_ok=True)
    report = {"definition": "signed error = first classifier alert time - first recorded failure-marker time",
              "mean_population": "Failure-labeled bags with complete aligned timelines, a recorded failure marker and an observed alert; one row per bag and input mode.",
              "note": "Positive is late; negative is an early alert. These are offline sampled alert times, not runtime latency or exact physical failure onset.",
              "modes": []}
    for mode in dict.fromkeys(input_modes):
        selected = [row for row in rows if row.get("input_mode") in (mode, None)]
        data = pd.DataFrame([{ "bag_name": row.get("bag_name"), "input_mode": mode,
                              "ground_truth_label": row.get("ground_truth_label"),
                              **{key: row.get(key) for key in TIMING_FIELDS},
                              "failure_timing_status": row.get("failure_timing_status") or "not_evaluated"}
                             for row in selected], columns=["bag_name", "input_mode", "ground_truth_label", *TIMING_FIELDS])
        failures = data.ground_truth_label.astype(str).str.lower().isin(["fail", "failure"])
        compared = data[failures & (data.failure_timing_status == "compared")]
        errors = pd.to_numeric(compared.failure_time_error_sec, errors="coerce").to_numpy(dtype=float)
        errors = errors[np.isfinite(errors)]
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", mode).strip("_").lower()
        stem = directory / f"failure_timing_{slug}_{method}"
        csv_path, plot_path = stem.with_suffix(".csv"), stem.with_suffix(".png")
        data.to_csv(csv_path, index=False)
        stats = {"input_mode": mode, "method": method, "evaluated_bags": len(data),
                 "failure_bags": int(failures.sum()),
                 "compared_bags": len(errors), "status_counts": dict(Counter(data.failure_timing_status)),
                 "mean_signed_error_sec": float(errors.mean()) if len(errors) else None,
                 "mean_absolute_error_sec": float(np.abs(errors).mean()) if len(errors) else None,
                 "median_absolute_error_sec": float(np.median(np.abs(errors))) if len(errors) else None,
                 "early_alerts": int((errors < 0).sum()), "late_alerts": int((errors > 0).sum()),
                 "exact_matches": int((errors == 0).sum()), "csv": str(csv_path), "plot": str(plot_path)}
        _plot_timing_report(data, stats, plot_path)
        report["modes"].append(stats)
    (directory / "failure_timing_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def _plot_timing_report(data, stats, path):
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    import matplotlib.patheffects as effects
    import textwrap

    failures = data.ground_truth_label.astype(str).str.lower().isin(["fail", "failure"])
    values = pd.to_numeric(data.failure_time_error_sec, errors="coerce")
    plotted = data[failures & (data.failure_timing_status == "compared") & np.isfinite(values)]
    deltas = pd.to_numeric(plotted.failure_time_error_sec).to_numpy(dtype=float)
    labels = ["\n".join(textwrap.wrap(str(name).replace("_", " ") if len(str(name)) > 38 else str(name), width=38))
              for name in plotted.bag_name]
    row_heights = np.asarray([max(1., .65 * (label.count("\n") + 1)) for label in labels])
    positions = np.cumsum(row_heights) - row_heights / 2
    total_rows = max(float(row_heights.sum()), 1.)
    height = max(4.8, 2.6 + .30 * total_rows)
    label_width = max((max(map(len, label.splitlines())) for label in labels), default=10)
    left = max(.16, min(.34, .075 + label_width * .0064))
    figure, axis = plt.subplots(figsize=(12, height))
    try:
        # Use physical margins so a long list does not create huge empty headers/footers.
        figure.subplots_adjust(left=left, right=.97, bottom=1.2 / height, top=1 - 1.3 / height)
        figure.text(.04, 1 - .33 / height, "Failure detection timing error", fontsize=19, weight="bold", color="#172b40")
        mean = stats["mean_signed_error_sec"]
        method_label = "kNN" if stats["method"] == "knn" else "Logistic regression"
        subtitle = (f"{len(plotted)} failure cases   |   Mean absolute error: {stats['mean_absolute_error_sec']:.2f} s"
                    if mean is not None else "No failure cases with a valid timing comparison")
        figure.text(.04, 1 - .65 / height, f"{method_label} / {stats['input_mode']}   |   {subtitle}", fontsize=11, color="#607080")
        early_color, late_color, mean_color = "#377eb8", "#d68739", "#7c2e86"
        axis.barh(positions, deltas, color=np.where(deltas < 0, early_color, late_color), height=.66, zorder=3)
        axis.axvline(0, color="#7c8995", linewidth=1.1, zorder=2)
        low = min(0., float(deltas.min())) if len(deltas) else 0.
        high = max(0., float(deltas.max())) if len(deltas) else 0.
        padding = max(1., high - low) * .18
        axis.set(xlim=(low - padding, high + padding), ylim=(total_rows + .25, -.25),
                 xlabel="Predicted time − recorded error time (s)")
        axis.set_yticks(positions, labels)
        axis.tick_params(axis="y", labelsize=9, length=0, pad=9)
        axis.tick_params(axis="x", labelsize=10, colors="#536273")
        axis.xaxis.label.set_size(11)
        handles = [Patch(facecolor=early_color, label="Early alert (−)"), Patch(facecolor=late_color, label="Late alert (+)")]
        if mean is not None:
            axis.axvline(mean, color=mean_color, linestyle="--", linewidth=2.5, zorder=5,
                         path_effects=[effects.Stroke(linewidth=4, foreground="white"), effects.Normal()])
            handles.append(Line2D([0], [0], color=mean_color, linestyle="--", linewidth=2.5,
                                  label=f"Mean difference: {mean:+.2f} s"))
        figure.legend(handles=handles, loc="upper left", bbox_to_anchor=(.033, 1 - .83 / height),
                      ncol=3, frameon=False, fontsize=10, handlelength=2.8, columnspacing=2.3)
        for position, value in zip(positions, deltas):
            axis.annotate(f"{value:+.2f}", (value, position), xytext=(5 if value >= 0 else -5, 0),
                          textcoords="offset points", ha="left" if value >= 0 else "right", va="center", fontsize=9,
                          color="#344454", zorder=6,
                          path_effects=[effects.Stroke(linewidth=2.5, foreground="white"), effects.Normal()])
        axis.grid(axis="x", color="#e7ecf0", linewidth=.8)
        axis.set_axisbelow(True)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.spines["bottom"].set_color("#c5ced6")
        if not len(plotted):
            axis.text(.5, .5, "No valid failure timing differences to plot", transform=axis.transAxes,
                      ha="center", color="#607080")
        excluded = int(failures.sum()) - len(plotted)
        figure.text(.04, .26 / height, f"Failure cases only. {excluded} failure cases without a valid timing difference are excluded from the mean.\n"
                    "All bag results, including nominal cases and missed detections, remain in the CSV report.", fontsize=9, color="#607080")
        figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    finally:
        plt.close(figure)
