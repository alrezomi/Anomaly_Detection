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
              "mean_population": "Complete aligned timelines with a recorded failure marker and an observed alert; one row per evaluated bag and input mode.",
              "note": "Positive is late; negative is an early alert. These are offline sampled alert times, not runtime latency or exact physical failure onset.",
              "modes": []}
    for mode in dict.fromkeys(input_modes):
        selected = [row for row in rows if row.get("input_mode") in (mode, None)]
        data = pd.DataFrame([{ "bag_name": row.get("bag_name"), "input_mode": mode,
                              "ground_truth_label": row.get("ground_truth_label"),
                              **{key: row.get(key) for key in TIMING_FIELDS},
                              "failure_timing_status": row.get("failure_timing_status") or "not_evaluated"}
                             for row in selected], columns=["bag_name", "input_mode", "ground_truth_label", *TIMING_FIELDS])
        compared = data[data.failure_timing_status == "compared"]
        errors = pd.to_numeric(compared.failure_time_error_sec, errors="coerce").to_numpy(dtype=float)
        errors = errors[np.isfinite(errors)]
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", mode).strip("_").lower()
        stem = directory / f"failure_timing_{slug}_{method}"
        csv_path, plot_path = stem.with_suffix(".csv"), stem.with_suffix(".png")
        data.to_csv(csv_path, index=False)
        stats = {"input_mode": mode, "method": method, "evaluated_bags": len(data),
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

    figure, (events, errors) = plt.subplots(1, 2, figsize=(13, max(5.5, 3.0 + .27 * len(data))),
                                           sharey=True, gridspec_kw={"width_ratios": [1.1, 1]})
    try:
        figure.subplots_adjust(left=.22, right=.97, bottom=.24, top=.80, wspace=.23)
        figure.text(.04, .95, "Failure timing against recorded error button", fontsize=18, weight="bold", color="#172b40")
        mean = stats["mean_signed_error_sec"]
        subtitle = (f"Mean absolute error: {stats['mean_absolute_error_sec']:.2f} s    "
                    f"Mean signed error: {mean:+.2f} s" if mean is not None else "No comparable detections; timing means are unavailable")
        figure.text(.04, .90, f"{stats['method']} / {stats['input_mode']}    {subtitle}", fontsize=10, color="#607080")
        positions = np.arange(len(data))
        recorded = pd.to_numeric(data.recorded_failure_time_sec, errors="coerce").to_numpy(dtype=float)
        predicted = pd.to_numeric(data.predicted_failure_time_sec, errors="coerce").to_numpy(dtype=float)
        deltas = pd.to_numeric(data.failure_time_error_sec, errors="coerce").to_numpy(dtype=float)
        compared = (data.failure_timing_status == "compared").to_numpy() & np.isfinite(deltas)
        for index in np.flatnonzero(np.isfinite(recorded) & np.isfinite(predicted)):
            events.plot([recorded[index], predicted[index]], [index, index], color="#b5c1cc", linewidth=2, zorder=1)
        events.scatter(recorded, positions, color="#27875f", s=38, marker="|", linewidths=2.5, label="Recorded failure", zorder=3)
        events.scatter(predicted, positions, color="#2f78c4", s=30, label="First classifier alert", zorder=3)
        events.set_yticks(positions, data.bag_name.tolist())
        events.tick_params(axis="y", labelsize=8)
        events.set(xlabel="Time from bag start (s)", title="Recorded time and first observed alert", xlim=(0, None))
        events.legend(loc="best", fontsize=8, frameon=False)
        colors = np.where(deltas[compared] < 0, "#2f78c4", "#c27524")
        errors.barh(positions[compared], deltas[compared], color=colors, height=.6)
        errors.axvline(0, color="#8292a0", linewidth=1)
        extent = max(1., float(np.max(np.abs(deltas[compared]))) if compared.any() else 1.) * 1.45
        errors.set(xlim=(-extent, extent), xlabel="Timing error (s): early < 0 / late > 0", title="Prediction minus recorded time")
        if mean is not None:
            errors.axvline(mean, color="#772d7c", linestyle="--", linewidth=1.5, label=f"Mean signed error: {mean:+.2f} s")
            errors.legend(loc="best", fontsize=8, frameon=False)
        for index, row in enumerate(data.itertuples()):
            if compared[index]:
                value = deltas[index]
                errors.annotate(f"{value:+.2f}", (value, index), xytext=(4 if value >= 0 else -4, 0),
                                textcoords="offset points", ha="left" if value >= 0 else "right", va="center", fontsize=8)
            else:
                errors.text(0, index, "  " + str(row.failure_timing_status).replace("_", " "), va="center", fontsize=8, color="#607080",
                            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
        for axis in (events, errors):
            axis.set_ylim(max(len(data), 1) - .5, -.5)
            axis.grid(axis="x", color="#e7ecf0", linewidth=.7)
            axis.set_axisbelow(True)
            axis.spines[["top", "right"]].set_visible(False)
        if data.empty:
            events.text(.5, .5, "No evaluated bags", transform=events.transAxes, ha="center")
        excluded = stats["evaluated_bags"] - stats["compared_bags"]
        figure.text(.04, .10, f"Mean population: {stats['compared_bags']} comparable detections / {stats['evaluated_bags']} evaluated bags; "
                    f"{excluded} without a valid timing comparison. Missing alerts are not zero-error detections.", fontsize=9, color="#607080")
        figure.text(.04, .045, "Reference = recorded button press. Prediction = first sampled classifier threshold crossing.\n"
                    "Offline timing comparison; this does not measure inference latency or interpolate physical failure onset.", fontsize=9, color="#607080")
        figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    finally:
        plt.close(figure)
