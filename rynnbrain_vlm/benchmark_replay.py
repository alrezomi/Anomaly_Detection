"""Select a threshold on causal replay scores, then export only its full reports."""

from copy import deepcopy
from datetime import datetime, timezone
import io
import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from .replay_policy import replay_settings, observation_interval
from .runtime import clock, feature_only_enabled, write_json


def threshold_search_enabled(selection):
    enabled = selection.get("threshold_search", False)
    if not isinstance(enabled, bool):
        raise ValueError("vlm_benchmark.threshold_search must be true or false.")
    return enabled


def validate_threshold_search(config, decision_mode, run_dino, skip_vlm):
    vlm = config["rynnbrain"]
    if decision_mode != "timeline" or skip_vlm or run_dino:
        raise ValueError("threshold_search requires timeline VLM evaluation without --with-dino.")
    if vlm.get("source") != "rosbag" or vlm.get("input_modes", ["raw"]) != ["raw"]:
        raise ValueError("threshold_search currently requires raw rosbag inputs, like browser VLM replay.")
    if vlm.get("cop_classifier", {}).get("method", "logistic") != "logistic":
        raise ValueError("threshold_search requires a logistic failure-probability classifier.")
    if not feature_only_enabled(vlm):
        raise ValueError("Set rynnbrain.inference.feature_only=true for sequential threshold selection.")


def _scan_bag(detector, config, bag, cache_path, frame_iterator=None):
    from rosbag_io import RosbagImageSource, iter_rosbag_image_frames

    vlm = config["rynnbrain"]
    policy = replay_settings(config)
    constant_interval = float(vlm.get("online_replay", {}).get("interval_sec", vlm.get("execution_interval_sec", 10.)))
    if not np.isfinite(constant_interval) or constant_interval <= 0:
        raise ValueError("online_replay.interval_sec must be finite and positive.")
    due = float(vlm.get("sampling_start_sec") or 0.)
    end = vlm.get("sampling_end_sec")
    limit = policy["max_observations"]
    frames = (iter(frame_iterator) if frame_iterator is not None else iter(iter_rosbag_image_frames(
        RosbagImageSource(bag, vlm["camera_topics"][0]), sample_fps=15., start_sec=due, end_sec=end)))
    rows, vectors, images = [], [], {}
    error, input_sec, processing_sec = None, 0., 0.
    bag_started = clock()
    detector.reset(bag)
    detector.capture_visual = False
    try:
        while limit is None or len(rows) < limit:
            started = clock()
            try:
                frame_id, timestamp, image = next(frames)
            except StopIteration:
                input_sec += clock() - started
                break
            input_sec += clock() - started
            if timestamp < due:
                continue
            if end is not None and timestamp > float(end):
                break
            started = clock()
            row, vector, _ = detector.append(image, float(timestamp), verify=False)
            row["processing_sec"] = clock() - started
            processing_sec += row["processing_sec"]
            score = row["anomaly_score"]
            if score is None or not np.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("Sequential threshold search requires a valid score at every observation.")
            interval = observation_interval(policy, score, constant_interval)
            row.update(frame_id=int(frame_id), input_elapsed_sec=input_sec,
                       processing_elapsed_sec=processing_sec, next_interval_sec=interval)
            rows.append(row)
            vectors.append(np.asarray(vector, dtype=np.float32))
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            images[f"image_{len(rows) - 1}"] = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
            row["bag_elapsed_sec"] = clock() - bag_started
            due = float(timestamp) + interval
            print(f"  {Path(bag).name}: observation {len(rows)}, t={timestamp:.3f}s, "
                  f"score={score:.1%}, next interval={interval:g}s", flush=True)
    except Exception as failure:
        error = f"{type(failure).__name__}: {failure}"
    finally:
        if hasattr(frames, "close"):
            frames.close()
    if not rows and error is None:
        error = "No eligible observations in the configured replay window."
    metadata = dict(rows=rows, error=error, complete=error is None,
                    input_sec=input_sec, processing_sec=processing_sec)
    np.savez_compressed(cache_path, vectors=np.stack(vectors) if vectors else np.empty((0, 0)),
        metadata_json=np.frombuffer(json.dumps(metadata, allow_nan=False).encode(), dtype=np.uint8), **images)
    return metadata


def _choose_threshold(report):
    overall = [item for item in report["comparisons"] if item["input_mode"] == "raw"
               and item["benchmark_decision_mode"] == "timeline" and item["failure_category"] == "all_failures"]
    if len(overall) != 1 or overall[0]["status"] != "created":
        raise ValueError("Best-threshold selection requires both nominal and failure bags with complete valid trajectories. "
                         "Selection scores/reports are saved; no best-threshold reports were generated.")
    # Lower cutoffs never reduce recall on the same trajectories. Keep every
    # accuracy tie in the report, and choose the lowest one deterministically.
    return min(overall[0]["best_accuracy_thresholds"]), overall[0]


def _read_timing_annotation(bag, config):
    from .visual_matching import read_corrected_failure_annotation
    try:
        return read_corrected_failure_annotation(Path(bag), config.get('stage_topic', '/recording_stage'))
    except Exception as error:
        return dict(status='unavailable', time_sec=None, error=f'{type(error).__name__}: {error}')


def _export_bag(detector, config, record, cache_path, output, threshold, startup_ignore_sec, *, full_csv_directory=None):
    from .cop_analysis import save_cop_vector
    from .cop_timeline import _plot_timeline, first_alert_interval
    from .failure_timing import evaluate_failure_timing
    from .run import write_multiturn_outputs

    vlm = config["rynnbrain"]
    bag = Path(record["bag_path"])
    policy = replay_settings(config)
    with np.load(cache_path, allow_pickle=False) as archive:
        saved = json.loads(archive["metadata_json"].tobytes())
        full = saved["rows"]
        evaluation_id = str(uuid4())
        alerts = [i for i, row in enumerate(full) if row["anomaly_score"] >= threshold]
        count = alerts[0] + 1 if alerts and policy["stop_on_alarm"] else len(full)
        prefix = [dict(row) for row in full[:count]]
        for i, row in enumerate(full):
            row.update(threshold=threshold, threshold_percent=threshold * 100,
                       above_threshold=row["anomaly_score"] >= threshold,
                       failure_probability=row["anomaly_score"], failure_percent=row["anomaly_score"] * 100,
                       sample_index=i, image_count=i + 1, complete_execution=i == len(full) - 1,
                       vector_source="cached_replay_prompt", score_kind="failure_probability",
                       classifier_model_path=str(detector.path), error=None,
                       bag_name=record["bag_name"], input_mode="raw", evaluation_id=evaluation_id,
                       lora_adapter_sha256=detector.model.adapter_identity,
                       threshold_source=getattr(detector, "score_source", "cached_replay_classifier"),
                       image_indices=json.dumps(list(range(i + 1))),
                       image_timestamps_sec=json.dumps([observation["timestamp_sec"] for observation in full[:i + 1]]))
        for i, row in enumerate(prefix):
            row.update(full[i], stop_at_selected_threshold=bool(alerts and i == alerts[0]),
                       next_interval_sec=None if alerts and i == alerts[0] and policy["stop_on_alarm"] else row["next_interval_sec"])
        directory = output / record["bag_name"] / "rynnbrain_multiturn"
        directory.mkdir(parents=True, exist_ok=True)
        timeline_csv = directory / "logistic_timeline_raw.csv"
        plot = directory / "logistic_timeline_raw.png"
        pd.DataFrame(prefix).to_csv(timeline_csv, index=False)
        full_csv = ((Path(full_csv_directory) / cache_path.name) if full_csv_directory else cache_path).with_suffix(".csv")
        pd.DataFrame(full).to_csv(full_csv, index=False)
        label = record["label"]
        peak = max((row["anomaly_score"] for row in full), default=None) if saved["complete"] else None
        selected_peak = max((observation["anomaly_score"] for observation in prefix), default=None)
        decision = "failure" if alerts else "success" if saved["complete"] else None
        expected = {"normal": "success", "fail": "failure"}.get(label)
        runtime = dict(runtime_input_sec=prefix[-1]["input_elapsed_sec"] if prefix else 0., runtime_reference_sec=0.,
            runtime_model_sec=sum(row["cached_feature_sec"] for row in prefix),
            runtime_classifier_sec=sum(row["classifier_sec"] for row in prefix), runtime_diagnostics_sec=0.)
        runtime["runtime_bag_sec"] = prefix[-1]["bag_elapsed_sec"] if prefix else 0.
        row = dict(bag_name=record["bag_name"], bag_path=str(bag), folder=record["folder"], ground_truth_label=label,
            input_mode="raw", benchmark_decision_mode="timeline", decision="not_generated", decision_correct=None,
            confidence=None, response=None, response_source="not_generated", evaluation_id=evaluation_id,
            generated_at_utc=datetime.now(timezone.utc).isoformat(), model_id=vlm["model"].get("model_id"),
            lora_adapter_path=vlm["model"].get("lora_adapter_path"),
            lora_adapter_sha256=detector.model.adapter_identity, inference_mode="feature_only",
            evaluation_error=None if decision else saved["error"], classifier_error=None if decision else saved["error"],
            classifier_decision=decision, classifier_decision_correct=decision == expected if decision and expected else None,
            classifier_failure_probability=selected_peak if decision else None,
            classifier_failure_percent=selected_peak * 100 if selected_peak is not None and decision else None,
            classifier_threshold=threshold, classifier_score_kind="failure_probability", classifier_model_path=str(detector.path),
            timeline_first_alarm_time_sec=full[alerts[0]]["timestamp_sec"] if alerts else None,
            timeline_peak_score=selected_peak, threshold_selection_peak_score=peak,
            roc_ranking_source="complete_uncensored_selection_trajectory", timeline_valid_points=len(prefix),
            timeline_total_points=len(prefix), timeline_error=saved["error"],
            logistic_timeline_status="created" if saved["complete"] else "partial",
            logistic_timeline_csv=str(timeline_csv), logistic_timeline_plot=str(plot),
            logistic_timeline_error=saved["error"], **runtime, runtime_total_bag_sec=runtime["runtime_bag_sec"],
            runtime_frame_count=len(prefix), runtime_frame_sec_total=sum(r["processing_sec"] for r in prefix),
            runtime_frame_sec_mean=float(np.mean([r["processing_sec"] for r in prefix])) if prefix else None,
            runtime_feature_frame_sec_mean=float(np.mean([r["cached_feature_sec"] for r in prefix])) if prefix else None,
            runtime_classifier_frame_sec_mean=float(np.mean([r["classifier_sec"] for r in prefix])) if prefix else None,
            stop_reason="alarm_threshold" if alerts and policy["stop_on_alarm"] else "end_of_observation_window")
        metadata = []
        for i, observation in enumerate(prefix):
            image_path = directory / "selected_inputs" / f"frame_{i + 1:03d}.png"
            image_path.parent.mkdir(exist_ok=True)
            image_path.write_bytes(archive[f"image_{i}"].tobytes())
            metadata.append(dict(topic=vlm["camera_topics"][0], sample_index=i,
                frame_id=observation["frame_id"], timestamp_sec=observation["timestamp_sec"], image_file=str(image_path)))
            if getattr(detector, "timestamp_inputs", True):
                metadata[-1]["model_label"] = (f"Frame {i + 1}, camera {vlm['camera_topics'][0]}, "
                                                f"time {observation['timestamp_sec']:.3f} s from this bag start:")
            else:
                from .run import _slug
                metadata[-1]["model_label"] = f"Time {i + 1}, camera {_slug(vlm['camera_topics'][0])}:"
        if prefix:
            vector_path, sidecar = save_cop_vector(directory, "raw", archive["vectors"][count - 1],
                {**row, "test_bag": str(bag), "comparison_signature": detector.signature,
                 "vector_timestamp_sec": prefix[-1]["timestamp_sec"], "vector_failure_probability": prefix[-1]["anomaly_score"]})
            row.update(cop_vector_path=str(vector_path), cop_metadata_path=str(sidecar))
    annotation = _read_timing_annotation(bag, config)
    # Evaluate early alarms against the complete scan window, even if the chosen
    # replay stops before the eventual recorded failure marker.
    row.update(evaluate_failure_timing(annotation,
        dict(status="created" if saved["complete"] else "partial", csv=str(full_csv)),
        source="rosbag", mode="raw", ground_truth=label, method="logistic"))
    if prefix:
        _plot_timeline(pd.DataFrame(prefix), plot, record["bag_name"], "raw", probability=True, failure_timing=row)
    else:
        from .report_artifacts import unavailable_plot
        unavailable_plot(plot, f"Timeline / {record['bag_name']}", saved["error"] or "No observations")
    raw_record = {**row, "nominal_response": detector.response, "comparison_signature": detector.signature,
                  "turns": [{"role": "user", "prompt": detector.signature["turn1_prompt"], "response": detector.response},
                            {"role": "user", "prompt": detector.signature["turn2_prompt"], "response": None,
                             "images": metadata}],
                  "logistic_timeline": dict(status=row["logistic_timeline_status"], csv=str(timeline_csv),
                                            plot=str(plot), first_alert=first_alert_interval(pd.DataFrame(prefix)) if prefix else None)}
    write_multiturn_outputs(directory, [row], metadata, [raw_record], vlm.get("task_description", ""))
    return row


def run_threshold_search(config, records, root, detector, update, startup_ignore_sec, *, select_threshold=True):
    selection = root / ("threshold_selection" if select_threshold else "replay_observations")
    selection.mkdir(parents=True, exist_ok=True)
    if select_threshold and not {"normal", "fail"}.issubset({record["label"] for record in records}):
        raise ValueError("threshold_search requires held-out bags labeled both normal and fail; "
                         "add a genuine failure category before running model inference.")
    output = root / "best_threshold" if select_threshold else root
    if select_threshold and output.exists() and any(output.iterdir()):
        raise ValueError(f"Best-threshold output already exists: {output}. Use a fresh output_dir or --benchmark-dir.")
    print("Sequential replay: same cached detector and interval rules as GUI, no real-time waits. "
          "Full scans retain scores for ROC; exported trajectories follow the configured stop policy.", flush=True)
    peaks, scans = [], []
    for index, record in enumerate(records, 1):
        bag = Path(record["bag_path"])
        cache = selection / f"{record['bag_name']}.npz"
        update(phase="threshold_scoring", current_bag=record["bag_name"], completed_bags=index - 1)
        scan = _scan_bag(detector, config, bag, cache)
        scans.append((record, cache, scan))
        peaks.append(dict(bag_name=record["bag_name"], bag_path=str(bag), input_mode="raw",
            ground_truth_label=record["label"], benchmark_decision_mode="timeline", decision="not_generated",
            classifier_failure_probability=max((row["anomaly_score"] for row in scan["rows"]), default=None)
                if scan["complete"] else None, classifier_threshold=float(detector.classifier.threshold),
            evaluation_error=scan["error"]))
        pd.DataFrame(peaks).to_csv(selection / "benchmark_summary.csv", index=False)
    if select_threshold:
        return _export_threshold_reports(config, records, selection, output, detector, scans, peaks,
                                         update, startup_ignore_sec)
    threshold = float(detector.classifier.threshold)
    results = [_export_bag(detector, config, record, cache, output, threshold, startup_ignore_sec,
                          full_csv_directory=selection) for record, cache, scan in scans]
    effective = deepcopy(config)
    effective["vlm_benchmark"]["train_lora"] = False
    effective["rynnbrain"]["cop_classifier"]["train_on_benchmark"] = False
    write_json(output / "pipeline_config.json", effective)
    write_json(output / "benchmark_protocol.json", dict(decision_mode="timeline", threshold_search=False,
               threshold=threshold, execution_interval_sec=config["rynnbrain"].get("execution_interval_sec"),
               note="Cached causal replay; full scans support ROC, outputs follow configured stopping policy."))
    update(completed_bags=len(results), benchmark_output_dir=str(output))
    return results, output, effective


def _export_threshold_reports(config, records, selection, output, detector, scans, peaks, update, startup_ignore_sec):
    from .threshold_sweep import write_threshold_sweep_report
    from .benchmark_roc import write_probability_roc_report
    from .temporal_threshold import write_temporal_cost_report

    selection.mkdir(parents=True, exist_ok=True)
    report = write_threshold_sweep_report(peaks, selection, input_modes=["raw"])
    accuracy_threshold, overall = _choose_threshold(report)
    annotations = {record['bag_name']: _read_timing_annotation(record['bag_path'], config)
                   for record in records if record['label'] == 'fail'}
    temporal = write_temporal_cost_report(scans, annotations, selection)
    if temporal['status'] != 'created':
        raise ValueError('Cannot select a timing-cost threshold: complete histories with known failure onsets '
                         'are required. See benchmark_temporal_cost/cost_summary.json; original reports remain available.')
    threshold = temporal['selected_threshold']
    threshold_rows = pd.read_csv(report['csv'])
    selected_accuracy = float(threshold_rows.loc[(threshold_rows.input_mode == 'raw') &
        (threshold_rows.benchmark_decision_mode == 'timeline') & (threshold_rows.failure_category == 'all_failures') &
        np.isclose(threshold_rows.threshold, threshold), 'accuracy'].iloc[0])
    choice = dict(threshold=threshold, threshold_percent=round(threshold * 100),
        selection_method='balanced_normalized_first_alarm_cost',
        accuracy=selected_accuracy, tied_thresholds=temporal['tied_thresholds'],
        minimum_cost=temporal['minimum_cost'], cost_definition=temporal,
        accuracy_only_threshold=accuracy_threshold, accuracy_only_best=overall['best_accuracy'],
        accuracy_only_tied_thresholds=overall['best_accuracy_thresholds'],
        tie_rule=temporal['tie_rule'],
        observation_policy=replay_settings(config),
        sampling="First recorded frame at or after the next due bag timestamp; no wall-clock waits or future frames.",
        ranking="ROC uses complete uncensored scan peaks; chosen-threshold timelines stop at the first alarm when requested.",
        validation_note="Threshold selected on these bags; use separate test bags to estimate generalization.")
    write_json(selection / "selected_threshold.json", choice)
    for row in peaks:
        row["classifier_threshold"] = threshold
    pd.DataFrame(peaks).to_csv(selection / "benchmark_summary.csv", index=False)
    write_probability_roc_report(peaks, selection, input_modes=["raw"])
    print(f"Selected threshold: {threshold:.0%}; timing cost={temporal['minimum_cost']:.4f}; "
          f"observed bag accuracy={selected_accuracy:.1%}. Accuracy-only cutoff={accuracy_threshold:.0%}. "
          "Generating one full set of outputs from saved observations.", flush=True)
    update(phase="best_threshold_outputs", selected_threshold=threshold, current_bag=None)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output / "benchmark_bag_selection.csv", index=False)
    effective = deepcopy(config)
    effective["vlm_benchmark"]["threshold_search"] = False
    effective["vlm_benchmark"]["train_lora"] = False
    effective["rynnbrain"]["cop_classifier"].setdefault("training", {})["threshold"] = threshold
    effective["rynnbrain"]["cop_classifier"]["train_on_benchmark"] = False
    write_json(output / "pipeline_config.json", effective)
    write_json(output / "selected_threshold.json", choice)
    results = [_export_bag(detector, config, record, cache, output, threshold, startup_ignore_sec,
                           full_csv_directory=selection)
               for record, cache, scan in scans]
    # Keep the cost graph and audit alongside all existing best-threshold reports.
    write_temporal_cost_report(scans, annotations, output)
    write_json(output / "benchmark_protocol.json", {**choice, "decision_mode": "timeline", "threshold_search": True})
    update(completed_bags=len(results), benchmark_output_dir=str(output))
    return results, output, effective


def refresh_saved_threshold_reports(root):
    """Re-select and export from saved scans without loading a model or classifier."""
    from types import SimpleNamespace
    from .runtime import record_run
    from run_benchmark import _write_benchmark_reports

    root = Path(root)
    previous = root / "best_threshold"
    config = json.loads((previous / "pipeline_config.json").read_text(encoding="utf-8"))
    records = pd.read_csv(previous / "benchmark_bag_selection.csv", keep_default_na=False).to_dict("records")
    peaks = pd.read_csv(root / "threshold_selection" / "benchmark_summary.csv").to_dict("records")
    output = root / "best_threshold_timing"
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Refreshed reports already exist: {output}. Existing reports were not changed.")
    scans = []
    provenance = None
    timestamp_inputs = True
    for record in records:
        cache = root / "threshold_selection" / f"{record['bag_name']}.npz"
        with np.load(cache, allow_pickle=False) as saved:
            scan = json.loads(saved["metadata_json"].tobytes())
        scans.append((record, cache, scan))
        if provenance is None and scan["rows"]:
            response_path = previous / record["bag_name"] / "rynnbrain_multiturn" / "rynnbrain_responses_multiturn.json"
            response = json.loads(response_path.read_text(encoding="utf-8"))
            provenance = response["results"][0]
            selected = response.get("selected_raw_frames", [])
            if selected:
                timestamp_inputs = selected[0].get("model_label", "").startswith("Frame ")
    if provenance is None:
        raise ValueError("No saved observation provenance is available; cannot export threshold reports.")
    detector = SimpleNamespace(model=SimpleNamespace(adapter_identity=provenance.get("lora_adapter_sha256")),
        path=Path(provenance["classifier_model_path"]), signature=provenance["comparison_signature"],
        response=provenance["nominal_response"], timestamp_inputs=timestamp_inputs,
        score_source="saved_cached_replay_classifier")
    if len(peaks) != len(records) or {row["bag_name"] for row in peaks} != {row["bag_name"] for row in records}:
        raise ValueError("Saved selection scores do not match the saved bag selection.")
    selection = root / "threshold_selection_timing"
    with record_run(selection, "report_refresh") as update:
        print("Selecting a timing-cost threshold over 0-100% from saved scans. No model inference or training.", flush=True)
        results, output, effective = _export_threshold_reports(config, records, selection, output, detector,
            scans, peaks, update, config.get("startup_ignore_sec", .1))
        folders = config.get("vlm_benchmark", {}).get("folders", [])
        _write_benchmark_reports(SimpleNamespace(skip_vlm=False), effective, effective["rynnbrain"], results,
                                 output, folders, records, "timeline", update)
    return results, output, effective


def write_frame_runtime_report(rows, root):
    table = pd.DataFrame([{key: row.get(key) for key in ("bag_name", "ground_truth_label", "runtime_frame_count",
        "runtime_frame_sec_total", "runtime_frame_sec_mean", "runtime_feature_frame_sec_mean", "runtime_classifier_frame_sec_mean")}
        for row in rows])
    table.to_csv(root / "benchmark_frame_runtime.csv", index=False)
    count = int(table.runtime_frame_count.fillna(0).sum())
    total = float(table.runtime_frame_sec_total.fillna(0).sum())
    summary = dict(observations=count, total_processing_sec=total,
                   average_processing_sec_per_frame=total / count if count else None,
                   definition="Observation-weighted cached feature extraction plus classifier processing; "
                   "excludes model/reference initialization, frame decoding, report export and waiting.")
    write_json(root / "benchmark_frame_runtime.json", summary)
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(max(9, len(rows) * .22), 5))
    axis.bar(table.bag_name, table.runtime_frame_sec_mean, color="#2f78c4")
    if count:
        axis.axhline(total / count, color="#c27524", linestyle="--", label=f"Weighted mean {total / count:.3f} s/frame")
        axis.legend()
    axis.set(ylabel="Processing seconds per observation", title="Average cached VLM processing time per frame")
    axis.tick_params(axis="x", labelrotation=75)
    figure.tight_layout()
    figure.savefig(root / "benchmark_frame_runtime.png", dpi=150)
    plt.close(figure)
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Refresh a completed threshold search from saved observations; no model inference.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Completed benchmark timeline directory")
    refresh_saved_threshold_reports(parser.parse_args().input_dir)
