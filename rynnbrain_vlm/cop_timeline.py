"""CoP score timelines: matched nominal kNN prefixes and logistic diagnostics."""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from functools import lru_cache
from zipfile import BadZipFile
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .cop_knn import CoPKNNDetector, fit_nominal_memory


PREFIX_PROTOCOL = "sample_index_prefixes_v1"
DENSE_PROTOCOL = "sample_index_capped_prefixes_v2"


def timeline_sampling(vlm: dict, full_count: int) -> dict:
    count = vlm.get("cop_classifier", {}).get("timeline_num_frames", full_count)
    if isinstance(count, bool) or not isinstance(count, int) or count < full_count:
        raise ValueError("cop_classifier.timeline_num_frames must be an integer >= rynnbrain.num_frames.")
    return {"protocol": PREFIX_PROTOCOL if count == full_count else DENSE_PROTOCOL,
            "step_count": count, "context_frames": full_count}


TIMELINE_NOTE = (
    "Each prefix is scored against nominal prefixes at the same sampled progress index. "
    "Thresholds are leave-one-bag-out nominal quantiles at that index. Alignment uses "
    "sampled progress, not robot stages or a shared wall-clock duration. This is offline "
    "analysis at sampled times, not an exact failure-onset annotation."
)


def timeline_paths(output_directory: Path, mode: str, method: str = "knn") -> tuple[Path, Path]:
    if method not in {"knn", "logistic"}:
        raise ValueError("Timeline method must be 'knn' or 'logistic'.")
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", mode).strip("_").lower()
    stem = output_directory / f"{method}_timeline_{slug}"
    return stem.with_suffix(".csv"), stem.with_suffix(".png")


def selected_frame_prefixes(frame_metadata: list[dict], image_count: int, step_count: int,
                            context_frames: int | None = None) -> list[dict]:
    """Use actual timestamps; no snapshot includes an image after its cutoff.

    Each sampled timestep contributes one point, with the cameras in that step
    and selected preceding steps. Asynchronous camera timestamps are not relabelled.
    The image order remains identical to the normal full-execution prompt.
    """
    if not frame_metadata or len(frame_metadata) != image_count:
        raise ValueError("Timeline requires a timestamp for every selected execution image.")
    times = np.asarray([float(row["timestamp_sec"]) for row in frame_metadata])
    if not np.isfinite(times).all() or (times < 0).any():
        raise ValueError("Timeline timestamps must be finite, nonnegative source times.")
    steps = [row.get("sample_index", index) for index, row in enumerate(frame_metadata)]
    if any(isinstance(value, bool) or not isinstance(value, int) for value in steps) or set(steps) != set(range(step_count)):
        raise ValueError(f"Timeline needs all {step_count} sampled timesteps to match nominal-prefix training.")
    snapshots = []
    for step in range(step_count):
        # Dense timelines retain a bounded, uniformly spaced history including
        # the current timestep. Select complete multi-camera groups, never future images.
        retained = min(step + 1, context_frames or step_count)
        selected_steps = {step} if retained == 1 else set(np.linspace(0, step, retained).round().astype(int))
        indices = [i for i, value in enumerate(steps) if value in selected_steps]
        cutoff = float(max(times[i] for i in indices))
        if snapshots and cutoff <= snapshots[-1]["timestamp_sec"]:
            raise ValueError("Timeline sampled times must strictly increase.")
        snapshots.append({"timestamp_sec": cutoff, "image_indices": indices, "sample_index": step})
    return snapshots


def _as_numpy(vector):
    return np.asarray(vector.numpy() if hasattr(vector, "numpy") else vector, dtype=np.float32)


def _prefix_cache_paths(record):
    directory = record.metadata_path.parent.parent / "cop_timeline"
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", record.metadata["input_mode"]).strip("_").lower()
    return directory / f"{slug}_prefix_vectors.npz", directory / f"{slug}_prefix_vectors.json"


def _load_prefix_cache(record, sampling=None):
    path, sidecar = _prefix_cache_paths(record)
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    signature = record.metadata["comparison_signature"]
    sampling = sampling or timeline_sampling({}, signature["num_frames"])
    with np.load(path, allow_pickle=False) as archive:
        vectors = np.asarray(archive["vectors"], dtype=np.float32)
    if (metadata.get("protocol") != sampling["protocol"]
            or metadata.get("sampling", timeline_sampling({}, signature["num_frames"])) != sampling
            or not record.metadata.get("evaluation_id")
            or metadata.get("full_evaluation_id") != record.metadata["evaluation_id"]
            or metadata.get("comparison_signature") != signature
            or vectors.shape != (sampling["step_count"], record.vector.size)
            or not np.isfinite(vectors).all()
            or not np.array_equal(vectors[-1], record.vector)):
        raise ValueError(f"Missing or stale nominal-prefix vectors for {record.metadata['bag_name']}.")
    return vectors


def prepare_nominal_prefixes(model, config, vlm, records, nominal_images, reference_response, generation, *, refresh=False):
    """Extract only missing/stale nominal prefixes, reusing full response/vector files."""
    from .run import _execution_inputs

    for record in records:
        sampling = timeline_sampling(vlm, record.metadata["comparison_signature"]["num_frames"])
        if not refresh:
            try:
                _load_prefix_cache(record, sampling)
                print(f"Reusing nominal-prefix vectors: {record.metadata['bag_name']}")
                continue
            except (ValueError, OSError, KeyError, EOFError, BadZipFile):
                pass
        signature = record.metadata["comparison_signature"]
        mode, count = record.metadata["input_mode"], sampling["step_count"]
        bag_config = {**config, "test_bag": record.metadata["test_bag"],
                      "output_dir": str(record.metadata_path.parent.parent.parent)}
        selected, _ = _execution_inputs(bag_config, vlm, count, [mode])
        images, frames = selected[mode]
        snapshots = selected_frame_prefixes(frames, len(images), count, sampling["context_frames"])
        turns = [{"role": "user", "images": nominal_images, "text": signature["turn1_prompt"]},
                 {"role": "user", "images": images, "text": signature["turn2_prompt"]}]
        vectors = []
        for snapshot in snapshots:
            if snapshot["sample_index"] == count - 1:
                vector = record.vector
            else:
                print(f"Extracting nominal prefix: {record.metadata['bag_name']} "
                      f"({snapshot['sample_index'] + 1}/{count})")
                prefix = {**turns[1], "images": [images[i] for i in snapshot["image_indices"]]}
                vector = _as_numpy(model.extract_multiturn_cop_vector(
                    [turns[0], prefix], generation, reference_response=reference_response,
                ))
            if vector.shape != record.vector.shape or not np.isfinite(vector).all():
                raise ValueError(f"Invalid prefix vector for {record.metadata['bag_name']}.")
            vectors.append(vector)
        path, sidecar = _prefix_cache_paths(record)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, vectors=np.stack(vectors))
        # The final cached vector comes from the original full execution. Its
        # source images need not equal a subset of a newly sampled dense grid.
        snapshots[-1] = {"sample_index": count - 1, "vector_source": "existing_full_execution",
                         "full_vector_file": str(record.vector_path)}
        sidecar.write_text(json.dumps({"protocol": sampling["protocol"], "sampling": sampling,
            "bag_name": record.metadata["bag_name"], "full_evaluation_id": record.metadata["evaluation_id"],
            "comparison_signature": signature, "snapshots": snapshots}, indent=2) + "\n", encoding="utf-8")


def prefix_model_path(complete_model_path: Path) -> Path:
    return complete_model_path.with_name(complete_model_path.stem + "_timeline.npz")


def _model_fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes() + path.with_suffix(".json").read_bytes()).hexdigest()


def train_prefix_detector(records, complete_model_path: Path, metadata: dict, *, timeline_num_frames=None) -> None:
    """Fit a separate nominal memory/cutoff at every sampled prefix length."""
    if any(value != "normal" for value in metadata["training_labels"].values()):
        raise ValueError("Prefix detectors require nominal training bags only.")
    # Shape: bags x progress points x feature dimensions. Cache identity is
    # tied to each complete vector's evaluation ID and full feature signature.
    options = {} if timeline_num_frames is None else {"cop_classifier": {"timeline_num_frames": timeline_num_frames}}
    sampling = timeline_sampling(options, metadata["comparison_signature"]["num_frames"])
    try:
        vectors = np.stack([_load_prefix_cache(record, sampling) for record in records])
    except (ValueError, OSError, KeyError, EOFError, BadZipFile) as error:
        raise ValueError("Nominal-prefix vectors are missing or stale. Run cop-classifier-train "
                         "without --saved-vectors-only, or run benchmark, to prepare them.") from error
    memories, thresholds = [], []
    for step in range(vectors.shape[1]):
        memory, threshold, _, _ = fit_nominal_memory(
            vectors[:, step, :], n_neighbors=metadata["n_neighbors"],
            threshold_quantile=metadata["threshold_estimation"]["quantile"],
        )
        memories.append(memory)
        thresholds.append(threshold)
    path = prefix_model_path(complete_model_path)
    np.savez_compressed(path, nominal_vectors=np.stack(memories), thresholds=np.asarray(thresholds))
    path.with_suffix(".json").write_text(json.dumps({
        "schema_version": 1, "protocol": sampling["protocol"], "sampling": sampling, "note": TIMELINE_NOTE,
        "complete_model_sha256": _model_fingerprint(complete_model_path),
        "comparison_signature": metadata["comparison_signature"],
        "training_bags": metadata["training_bags"], "n_neighbors": metadata["n_neighbors"],
        "threshold_quantile": metadata["threshold_estimation"]["quantile"],
        "step_count": len(thresholds), "thresholds": thresholds,
    }, indent=2) + "\n", encoding="utf-8")
    _load_prefix_detector.cache_clear()
    print(f"Saved nominal-prefix kNN detector: {path}")


@dataclass(frozen=True)
class PrefixDetector:
    memories: np.ndarray
    thresholds: np.ndarray
    sampling: dict

    def at_step(self, step, complete_classifier):
        return CoPKNNDetector(self.memories[step], complete_classifier.n_neighbors,
                              float(self.thresholds[step]), complete_classifier.input_mode,
                              complete_classifier.representation_id, complete_classifier.comparison_signature,
                              complete_classifier.metadata)


@lru_cache(maxsize=8)
def _load_prefix_detector(path: Path, fingerprint: str) -> PrefixDetector:
    if not path.is_file() or not path.with_suffix(".json").is_file():
        raise ValueError("Nominal-prefix detector is missing. Run benchmark or cop-classifier-train once to prepare it.")
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if metadata.get("protocol") not in (PREFIX_PROTOCOL, DENSE_PROTOCOL) or metadata.get("complete_model_sha256") != fingerprint:
        raise ValueError("Nominal-prefix detector is stale; rerun classifier preparation.")
    with np.load(path, allow_pickle=False) as archive:
        memory = np.asarray(archive["nominal_vectors"], dtype=np.float64)
        thresholds = np.asarray(archive["thresholds"], dtype=np.float64)
    count = metadata["step_count"]
    width = metadata["comparison_signature"]["hidden_size"]
    if (memory.shape != (count, len(metadata["training_bags"]), width)
            or thresholds.shape != (count,) or not np.isfinite(memory).all()
            or not np.allclose(np.linalg.norm(memory, axis=2), 1.0, atol=1e-8)
            or not np.isfinite(thresholds).all() or (thresholds <= 0).any()
            or not np.array_equal(thresholds, metadata["thresholds"])):
        raise ValueError("Invalid nominal-prefix classifier arrays.")
    sampling = metadata.get("sampling", timeline_sampling({}, metadata["comparison_signature"]["num_frames"]))
    if sampling.get("protocol") != metadata["protocol"] or sampling.get("step_count") != count:
        raise ValueError("Invalid nominal-prefix sampling metadata.")
    return PrefixDetector(memory, thresholds, sampling)


def load_prefix_detector(complete_model_path: Path, classifier, sampling=None) -> PrefixDetector:
    detector = _load_prefix_detector(prefix_model_path(complete_model_path), _model_fingerprint(complete_model_path))
    sampling = sampling or timeline_sampling({}, classifier.comparison_signature["num_frames"])
    if detector.sampling != sampling:
        raise ValueError("Timeline sampling changed; rerun cop-classifier-train to prepare matching nominal prefixes.")
    if (len(detector.thresholds) != sampling["step_count"]
            or not np.array_equal(detector.memories[-1], classifier.nominal_vectors)
            or detector.thresholds[-1] != classifier.threshold):
        raise ValueError("Nominal-prefix detector does not match the full-execution classifier.")
    return detector


def write_knn_timeline(
    *, model, classifier, turns, generation, nominal_response: str,
    frame_metadata: list[dict], final_vector, comparison_signature: dict,
    output_directory: Path, mode: str, bag_name: str, provenance: dict,
    classifier_model_path: str,
    config: dict | None = None, vlm: dict | None = None,
) -> dict:
    """Score against progress-matched nominal prefixes, preserving the final result."""
    csv_path, plot_path = timeline_paths(output_directory, mode)
    # A failed rerun must not leave a previous timeline looking current.
    csv_path.unlink(missing_ok=True)
    plot_path.unlink(missing_ok=True)
    full_count = classifier.comparison_signature["num_frames"]
    sampling = timeline_sampling(vlm or {}, full_count)
    detector = load_prefix_detector(Path(classifier_model_path), classifier, sampling)
    images, timeline_frames = turns[1]["images"], frame_metadata
    if sampling["step_count"] != full_count:
        from .run import _execution_inputs
        if config is None:
            raise ValueError("Dense timeline requires the execution's input configuration.")
        selected, _ = _execution_inputs(config, vlm, sampling["step_count"], [mode])
        images, timeline_frames = selected[mode]
    snapshots = selected_frame_prefixes(timeline_frames, len(images), len(detector.thresholds), sampling["context_frames"])
    # The final point uses the original full prompt/vector, not a resampled approximation.
    snapshots[-1] = selected_frame_prefixes(frame_metadata, len(turns[1]["images"]), full_count)[-1]
    if len(snapshots) > 1 and snapshots[-1]["timestamp_sec"] <= snapshots[-2]["timestamp_sec"]:
        raise ValueError("Dense and full-execution sampling endpoints do not match; check source timestamps.")
    # Validate the full model/adapter/prompt/frame settings before doing extras.
    final_score = classifier.predict_anomaly_score(
        _as_numpy(final_vector), input_mode=mode, comparison_signature=comparison_signature,
    )
    rows = []
    for index, snapshot in enumerate(snapshots):
        indices = snapshot["image_indices"]
        complete = index == len(snapshots) - 1
        step_classifier = detector.at_step(index, classifier)
        score, error = None, None
        print(f"kNN timeline ({mode}): snapshot {index + 1}/{len(snapshots)} "
              f"at {snapshot['timestamp_sec']:.3f}s ({len(indices)} images)")
        try:
            if complete:
                score = final_score
            else:
                partial_turns = [turns[0], {**turns[1], "images": [images[i] for i in indices]}]
                vector = model.extract_multiturn_cop_vector(
                    partial_turns, generation, reference_response=nominal_response,
                )
                score = step_classifier.predict_anomaly_score(
                    _as_numpy(vector), input_mode=mode, comparison_signature=comparison_signature,
                )
        except (RuntimeError, ValueError) as failure:
            error = str(failure)
            print(f"kNN timeline snapshot unavailable; retaining the full evaluation: {error}")
        rows.append({
            "bag_name": bag_name, "input_mode": mode,
            "evaluation_id": provenance["evaluation_id"],
            "lora_adapter_sha256": provenance.get("lora_adapter_sha256"),
            "timestamp_sec": snapshot["timestamp_sec"],
            "sample_index": index,
            "sampled_progress": index / (len(snapshots) - 1) if len(snapshots) > 1 else 1.0,
            "image_count": len(indices), "image_indices": json.dumps(indices),
            "image_timestamps_sec": json.dumps([float((frame_metadata if complete else timeline_frames)[i]["timestamp_sec"]) for i in indices]),
            "anomaly_score": score, "threshold": step_classifier.threshold,
            "above_threshold": score >= step_classifier.threshold if score is not None else None,
            "complete_execution": complete,
            "vector_source": "existing_full_execution" if complete else "prefix_prompt",
            "threshold_source": "matched_nominal_prefix_training",
            "classifier_model_path": classifier_model_path, "error": error,
        })
    dataframe = pd.DataFrame(rows)
    alert = first_alert_interval(dataframe)
    dataframe.to_csv(csv_path, index=False)
    _plot_timeline(dataframe, plot_path, bag_name, mode)
    failed = sum(row["error"] is not None for row in rows)
    return {"status": "partial" if failed else "created", "csv": str(csv_path), "plot": str(plot_path),
            "point_count": len(rows), "failed_point_count": failed,
            "protocol": sampling["protocol"], "sampling": sampling, "first_alert": alert,
            "prefix_model_path": str(prefix_model_path(Path(classifier_model_path))),
            "note": TIMELINE_NOTE}


def write_logistic_timeline(
    *, model, classifier, turns, generation, nominal_response: str,
    frame_metadata: list[dict], final_vector, comparison_signature: dict,
    output_directory: Path, mode: str, bag_name: str, provenance: dict,
    classifier_model_path: str,
) -> dict:
    """Apply the existing full-execution classifier to the original sampled prefixes.

    Bag-level failure labels do not establish when a failure began. Do not
    propagate them onto early prefixes or fit onset classifiers without temporal
    supervision. These partial-context scores are explicitly diagnostic.
    """
    csv_path, plot_path = timeline_paths(output_directory, mode, "logistic")
    csv_path.unlink(missing_ok=True)
    plot_path.unlink(missing_ok=True)
    images = turns[1]["images"]
    snapshots = selected_frame_prefixes(frame_metadata, len(images), comparison_signature["num_frames"])
    final_score = classifier.predict_failure_probability(
        _as_numpy(final_vector), input_mode=mode, comparison_signature=comparison_signature,
    )
    rows = []
    for index, snapshot in enumerate(snapshots):
        indices = snapshot["image_indices"]
        complete = index == len(snapshots) - 1
        score, error = None, None
        print(f"Logistic timeline ({mode}): snapshot {index + 1}/{len(snapshots)} "
              f"at {snapshot['timestamp_sec']:.3f}s ({len(indices)} images)")
        try:
            if complete:
                score = final_score
            else:
                prefix = {**turns[1], "images": [images[i] for i in indices]}
                vector = model.extract_multiturn_cop_vector(
                    [turns[0], prefix], generation, reference_response=nominal_response,
                )
                score = classifier.predict_failure_probability(
                    _as_numpy(vector), input_mode=mode, comparison_signature=comparison_signature,
                )
        except (RuntimeError, ValueError) as failure:
            error = str(failure)
            print(f"Logistic timeline snapshot unavailable; retaining the full evaluation: {error}")
        rows.append({
            "bag_name": bag_name, "input_mode": mode, "evaluation_id": provenance["evaluation_id"],
            "lora_adapter_sha256": provenance.get("lora_adapter_sha256"),
            "timestamp_sec": snapshot["timestamp_sec"], "sample_index": index,
            "image_count": len(indices), "image_indices": json.dumps(indices),
            "image_timestamps_sec": json.dumps([float(frame_metadata[i]["timestamp_sec"]) for i in indices]),
            "score_kind": "failure_probability", "anomaly_score": score, "failure_probability": score,
            "failure_percent": score * 100 if score is not None else None,
            "threshold": classifier.threshold, "threshold_percent": classifier.threshold * 100,
            "above_threshold": score >= classifier.threshold if score is not None else None,
            "complete_execution": complete,
            "vector_source": "existing_full_execution" if complete else "prefix_prompt",
            "threshold_source": "full_execution_classifier",
            "classifier_model_path": classifier_model_path, "error": error,
        })
    dataframe = pd.DataFrame(rows)
    dataframe.to_csv(csv_path, index=False)
    _plot_timeline(dataframe, plot_path, bag_name, mode, probability=True)
    failed = sum(row["error"] is not None for row in rows)
    return {"status": "partial" if failed else "created", "csv": str(csv_path), "plot": str(plot_path),
            "point_count": len(rows), "failed_point_count": failed,
            "protocol": "full_execution_logistic_on_prefixes_v1",
            "score_kind": "failure_probability", "first_alert": first_alert_interval(dataframe),
            "note": "The same complete-execution logistic classifier scores every prefix. "
                    "Prefix probabilities are diagnostic and are not calibrated failure-onset estimates. "
                    "No onset labels or prefix training are used; the last point matches the complete score."}


def first_alert_interval(dataframe: pd.DataFrame) -> dict:
    """Bound a sampled score crossing; never call it physical failure onset."""
    scores = dataframe.anomaly_score.to_numpy(dtype=float)
    cutoffs = dataframe.threshold.to_numpy(dtype=float)
    times = dataframe.timestamp_sec.to_numpy(dtype=float)
    above = np.flatnonzero(np.isfinite(scores) & (scores >= cutoffs))
    if not len(above):
        return {"status": "not_observed", "last_below_sec": None, "first_above_sec": None}
    index = int(above[0])
    below = np.flatnonzero(np.isfinite(scores[:index]) & (scores[:index] < cutoffs[:index]))
    previous = int(below[-1]) if len(below) else None
    return {"status": "bracketed" if previous is not None else "no_prior_below",
            "last_below_sec": float(times[previous]) if previous is not None else None,
            "first_above_sec": float(times[index]),
            "unavailable_samples_in_interval": int(np.sum(~np.isfinite(scores[(previous + 1 if previous is not None else 0):index]))),
            "note": "Interval brackets the first observed threshold crossing, not the physical failure onset."}


def _plot_timeline(dataframe: pd.DataFrame, plot_path: Path, bag_name: str, mode: str, *, probability=False) -> None:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    times = dataframe.timestamp_sec.to_numpy(dtype=float)
    scale = 100 if probability else 1
    scores = dataframe.anomaly_score.to_numpy(dtype=float) * scale
    thresholds = dataframe.threshold.to_numpy(dtype=float) * scale
    above = np.isfinite(scores) & (scores >= thresholds)
    figure, axis = plt.subplots(figsize=(10, 5.6))
    try:
        figure.subplots_adjust(left=0.11, right=0.97, top=0.81, bottom=0.26)
        title = "Logistic failure score over time" if probability else "kNN anomaly score over time"
        figure.text(0.11, 0.94, title, fontsize=18, weight="bold", color="#172b40")
        figure.text(0.11, 0.875, f"{bag_name}  /  {mode}  /  {len(times)} sampled snapshots", fontsize=10, color="#607080")
        axis.plot(times, scores, "o-", color="#2f78c4", linewidth=1.8, markersize=5,
                  label="Failure probability" if probability else "Anomaly distance")
        axis.plot(times, thresholds, "s--", color="#c27524", linewidth=1.6, markersize=4,
                  label="Classifier threshold" if probability else "Nominal threshold (matched progress)")
        if above.any():
            axis.scatter(times[above], scores[above], color="#c44850", s=45, zorder=4, label="Above threshold")
        alert = first_alert_interval(dataframe)
        if alert["status"] == "bracketed":
            start, end = alert["last_below_sec"], alert["first_above_sec"]
            axis.axvspan(start, end, color="#c44850", alpha=0.09,
                         label=f"First observed crossing: {start:.1f}–{end:.1f} s")
        elif alert["status"] == "no_prior_below":
            axis.axvline(alert["first_above_sec"], color="#c44850", alpha=0.6, linestyle=":",
                         label="First alert; no earlier below-threshold sample")
        missing = ~np.isfinite(scores)
        if missing.any():
            axis.scatter(times[missing], np.zeros(missing.sum()), transform=axis.get_xaxis_transform(),
                         marker="x", color="#777777", clip_on=False, label="Snapshot unavailable")
        axis.set(xlabel="Time from bag / video start (s)",
                 ylabel="Failure probability (%)" if probability else "Anomaly distance",
                 ylim=(0, 103 if probability else None))
        if probability:
            axis.set_yticks(np.arange(0, 101, 20))
        if len(times) == 1:
            axis.set_xlim(times[0] - 0.5, times[0] + 0.5)
        axis.grid(axis="y", color="#e7ecf0", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(loc="best", fontsize=9, frameon=False)
        figure.text(0.11, 0.12, "Offline diagnostic: each point uses only selected execution images available by that time.",
                    fontsize=9, color="#607080")
        note = ("The classifier was trained on complete executions. Prefix probabilities are diagnostic,\n"
                "not calibrated estimates of whether a failure has already started." if probability else
                "Thresholds use nominal prefixes at the same sampled progress index, not robot-stage alignment.\n"
                "Lines connect evaluated snapshots only; they do not locate the exact failure onset.")
        figure.text(0.11, 0.065, note, fontsize=8.5, color="#607080")
        figure.savefig(plot_path, dpi=180, bbox_inches="tight", facecolor="white")
    finally:
        plt.close(figure)
