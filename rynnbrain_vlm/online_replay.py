"""Persistent detector replay with cached VLM features and a web UI.

Run with python -m rynnbrain_vlm.online_replay --config ... --interval-sec 10.
VLM samples the latest available image up to eight times. DINO processes every
image at its native sampling rate; GMR replays every prepared signal sample.
"""

import argparse
import csv
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import math
import os
from pathlib import Path
import queue
import signal
import threading
import time
import traceback
from uuid import uuid4

import numpy as np

from pipeline_settings import load_pipeline_config


SAMPLING_NOTE = ("Live elapsed-time sampling differs from whole-bag training sampling. "
                 "Scores are diagnostic; online accuracy and thresholds are not yet validated.")


def replay_bags(config, explicit=None, data_root=None):
    """Offer recorded bags without changing the shared single-test config."""
    if explicit:
        return list(dict.fromkeys(Path(bag).resolve() for bag in explicit))
    selected = config.get("test_bag")
    if not isinstance(selected, str) or not selected.strip():
        raise ValueError("Keep test_bag as one bag-path string. Replay automatically lists ROS bags "
                         "under the mounted data folder; do not replace test_bag with a list.")
    root = Path(data_root if data_root is not None else os.environ.get("ANOMALY_DATA_ROOT", "/data"))
    discovered = sorted({path.parent.resolve() for path in root.glob("**/metadata.yaml")})
    return list(dict.fromkeys([Path(selected).resolve(), *discovered]))


def signature_differences(saved, current, prefix="comparison_signature"):
    """Explain the existing strict comparison without weakening compatibility."""
    missing = object()
    def display(value):
        if value is missing:
            return "<missing>"
        text = json.dumps(value, ensure_ascii=False)
        return text if len(text) <= 240 else text[:237] + "..."
    if isinstance(saved, dict) and isinstance(current, dict):
        differences = []
        for key in sorted(saved.keys() | current.keys()):
            before, after = saved.get(key, missing), current.get(key, missing)
            path = f"{prefix}.{key}"
            if before is missing or after is missing:
                differences.append(f"{path}: saved={display(before)}; current={display(after)}")
            else:
                differences.extend(signature_differences(before, after, path))
        return differences
    if saved == current:
        return []
    return [f"{prefix}: saved={display(saved)}; current={display(current)}"]


def validate_online_classifier(classifier, signature, representation_id, path):
    differences = signature_differences(classifier.comparison_signature, signature)
    differences += signature_differences(classifier.representation_id, representation_id, "representation_id")
    if differences:
        raise ValueError(
            f"Saved classifier does not match the current feature extractor: {path}\n"
            + "\n".join("- " + item for item in differences)
            + "\nBoth kNN and logistic replay are supported. Use the feature settings saved with this classifier, "
              "or train a new classifier and its prefix companion with the current settings in a new output path. "
              "Replay does not train models; train_on_benchmark applies only to benchmarks.")


def finish_verification(check, cached_score, uncached_score, threshold):
    """Keep independent feature, score and decision checks visible."""
    difference = abs(uncached_score - cached_score)
    check.update(vector_within_tolerance=bool(check["within_tolerance"]),
                 cached_score=cached_score, uncached_score=uncached_score,
                 score_difference=difference, score_tolerance=.001,
                 score_within_tolerance=difference <= .001,
                 cached_above_threshold=cached_score >= threshold,
                 uncached_above_threshold=uncached_score >= threshold,
                 decisions_match=(cached_score >= threshold) == (uncached_score >= threshold))
    check["failed_checks"] = [name for name, key in (
        ("feature_vector", "vector_within_tolerance"), ("classifier_score", "score_within_tolerance"),
        ("threshold_decision", "decisions_match")) if not check[key]]
    check["within_tolerance"] = not check["failed_checks"]
    return check


def verification_failure_message(check):
    detail = []
    if "relative_l2_difference" in check:
        detail.append(f"feature difference {check['relative_l2_difference']:.3%} "
                      f"(limit {check['relative_l2_tolerance']:.3%})")
    if "score_difference" in check:
        detail.append(f"score difference {check['score_difference']:.6g} (limit {check['score_tolerance']:.6g})")
    if "decisions_match" in check:
        detail.append("decisions match" if check["decisions_match"] else "DECISIONS DIFFER")
    return "Cached/uncached verification failed" + (": " + "; ".join(detail) if detail else "")


class ReplayFeed:
    """A single latest-frame mailbox, not a queue of camera frames."""
    def __init__(self, frames, publish, stop, clock=time.monotonic):
        self.frames, self.publish, self.stop, self.clock = frames, publish, stop, clock
        self.condition = threading.Condition()
        self.latest = None
        self.finished = False
        self.error = None
        self.origin = clock()

    def run(self):
        try:
            for frame_id, timestamp, image in self.frames:
                if not math.isfinite(timestamp) or timestamp < 0:
                    raise ValueError("Replay frame has an invalid timestamp.")
                if self.stop.wait(max(0., self.origin + timestamp - self.clock())):
                    break
                with self.condition:
                    if self.latest and timestamp <= self.latest[1]:
                        continue
                    self.latest = (frame_id, timestamp, image)
                    self.condition.notify_all()
                self.publish(image, timestamp)
        except Exception as error:
            self.error = str(error)
        finally:
            if hasattr(self.frames, "close"):
                self.frames.close()
            with self.condition:
                self.finished = True
                self.condition.notify_all()

    def next_frame(self, earliest):
        with self.condition:
            while not self.stop.is_set():
                if self.error:
                    raise RuntimeError(self.error)
                if self.latest is not None and self.latest[1] >= earliest:
                    return self.latest
                if self.finished:
                    return None
                self.condition.wait(.1)
        return None


def synchronized_time():
    import torch
    if torch.cuda.is_available():
        for device in range(torch.cuda.device_count()):
            torch.cuda.synchronize(device)
    return time.monotonic()


class SequentialReplayFeed:
    """Pace every native detector sample without dropping samples when scoring lags.

    Unlike the VLM latest-frame mailbox this advances on demand, so memory is
    bounded and a slow detector makes replay slower instead of skipping inputs.
    An optional fourth packet field gives the actual accompanying camera time.
    """
    def __init__(self, frames, publish, stop, clock=time.monotonic, allow_equal_timestamps=False):
        self.frames, self.publish, self.stop, self.clock = iter(frames), publish, stop, clock
        self.origin = clock()
        self.previous = None
        self.camera_time = None
        self.allow_equal_timestamps = allow_equal_timestamps

    def run(self):
        pass  # Consumption and pacing happen in next_frame, not a producer thread.

    def next_frame(self, earliest):
        if self.stop.is_set():
            return None
        packet = next(self.frames, None)
        if packet is None:
            return None
        frame_id, timestamp, image = packet[:3]
        if not math.isfinite(timestamp) or timestamp < 0 or (self.previous is not None and
                (timestamp < self.previous or (timestamp == self.previous and not self.allow_equal_timestamps))):
            raise ValueError('Replay samples must have finite, strictly increasing bag timestamps.')
        if self.stop.wait(max(0., self.origin + timestamp - self.clock())):
            return None
        self.previous = timestamp
        camera_time = packet[3] if len(packet) == 4 else timestamp
        camera_stamp = (image.info.get('dino_camera_topic'), camera_time) if image is not None else camera_time
        if image is not None and camera_stamp != self.camera_time:
            self.publish(image, camera_time)
            self.camera_time = camera_stamp
        return frame_id, timestamp, image

    def close(self):
        if hasattr(self.frames, 'close'):
            self.frames.close()


def gmr_sample_frames(frames, timestamps):
    """Pair every GMR sample with the latest camera image at or before its time.

    The signal clock drives scoring, even before/after camera coverage. No image
    is required to score force/torque, and no signal sample is silently dropped.
    """
    frames = iter(frames)
    try:
        upcoming = next(frames, None)
        latest = None
        for timestamp in timestamps:
            while upcoming is not None and upcoming[1] <= timestamp:
                latest = upcoming
                upcoming = next(frames, None)
            yield (latest[0] if latest else None, float(timestamp),
                   latest[2] if latest else None, latest[1] if latest else None)
    finally:
        if hasattr(frames, 'close'):
            frames.close()


class OnlineDetector:
    """Load once, reset execution cache between bags, never train artifacts."""
    def __init__(self, config, progress=lambda message: None):
        from .model import RynnBrainModel, COP_REPRESENTATION_ID
        from .cop_classifier import classifier_model_paths, classifier_method, load_classifier
        from .cop_knn import CoPKNNDetector
        from .cop_timeline import load_prefix_detector
        from .cop_onset import onset_enabled, load_onset_classifier
        from .run import _raw_inputs, cop_comparison_signature
        from .temporal_lora import timestamped_images
        from .prompts import task_context_prompt, evaluation_prompt_multiturn
        from .online_cache import CachedExecution
        from .visual_matching import nominal_matching_bags

        self.config = config
        self.vlm = vlm = config["rynnbrain"]
        if vlm.get("source") != "rosbag" or vlm.get("input_modes", ["raw"]) != ["raw"]:
            raise ValueError("Online replay currently requires raw rosbag inputs.")
        if len(vlm["camera_topics"]) != 1:
            raise ValueError("The first online version supports one execution camera.")
        self.frame_count = int(vlm.get("num_frames", 8))
        if not 1 <= self.frame_count <= 8:
            raise ValueError("Online replay supports at most eight selected frames per execution.")
        settings = vlm.get("cop_classifier", {})
        if not settings.get("enabled", False):
            raise ValueError("Enable a trained cop_classifier for online replay.")
        self.path = Path(classifier_model_paths(settings)["raw"])
        self.complete = load_classifier(self.path)
        self.knn = isinstance(self.complete, CoPKNNDetector)
        if self.knn != (classifier_method(settings) == "knn") or self.complete.input_mode != "raw":
            raise ValueError("Saved classifier method/input mode does not match the config.")
        self.response = self.complete.comparison_signature.get("nominal_response")
        if not isinstance(self.response, str):
            raise ValueError("The classifier must contain its saved nominal-reference response.")
        progress("Loading VLM and LoRA adapter")
        self.model = RynnBrainModel(vlm["model"])
        generation = vlm.get("generation", {})
        self.signature = cop_comparison_signature(self.model, config, vlm, "raw", self.response,
                                                 self.frame_count, generation)
        validate_online_classifier(self.complete, self.signature, COP_REPRESENTATION_ID, self.path)
        self.prefix = load_prefix_detector(self.path, self.complete) if self.knn else None
        self.onset = onset_enabled(settings)
        self.classifier = self.complete
        if self.onset:
            self.classifier, _ = load_onset_classifier(self.path, self.complete, bag_name="")
        self.score_source = ("nominal_prefix_index_diagnostic" if self.knn else
                             "onset_prefix_classifier" if self.onset else "whole_execution_classifier_on_prefixes")
        self.excluded = set(self.model.adapter_training_bags)
        self.excluded.update(self.complete.metadata.get("training_bags", []))
        self.excluded.update(self.classifier.metadata.get("training_bags", []))
        self.excluded.update(Path(value).name for value in vlm.get("reference_bags", []))
        self.excluded.update(Path(value).name for value in nominal_matching_bags(config))
        self.visual_memory = None
        self.capture_visual = False
        self.matching_stages = None
        progress("Reading nominal reference images")
        nominal = []
        temporal = bool(self.model.temporal_profile)
        for bag in vlm.get("reference_bags", []):
            images, metadata = _raw_inputs(Path(bag), vlm["memory_camera_topics"], self.frame_count)
            if temporal:
                images = timestamped_images(images, metadata, vlm["memory_camera_topics"], self.frame_count)
            nominal.extend(images)
        if not nominal:
            raise ValueError("No nominal reference images were loaded.")
        task = vlm.get("task_description", "Robot manipulation task")
        progress("Preparing reusable reference cache (first model inference)")
        self.cache = CachedExecution(self.model,
            dict(role="user", images=nominal, text=task_context_prompt(task, temporal=temporal)),
            self.response, evaluation_prompt_multiturn(task, "raw", temporal=temporal), generation,
            self.frame_count)

    def reset(self, bag):
        if Path(bag).name in self.excluded:
            raise ValueError("This bag was used for LoRA/classifier training or as the reference; select a held-out bag.")
        self.cache.reset()
        self.last_timestamp = None

    def prepare_matching(self, bag, progress, stop):
        from .visual_matching import read_recorded_stages, build_visual_memory
        self.capture_visual = False
        self.matching_stages = None
        if self.config.get("stage_count", 3) != 3 or self.config.get("ignore_recorded_stages", False):
            raise ValueError("Stage preview requires three recorded stages and ignore_recorded_stages=false.")
        stages = read_recorded_stages(bag, self.config.get("stage_topic", "/recording_stage"))
        if self.visual_memory is None:
            self.visual_memory = build_visual_memory(self, progress, stop)
        self.matching_stages = stages
        self.capture_visual = True
        return {"status": "ready", "stage_annotation": stages.metadata(), "memory": self.visual_memory.report}

    def score(self, vector, step):
        classifier = self.prefix.at_step(step, self.complete) if self.knn else self.classifier
        predict = classifier.predict_anomaly_score if self.knn else classifier.predict_failure_probability
        score = float(predict(vector, input_mode="raw", comparison_signature=self.signature))
        return score, float(classifier.threshold)

    def append(self, image, timestamp, verify=False):
        import torch
        from .feature_verification import vector_difference
        from .run import _slug
        if not math.isfinite(timestamp) or timestamp < 0 or (
                self.last_timestamp is not None and timestamp <= self.last_timestamp):
            raise ValueError("Observation timestamps must increase.")
        index = len(self.cache.images)
        topic = self.vlm["camera_topics"][0]
        label = (f"Frame {index + 1}, camera {topic}, time {timestamp:.3f} s from this bag start:"
                 if self.model.temporal_profile else f"Time {index + 1}, camera {_slug(topic)}:")
        started = synchronized_time()
        if torch.cuda.is_available():
            for device in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(device)
        vector = self.cache.append(label, image, capture_visual=self.capture_visual).numpy()
        feature_sec = synchronized_time() - started
        memory = {str(device): torch.cuda.max_memory_allocated(device)
                  for device in range(torch.cuda.device_count())} if torch.cuda.is_available() else {}
        started = time.monotonic()
        score, threshold = self.score(vector, index)
        classifier_sec = time.monotonic() - started
        matching = {"status": "disabled"}
        matching_started = time.monotonic()
        if self.capture_visual:
            try:
                descriptor = self.cache.last_visual_descriptor
                if descriptor is None:
                    raise ValueError(self.cache.last_visual_error or "No visual descriptor was captured.")
                matching = self.visual_memory.match(descriptor, self.matching_stages.at(timestamp))
            except (ValueError, RuntimeError) as error:
                matching = {"status": "unavailable", "reason": str(error)}
        matching_sec = time.monotonic() - matching_started
        check = None
        baseline = None
        if verify:
            started = synchronized_time()
            baseline = self.cache.uncached_vector().numpy()
            uncached_sec = synchronized_time() - started
            check = vector_difference(baseline, vector)
            baseline_score, _ = self.score(baseline, index)
            check["uncached_feature_sec"] = uncached_sec
            finish_verification(check, score, baseline_score, threshold)
        self.last_timestamp = timestamp
        return dict(sample_index=index, timestamp_sec=timestamp, anomaly_score=score,
                    threshold=threshold, above_threshold=score >= threshold,
                    cached_feature_sec=feature_sec, classifier_sec=classifier_sec,
                    peak_gpu_allocated_bytes=memory,
                    score_kind="knn_distance" if self.knn else "failure_probability",
                    visual_matching=matching, visual_matching_sec=matching_sec,
                    verification=check), vector, baseline


def save_plot(rows, path, probability):
    if not rows:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    scale = 100 if probability else 1
    figure, axis = plt.subplots(figsize=(10, 5))
    times = [row["timestamp_sec"] for row in rows]
    axis.plot(times, [row["anomaly_score"] * scale for row in rows], "o-", markersize=2 if len(rows) > 50 else 4,
              label="Score", color="#2f78c4")
    axis.plot(times, [row["threshold"] * scale for row in rows], "--", label="Threshold", color="#c27524")
    fallback = any(row.get("score_computation") == "uncached_verification_fallback" for row in rows)
    kind = rows[0].get('score_kind', '')
    title = (('GMR causal anomaly scores' if rows[0].get('score_computation') == 'causal_ts' else 'GMR offline score playback') if kind in {'gmr_epsilon', 'gmr_distance'} else
             'DINO per-image replay' if kind == 'dino_distance' else
             'Online replay · uncached fallback on failed checks' if fallback else 'Online replay · cached execution history')
    axis.set(xlabel="Observation time from bag start (s)",
             ylabel="Failure score (%)" if probability else 'Smoothed epsilon' if kind == 'gmr_epsilon' else "Anomaly distance",
             title=title)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(alpha=.15)
    axis.legend(frameon=False)
    caption = ("Joint distance and nominal adaptive limit. Filtered alarm uses trailing epsilon smoothing and voting."
               if kind == 'gmr_distance' else
               "Elapsed-time replay; offline-trained scores require online validation. Lines connect scored observations.")
    figure.text(.1, .01, caption, fontsize=8)
    figure.tight_layout(rect=(0, .04, 1, 1))
    figure.savefig(path, dpi=160)
    plt.close(figure)


class ReplayApplication:
    def __init__(self, config, bags, interval, verify, diagnose=False):
        self.config, self.bags, self.interval = config, bags, interval
        self.verify, self.diagnose = bool(verify or diagnose), diagnose
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.shutdown = threading.Event()
        self.jobs = queue.Queue(maxsize=1)
        self.state = dict(status="loading", rows=[], bags=[str(bag) for bag in bags],
                          note=SAMPLING_NOTE, verification_mode=self.verify, diagnostic_mode=diagnose,
                          warning=None, cache_verification_status="not_requested", video_time_sec=None,
                          interval_sec=interval, output_directory=None)
        self.state["replay_mode"] = "diagnostic" if diagnose else "strict" if verify else "cached"
        self.state["initialization_phase"] = "Starting model worker"
        self.state["excluded_bag_indices"] = []
        self.state["match_preview"] = True
        self.state["matching_status"] = "Not prepared"
        self.state['detector_type'] = 'vlm'
        self.state['requested_detector'] = 'vlm'
        self.state['gmr_signal'] = 'pose'
        self.jpeg = None
        self.matched_jpeg = None
        self.observed_jpeg = None
        self.camera_jpegs = {}
        self.state['dino_cameras'] = []
        self.state['camera_times'] = {}
        self.worker = threading.Thread(target=self._worker, daemon=True)

    def update(self, **values):
        with self.lock:
            self.state.update(values)

    def snapshot(self):
        with self.lock:
            return json.loads(json.dumps({**self.state, "model_ready": bool(getattr(self, "detector", None))}))

    def start(self, index, interval=None, mode=None, match_preview=None, detector_type=None, gmr_signal=None):
        with self.lock:
            if self.state["status"] not in {"ready", "completed", "stopped", "error"}:
                raise ValueError("Wait for initialization or stop the current execution first.")
            detector_type = detector_type or self.state['detector_type']
            if detector_type not in {'vlm', 'dino', 'gmr'}:
                raise ValueError('Choose VLM, DINO or GMR online TS.')
            gmr_signal = self.state['gmr_signal'] if gmr_signal is None else gmr_signal
            if not isinstance(gmr_signal, str) or gmr_signal not in {'pose', 'joints', 'gripper', 'pose_gripper', 'force'}:
                raise ValueError('Select TCP pose, joint positions, gripper, pose + gripper, or force/torque.')
            switching = detector_type != self.state['detector_type']
            if type(index) is not int or not 0 <= index < len(self.bags):
                raise ValueError("Select a listed bag.")
            if not switching and index in self.state["excluded_bag_indices"]:
                raise ValueError("This is a training/reference bag; select a held-out execution.")
            interval = self.interval if interval is None else interval
            mode = self.state["replay_mode"] if mode is None else mode
            if type(interval) not in (int, float) or not math.isfinite(interval) or interval <= 0:
                raise ValueError("Observation interval must be a finite positive number of seconds.")
            if mode not in ("cached", "diagnostic", "strict"):
                raise ValueError("Select cached replay, diagnostics, or strict verification.")
            if detector_type != 'vlm':
                mode = 'cached'
            match_preview = self.state["match_preview"] if match_preview is None else match_preview
            if not isinstance(match_preview, bool):
                raise ValueError("Stage match preview must be true or false.")
            self.interval = float(interval)
            self.verify, self.diagnose = mode != "cached", mode == "diagnostic"
            self.stop.clear()
            self.jpeg = None
            self.matched_jpeg = self.observed_jpeg = None
            self.camera_jpegs = {}
            self.state.update(status="starting", rows=[], error=None, video_time_sec=None,
                              output_directory=None, warning=None, cache_verification_status="pending" if self.verify else "not_requested",
                              bag=str(self.bags[index]), interval_sec=self.interval,
                              match_preview=match_preview, matching_status="Not prepared",
                              requested_detector=detector_type,
                              gmr_signal=gmr_signal,
                              signal_description=None, signal_columns=[],
                              detector_preparation_sec=None,
                              camera_times={},
                              replay_mode=mode, verification_mode=self.verify, diagnostic_mode=self.diagnose)
            self.jobs.put_nowait(index)

    def publish(self, image, timestamp):
        preview = image.copy()
        preview.thumbnail((960, 640))
        buffer = io.BytesIO()
        preview.save(buffer, format="JPEG", quality=80)
        with self.lock:
            self.jpeg = buffer.getvalue()
            self.state["video_time_sec"] = timestamp
            topic = image.info.get('dino_camera_topic')
            if topic:
                self.camera_jpegs.setdefault(topic, {})['frame'] = self.jpeg
                self.state['camera_times'][topic] = timestamp

    def _worker(self):
        def progress(message):
            self.update(initialization_phase=message)
            print(f"Initialization: {message}", flush=True)
        try:
            progress("Loading inference libraries")
            started = synchronized_time()
            self.detector = OnlineDetector(self.config, progress=progress)
            excluded = getattr(self.detector, "excluded", set())
            excluded_indices = [index for index, bag in enumerate(self.bags) if Path(bag).name in excluded]
            self.update(status="ready", initialization_sec=synchronized_time() - started,
                        initialization_phase="Ready — choose a bag and click Start execution",
                        score_source=self.detector.score_source,
                        score_kind="knn_distance" if self.detector.knn else "failure_probability",
                        excluded_bag_indices=excluded_indices,
                        max_frames=self.detector.frame_count)
            print(f"Replay selector: {len(self.bags) - len(excluded_indices)} available bags; "
                  f"{len(excluded_indices)} training/reference bags disabled.", flush=True)
            print("Model ready. Waiting for Start execution in the browser; no bag runs automatically.", flush=True)
        except Exception as error:
            self.update(status="error", error=str(error))
            traceback.print_exc()
        while not self.shutdown.is_set():
            try:
                index = self.jobs.get(timeout=.2)
            except queue.Empty:
                continue
            try:
                kind = self.state['requested_detector']
                if kind != self.state['detector_type'] or not getattr(self, 'detector', None):
                    self.switch_detector(kind)
                self.run_bag(self.bags[index])
            except Exception as error:
                self.update(status="error", error=str(error))
                traceback.print_exc()

    def switch_detector(self, kind):
        """Release the previous model before loading another GPU detector."""
        import gc
        self.detector = None
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.update(status='starting', detector_type=kind, excluded_bag_indices=[],
                    dino_cameras=[],
                    initialization_phase=f'Loading {kind.upper()}')
        def progress(message):
            self.update(initialization_phase=message, matching_status=message)
            print(message, flush=True)
        if kind == 'vlm':
            detector = OnlineDetector(self.config, progress=progress)
        else:
            from .dino_cameras import DinoCameraReplay
            from time_series_gmr_scripts.online_gmr import GmrOnlineDetector
            detector = (DinoCameraReplay if kind == 'dino' else GmrOnlineDetector)(self.config, progress)
        self.detector = detector
        self.update(note=getattr(detector, 'note', SAMPLING_NOTE), score_source=detector.score_source,
                    dino_cameras=detector.catalogue() if kind == 'dino' else [],
                    score_kind=getattr(detector, 'score_kind', 'knn_distance' if detector.knn else 'failure_probability'),
                    max_frames=detector.frame_count,
                    excluded_bag_indices=[i for i, bag in enumerate(self.bags) if Path(bag).name in detector.excluded])

    def run_bag(self, bag):
        from rosbag_io import RosbagImageSource, iter_rosbag_image_frames
        if self.state['detector_type'] == 'gmr':
            self.detector.signal_mode = self.state['gmr_signal']
        self.detector.reset(bag)
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
        output = Path(self.config["output_dir"]) / "online_replay" / run_id / Path(bag).stem
        output.mkdir(parents=True, exist_ok=False)
        vlm = self.config["rynnbrain"]
        kind = self.state['detector_type']
        causal_gmr = kind == 'gmr' and hasattr(self.detector, 'create_feed')
        if causal_gmr:
            self.update(gmr_stage=0)
        preparation_metadata = {}
        if hasattr(self.detector, 'prepare_replay'):
            def preparation_progress(message):
                self.update(status='preparing', matching_status=message, output_directory=str(output))
                print(message, flush=True)
            started = time.monotonic()
            try:
                preparation_metadata = self.detector.prepare_replay(preparation_progress, self.stop)
            except Exception as failure:
                status = 'stopped' if isinstance(failure, InterruptedError) else 'error'
                (output / 'online_results.json').write_text(json.dumps(dict(bag=str(bag), detector_type=kind,
                    status=status, error=str(failure), rows=[]), indent=2), encoding='utf-8')
                self.update(status=status, error=None if status == 'stopped' else str(failure))
                return
            preparation_metadata['duration_sec'] = time.monotonic() - started
            self.update(detector_preparation_sec=preparation_metadata['duration_sec'],
                        signal_description=preparation_metadata.get('signal_description'),
                        signal_columns=preparation_metadata.get('output_columns', []))
        matching_metadata = {"status": "disabled"}
        self.detector.capture_visual = False
        preparation_started = time.monotonic()
        if kind == 'vlm' and self.state["match_preview"] and hasattr(self.detector, "prepare_matching"):
            def matching_progress(message):
                self.update(status="preparing", matching_status=message, output_directory=str(output))
                print(message, flush=True)
            try:
                matching_progress("Preparing stage-matched visual preview")
                matching_metadata = self.detector.prepare_matching(bag, matching_progress, self.stop)
                self.update(matching_status="Ready: corrected recorded stages (retrospective preview)")
            except InterruptedError:
                (output / "online_results.json").write_text(json.dumps({
                    "bag": str(bag), "status": "stopped", "rows": [],
                    "visual_matching": {"status": "preparation_stopped"},
                    "matching_preparation_sec": time.monotonic() - preparation_started,
                }, indent=2), encoding="utf-8")
                self.update(status="stopped", matching_status="Preparation stopped")
                return
            except Exception as failure:
                self.detector.capture_visual = False
                matching_metadata = {"status": "unavailable", "reason": str(failure)}
                self.update(matching_status=f"Stage preview unavailable: {failure}")
                print(f"Stage preview unavailable; classifier replay continues: {failure}", flush=True)
        else:
            self.update(matching_status='DINO uses its native nearest-frame matching for scoring.' if kind == 'dino'
                        else 'GMR nominal model ready; no image matching.' if kind == 'gmr' else 'Disabled')
        # Annotation is saved only for evaluation, never passed into matching/scoring.
        from .visual_matching import read_corrected_failure_annotation
        failure_annotation = (None if causal_gmr else
            read_corrected_failure_annotation(bag, self.config.get("stage_topic", "/recording_stage")))
        start_sec = (float(vlm.get('sampling_start_sec') or 0.) if kind == 'vlm'
                     else getattr(self.detector, 'start_sec', 0.))
        end_sec = vlm.get('sampling_end_sec') if kind == 'vlm' else getattr(self.detector, 'end_sec', None)
        sample_fps = self.detector.sample_fps if kind == 'dino' else 15.
        if causal_gmr:
            feed = self.detector.create_feed(self.publish, self.stop, self.update)
        elif kind == 'dino' and hasattr(self.detector, 'iter_frames'):
            feed = SequentialReplayFeed(self.detector.iter_frames(), self.publish, self.stop,
                                        allow_equal_timestamps=True)
        else:
            frames = iter_rosbag_image_frames(RosbagImageSource(bag, vlm["camera_topics"][0]),
                sample_fps=sample_fps, start_sec=0. if kind == 'gmr' else start_sec, end_sec=end_sec)
            if kind == 'gmr':
                frames = gmr_sample_frames(frames, self.detector.results.time.to_numpy())
            feed_class = ReplayFeed if kind == 'vlm' else SequentialReplayFeed
            feed = feed_class(frames, self.publish, self.stop)
        producer = threading.Thread(target=feed.run, daemon=True)
        rows, status, error = [], "running", None
        max_frames = self.detector.frame_count if kind == 'vlm' else None
        total_samples = len(self.detector.results) if kind == 'gmr' and not causal_gmr else max_frames
        self.update(status=status, output_directory=str(output), max_frames=max_frames,
                    total_samples=total_samples, sample_fps=sample_fps if kind == 'dino' else None)
        protocol = ('online_elapsed_time_latest_frame_v1' if kind == 'vlm' else
                    'dino_all_sampled_frames_v1' if kind == 'dino' else
                    self.detector.signature['protocol'] if causal_gmr else 'gmr_all_prepared_samples_v1')
        metadata = dict(bag=str(bag), protocol=protocol, interval_sec=self.interval if kind == 'vlm' else None,
            sample_fps=sample_fps if kind == 'dino' else None, total_samples=total_samples,
            max_frames=max_frames, note=self.state['note'], classifier_path=str(self.detector.path),
            score_source=self.detector.score_source, comparison_signature=self.detector.signature,
            verification_mode=self.verify, diagnostic_mode=self.diagnose)
        metadata.update(visual_matching=matching_metadata, recorded_failure_annotation=failure_annotation,
                        detector_type=kind, detector_preparation=preparation_metadata,
                        matching_preparation_sec=time.monotonic() - preparation_started)

        def verification_status():
            checks = [row["verification"] for row in rows if row.get("verification")]
            if any(not check["within_tolerance"] for check in checks):
                return "failed"
            return "passed_observed_prefixes" if checks else "pending" if self.verify else "not_requested"

        last_persist = [0.]
        def persist(force=True):
            if not force and causal_gmr and time.monotonic() - last_persist[0] < 1.:
                return
            last_persist[0] = time.monotonic()
            if causal_gmr:
                stage_clock = self.detector.sampler.clock
                metadata['stage_boundaries_sec'] = dict(stage_clock.boundaries)
                metadata['recorded_failure_annotation'] = list(stage_clock.failures)
            data = {**metadata, "status": status, "error": error, "rows": rows,
                    "cache_verification_status": verification_status(),
                    "elapsed_sec": time.monotonic() - feed.origin}
            temporary = output / "online_results.tmp"
            temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
            temporary.replace(output / "online_results.json")

        persist()
        producer.start()
        try:
            with (output / "online_scores.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = None
                earliest = start_sec
                while max_frames is None or len(rows) < max_frames:
                    packet = feed.next_frame(earliest)
                    if packet is None:
                        break
                    frame_id, timestamp, image = packet
                    active_detector = self.detector
                    camera_topic = None
                    if kind == 'dino' and hasattr(self.detector, 'cameras'):
                        camera_topic, frame_id = frame_id
                        active_detector = self.detector.cameras[camera_topic]
                    self.update(status="scoring", scoring_time_sec=timestamp)
                    row, vector, baseline = active_detector.append(image, timestamp, self.verify)
                    if kind == 'dino':
                        row['camera_topic'] = camera_topic or getattr(active_detector, 'topic', None)
                    check = row.get("verification")
                    failed = bool(check and not check["within_tolerance"])
                    row["score_computation"] = 'cached' if kind == 'vlm' else 'single_image' if kind == 'dino' else 'causal_ts' if causal_gmr else 'offline_lookup'
                    # Diagnostics may continue, but a rejected cached score is
                    # never used as the displayed/recorded detector result.
                    if failed and self.diagnose:
                        fallback_score = check.get("uncached_score")
                        if baseline is None or fallback_score is None or not math.isfinite(fallback_score):
                            raise RuntimeError("Cannot continue diagnostics without a finite uncached baseline.")
                        row["cached_vector_file"] = f"frame_{len(rows) + 1:02d}_cached.npy"
                        np.save(output / row["cached_vector_file"], vector)
                        vector = baseline
                        row.update(anomaly_score=fallback_score,
                                   above_threshold=fallback_score >= row["threshold"],
                                   score_computation="uncached_verification_fallback")
                    else:
                        row["cached_vector_file"] = None
                    row.update(frame_id=frame_id, result_time_sec=time.monotonic() - feed.origin,
                               observation_to_result_sec=time.monotonic() - feed.origin - timestamp)
                    row["vector_file"] = f"frame_{len(rows) + 1:02d}.npy" if kind != 'gmr' else None
                    if kind != 'gmr':
                        np.save(output / row["vector_file"], vector)
                        image.save(output / f"frame_{len(rows) + 1:02d}.jpg")
                    match = row.get("visual_matching", {})
                    if match.get("status") == "matched":
                        try:
                            if getattr(active_detector, 'matched_image', None) is not None:
                                match_buffer = io.BytesIO()
                                active_detector.matched_image.save(match_buffer, format='JPEG', quality=85)
                                matched_bytes = match_buffer.getvalue()
                            else:
                                matched_bytes = Path(match["best"]["image_file"]).read_bytes()
                            match["saved_preview"] = f"frame_{len(rows) + 1:02d}_nominal.jpg"
                            (output / match["saved_preview"]).write_bytes(matched_bytes)
                        except OSError as failure:
                            matched_bytes = None
                            match["preview_error"] = str(failure)
                    else:
                        matched_bytes = None
                    observed = getattr(active_detector, 'preview_image', None)
                    observed = observed.copy() if observed is not None else image.copy() if image is not None else None
                    if kind == 'dino':
                        observed.save(output / f'frame_{len(rows) + 1:02d}_heatmap.jpg')
                    additional_previews = {}
                    if camera_topic:
                        for view, attribute in [('scored', 'scored_preview_image'), ('attention', 'attention_preview_image')]:
                            preview = getattr(active_detector, attribute, None)
                            if preview is not None:
                                preview = preview.copy()
                                preview.thumbnail((640, 480))
                                extra_buffer = io.BytesIO()
                                preview.save(extra_buffer, format='JPEG', quality=85)
                                additional_previews[view] = extra_buffer.getvalue()
                                (output / f'frame_{len(rows) + 1:02d}_{view}.jpg').write_bytes(extra_buffer.getvalue())
                    if camera_topic and getattr(active_detector, 'diagnostic_data', None):
                        additional_previews['diagnostics'] = json.dumps(active_detector.diagnostic_data).encode()
                        for name, preview in active_detector.diagnostic_previews.items():
                            if name not in {'excess', 'standardized', 'weighted'}:
                                continue  # Other maps already use the normal preview endpoints.
                            preview = preview.copy()
                            preview.thumbnail((640, 480))
                            extra_buffer = io.BytesIO()
                            preview.save(extra_buffer, format='JPEG', quality=85)
                            additional_previews['diag_' + name] = extra_buffer.getvalue()
                    buffer = io.BytesIO()
                    if observed is not None and kind != 'gmr':
                        observed.thumbnail((640, 480))
                        observed.save(buffer, format="JPEG", quality=85)
                    if getattr(self.detector, "capture_visual", False) and self.detector.cache.last_visual_descriptor is not None:
                        np.save(output / f"frame_{len(rows) + 1:02d}_visual.npy", self.detector.cache.last_visual_descriptor)
                    if baseline is not None:
                        np.save(output / f"frame_{len(rows) + 1:02d}_uncached.npy", baseline)
                    rows.append(row)
                    if writer is None:
                        writer = csv.DictWriter(stream, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow({key: json.dumps(value) if isinstance(value, (dict, list)) else value
                                     for key, value in row.items()})
                    stream.flush()
                    persist(force=False)
                    with self.lock:
                        self.observed_jpeg, self.matched_jpeg = buffer.getvalue(), matched_bytes
                        if camera_topic:
                            self.camera_jpegs.setdefault(camera_topic, {}).update(
                                observed=self.observed_jpeg, matched=matched_bytes, **additional_previews)
                        self.state.update(status="observing", rows=list(rows), scoring_time_sec=None)
                    if kind == 'vlm' or len(rows) == 1 or len(rows) % 25 == 0:
                        print(f"{Path(bag).name}: {len(rows)}/{total_samples or 'end of recording'}, "
                              f"t={timestamp:.3f}s, score={row['anomaly_score']:.5f}, "
                              f"threshold={row['threshold']:.5f}, processing={row['cached_feature_sec']:.3f}s", flush=True)
                    if failed:
                        message = verification_failure_message(check)
                        if not self.diagnose:
                            raise RuntimeError(message + ". Replay stopped. Select Diagnostics in the browser to collect all checks with uncached fallback scores.")
                        warning = message + ". Diagnostic replay continues using the uncached score; cache remains UNVERIFIED."
                        self.update(warning=warning)
                        print(warning, flush=True)
                    self.update(cache_verification_status=verification_status())
                    earliest = timestamp + self.interval
            status = "stopped" if self.stop.is_set() else "completed"
            if not rows and not self.stop.is_set():
                raise ValueError("No eligible image frames were available in the replay window.")
        except Exception as failure:
            status, error = "error", str(failure)
        finally:
            self.stop.set()
            producer.join(timeout=5)
            if hasattr(feed, 'close'):
                feed.close()
            persist()
            try:
                if kind == 'dino' and hasattr(self.detector, 'cameras'):
                    from .run import _slug
                    for topic in self.detector.cameras:
                        save_plot([row for row in rows if row.get('camera_topic') == topic],
                                  output / f'{_slug(topic)}_timeline.png', False)
                else:
                    save_plot(rows, output / "online_timeline.png", not self.detector.knn)
            except Exception as failure:
                error = f"{error or ''} Plot export failed: {failure}".strip()
                status = "error"
                persist()
            self.update(status=status, error=error, rows=rows, scoring_time_sec=None,
                        cache_verification_status=verification_status())


def make_handler(app):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def send(self, code, data, content_type):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path == "/":
                self.send(200, Path(__file__).with_name("online_replay.html").read_bytes(), "text/html; charset=utf-8")
            elif path == "/state":
                self.send(200, json.dumps(app.snapshot()).encode(), "application/json")
            elif path == '/dino-camera':
                from urllib.parse import urlsplit, parse_qs
                query = parse_qs(urlsplit(self.path).query)
                topic, view = query.get('topic', [''])[0], query.get('view', ['frame'])[0]
                with app.lock:
                    jpeg = app.camera_jpegs.get(topic, {}).get(view) if view in {'frame', 'observed', 'matched', 'scored', 'attention', 'diagnostics', 'diag_raw', 'diag_excess', 'diag_standardized', 'diag_weighted', 'diag_pooled', 'diag_multiplier'} else None
                self.send(200 if jpeg else 204, jpeg or b'', 'application/json' if view == 'diagnostics' else 'image/jpeg')
            elif path in {"/frame", "/observed-frame", "/matched-frame"}:
                with app.lock:
                    jpeg = getattr(app, {"/frame": "jpeg", "/observed-frame": "observed_jpeg",
                                         "/matched-frame": "matched_jpeg"}[path])
                self.send(200 if jpeg else 204, jpeg or b"", "image/jpeg")
            else:
                self.send(404, b"Not found", "text/plain")

        def do_POST(self):
            # No cross-origin controls or arbitrary filesystem paths.
            origin = self.headers.get("Origin")
            if origin and origin != "http://" + self.headers.get("Host", ""):
                self.send(403, b"Cross-origin request rejected", "text/plain")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 <= length <= 1024:
                    raise ValueError("Invalid request length")
                data = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(data, dict):
                    raise ValueError("Expected a JSON object")
                if self.path == "/start":
                    app.start(data.get("bag_index"), data.get("interval_sec"), data.get("mode"), data.get("match_preview"), data.get('detector_type'), data.get('gmr_signal'))
                elif self.path == "/stop":
                    app.stop.set()
                else:
                    raise ValueError("Unknown action")
                self.send(200, b"{}", "application/json")
            except (ValueError, queue.Full) as error:
                self.send(400, json.dumps({"error": str(error)}).encode(), "application/json")
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bag", type=Path, action="append", help="Optional explicit selection; otherwise lists test_bag and all ROS bags under ANOMALY_DATA_ROOT (/data).")
    parser.add_argument("--interval-sec", type=float, default=10., help="Minimum elapsed time between selected observations; default 10 s.")
    parser.add_argument("--verify-cache", action="store_true", help="Recompute each prefix and stop on feature/score/decision mismatch. Adds diagnostic runtime.")
    parser.add_argument("--diagnose-cache", action="store_true", help="Check every prefix, continuing mismatches with uncached scores. Cache remains unverified if any check fails.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not math.isfinite(args.interval_sec) or args.interval_sec <= 0:
        parser.error("--interval-sec must be finite and positive")
    config = load_pipeline_config(args.config)
    bags = replay_bags(config, args.bag)
    app = ReplayApplication(config, bags, args.interval_sec, args.verify_cache, args.diagnose_cache)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    print(f"Replay HTTP server listening on {args.host}:{args.port}. Open http://127.0.0.1:{args.port} on the lab PC.", flush=True)
    print("From another PC, forward this port through SSH or VS Code first. Open the LOCAL forwarded port in your browser "
          "(e.g. http://127.0.0.1:18765 for a tunnel from local 18765 to lab 8765). "
          "The page is available while the model loads.", flush=True)
    print(SAMPLING_NOTE, flush=True)
    app.worker.start()
    # Compose sends SIGTERM, whereas an interactive Python session gets SIGINT.
    # Route both through the same cleanup so in-flight results can be saved.
    def terminate(_signum, _frame):
        raise KeyboardInterrupt
    previous_sigterm = signal.signal(signal.SIGTERM, terminate)
    try:
        server.serve_forever(poll_interval=.2)
    except KeyboardInterrupt:
        pass
    finally:
        app.shutdown.set()
        app.stop.set()
        server.server_close()
        app.worker.join()
        signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == "__main__":
    main()
