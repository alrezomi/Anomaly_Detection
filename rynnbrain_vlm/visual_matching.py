"""Experimental image retrieval within recorded stages; never a failure score.

The corrected final-three stage sequence is retrospective bag annotation, not
a causal stage estimator. Error markers never select candidates or descriptors.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np


DESCRIPTOR_ID = "qwen3vl_visual_merger_mean_l2_v1"
STAGE_POLICY = "final_three_numeric_markers_retrospective_v1"


def pool_visual_tokens(tokens):
    """One image's merged spatial tokens -> one unit vector, in float32."""
    import torch
    with torch.no_grad():
        if tokens.ndim != 2 or not tokens.shape[0]:
            raise ValueError("Expected nonempty visual tokens for one image.")
        pooled = tokens.detach().float().mean(dim=0)
        norm = torch.linalg.vector_norm(pooled)
        if not bool(torch.isfinite(pooled).all()) or not bool(norm > 1e-12):
            raise ValueError("Cannot normalize a zero or non-finite visual descriptor.")
        return (pooled / norm).cpu().numpy().copy()


@dataclass(frozen=True)
class RecordedStages:
    starts: tuple
    failure_markers: tuple = ()

    def at(self, timestamp):
        if not np.isfinite(timestamp):
            raise ValueError("Invalid observation timestamp.")
        index = int(np.searchsorted(self.starts, timestamp, side="right"))
        return index or None  # No guessed stage before the first valid marker.

    def metadata(self):
        return {"policy": STAGE_POLICY, "starts_sec": list(self.starts),
                "retrospective": True, "failure_markers_used_for_matching": False}

    def failure_annotation(self):
        first = self.failure_markers[0] if self.failure_markers else None
        return {"status": "available" if first else "missing_marker",
                "time_sec": first[0] if first else None, "marker": first[1] if first else None,
                "marker_count": len(self.failure_markers),
                "policy": "first_failure_at_or_after_corrected_stage_1", "stage_policy": STAGE_POLICY}


def final_recorded_stages(events):
    """Use final three numbered markers; inserted error/fail events never advance stages."""
    from build_dataset_manifest import FAIL_PATTERN
    numeric, failures = [], []
    for row in events.itertuples(index=False):
        label = str(row.stage).strip()
        timestamp = float(row.time)
        if not np.isfinite(timestamp) or timestamp < 0:
            raise ValueError("Stage markers contain invalid timestamps.")
        if label in {'1', '2', '3'}:
            numeric.append((timestamp, label))
        elif FAIL_PATTERN.search(label):
            failures.append((timestamp, label))
    selected = sorted(numeric, key=lambda item: item[0])[-3:]
    if ([stage for _, stage in selected] != ['1', '2', '3']
            or any(b[0] <= a[0] for a, b in zip(selected, selected[1:]))):
        raise ValueError("The final three numbered stage markers must be 1, 2, 3 in time order. "
                         "Error/Fail messages do not replace missing stage starts; no equal-time fallback is used.")
    return RecordedStages(tuple(timestamp for timestamp, _ in selected),
                          tuple(sorted(event for event in failures if event[0] >= selected[0][0])))


def read_recorded_stages(bag, topic):
    from rosbag_io import read_stage_events
    return final_recorded_stages(read_stage_events(bag, topic))


def read_corrected_failure_annotation(bag, topic):
    """Evaluation annotation only, excluding errors before the corrected execution."""
    try:
        return {**read_recorded_stages(bag, topic).failure_annotation(), "topic": topic}
    except Exception as error:
        return {"status": "unavailable", "time_sec": None, "topic": topic, "error": str(error)}


def nominal_matching_bags(config):
    """Use explicit nominal training sources, falling back to nominal references."""
    vlm = config["rynnbrain"]
    selected = vlm.get("cop_classifier", {}).get("training", {}).get("normal_bags", [])
    selected = selected or vlm.get("reference_bags", [])
    if not isinstance(selected, list) or not all(isinstance(bag, str) for bag in selected):
        raise ValueError("Nominal matching sources must be a list of bag paths.")
    return list(dict.fromkeys(selected))


class VisualMemory:
    def __init__(self, entries, vectors, report):
        self.entries = entries
        self.vectors = np.asarray(vectors, dtype=np.float32)
        if (self.vectors.ndim != 2 or len(entries) != len(self.vectors) or not entries
                or not np.isfinite(self.vectors).all()
                or not np.allclose(np.linalg.norm(self.vectors, axis=1), 1., atol=1e-4)):
            raise ValueError("Invalid visual memory.")
        self.report = report

    def match(self, vector, stage):
        if stage is None:
            return {"status": "unavailable", "stage": None, "reason": "Before first valid stage marker."}
        indices = [i for i, item in enumerate(self.entries) if item["stage"] == stage]
        if not indices:
            return {"status": "unavailable", "stage": stage, "reason": "No nominal images for this stage."}
        query = np.asarray(vector, dtype=np.float32)
        if (query.shape != self.vectors.shape[1:] or not np.isfinite(query).all()
                or not np.isclose(np.linalg.norm(query), 1., atol=1e-4)):
            raise ValueError("Invalid query visual descriptor.")
        similarity = np.clip(self.vectors[indices] @ query, -1., 1.)
        order = np.argsort(-similarity, kind="stable")[:3]
        candidates = [{**self.entries[indices[int(i)]], "cosine_distance": float(1. - similarity[i])} for i in order]
        return {"status": "matched", "stage": stage, "candidate_count": len(indices),
                "descriptor_id": DESCRIPTOR_ID, "best": candidates[0], "top_matches": candidates}


def _source_identity(path):
    """Invalidate on local bag metadata or recording-file changes, without hashing GBs."""
    files = [file for file in path.rglob("*") if file.is_file() and
             (file.name == "metadata.yaml" or file.suffix in {".db3", ".mcap", ".zstd"})]
    if not files:
        raise ValueError(f"No ROS bag files at {path}")
    return [[str(file.relative_to(path)), file.stat().st_size, file.stat().st_mtime_ns] for file in sorted(files)]


def _visual_descriptor(detector, image):
    """Encode a nominal image once, without language decoding or KV mutations."""
    import torch
    wrapper = detector.model
    message = wrapper.conversation_message(dict(role="user", images=[("Image", image)], text=""))
    inputs = wrapper.tokenize_conversation([message], detector.vlm.get("generation", {})).to(wrapper.input_device)
    with torch.inference_mode():
        embeds, _ = detector.cache.base.get_image_features(inputs["pixel_values"], inputs["image_grid_thw"])
        if len(embeds) != 1:
            raise ValueError("Expected one nominal image.")
        return pool_visual_tokens(embeds[0])


def build_visual_memory(detector, progress, stop):
    """Cache up to eight nominal samples per stage per bag, separately from classifiers."""
    from rosbag_io import RosbagImageSource, bag_duration_seconds, sample_rosbag_image_frames_uniform
    config, vlm = detector.config, detector.vlm
    if config.get("stage_count", 3) != 3 or config.get("ignore_recorded_stages", False):
        raise ValueError("Stage preview requires three recorded stages and ignore_recorded_stages=false.")
    topic = config.get("stage_topic", "/recording_stage")
    image_topic = vlm["camera_topics"][0]
    root = Path(config["output_dir"]) / "online_replay" / "visual_memory"
    root.mkdir(parents=True, exist_ok=True)
    sources = nominal_matching_bags(config)
    entries, vectors, skipped = [], [], []
    from PIL import Image
    for index, value in enumerate(sources):
        if stop.is_set():
            raise InterruptedError("Nominal matching preparation stopped.")
        path = Path(value)
        progress(f"Nominal matching memory: {index + 1}/{len(sources)} — {path.name}")
        try:
            stages = read_recorded_stages(path, topic)
            end = bag_duration_seconds(path)
            if stages.starts[-1] >= end:
                raise ValueError("Final stage starts outside the bag duration.")
            identity = {"descriptor_id": DESCRIPTOR_ID, "stages": stages.metadata(), "stage_topic": topic,
                        "camera_topic": image_topic, "bag": str(path.resolve()), "source_files": _source_identity(path),
                        "feature_signature": detector.signature, "samples_per_stage": 8}
            key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
            directory = root / key
            metadata_path = directory / "memory.json"
            loaded = False
            if metadata_path.is_file():
                try:
                    data = json.loads(metadata_path.read_text(encoding="utf-8"))
                    with np.load(directory / "vectors.npz", allow_pickle=False) as archive:
                        cached = archive["vectors"]
                    VisualMemory(data["entries"], cached, {})
                    if data["identity"] != identity or not all(Path(item["image_file"]).is_file() for item in data["entries"]):
                        raise ValueError("Stale visual memory.")
                    entries.extend(data["entries"])
                    vectors.extend(cached)
                    loaded = True
                except (ValueError, OSError, KeyError, EOFError):
                    pass
            if loaded:
                continue
            directory.mkdir(parents=True, exist_ok=True)
            bag_entries, bag_vectors = [], []
            for stage, start in enumerate(stages.starts, 1):
                end_stage = stages.starts[stage] if stage < 3 else end
                frames = sample_rosbag_image_frames_uniform(RosbagImageSource(path, image_topic), 8,
                    start_sec=start, end_sec=np.nextafter(end_stage, -np.inf))
                for frame_id, timestamp, image in frames:
                    if stop.is_set():
                        raise InterruptedError("Nominal matching preparation stopped.")
                    descriptor = _visual_descriptor(detector, image)
                    image_file = directory / f"stage_{stage}_frame_{frame_id}.jpg"
                    preview = image.copy()
                    preview.thumbnail((640, 480), Image.Resampling.LANCZOS)
                    preview.save(image_file, quality=85)
                    bag_entries.append({"bag": str(path), "frame_id": frame_id,
                                        "timestamp_sec": timestamp, "stage": stage, "image_file": str(image_file)})
                    bag_vectors.append(descriptor)
            VisualMemory(bag_entries, bag_vectors, {})
            np.savez_compressed(directory / "vectors.npz", vectors=np.stack(bag_vectors))
            metadata_path.write_text(json.dumps({"identity": identity, "entries": bag_entries}, indent=2), encoding="utf-8")
            entries.extend(bag_entries)
            vectors.extend(bag_vectors)
        except InterruptedError:
            raise
        except (ValueError, OSError, KeyError) as error:
            skipped.append({"bag": str(path), "reason": str(error)})
            progress(f"Skipping nominal matching source {path.name}: {error}")
    report = {"descriptor_id": DESCRIPTOR_ID, "stage_policy": STAGE_POLICY, "nominal_sources": sources,
              "skipped_bags": skipped, "image_count": len(entries), "samples_per_stage_per_bag": 8,
              "note": "Visual retrieval preview only; does not change classifier scores or thresholds."}
    (root / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return VisualMemory(entries, vectors, report)
