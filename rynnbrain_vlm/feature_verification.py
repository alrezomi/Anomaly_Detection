"""Controlled, one-bag comparison of generation and direct CoP extraction."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np


def vector_difference(reference, candidate):
    a, b = (np.asarray(value, dtype=np.float64) for value in (reference, candidate))
    if a.ndim != 1 or a.shape != b.shape or not a.size:
        raise ValueError("Feature comparison requires two vectors of the same nonempty shape.")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Feature comparison contains non-finite values.")
    an, bn = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if min(an, bn) == 0:
        raise ValueError("Feature comparison cannot use a zero vector.")
    relative = float(np.linalg.norm(a - b) / an)
    return {"max_absolute_difference": float(np.max(np.abs(a - b))),
            "relative_l2_difference": relative,
            "cosine_similarity": float(np.clip(a @ b / (an * bn), -1, 1)),
            "within_tolerance": relative <= 1e-3,
            "relative_l2_tolerance": 1e-3}


def score_comparison(classifier, vectors, mode, signature):
    from .cop_knn import CoPKNNDetector

    is_knn = isinstance(classifier, CoPKNNDetector)
    predict = classifier.predict_anomaly_score if is_knn else classifier.predict_failure_probability
    scores = {name: float(predict(vector, input_mode=mode, comparison_signature=signature))
              for name, vector in vectors.items()}
    labels = {name: classifier.predict_label(score) for name, score in scores.items()}
    maximum_delta = max(abs(score - scores["generation"]) for score in scores.values())
    return {"score_kind": "knn_distance" if is_knn else "failure_probability",
            "scores": scores, "decisions": labels, "threshold": float(classifier.threshold),
            "maximum_score_difference": maximum_delta, "score_tolerance": 1e-3,
            "within_tolerance": maximum_delta <= 1e-3 and len(set(labels.values())) == 1}


def signature_changes(old, new, prefix=""):
    changes = {}
    for key in sorted(old.keys() | new.keys()):
        name = f"{prefix}.{key}" if prefix else key
        a, b = old.get(key), new.get(key)
        if isinstance(a, dict) and isinstance(b, dict):
            changes.update(signature_changes(a, b, name))
        elif a != b:
            changes[name] = {"baseline": a, "current": b}
    return changes


def saved_baseline(root, bag_name, mode, signature, vector):
    path = root / bag_name / "rynnbrain_multiturn" / "cop_vectors" / f"{mode}.json"
    if not path.is_file():
        return {"status": "bag_vector_not_found", "metadata_path": str(path)}
    metadata = json.loads(path.read_text(encoding="utf-8"))
    old_signature = metadata.get("comparison_signature")
    changes = signature_changes(old_signature or {}, signature)
    report = {"status": "read", "metadata_path": str(path),
              "same_representation_settings": old_signature is not None and not changes,
              "changed_settings": changes,
              "saved_failure_probability": metadata.get("classifier_failure_probability"),
              "saved_anomaly_score": metadata.get("classifier_anomaly_score"),
              "saved_classifier_path": metadata.get("classifier_model_path"),
              "note": "Different settings make this an uncontrolled historical comparison. Scores may also use a different trained classifier."}
    vector_path = path.parent / Path(metadata.get("vector_file", f"{mode}.npy")).name
    if vector_path.is_file():
        report["vector_difference"] = vector_difference(np.load(vector_path, allow_pickle=False), vector)
    return report


def compare_populations(baseline_dir, current_dir):
    def bags(root):
        # Use completed rows, not a selection list that might include unfinished bags.
        path = root / "benchmark_summary.csv"
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return {row["bag_name"] for row in csv.DictReader(handle)}

    try:
        old, new = bags(baseline_dir), bags(current_dir)
    except (OSError, KeyError) as error:
        return {"status": "unavailable", "error": str(error)}
    return {"status": "compared", "baseline_bags": len(old), "current_bags": len(new),
            "common_bags": len(old & new), "added_bags": sorted(new - old),
            "removed_bags": sorted(old - new)}


def run_verification(config, vlm, frame_count, generation, *, baseline_dir=None):
    from .cop_classifier import classifier_method, classifier_model_paths, load_classifier
    from .cop_knn import CoPKNNDetector
    from .model import RynnBrainModel
    from .prompts import task_context_prompt, evaluation_prompt_multiturn
    from .run import _execution_inputs, _raw_inputs, cop_comparison_signature
    from .runtime import clock, record_run
    from .temporal_lora import timestamped_images, validate_profile

    if generation.get("do_sample", False):
        raise ValueError("Feature verification requires generation.do_sample=false.")
    classifier_config = vlm.get("cop_classifier", {})
    if not classifier_config.get("enabled", False):
        raise ValueError("Feature verification requires an enabled, already trained CoP classifier.")
    base_output = Path(vlm.get("output_dir", Path(config["output_dir"]) / "rynnbrain"))
    bag_name = Path(config["test_bag"]).name
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = base_output / "feature_verification" / bag_name / stamp
    modes = list(vlm.get("input_modes", ["raw"]))
    paths = classifier_model_paths(classifier_config)
    classifiers = {mode: load_classifier(Path(paths[mode])) for mode in modes}
    if any(isinstance(value, CoPKNNDetector) != (classifier_method(classifier_config) == "knn")
           for value in classifiers.values()):
        raise ValueError("Saved classifier type does not match cop_classifier.method.")
    with record_run(output, "verification") as update:
        print("Verification only: existing classifier, identical inputs; no training or benchmark overwrite.")
        update(phase="model_loading")
        model = RynnBrainModel(vlm["model"])
        if bag_name in model.adapter_training_bags:
            raise ValueError("Select a held-out test_bag for feature verification.")
        profile = model.temporal_profile
        validate_profile(profile, config, vlm, frame_count, modes)
        vlm = {**vlm, "_temporal_profile": profile}
        selected, _ = _execution_inputs(config, vlm, frame_count, modes)
        topics = list(vlm.get("memory_camera_topics", vlm.get("camera_topics", config["camera_topics"])))
        nominal_images = []
        for bag in vlm.get("reference_bags", []):
            images, frames = _raw_inputs(Path(bag), topics, frame_count)
            if profile:
                images = timestamped_images(images, frames, topics, frame_count)
            nominal_images.extend(images)
        if not nominal_images:
            raise ValueError("No nominal reference images were loaded.")
        task = vlm.get("task_description", "Robot manipulation task")
        nominal_turn = {"role": "user", "images": nominal_images,
                        "text": task_context_prompt(task, temporal=bool(profile))}
        reference = model.generate_nominal(nominal_turn, generation)
        report = {"test_bag": config["test_bag"], "results": [],
                  "versions": {name: version(name) for name in ("torch", "transformers", "peft", "accelerate")},
                  "note": "Single-bag numerical check, not an accuracy guarantee or a warmed-up latency benchmark. Identical reference text is used in every path."}
        if baseline_dir is not None:
            report["population_comparison"] = compare_populations(baseline_dir, Path(f"{base_output}_benchmark"))
        for mode in modes:
            update(phase="comparing", input_mode=mode)
            images, frames = selected[mode]
            turns = [nominal_turn, {"role": "user", "images": images,
                     "text": evaluation_prompt_multiturn(task, mode, temporal=bool(profile))}]
            signature = cop_comparison_signature(model, config, vlm, mode, reference, frame_count, generation)
            vectors, timings = {}, {}
            # Check both before and after generate(): mutable model state must not
            # make the direct path depend on which call happened previously.
            for name in ("direct_before", "generation", "direct_after"):
                started = clock()
                if name == "generation":
                    generated = model.generate_multiturn_with_cop_vector(turns, generation, reference_response=reference)
                    vector = generated.cop_vector
                else:
                    vector = model.extract_multiturn_cop_vector(turns, generation, reference_response=reference)
                timings[name] = clock() - started
                vectors[name] = vector.detach().float().cpu().numpy()
                np.save(output / f"{mode}_{name}.npy", vectors[name], allow_pickle=False)
            differences = {name: vector_difference(vectors["generation"], vectors[name])
                           for name in ("direct_before", "direct_after")}
            result = {"input_mode": mode, "comparison_signature": signature,
                      "vector_comparisons": differences, "runtime_sec": timings,
                      "generated_response": generated.evaluation_response,
                      "selected_frames": frames, "classifier_path": str(paths[mode]),
                      "classifier_sha256": hashlib.sha256(Path(paths[mode]).read_bytes()).hexdigest()}
            try:
                result["classifier_comparison"] = score_comparison(classifiers[mode], vectors, mode, signature)
                result["status"] = ("match" if all(item["within_tolerance"] for item in differences.values())
                                    and result["classifier_comparison"]["within_tolerance"] else "mismatch")
            except ValueError as error:
                result.update(status="classifier_incompatible", classifier_error=str(error))
            if baseline_dir is not None:
                try:
                    result["baseline"] = saved_baseline(baseline_dir, bag_name, mode, signature, vectors["generation"])
                except (OSError, ValueError, TypeError) as error:
                    result["baseline"] = {"status": "unavailable", "error": str(error)}
            report["results"].append(result)
            report_path = output / "comparison.json"
            report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
            print(f"{mode}: feature verification {result['status']}")
            print(json.dumps(result.get("classifier_comparison", result.get("classifier_error")), indent=2))
            if "baseline" in result:
                print("Changed baseline settings: " + ", ".join(result["baseline"].get("changed_settings", {})))
            print(f"Comparison saved to: {report_path}")
        update(phase="finished", verification_status={item["input_mode"]: item["status"] for item in report["results"]})
