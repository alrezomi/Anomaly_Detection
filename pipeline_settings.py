"""Resolve optional pipeline paths and camera inheritance without rewriting JSON."""

from copy import deepcopy
import json
from pathlib import Path


def resolve_pipeline_config(config):
    """Explicit overrides win; omitted/null output paths use the experiment root.

    The loaded adapter is deliberately never inferred: training starts with a
    null adapter, while evaluation explicitly selects the trained artifact.
    Training bags and inference/training hyperparameters are not changed.
    """
    result = deepcopy(config)
    if not result.get("rynnbrain"):
        return result
    vlm = result["rynnbrain"]
    root = Path(result.get("output_dir") or "outputs")
    if not result.get("output_dir"):
        result["output_dir"] = str(root)
    vlm["output_dir"] = str(vlm.get("output_dir") or root / "rynnbrain")
    if "camera_topics" not in vlm:
        vlm["camera_topics"] = deepcopy(result.get("camera_topics", []))
    if "memory_camera_topics" not in vlm:
        vlm["memory_camera_topics"] = deepcopy(vlm["camera_topics"])

    training = vlm.setdefault("lora", {}).setdefault("training", {})
    folder = "lora_temporal_adapter" if training.get("temporal_supervision", False) else "lora_adapter"
    training["output_dir"] = str(training.get("output_dir") or Path(vlm["output_dir"]) / folder)

    classifier = vlm.setdefault("cop_classifier", {})
    classifier_training = classifier.setdefault("training", {})
    classifier_training["input_dir"] = str(classifier_training.get("input_dir") or f"{vlm['output_dir']}_benchmark")
    method = classifier.get("method", "logistic")
    if method not in {"logistic", "knn"}:
        raise ValueError("cop_classifier.method must be 'logistic' or 'knn'.")
    paths = classifier.setdefault("model_paths", {})
    if not isinstance(paths, dict):
        raise ValueError("cop_classifier.model_paths must be an object.")
    modes = set(vlm.get("input_modes", ["raw"])) | {classifier_training.get("input_mode", "raw")}
    for mode in sorted(modes):
        if not paths.get(mode):
            paths[mode] = str(root / "cop_classifier" / f"{mode}_{method}.npz")
    return result


def load_pipeline_config(path):
    return resolve_pipeline_config(json.loads(Path(path).read_text(encoding="utf-8")))
