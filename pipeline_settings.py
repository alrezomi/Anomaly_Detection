"""Resolve optional pipeline paths and camera inheritance without rewriting JSON."""

from copy import deepcopy
import json
from pathlib import Path


def _prompt_b_path(value, *, file=False):
    path = Path(value)
    name = path.stem if file else path.name
    if name.endswith('_prompt_b') or any(part.endswith(('_prompt_b', '_prompt_b_benchmark')) for part in path.parts):
        return str(path)
    return str(path.with_name(name + '_prompt_b' + (path.suffix if file else '')))


def resolve_pipeline_config(config):
    """Resolve paths; prompt B gets separate artifacts, including explicit paths.

    The loaded adapter is deliberately never inferred: training starts with a
    null adapter, while evaluation explicitly selects the trained artifact.
    Training bags and inference/training hyperparameters are not changed.
    """
    result = deepcopy(config)
    if not result.get("rynnbrain"):
        return result
    vlm = result["rynnbrain"]
    variant = vlm.get('prompt_variant', 'A')
    if variant not in ('A', 'B'):
        raise ValueError('rynnbrain.prompt_variant must be A or B.')
    root = Path(result.get("output_dir") or "outputs")
    if not result.get("output_dir"):
        result["output_dir"] = str(root)
    vlm["output_dir"] = str(vlm.get("output_dir") or root / "rynnbrain")
    if variant == 'B':
        vlm['output_dir'] = _prompt_b_path(vlm['output_dir'])
    if "camera_topics" not in vlm:
        vlm["camera_topics"] = deepcopy(result.get("camera_topics", []))
    if "memory_camera_topics" not in vlm:
        vlm["memory_camera_topics"] = deepcopy(vlm["camera_topics"])

    training = vlm.setdefault("lora", {}).setdefault("training", {})
    folder = "lora_temporal_adapter" if training.get("temporal_supervision", False) else "lora_adapter"
    training["output_dir"] = str(training.get("output_dir") or Path(vlm["output_dir"]) / folder)
    if variant == 'B' and config['rynnbrain'].get('lora', {}).get('training', {}).get('output_dir'):
        training['output_dir'] = _prompt_b_path(training['output_dir'])

    classifier = vlm.setdefault("cop_classifier", {})
    classifier_training = classifier.setdefault("training", {})
    classifier_training["input_dir"] = str(classifier_training.get("input_dir") or f"{vlm['output_dir']}_benchmark")
    if variant == 'B' and config['rynnbrain'].get('cop_classifier', {}).get('training', {}).get('input_dir'):
        classifier_training['input_dir'] = _prompt_b_path(classifier_training['input_dir'])
    if variant == 'B' and classifier_training.get('output_file'):
        classifier_training['output_file'] = _prompt_b_path(classifier_training['output_file'], file=True)
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
        if variant == 'B':
            paths[mode] = _prompt_b_path(paths[mode], file=True)
    return result


def load_pipeline_config(path):
    return resolve_pipeline_config(json.loads(Path(path).read_text(encoding="utf-8")))
