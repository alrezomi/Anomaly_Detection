"""Optional isolated LoRA -> classifier -> benchmark orchestration."""

from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

from pipeline_settings import load_pipeline_config, resolve_pipeline_config
from .runtime import write_json


def train_lora_enabled(config):
    selection = config.get("vlm_benchmark") or {}
    if not isinstance(selection, dict):
        raise ValueError("vlm_benchmark must be an object.")
    enabled = selection.get("train_lora", False)
    if not isinstance(enabled, bool):
        raise ValueError("vlm_benchmark.train_lora must be true or false.")
    return enabled


def prepare_pipeline_run(arguments, config):
    """Create fresh artifacts, retaining the requested config and existing models."""
    from .benchmark_protocol import configure_decision_mode
    from .cop_classifier import classifier_method
    from .lora import resolve_bag_selection
    from .train_lora import training_settings

    if arguments.skip_vlm:
        raise ValueError("vlm_benchmark.train_lora requires VLM evaluation; remove --skip-vlm.")
    if not config.get("rynnbrain"):
        raise ValueError("Add a rynnbrain section before enabling train_lora.")
    requested = deepcopy(config)
    result = deepcopy(config)
    vlm = result["rynnbrain"]
    old_output = Path(vlm["output_dir"])
    base = arguments.benchmark_dir or Path(f"{old_output}_benchmark")
    variant = vlm.get("prompt_variant", "A").lower()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f") + f"_prompt_{variant}"
    root = (base / "pipeline_runs" / run_id).resolve()
    result["output_dir"] = str(root)
    vlm["output_dir"] = str(root / "rynnbrain")
    training = vlm.setdefault("lora", {}).setdefault("training", {})
    folder = "lora_temporal_adapter" if training.get("temporal_supervision", False) else "lora_adapter"
    training["output_dir"] = str(root / folder)
    # Prepared videos, when used, still belong to the originally configured source.
    training["benchmark_dir"] = training.get("benchmark_dir") or f"{old_output}_benchmark"
    vlm["model"] = {**vlm.get("model", {}), "lora_adapter_path": None}
    classifier = vlm.setdefault("cop_classifier", {})
    classifier.update(enabled=True, train_on_benchmark=True)
    classifier_training = classifier.setdefault("training", {})
    classifier_training["input_dir"] = str(root / "classifier_training")
    classifier_training.pop("output_file", None)
    method = classifier_method(classifier)
    modes = list(vlm.get("input_modes", ["raw"]))
    classifier["model_paths"] = {mode: str(root / "cop_classifier" / f"{mode}_{method}.npz") for mode in modes}
    vlm["cop_vectors"] = {**vlm.get("cop_vectors", {}), "enabled": True}
    result["vlm_benchmark"] = {**result.get("vlm_benchmark", {}), "train_lora": False}
    result = resolve_pipeline_config(result)
    mode = configure_decision_mode(result["vlm_benchmark"], result["rynnbrain"])

    # Check explicit LoRA lists before allocating the GPU or creating run artifacts.
    settings = training_settings(result)
    data_root = arguments.data_root or Path(os.environ.get("ANOMALY_DATA_ROOT", "/data"))
    resolve_bag_selection(settings, data_root, result["rynnbrain"]["reference_bags"])
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "requested_config.json", requested)
    write_json(root / "lora_training_config.json", result)
    adapter = settings["output_dir"]
    result["rynnbrain"]["model"]["lora_adapter_path"] = adapter
    effective_path = root / "pipeline_config.json"
    write_json(effective_path, result)
    arguments = deepcopy(arguments)
    arguments.config = effective_path
    arguments.benchmark_dir = root
    arguments.lora_training_config = root / "lora_training_config.json"
    return arguments, result, root / mode


def run_lora_training(config_path, data_root, adapter_path, update, startup_ignore_sec):
    """Use a child process so all training GPU memory is freed before inference."""
    started = perf_counter()
    adapter = Path(adapter_path)
    update(phase="lora_training", lora_training_status="running",
           trained_adapter_path=str(adapter), lora_training_config=str(config_path))
    command = [sys.executable, "-u", "-m", "rynnbrain_vlm.train_lora",
               "--config", str(config_path), "--data-root", str(data_root)]
    print(f"Training a fresh LoRA adapter before classifier preparation: {adapter}", flush=True)
    try:
        with subprocess.Popen(command, cwd=Path(__file__).resolve().parents[1],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding="utf-8", errors="replace",
                              env={**os.environ, "STAGE_STARTUP_IGNORE_SEC": str(startup_ignore_sec)}) as process:
            for line in process.stdout:
                print(line, end="", flush=True)
            if process.wait():
                raise subprocess.CalledProcessError(process.returncode, command)
        required = [adapter / "adapter_config.json", adapter / "training_summary.json",
                    adapter / "training_manifest.json"]
        if not all(path.is_file() for path in required) or not any(
                (adapter / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")):
            raise RuntimeError(f"LoRA process finished without complete adapter artifacts: {adapter}")
        summary = json.loads((adapter / "training_summary.json").read_text(encoding="utf-8"))
        if not summary.get("saved_epoch") or summary.get("optimizer_steps", 0) < 1:
            raise RuntimeError(f"LoRA process produced no successfully trained checkpoint: {adapter}")
    except BaseException as error:
        update(lora_training_status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
               lora_training_sec=perf_counter() - started)
        raise
    update(lora_training_status="completed", lora_training_sec=perf_counter() - started)
    print(f"LoRA training completed; automatically loading adapter: {adapter}", flush=True)


def save_reusable_config(path, config):
    """The saved effective config loads trained artifacts without training again."""
    saved = deepcopy(config)
    saved["vlm_benchmark"]["train_lora"] = False
    saved["rynnbrain"]["cop_classifier"]["train_on_benchmark"] = False
    write_json(path, saved)


def resolve_trained_artifacts(config, progress=print):
    """Find trained artifacts for inference without rewriting the user's config.

    Disabling automatic LoRA training keeps all explicitly configured paths,
    including a null adapter selecting the base model, even if files are missing.
    Current prompts, inference options and threshold remain authoritative; replay
    still checks their compatibility with the selected classifier before scoring.
    """
    if not train_lora_enabled(config):
        return config

    from .cop_classifier import classifier_method, classifier_model_paths

    vlm = config.get("rynnbrain", {})
    classifier = vlm.get("cop_classifier", {})
    modes = list(vlm.get("input_modes", ["raw"]))
    output = vlm.get("output_dir")
    if not output:
        return config
    runs = Path(f"{output}_benchmark") / "pipeline_runs"
    for path in sorted(runs.glob("*/pipeline_config.json"), reverse=True):
        try:
            saved = load_pipeline_config(path)
            trained = saved.get("rynnbrain", {})
            if (trained.get("prompt_variant", "A") != vlm.get("prompt_variant", "A")
                    or classifier_method(trained.get("cop_classifier", {})) != classifier_method(classifier)):
                continue
            mode = saved.get("vlm_benchmark", {}).get("decision_mode", "whole_demo")
            runtime = json.loads((path.parent / mode / "benchmark_runtime.json").read_text(encoding="utf-8"))
            if runtime.get("status") != "completed":
                continue
        except (OSError, ValueError, TypeError):
            continue  # Failed/interrupted runs are not usable completed benchmarks.
        paths = classifier_model_paths(trained.get("cop_classifier", {}))
        adapter = trained.get("model", {}).get("lora_adapter_path")
        required = [Path(paths[name]) for name in modes if paths.get(name)]
        if len(required) != len(modes):
            continue
        required += [item.with_suffix(".json") for item in list(required)]
        if adapter:
            required += [Path(adapter) / "adapter_config.json", Path(adapter) / "training_manifest.json"]
        if not adapter or not any((Path(adapter) / name).is_file()
                                  for name in ("adapter_model.safetensors", "adapter_model.bin")):
            raise FileNotFoundError(f"Completed training run is missing its LoRA weights: {path.parent}")
        missing = [str(item) for item in required if not item.is_file()]
        if missing:
            raise FileNotFoundError(f"Completed training run has missing artifacts: {', '.join(missing)}")
        result = deepcopy(config)
        target = result["rynnbrain"]
        target["model"] = {**target.get("model", {}), "lora_adapter_path": adapter}
        target["cop_classifier"] = {**classifier, "enabled": True,
                                    "model_paths": {name: paths[name] for name in modes}}
        target["cop_classifier"]["training"] = {
            **classifier.get("training", {}),
            "input_dir": trained["cop_classifier"].get("training", {}).get("input_dir"),
        }
        progress(f"Using completed training run: {path.parent}")
        progress(f"Trained LoRA adapter: {adapter}")
        progress(f"Trained classifier(s): {target['cop_classifier']['model_paths']}")
        return result
    return config
