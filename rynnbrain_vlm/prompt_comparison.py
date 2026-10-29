"""Run independent prompt training pipelines through the ordinary benchmark CLI."""

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from .execution_sampling import execution_sampling
from .runtime import record_run, write_json


def comparison_variants(config):
    values = (config.get("vlm_benchmark") or {}).get("prompt_variants")
    if values is None:
        return None
    if (not isinstance(values, list) or len(values) < 2 or any(item not in ("A", "B") for item in values)
            or len(set(values)) != len(values)):
        raise ValueError("vlm_benchmark.prompt_variants must be ['A', 'B'] (or ['B', 'A']).")
    if not config["vlm_benchmark"].get("train_lora", False):
        raise ValueError("prompt_variants requires vlm_benchmark.train_lora=true for independent fresh adapters.")
    return values


def comparison_prompts(config, variants):
    from .prompts import task_context_prompt, evaluation_prompt_multiturn
    from .temporal_lora import training_profile
    vlm = config["rynnbrain"]
    if (not vlm.get("lora", {}).get("training", {}).get("temporal_supervision", False)
            or not execution_sampling(vlm)):
        raise ValueError("Prompt interval comparison requires temporal_supervision=true and execution_interval_sec.")
    training_profile(config, {"input_mode": "raw", **vlm["lora"]["training"]})
    if (config["vlm_benchmark"].get("decision_mode") != "timeline"
            or vlm.get("input_modes", ["raw"]) != ["raw"]
            or not vlm.get("inference", {}).get("feature_only", False)
            or not vlm.get("cop_classifier", {}).get("logistic_onset", {}).get("enabled", False)):
        raise ValueError("Prompt interval comparison requires timeline, feature_only=true and logistic_onset.enabled=true.")
    result = {}
    for variant in variants:
        task = vlm.get("task_description", "Robot manipulation task")
        texts = dict(turn1_prompt=task_context_prompt(task, temporal=True, variant=variant),
                     turn2_prompt=evaluation_prompt_multiturn(task, "raw", temporal=True, variant=variant))
        result[variant] = dict(texts=texts,
            sha256=hashlib.sha256(json.dumps(texts, sort_keys=True).encode()).hexdigest())
    if len({item["sha256"] for item in result.values()}) != len(variants):
        raise ValueError("Prompt A and B produce identical temporal prompt text. Edit the temporal branches "
                         "of your A/B functions before starting a comparison; no training was started.")
    return result


def _child_command(arguments, path, root):
    command = [sys.executable, "-u", str(Path(__file__).resolve().parents[1] / "run_benchmark.py"),
               "--config", str(path), "--benchmark-dir", str(root)]
    for key in ("data_root", "startup_ignore_sec", "limit"):
        value = getattr(arguments, key, None)
        if value is not None:
            command.extend(["--" + key.replace("_", "-"), str(value)])
    for key in ("folder", "bag", "normal_bag", "failure_bag"):
        for value in getattr(arguments, key, None) or []:
            command.extend(["--" + key.replace("_", "-"), str(value)])
    for key in ("no_recursive", "include_nominal_bags", "skip_dino", "with_dino", "prepare_classifier"):
        if getattr(arguments, key, False):
            command.append("--" + key.replace("_", "-"))
    return command


def run_prompt_comparison(arguments, config):
    import pandas as pd
    variants = comparison_variants(config)
    if getattr(arguments, "skip_vlm", False) or getattr(arguments, "with_dino", False):
        raise ValueError("Prompt interval comparison requires VLM evaluation without --with-dino.")
    prompts = comparison_prompts(config, variants)
    vlm = config["rynnbrain"]
    interval = execution_sampling(vlm)["interval_sec"]
    base = arguments.benchmark_dir or Path(f"{vlm['output_dir']}_benchmark")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    root = (base / "prompt_comparisons" / f"{stamp}_interval_{interval:g}s").resolve()
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "requested_config.json", config)
    write_json(root / "prompt_texts.json", prompts)
    rows, first_selection = [], None
    with record_run(root, "benchmark") as update:
        print(f"Prompt comparison: independent {variants} pipelines at {interval:g}s. Output: {root}", flush=True)
        for variant in variants:
            update(phase="prompt_pipeline", current_prompt=variant, completed_prompts=len(rows))
            child = deepcopy(config)
            child["vlm_benchmark"].pop("prompt_variants")
            child["vlm_benchmark"]["expected_prompt_texts"] = prompts[variant]["texts"]
            child["rynnbrain"]["prompt_variant"] = variant
            child["rynnbrain"]["model"]["lora_adapter_path"] = None
            child["rynnbrain"]["online_replay"] = {
                **child["rynnbrain"].get("online_replay", {}), "interval_mode": "constant",
                "interval_sec": interval, "max_observations": None}
            directory = root / f"prompt_{variant.lower()}"
            directory.mkdir()
            path = directory / "requested_config.json"
            write_json(path, child)
            print(f"Starting prompt {variant}: fresh base model -> LoRA -> classifier -> held-out benchmark", flush=True)
            # A separate process releases the preceding VLM's GPU memory completely.
            with subprocess.Popen(_child_command(arguments, path, directory),
                    cwd=Path(__file__).resolve().parents[1], stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace") as process:
                for line in process.stdout:
                    print(line, end="", flush=True)
                exit_code = process.wait()
                if exit_code:
                    raise subprocess.CalledProcessError(exit_code, process.args)
            snapshots = list(directory.glob("pipeline_runs/*/pipeline_config.json"))
            if len(snapshots) != 1:
                raise RuntimeError(f"Expected one completed model pair for prompt {variant}: {directory}")
            saved = json.loads(snapshots[0].read_text(encoding="utf-8"))
            runtime_path = snapshots[0].parent / "timeline" / "benchmark_runtime.json"
            runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
            if runtime.get("status") != "completed":
                raise RuntimeError(f"Prompt {variant} benchmark did not complete: {runtime_path}")
            reports = Path(runtime["benchmark_output_dir"])
            summary = pd.read_csv(reports / "benchmark_summary.csv")
            selected = set(zip(summary.bag_path.astype(str), summary.ground_truth_label.astype(str)))
            if first_selection is not None and selected != first_selection:
                raise RuntimeError("A/B held-out bag selections differ. Individual reports are saved; comparison is invalid.")
            first_selection = selected
            statistics = json.loads((reports / "benchmark_statistics.json").read_text(encoding="utf-8"))
            timing_path = reports / "benchmark_frame_runtime.json"
            timing = json.loads(timing_path.read_text(encoding="utf-8")) if timing_path.is_file() else {}
            row = dict(prompt_variant=variant, interval_sec=interval, status="completed",
                prompt_sha256=prompts[variant]["sha256"], pipeline_config=str(snapshots[0]), reports_dir=str(reports),
                lora_adapter_path=saved["rynnbrain"]["model"]["lora_adapter_path"],
                classifier_path=saved["rynnbrain"]["cop_classifier"]["model_paths"]["raw"],
                threshold=saved["rynnbrain"]["cop_classifier"]["training"]["threshold"],
                average_processing_sec_per_frame=timing.get("average_processing_sec_per_frame"),
                selected_bags=len(selected), accuracy=statistics.get("accuracy"),
                correct_rows=statistics.get("correct_rows"), incorrect_rows=statistics.get("incorrect_rows"))
            row["benchmark_statistics"] = statistics
            rows.append(row)
            write_json(root / "prompt_comparison.json", dict(interval_sec=interval, runs=rows))
            pd.DataFrame([{key: value for key, value in item.items() if key != "benchmark_statistics"}
                          for item in rows]).to_csv(root / "prompt_comparison.csv", index=False)
            print(f"Prompt {variant} completed. Reusable config: {snapshots[0]}", flush=True)
        update(completed_prompts=len(rows), comparison_file=str(root / "prompt_comparison.json"))
    return root
