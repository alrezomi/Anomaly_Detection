"""Train a decision LoRA adapter on explicitly selected nominal/failure ROS bags."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import random
from typing import Any

from .lora import answer_loss, attach_lora, resolve_bag_selection, supervised_example
from .prompts import evaluation_prompt_multiturn, task_context_prompt


def training_settings(config: dict[str, Any]) -> dict[str, Any]:
    vlm = config["rynnbrain"]
    supplied = dict(vlm.get("lora", {}).get("training", {}))
    settings = {
        "epochs": 3, "learning_rate": 1e-4, "gradient_accumulation_steps": 8,
        "max_grad_norm": 1.0, "max_tokens": 16384, "seed": 42,
        "precision": "bfloat16", "gradient_checkpointing": True,
        "class_weight": "balanced", "input_mode": "raw", **supplied,
    }
    settings["output_dir"] = supplied.get("output_dir") or str(Path(vlm["output_dir"]) / "lora_adapter")
    settings["benchmark_dir"] = supplied.get("benchmark_dir") or f"{vlm['output_dir']}_benchmark"
    for key in ("epochs", "gradient_accumulation_steps", "max_tokens"):
        if not isinstance(settings[key], int) or settings[key] < 1:
            raise ValueError(f"lora.training.{key} must be a positive integer.")
    for key in ("learning_rate", "max_grad_norm"):
        if not math.isfinite(float(settings[key])) or float(settings[key]) <= 0:
            raise ValueError(f"lora.training.{key} must be positive and finite.")
    if settings["precision"] not in {"bfloat16", "float16"}:
        raise ValueError("LoRA training precision must be bfloat16 or float16.")
    if settings["class_weight"] not in {"balanced", "none"}:
        raise ValueError("LoRA class_weight must be balanced or none.")
    if settings["input_mode"] not in {"raw", "heatmap", "raw_heatmap"}:
        raise ValueError("LoRA input_mode must be raw, heatmap or raw_heatmap.")
    if vlm.get("source", "generated_videos") not in {"rosbag", "generated_videos"}:
        raise ValueError("rynnbrain.source must be rosbag or generated_videos.")
    if vlm.get("model", {}).get("lora_adapter_path"):
        raise ValueError("Set model.lora_adapter_path to null before training a new adapter from the base model.")
    if vlm.get("generation", {}).get("do_sample", False):
        raise ValueError("LoRA requires generation.do_sample=false for repeatable nominal context.")
    if not vlm.get("reference_bags") or int(vlm.get("num_frames", 4)) < 1:
        raise ValueError("LoRA needs reference_bags and a positive num_frames.")
    return settings


def _load_execution_images(row: dict[str, str], config: dict[str, Any], settings: dict[str, Any]) -> list:
    # Reuse the inference sampling, camera ordering and raw/heatmap pairing.
    from .run import _heatmap_inputs, _raw_inputs, _slug

    vlm = config["rynnbrain"]
    topics = list(vlm.get("camera_topics", config.get("camera_topics", [])))
    if not topics:
        raise ValueError("At least one test camera topic is required.")
    mode = settings["input_mode"]
    count = int(vlm.get("num_frames", 4))
    source = vlm.get("source", "generated_videos")
    start = float(vlm.get("sampling_start_sec", 0.0))
    end = vlm.get("sampling_end_sec")
    end = float(end) if end is not None else None
    directory = Path(settings["benchmark_dir"]) / row["bag_name"]
    raw = []
    if mode in {"raw", "raw_heatmap"}:
        if source == "rosbag":
            raw, _ = _raw_inputs(Path(row["bag_path"]), topics, count)
        else:
            paths = {topic: str(directory / f"{_slug(topic)}_raw_original.mp4") for topic in topics}
            raw, _ = _heatmap_inputs(paths, topics, count, start, end)
            raw = [(label.replace("heatmap", "raw frame"), image) for label, image in raw]
    if mode == "raw":
        return raw
    paths = {topic: str(directory / f"{_slug(topic)}_heatmap.mp4") for topic in topics}
    heatmaps, _ = _heatmap_inputs(paths, topics, count, start, end)
    if mode == "heatmap":
        return heatmaps
    if len(raw) != len(heatmaps):
        raise ValueError(f"Raw/heatmap frame counts differ for {row['bag_name']}.")
    return [item for pair in zip(raw, heatmaps) for item in pair]


def train(config: dict[str, Any], settings: dict[str, Any], rows: list[dict[str, str]]) -> None:
    import torch
    from PIL import Image
    from .model import RynnBrainModel
    from .run import _raw_inputs

    if not torch.cuda.is_available():
        raise RuntimeError("LoRA training requires the NVIDIA lab GPU; use --validate-only to check bag selection here.")
    if settings["precision"] == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise ValueError("This GPU does not support bfloat16; set lora.training.precision to float16.")
    vlm = config["rynnbrain"]
    output = Path(settings["output_dir"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"LoRA output is not empty: {output}. Choose a new training.output_dir for a new run.")
    if vlm.get("raw_video_paths") or vlm.get("heatmap_video_paths"):
        raise ValueError("LoRA uses per-bag benchmark videos; global video-path overrides cannot identify multiple bags.")
    if len({row["bag_name"] for row in rows}) != len(rows):
        raise ValueError("LoRA training requires unique bag directory names for saved sample provenance.")
    output.mkdir(parents=True, exist_ok=True)
    random.seed(int(settings["seed"]))
    torch.manual_seed(int(settings["seed"]))
    torch.cuda.manual_seed_all(int(settings["seed"]))
    model_config = dict(vlm["model"])
    model_config.update(dtype=settings["precision"], device_map={"": "cuda:0"}, input_device="cuda:0")
    model_config.pop("max_memory", None)  # Inference CPU offload is not a training strategy.
    wrapper = RynnBrainModel(model_config)
    generation = dict(vlm.get("generation", {}))
    topics = list(vlm.get("memory_camera_topics", vlm.get("camera_topics", config.get("camera_topics", []))))
    nominal_images = []
    for bag in vlm["reference_bags"]:
        images, _ = _raw_inputs(Path(bag), topics, int(vlm.get("num_frames", 4)))
        nominal_images.extend(images)
    if not nominal_images:
        raise ValueError("No nominal reference images were loaded.")
    task = vlm.get("task_description", "Robot manipulation task")
    nominal_turn = {"role": "user", "images": nominal_images, "text": task_context_prompt(task)}
    nominal_response = wrapper.generate_nominal(nominal_turn, generation)
    nominal_message = wrapper.conversation_message(nominal_turn)

    # Decode each ROS bag only once; retain resized lossless frames for all epochs.
    prepared = []
    for index, row in enumerate(rows):
        images = _load_execution_images(row, config, settings)
        if not images:
            raise ValueError(f"No input images for {row['bag_name']}.")
        folder = output / "training_frames" / f"{index:04d}"
        folder.mkdir(parents=True)
        files = []
        for image_index, (label, image) in enumerate(images):
            path = folder / f"{image_index:03d}.png"
            wrapper._resize(image).save(path)
            files.append((label, path))
        prepared.append({**row, "images": files})
        print(f"Prepared {row['split']} {row['bag_name']} ({row['label']}, {len(files)} images)", flush=True)

    def example(sample: dict[str, Any]) -> dict[str, Any]:
        images = []
        for label, path in sample["images"]:
            with Image.open(path) as image:
                images.append((label, image.convert("RGB")))
        turn = {"role": "user", "images": images, "text": evaluation_prompt_multiturn(task, settings["input_mode"])}
        conversation = [nominal_message, {"role": "assistant", "content": [{"type": "text", "text": nominal_response}]}, wrapper.conversation_message(turn)]
        prefix = wrapper.tokenize_conversation(conversation, generation)
        data = supervised_example(prefix, wrapper.processor.tokenizer, sample["label"], settings["max_tokens"])
        return {key: value.to(wrapper.input_device) if isinstance(value, torch.Tensor) else value for key, value in data.items()}

    wrapper.model = attach_lora(wrapper.model, dict(vlm.get("lora", {})))
    model = wrapper.model
    model.config.use_cache = False
    if settings["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    trainable_count = sum(parameter.numel() for parameter in parameters)
    print(f"Training {trainable_count:,} LoRA parameters; base model and vision encoder are frozen.", flush=True)
    manifest = {
        "base_model": vlm["model"]["model_id"], "bags": rows,
        "input_mode": settings["input_mode"], "settings": settings,
        "vlm_config": vlm, "nominal_response": nominal_response,
        "trainable_parameters": trainable_count,
        "supervision": "Final Decision: success/failure plus EOS only; context and images masked from loss.",
        "nominal_turn": "Frozen base model; adapter applies only to the execution decision turn.",
    }
    (output / "training_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    training = [sample for sample in prepared if sample["split"] == "train"]
    validation = [sample for sample in prepared if sample["split"] == "validation"]
    counts = {label: sum(sample["label"] == label for sample in training) for label in ("normal", "fail")}
    weights = {label: len(training) / (2 * count) if settings["class_weight"] == "balanced" else 1.0 for label, count in counts.items()}
    optimizer = torch.optim.AdamW(parameters, lr=float(settings["learning_rate"]), weight_decay=0.0)
    dtype = getattr(torch, settings["precision"])
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    step, best_loss = 0, float("inf")
    best_epoch = None
    history = []
    with (output / "loss_history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["epoch", "optimizer_step", "training_loss", "update_applied"])
        writer.writeheader()
        for epoch in range(1, settings["epochs"] + 1):
            model.train()
            order = list(training)
            random.Random(settings["seed"] + epoch).shuffle(order)
            epoch_loss = 0.0
            accumulation = settings["gradient_accumulation_steps"]
            for start in range(0, len(order), accumulation):
                group = order[start:start + accumulation]
                optimizer.zero_grad(set_to_none=True)
                group_loss = 0.0
                for sample in group:
                    with torch.autocast("cuda", dtype=dtype):
                        loss = answer_loss(model, example(sample)) * weights[sample["label"]]
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"Non-finite training loss for {sample['bag_name']}.")
                    group_loss += float(loss.detach())
                    scaler.scale(loss / len(group)).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(parameters, float(settings["max_grad_norm"]), error_if_nonfinite=not scaler.is_enabled())
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                applied = scaler.get_scale() >= previous_scale
                step += int(applied)
                epoch_loss += group_loss
                writer.writerow({"epoch": epoch, "optimizer_step": step, "training_loss": group_loss / len(group), "update_applied": applied})
                stream.flush()
                print(f"Epoch {epoch}, step {step}: loss={group_loss / len(group):.6f}, update_applied={applied}", flush=True)
            model.eval()
            validation_loss = None
            if validation:
                losses = []
                with torch.no_grad():
                    for sample in validation:
                        with torch.autocast("cuda", dtype=dtype):
                            losses.append(float(answer_loss(model, example(sample))))
                validation_loss = sum(losses) / len(losses)
                if not math.isfinite(validation_loss):
                    raise RuntimeError("Non-finite validation loss.")
            if step == 0:
                raise RuntimeError("No optimizer update succeeded; check precision and learning rate.")
            # With validation keep its best epoch; otherwise keep the latest epoch.
            if validation_loss is None or validation_loss < best_loss:
                model.save_pretrained(output, safe_serialization=True)
                best_epoch = epoch
                best_loss = validation_loss if validation_loss is not None else float("inf")
            history.append({"epoch": epoch, "training_loss": epoch_loss / len(training), "validation_loss": validation_loss})
            (output / "training_summary.json").write_text(json.dumps({
                "saved_epoch": best_epoch, "optimizer_steps": step, "epochs": history,
                "selection": "lowest validation loss" if validation else "last epoch (no validation supplied)",
            }, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved LoRA adapter: {output}")
    print(f"For evaluation set rynnbrain.model.lora_adapter_path to {str(output)!r}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("pipeline_config.json"))
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("ANOMALY_DATA_ROOT", "/data")))
    parser.add_argument("--validate-only", action="store_true", help="Check explicit bag lists without loading the model or training.")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    settings = training_settings(config)
    rows = resolve_bag_selection(settings, args.data_root, config["rynnbrain"]["reference_bags"])
    print(json.dumps({"bags": rows, "output_dir": settings["output_dir"]}, indent=2))
    if not args.validate_only:
        train(config, settings, rows)


if __name__ == "__main__":
    main()
