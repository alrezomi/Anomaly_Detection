"""LoRA adapter utilities; imports of GPU/PEFT dependencies stay lazy."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def resolve_bag_selection(training: dict[str, Any], data_root: Path, references: list[str]) -> list[dict[str, str]]:
    """Resolve explicit, labeled train/validation lists without an automatic split."""
    discovered: list[Path] | None = None

    def resolve(value: str) -> Path:
        nonlocal discovered
        path = Path(value)
        candidate = path if path.is_absolute() else data_root / path
        if (candidate / "metadata.yaml").is_file():
            return candidate.resolve()
        if len(path.parts) == 1:
            if discovered is None:
                discovered = sorted({p.parent.resolve() for p in data_root.glob("**/metadata.yaml")})
            matches = [p for p in discovered if p.name == value]
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                raise ValueError(f"Ambiguous bag name {value!r}; use its full path.")
        raise FileNotFoundError(f"ROS bag not found: {value}")

    reference_names = {Path(value).name for value in references}
    rows = []
    seen: set[Path] = set()
    for split, prefix in (("train", ""), ("validation", "validation_")):
        for label, suffix in (("normal", "normal_bags"), ("fail", "failure_bags")):
            values = training.get(prefix + suffix, [])
            if not isinstance(values, list) or any(not isinstance(v, str) or not v.strip() for v in values):
                raise ValueError(f"lora.training.{prefix + suffix} must be a list of bag paths/names.")
            for value in values:
                path = resolve(value)
                if path in seen:
                    raise ValueError(f"Bag selected more than once, across labels or splits: {path}")
                if path.name in reference_names:
                    raise ValueError(f"Reference demonstration cannot be a supervised sample: {path}")
                seen.add(path)
                rows.append({"bag_name": path.name, "bag_path": str(path), "label": label, "split": split})
    for split in ("train", "validation"):
        labels = {r["label"] for r in rows if r["split"] == split}
        if (split == "train" or labels) and labels != {"normal", "fail"}:
            raise ValueError(f"Select both normal and failure bags for {split}; no automatic selection is performed.")
    return rows


def attach_lora(model: Any, config: dict[str, Any]) -> Any:
    """Attach LoRA only to language-decoder attention projections."""
    import torch
    from peft import LoraConfig, TaskType, get_peft_model

    projections = config.get("target_modules", ["q_proj", "v_proj"])
    if not isinstance(projections, list) or not projections or not set(projections) <= {"q_proj", "k_proj", "v_proj", "o_proj"}:
        raise ValueError("LoRA target_modules must select q_proj/k_proj/v_proj/o_proj.")
    targets = [
        name for name, module in model.named_modules()
        if ".language_model.layers." in name and ".self_attn." in name
        and name.rsplit(".", 1)[-1] in projections and isinstance(module, torch.nn.Linear)
    ]
    if not targets or {name.rsplit(".", 1)[-1] for name in targets} != set(projections):
        raise ValueError("Cannot find the requested Qwen3-VL language attention projections.")
    rank, alpha = int(config.get("rank", 8)), int(config.get("alpha", 16))
    dropout = float(config.get("dropout", 0.05))
    if rank < 1 or alpha < 1 or not 0 <= dropout < 1:
        raise ValueError("LoRA rank/alpha must be positive and dropout must be in [0, 1).")
    model.requires_grad_(False)
    adapted = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=rank, lora_alpha=alpha,
        lora_dropout=dropout, target_modules=targets, bias="none",
    ))
    trainable = [name for name, parameter in adapted.named_parameters() if parameter.requires_grad]
    if not trainable or any("lora_" not in name or ".language_model." not in name for name in trainable):
        raise RuntimeError("Expected only language-model LoRA parameters to be trainable.")
    return adapted


def supervised_example(prefix: dict[str, Any], tokenizer: Any, label: str, max_tokens: int) -> dict[str, Any]:
    """Keep images/context unmasked as inputs, but supervise only the final answer."""
    import torch

    decisions = {"normal": "success", "fail": "failure"}
    if label not in decisions:
        raise ValueError(f"Unknown training label: {label}")
    if tokenizer.eos_token_id is None:
        raise ValueError("The model tokenizer must define an EOS token.")
    answer = tokenizer.encode(f"Decision: {decisions[label]}", add_special_tokens=False)
    answer.append(tokenizer.eos_token_id)
    answer_ids = torch.tensor([answer], dtype=prefix["input_ids"].dtype)
    prefix_length = prefix["input_ids"].shape[1]
    if prefix["input_ids"].shape[0] != 1 or prefix_length < 1:
        raise ValueError("LoRA training expects one nonempty conversation per batch.")
    if prefix_length + len(answer) > max_tokens:
        raise ValueError(
            f"Training example needs {prefix_length + len(answer)} tokens (limit {max_tokens}). "
            "Reduce num_frames/max_image_size or increase lora.training.max_tokens; images are never truncated."
        )
    result = dict(prefix)
    result["input_ids"] = torch.cat([prefix["input_ids"], answer_ids], dim=1)
    result["attention_mask"] = torch.cat([prefix["attention_mask"], torch.ones_like(answer_ids)], dim=1)
    result["labels"] = result["input_ids"].clone()
    result["labels"][:, :prefix_length] = -100
    # Position IDs must be recomputed by Qwen3-VL after appending the answer.
    result.pop("position_ids", None)
    result.pop("token_type_ids", None)
    return result


def answer_loss(model: Any, example: dict[str, Any]) -> Any:
    """Causal answer-token loss without materializing vocabulary logits for images."""
    import torch
    from torch.nn import functional as F

    inputs = dict(example)
    labels = inputs.pop("labels")
    positions = torch.nonzero(labels[0, 1:] != -100, as_tuple=False).flatten()
    if positions.numel() == 0:
        raise ValueError("The training example has no supervised answer tokens.")
    outputs = model(**inputs, use_cache=False, logits_to_keep=positions)
    return F.cross_entropy(outputs.logits[0].float(), labels[0, positions + 1])


def adapter_training_bag_names(adapter_path: str | Path) -> set[str]:
    manifest_path = Path(adapter_path) / "training_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing LoRA training provenance: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {row["bag_name"] for row in manifest["bags"] if row["split"] == "train"}


def load_lora_adapter(model: Any, adapter_path: str | Path, model_id: str) -> tuple[Any, str, set[str]]:
    from peft import PeftConfig, PeftModel

    root = Path(adapter_path)
    config = PeftConfig.from_pretrained(str(root), local_files_only=True)
    if config.base_model_name_or_path != model_id:
        raise ValueError(f"LoRA adapter was trained for {config.base_model_name_or_path}, not {model_id}.")
    training_bags = adapter_training_bag_names(root)
    digest = hashlib.sha256()
    for filename in ("adapter_config.json", "adapter_model.safetensors"):
        with (root / filename).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return PeftModel.from_pretrained(model, str(root), is_trainable=False), digest.hexdigest(), training_bags
