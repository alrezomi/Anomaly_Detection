"""RynnValue model loading and inference."""

from __future__ import annotations

import re
from typing import Any

from PIL import Image
import torch
from transformers import AutoConfig, AutoModel, AutoProcessor


DEFAULT_MODEL_ID = "Alibaba-DAMO-Academy/RynnValue-8B"


def _match_first_input_dtype(
    module: torch.nn.Module,
    inputs: tuple[Any, ...],
) -> tuple[Any, ...]:
    """Match a custom value head's activation to its parameter dtype."""
    if not inputs or not isinstance(inputs[0], torch.Tensor):
        return inputs
    parameter = next(module.parameters(), None)
    value = inputs[0]
    if parameter is None or not value.is_floating_point():
        return inputs
    if value.dtype == parameter.dtype:
        return inputs
    return (value.to(dtype=parameter.dtype), *inputs[1:])


def _parse_analysis(text: str) -> dict[str, str | None]:
    patterns = {
        "description": r"(?:^|\n)\s*-?\s*Video Description\s*:\s*(.+)",
        "match": r"(?:^|\n)\s*-?\s*Match\s*:\s*(Yes|No)\b",
        "success": r"(?:^|\n)\s*-?\s*Success\s*:\s*(Yes|No)\b",
    }
    parsed: dict[str, str | None] = {}
    for name, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        parsed[name] = match.group(1).strip() if match else None
    return parsed


class RynnValueModel:
    def __init__(self, model_config: dict[str, Any]) -> None:
        self.model_id = model_config.get("model_id", DEFAULT_MODEL_ID)
        dtype_name = model_config.get("dtype", "bfloat16")
        dtype = getattr(torch, dtype_name)
        config = AutoConfig.from_pretrained(self.model_id, trust_remote_code=True)
        config._attn_implementation = "pred_slot_isolated_eager"

        load_options: dict[str, Any] = {
            "config": config,
            "trust_remote_code": True,
            "dtype": dtype,
            "low_cpu_mem_usage": True,
        }
        device_map = model_config.get("device_map")
        if device_map is not None:
            load_options["device_map"] = device_map
        if model_config.get("max_memory"):
            load_options["max_memory"] = {
                (int(key) if str(key).isdigit() else key): value
                for key, value in model_config["max_memory"].items()
            }

        self.processor = AutoProcessor.from_pretrained(
            self.model_id, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(self.model_id, **load_options)
        if device_map is None:
            device = model_config.get(
                "input_device", "cuda" if torch.cuda.is_available() else "cpu"
            )
            self.model = self.model.to(device=device, dtype=dtype)

        # The released custom value heads initialize in FP32. With
        # device_map="auto", Accelerate can offload them without converting
        # their weights, while the backbone emits FP16/BF16 activations.
        # Match only those head inputs to their actual parameter dtype.
        for head in getattr(self.model, "value_heads", None) or []:
            head.register_forward_pre_hook(_match_first_input_dtype)
        relative_head = getattr(self.model, "relative_value_head", None)
        if relative_head is not None:
            relative_head.register_forward_pre_hook(_match_first_input_dtype)

        self.device = torch.device(
            model_config.get("input_device", self.model.device)
        )
        self.max_image_size = int(model_config.get("max_image_size", 640))
        self.model.eval()

    @staticmethod
    def _values(tensor: torch.Tensor | None, count: int) -> list[float]:
        if tensor is None or count < 1:
            return []
        values = tensor.detach().float()
        if values.numel() % count:
            raise ValueError(
                f"Unexpected RynnValue output shape {tuple(values.shape)} "
                f"for {count} frames"
            )
        return values.reshape(-1, count).mean(dim=0).cpu().tolist()

    def predict(
        self,
        instruction: str,
        images: list[Image.Image],
        robot_description: str | None,
        camera_description: str | None,
    ) -> dict[str, Any]:
        resized: list[Image.Image] = []
        for image in images:
            output = image.convert("RGB").copy()
            output.thumbnail((self.max_image_size, self.max_image_size))
            resized.append(output)
        processed = self.processor.process_episode(
            instruction=instruction,
            images=resized,
            robot_description=robot_description,
            camera_description=camera_description,
        )

        inputs: dict[str, torch.Tensor] = {}
        for key in ("input_ids", "attention_mask"):
            inputs[key] = processed[key].to(self.device).long()
        for key in ("pixel_values", "pixel_values_videos"):
            value = processed.get(key)
            if value is not None:
                inputs[key] = value.flatten(0, 1).to(self.device)
        for key in ("image_grid_thw", "video_grid_thw"):
            value = processed.get(key)
            if value is not None:
                inputs[key] = value.flatten(0, 1).to(self.device).long()

        with torch.inference_mode():
            output = self.model(**inputs)
        remaining = self._values(output.value.pred_value, len(images))
        entropy = self._values(output.value.entropy, len(images))
        relative = self._values(
            output.relative.pred_value, max(0, len(images) - 1)
        )
        del output

        tokenizer = self.processor.tokenizer
        eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        input_length = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                num_beams=1,
                eos_token_id=eos_token_id,
                pad_token_id=eos_token_id,
                use_cache=True,
            )
        analysis_text = tokenizer.decode(
            generated[0, input_length:], skip_special_tokens=True
        ).strip()
        return {
            "remaining": remaining,
            "relative": relative,
            "entropy": entropy,
            "analysis_text": analysis_text,
            **_parse_analysis(analysis_text),
        }
