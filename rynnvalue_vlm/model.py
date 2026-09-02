"""Hugging Face wrapper for the released RynnValue checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image
import torch
from transformers import AutoConfig, AutoModel, AutoProcessor

from .analysis import parse_analysis


@dataclass(frozen=True)
class RynnValuePrediction:
    """Native model outputs for one instruction-conditioned trajectory."""

    remaining_time_seconds: list[float]
    relative_time_seconds: list[float]
    entropy: list[float]
    analysis_text: str
    video_description: str | None
    match: str | None
    success: str | None


class RynnValueModel:
    """Load RynnValue once and evaluate multiple trajectories."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.model_id = config.get(
            "model_id", "Alibaba-DAMO-Academy/RynnValue-8B"
        )
        self.max_image_side = int(config.get("max_image_side", 640))
        self.trust_remote_code = bool(config.get("trust_remote_code", True))
        dtype_name = str(config.get("dtype", "bfloat16"))
        if not hasattr(torch, dtype_name):
            raise ValueError(f"Unsupported torch dtype: {dtype_name}")
        self.dtype = getattr(torch, dtype_name)

        hf_config = AutoConfig.from_pretrained(
            self.model_id, trust_remote_code=self.trust_remote_code
        )
        # Released checkpoints require the custom prediction-slot isolation
        # attention for valid temporal values. The setting is not persisted by
        # every exported checkpoint, so force the official inference default.
        hf_config._attn_implementation = config.get(
            "attn_implementation", "pred_slot_isolated_eager"
        )

        load_kwargs: dict[str, Any] = {
            "config": hf_config,
            "trust_remote_code": self.trust_remote_code,
            "torch_dtype": self.dtype,
            "low_cpu_mem_usage": True,
        }
        device_map = config.get("device_map")
        if device_map is not None:
            load_kwargs["device_map"] = device_map
        if config.get("max_memory"):
            load_kwargs["max_memory"] = {
                (int(key) if str(key).isdigit() else key): value
                for key, value in config["max_memory"].items()
            }

        self.processor = AutoProcessor.from_pretrained(
            self.model_id, trust_remote_code=self.trust_remote_code
        )
        self.model = AutoModel.from_pretrained(self.model_id, **load_kwargs)

        if device_map is None:
            requested_device = str(
                config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
            )
            self.model = self.model.to(device=requested_device, dtype=self.dtype)
            self.input_device = torch.device(requested_device)
        else:
            default_input_device = getattr(self.model, "device", None)
            self.input_device = torch.device(
                config.get("input_device", default_input_device or "cuda")
            )
        self.model.eval()

    def _resize(self, image: Image.Image) -> Image.Image:
        output = image.convert("RGB").copy()
        if self.max_image_side > 0:
            output.thumbnail((self.max_image_side, self.max_image_side))
        return output

    def _model_inputs(self, processed: dict[str, Any]) -> dict[str, torch.Tensor]:
        model_inputs: dict[str, torch.Tensor] = {}
        for key in ("input_ids", "attention_mask"):
            value = processed.get(key)
            if value is not None:
                model_inputs[key] = value.to(self.input_device).long()

        for key in ("pixel_values", "pixel_values_videos"):
            value = processed.get(key)
            if value is not None:
                if value.ndim >= 3:
                    value = value.flatten(0, 1)
                model_inputs[key] = value.to(self.input_device)

        for key in ("image_grid_thw", "video_grid_thw"):
            value = processed.get(key)
            if value is not None:
                if value.ndim >= 3:
                    value = value.flatten(0, 1)
                model_inputs[key] = value.to(self.input_device).long()
        return model_inputs

    @staticmethod
    def _reduce_slots(
        value: torch.Tensor | None, expected_slots: int
    ) -> list[float]:
        """Reduce optional head/batch axes to one scalar per query slot."""

        if value is None or expected_slots < 1:
            return []
        tensor = value.detach().float()
        if tensor.numel() % expected_slots != 0:
            raise ValueError(
                "RynnValue returned an unexpected number of predictions: "
                f"shape={tuple(tensor.shape)}, expected_slots={expected_slots}"
            )
        tensor = tensor.reshape(-1, expected_slots).mean(dim=0)
        return [float(item) for item in tensor.cpu().tolist()]

    def predict(
        self,
        instruction: str,
        images: list[Image.Image],
        generation: dict[str, Any] | None = None,
        robot_description: str | None = None,
        camera_description: str | None = None,
    ) -> RynnValuePrediction:
        """Predict temporal values and native verification for one trajectory."""

        if not images:
            raise ValueError("RynnValue requires at least one image.")
        generation = generation or {}
        processed = self.processor.process_episode(
            instruction=instruction,
            images=[self._resize(image) for image in images],
            robot_description=robot_description,
            camera_description=camera_description,
        )
        model_inputs = self._model_inputs(processed)

        with torch.inference_mode():
            output = self.model(**model_inputs)

        remaining = self._reduce_slots(
            getattr(output.value, "pred_value", None), len(images)
        )
        entropy = self._reduce_slots(
            getattr(output.value, "entropy", None), len(images)
        )
        relative_output = getattr(output, "relative", None)
        relative_tensor = (
            getattr(relative_output, "pred_value", None)
            if relative_output is not None
            else None
        )
        relative = self._reduce_slots(relative_tensor, max(0, len(images) - 1))
        # The forward output can retain a large KV cache. Temporal values have
        # already been copied to CPU, so release it before language generation.
        del relative_tensor, output

        analysis_text = ""
        if bool(generation.get("enabled", True)):
            tokenizer = self.processor.tokenizer
            eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
            input_length = model_inputs["input_ids"].shape[1]
            with torch.inference_mode():
                generated = self.model.generate(
                    **model_inputs,
                    max_new_tokens=int(generation.get("max_new_tokens", 128)),
                    do_sample=False,
                    num_beams=1,
                    eos_token_id=eos_token_id,
                    pad_token_id=eos_token_id,
                    use_cache=True,
                )
            generated_ids = generated[0, input_length:]
            analysis_text = tokenizer.decode(
                generated_ids, skip_special_tokens=True
            ).strip()

        parsed = parse_analysis(analysis_text)
        return RynnValuePrediction(
            remaining_time_seconds=remaining,
            relative_time_seconds=relative,
            entropy=entropy,
            analysis_text=analysis_text,
            video_description=parsed["description"],
            match=parsed["match"],
            success=parsed["success"],
        )
