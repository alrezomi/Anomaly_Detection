"""Configurable RynnBrain inference wrapper."""

from __future__ import annotations

import gc
from typing import Any

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


class RynnBrainModel:
    def __init__(self, config: dict[str, Any]) -> None:
        self.model_id = config["model_id"]
        self.max_image_size = int(config.get("max_image_size", 640))
        dtype_name = config.get("dtype", "float16")
        dtype = getattr(torch, dtype_name)
        kwargs: dict[str, Any] = {
            "device_map": config.get("device_map", "auto"),
            "dtype": dtype,
            "low_cpu_mem_usage": True,
            "trust_remote_code": bool(config.get("trust_remote_code", True)),
        }
        if config.get("max_memory"):
            kwargs["max_memory"] = {
                (int(key) if str(key).isdigit() else key): value
                for key, value in config["max_memory"].items()
            }
        self.processor = AutoProcessor.from_pretrained(
            self.model_id, trust_remote_code=kwargs["trust_remote_code"]
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id, **kwargs
        )
        self.model.eval()
        self.input_device = config.get(
            "input_device", "cuda" if torch.cuda.is_available() else "cpu"
        )

    def _resize(self, image: Image.Image) -> Image.Image:
        output = image.convert("RGB").copy()
        output.thumbnail((self.max_image_size, self.max_image_size))
        return output

    def generate(
        self,
        images: list[tuple[str, Image.Image]],
        prompt: str,
        generation: dict[str, Any],
    ) -> str:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        content: list[dict[str, Any]] = []
        for label, image in images:
            content.append({"type": "text", "text": label})
            content.append({"type": "image", "image": self._resize(image)})
        content.append({"type": "text", "text": prompt})
        conversation = [{"role": "user", "content": content}]
        template_kwargs = {
            "add_generation_prompt": True,
            "tokenize": True,
            "return_dict": True,
            "return_tensors": "pt",
        }
        try:
            inputs = self.processor.apply_chat_template(
                conversation,
                enable_thinking=bool(generation.get("enable_thinking", False)),
                **template_kwargs,
            )
        except TypeError:
            inputs = self.processor.apply_chat_template(
                conversation, **template_kwargs
            )
        inputs = inputs.to(self.input_device)
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=int(generation.get("max_new_tokens", 300)),
                do_sample=bool(generation.get("do_sample", False)),
                repetition_penalty=float(generation.get("repetition_penalty", 1.1)),
                no_repeat_ngram_size=int(generation.get("no_repeat_ngram_size", 6)),
                use_cache=True,
            )
        new_tokens = output_ids[:, inputs["input_ids"].shape[1]:]
        return self.processor.decode(
            new_tokens[0], skip_special_tokens=True
        ).strip()

    def text(self, prompt: str, generation: dict[str, Any]) -> str:
        return self.generate([], prompt, generation)
