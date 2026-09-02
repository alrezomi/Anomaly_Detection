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

    def generate_multiturn(
        self,
        turns: list[dict[str, Any]],
        generation: dict[str, Any],
    ) -> tuple[str, str]:
        """
        Generate response using multi-turn conversation.
        
        Args:
            turns: List of conversation turns, each with:
                - "role": "user" or "assistant"
                - "images": list of (label, Image) tuples (optional, for user turns)
                - "text": text response (for assistant turns) or prompt (for user turns)
            generation: Generation config dictionary
            
        Returns:
            A tuple containing the first and final model responses.
        """
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        conversation: list[dict[str, Any]] = []
        responses: list[str] = []

        for turn in turns:
            role = turn["role"]
            content: list[dict[str, Any]] = []
            
            # Add images if provided (typically for user turns)
            if "images" in turn and turn["images"]:
                for label, image in turn["images"]:
                    content.append({"type": "text", "text": label})
                    content.append({"type": "image", "image": self._resize(image)})
            
            # Add text (prompt or assistant response)
            content.append({"type": "text", "text": turn["text"]})
            conversation.append({"role": role, "content": content})

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
            response = self.processor.decode(
                new_tokens[0], skip_special_tokens=True
            ).strip()
            responses.append(response)

            conversation.append(
                {"role": "assistant", "content": [{"type": "text", "text": response}]}
            )

        if len(responses) != 2:
            raise ValueError("generate_multiturn requires exactly two user turns")
        return responses[0], responses[1]
