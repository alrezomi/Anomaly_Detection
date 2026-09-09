"""Configurable RynnBrain inference wrapper."""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


COP_REPRESENTATION_ID = "qwen3vl_final_norm_last_turn2_prompt_token_v1"


@dataclass(frozen=True)
class MultiturnGeneration:
    """Text responses plus an optional fixed-size CoP context vector."""

    nominal_response: str
    evaluation_response: str
    cop_vector: torch.Tensor | None = None


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

    def _final_language_norm(self) -> Any:
        """Return Qwen3-VL's final language normalization module."""
        base_model = getattr(self.model, "model", None)
        language_model = getattr(base_model, "language_model", None)
        norm = getattr(language_model, "norm", None)
        if norm is None or not hasattr(norm, "register_forward_hook"):
            raise RuntimeError(
                "Could not locate model.model.language_model.norm; "
                "CoP-vector extraction requires the Qwen3-VL architecture."
            )
        return norm

    def _generate_multiturn(
        self,
        turns: list[dict[str, Any]],
        generation: dict[str, Any],
        capture_cop_vector: bool,
    ) -> MultiturnGeneration:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        conversation: list[dict[str, Any]] = []
        responses: list[str] = []
        cop_vector: torch.Tensor | None = None

        for turn_index, turn in enumerate(turns):
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

            capture_this_turn = capture_cop_vector and turn_index == len(turns) - 1
            captured: list[torch.Tensor] = []
            hook_handle = None
            if capture_this_turn:
                prompt_length = int(inputs["input_ids"].shape[1])
                attention_mask = inputs.get("attention_mask")
                if attention_mask is None:
                    last_prompt_index = prompt_length - 1
                else:
                    valid_positions = torch.nonzero(
                        attention_mask[0], as_tuple=False
                    ).flatten()
                    if valid_positions.numel() == 0:
                        raise ValueError("The turn-2 prompt has no unmasked tokens.")
                    last_prompt_index = int(valid_positions[-1].item())

                def capture_last_prompt_state(
                    _module: Any, _arguments: tuple[Any, ...], output: Any
                ) -> None:
                    hidden = output[0] if isinstance(output, tuple) else output
                    if (
                        not captured
                        and isinstance(hidden, torch.Tensor)
                        and hidden.ndim == 3
                        and hidden.shape[0] == 1
                        and hidden.shape[1] == prompt_length
                    ):
                        captured.append(
                            hidden[0, last_prompt_index].detach().float().cpu().clone()
                        )

                hook_handle = self._final_language_norm().register_forward_hook(
                    capture_last_prompt_state
                )

            try:
                with torch.inference_mode():
                    output_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=int(generation.get("max_new_tokens", 500)),
                        do_sample=bool(generation.get("do_sample", False)),
                        use_cache=True,
                    )
            finally:
                if hook_handle is not None:
                    hook_handle.remove()

            if capture_this_turn:
                if not captured:
                    raise RuntimeError(
                        "The final language layer did not expose the full turn-2 "
                        "prompt state during generation."
                    )
                vector = captured[0]
                text_config = getattr(self.model.config, "text_config", None)
                expected_size = getattr(text_config, "hidden_size", None)
                if vector.ndim != 1:
                    raise RuntimeError(
                        f"Expected a one-dimensional CoP vector, got {tuple(vector.shape)}."
                    )
                if expected_size is not None and vector.numel() != int(expected_size):
                    raise RuntimeError(
                        f"Expected CoP vector width {expected_size}, got {vector.numel()}."
                    )
                if not torch.isfinite(vector).all():
                    raise RuntimeError("The captured CoP vector contains non-finite values.")
                cop_vector = vector

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
        return MultiturnGeneration(responses[0], responses[1], cop_vector)

    def generate_multiturn(
        self,
        turns: list[dict[str, Any]],
        generation: dict[str, Any],
    ) -> tuple[str, str]:
        """Generate the two responses without exporting an internal vector."""
        result = self._generate_multiturn(turns, generation, capture_cop_vector=False)
        return result.nominal_response, result.evaluation_response

    def generate_multiturn_with_cop_vector(
        self,
        turns: list[dict[str, Any]],
        generation: dict[str, Any],
    ) -> MultiturnGeneration:
        """Generate responses and capture the final turn-2 prompt representation."""
        result = self._generate_multiturn(turns, generation, capture_cop_vector=True)
        if result.cop_vector is None:
            raise RuntimeError("CoP-vector extraction was requested but returned no vector.")
        return result
