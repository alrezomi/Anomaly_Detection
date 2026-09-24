"""Configurable RynnBrain inference wrapper."""

from __future__ import annotations

import gc
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


COP_REPRESENTATION_ID = "qwen3vl_final_norm_last_turn2_prompt_token_v1"


def _input_execution_device(model: Any) -> torch.device:
    """Follow the final placement, including Accelerate's offload hooks."""
    embeddings = model.get_input_embeddings()

    def hook_device(hook: Any) -> Any:
        device = getattr(hook, "execution_device", None)
        if device is not None:
            return device
        for child in getattr(hook, "hooks", ()):
            device = hook_device(child)
            if device is not None:
                return device
        return None

    modules = dict(model.named_modules())
    name = next((name for name, module in modules.items() if module is embeddings), None)
    # A hook on the embedding or its containing block can materialize weights
    # stored on CPU/meta on a different device just before the forward pass.
    if name is not None:
        parts = name.split(".") if name else []
        for depth in range(len(parts), -1, -1):
            module = modules[".".join(parts[:depth])]
            device = hook_device(getattr(module, "_hf_hook", None))
            if device is not None:
                return torch.device(f"cuda:{device}" if isinstance(device, int) else device)
    device = embeddings.weight.device
    if device.type == "meta":
        raise RuntimeError("Input embeddings are on meta without an execution-device hook.")
    return device


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
        self.adapter_identity = None
        self.lora_adapter_path = config.get("lora_adapter_path")
        self.adapter_training_bags: set[str] = set()
        if config.get("lora_adapter_path"):
            from .lora import load_lora_adapter

            self.model, self.adapter_identity, self.adapter_training_bags = load_lora_adapter(
                self.model, config["lora_adapter_path"], self.model_id
            )
        self.model.eval()
        if self.adapter_identity:
            print(f"Execution model: {self.model_id} + LoRA {self.lora_adapter_path}")
            print(f"LoRA adapter SHA256: {self.adapter_identity}")
        else:
            print(f"Execution model: {self.model_id} (base model; no LoRA adapter)")
        # PEFT can redispatch an automatically placed model during adapter load.
        # Resolve after loading; the configured preference may now be stale.
        self.input_device = _input_execution_device(self.model)
        print(f"Model input execution device: {self.input_device}")
        if self.input_device.type == "cpu" and str(config.get("input_device", "")).startswith("cuda"):
            print("Model inputs are on CPU after loading; inference may be slow. Check available GPU memory with nvidia-smi.")

    def _resize(self, image: Image.Image) -> Image.Image:
        output = image.convert("RGB").copy()
        output.thumbnail((self.max_image_size, self.max_image_size))
        return output

    def _final_language_norm(self) -> Any:
        """Return Qwen3-VL's final language normalization module."""
        unwrapped = self.model.get_base_model() if self.adapter_identity else self.model
        base_model = getattr(unwrapped, "model", None)
        language_model = getattr(base_model, "language_model", None)
        norm = getattr(language_model, "norm", None)
        if norm is None or not hasattr(norm, "register_forward_hook"):
            raise RuntimeError(
                "Could not locate model.model.language_model.norm; "
                "CoP-vector extraction requires the Qwen3-VL architecture."
            )
        return norm

    def conversation_message(self, turn: dict[str, Any]) -> dict[str, Any]:
        """Shared image order and resizing for inference and supervised training."""
        content: list[dict[str, Any]] = []
        for label, image in turn.get("images", []):
            content.append({"type": "text", "text": label})
            content.append({"type": "image", "image": self._resize(image)})
        content.append({"type": "text", "text": turn["text"]})
        return {"role": turn["role"], "content": content}

    def tokenize_conversation(self, conversation: list[dict[str, Any]], generation: dict[str, Any]) -> Any:
        template_kwargs = {
            "add_generation_prompt": True, "tokenize": True,
            "return_dict": True, "return_tensors": "pt",
        }
        try:
            return self.processor.apply_chat_template(
                conversation, enable_thinking=bool(generation.get("enable_thinking", False)),
                **template_kwargs,
            )
        except TypeError:
            return self.processor.apply_chat_template(conversation, **template_kwargs)

    def generate_nominal(self, turn: dict[str, Any], generation: dict[str, Any]) -> str:
        """Generate the frozen base model's nominal response for training context."""
        inputs = self.tokenize_conversation([self.conversation_message(turn)], generation).to(self.input_device)
        context = self.model.disable_adapter() if self.adapter_identity else nullcontext()
        with context, torch.inference_mode():
            output = self.model.generate(
                **inputs, max_new_tokens=int(generation.get("max_new_tokens", 500)),
                do_sample=bool(generation.get("do_sample", False)), use_cache=True,
            )
        return self.processor.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def generate_visual_evidence(
        self, turns: list[dict[str, Any]], nominal_response: str,
        evidence_prompt: str, generation: dict[str, Any],
    ) -> str:
        """Independent base-model observations, without the adapted decision."""
        conversation = [
            self.conversation_message(turns[0]),
            {"role": "assistant", "content": [{"type": "text", "text": nominal_response}]},
            self.conversation_message({**turns[1], "text": evidence_prompt}),
        ]
        inputs = self.tokenize_conversation(conversation, generation).to(self.input_device)
        context = self.model.disable_adapter() if self.adapter_identity else nullcontext()
        with context, torch.inference_mode():
            output = self.model.generate(
                **inputs, max_new_tokens=int(generation.get("max_new_tokens", 500)),
                do_sample=bool(generation.get("do_sample", False)), use_cache=True,
            )
        return self.processor.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def _generate_multiturn(
        self,
        turns: list[dict[str, Any]],
        generation: dict[str, Any],
        capture_cop_vector: bool,
        reference_response: str | None = None,
    ) -> MultiturnGeneration:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        conversation: list[dict[str, Any]] = []
        responses: list[str] = []
        cop_vector: torch.Tensor | None = None

        for turn_index, turn in enumerate(turns):
            conversation.append(self.conversation_message(turn))
            if turn_index == 0 and reference_response is not None:
                responses.append(reference_response)
                conversation.append({"role": "assistant", "content": [{"type": "text", "text": reference_response}]})
                continue
            inputs = self.tokenize_conversation(conversation, generation)
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
                # Keep the reference turn on the frozen base model as in training.
                # The learned adapter applies only to the test turn.
                adapter_context = (
                    self.model.disable_adapter()
                    if self.adapter_identity and turn_index == 0 else nullcontext()
                )
                with adapter_context, torch.inference_mode():
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
        *,
        reference_response: str | None = None,
    ) -> MultiturnGeneration:
        """Generate responses and capture the final turn-2 prompt representation."""
        result = self._generate_multiturn(
            turns, generation, capture_cop_vector=True, reference_response=reference_response
        )
        if result.cop_vector is None:
            raise RuntimeError("CoP-vector extraction was requested but returned no vector.")
        return result
