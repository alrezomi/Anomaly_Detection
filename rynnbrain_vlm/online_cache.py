"""Append-only Qwen3-VL image history with the existing prompt-end readout.

Single worker only. Evaluation suffixes are temporary; their KV states are
removed before another image is appended. No weight or prompt changes.
"""

from copy import deepcopy

import torch


class CachedExecution:
    def __init__(self, wrapper, reference_turn, reference_response, evaluation_text,
                 generation, max_frames=8):
        if not 1 <= max_frames <= 8:
            raise ValueError("Online execution supports one to eight frames.")
        self.wrapper = wrapper
        self.max_frames = max_frames
        self.reference_turn = reference_turn
        self.reference_response = reference_response
        self.evaluation_text = evaluation_text
        self.generation = generation
        unwrapped = wrapper.model.get_base_model() if wrapper.adapter_identity else wrapper.model
        self.base = unwrapped.model
        self.vision_end = unwrapped.config.vision_end_token_id
        self.reference_conversation = [wrapper.conversation_message(reference_turn),
            {"role": "assistant", "content": [{"type": "text", "text": reference_response}]}]
        kwargs = dict(add_generation_prompt=False, tokenize=True, return_dict=True, return_tensors="pt")
        try:
            inputs = wrapper.processor.apply_chat_template(self.reference_conversation,
                enable_thinking=bool(generation.get("enable_thinking", False)), **kwargs)
        except TypeError:
            inputs = wrapper.processor.apply_chat_template(self.reference_conversation, **kwargs)
        inputs = inputs.to(wrapper.input_device)
        self._validate(inputs)
        positions = self._positions(inputs)
        with torch.inference_mode():
            output = wrapper.model(**self._chunk(inputs, positions, 0, inputs["input_ids"].shape[1], 0),
                                   use_cache=True, logits_to_keep=1)
        self.reference_cache = output.past_key_values
        if self.reference_cache is None or not hasattr(self.reference_cache, "crop"):
            raise RuntimeError("This model did not return a supported KV cache.")
        self.reference_ids = inputs["input_ids"].clone()
        self.reference_image_count = len(inputs.get("image_grid_thw", []))
        self.reset()

    def reset(self):
        # The reference cache must remain pristine across executions.
        self.cache = deepcopy(self.reference_cache)
        self.cached_ids = self.reference_ids.clone()
        self.images = []
        self.failed = False

    @staticmethod
    def _validate(inputs):
        ids = inputs["input_ids"]
        if ids.ndim != 2 or ids.shape[0] != 1 or not bool(inputs["attention_mask"].all()):
            raise ValueError("Online cache requires one unpadded conversation.")
        if inputs.get("pixel_values_videos") is not None:
            raise ValueError("Online cache accepts individual images, not packed video tokens.")

    def _positions(self, inputs):
        # Compute full multimodal coordinates cheaply, then slice them. The
        # ordinary generation continuation assumes TEXT, not newly arriving images.
        positions, _ = self.base.get_rope_index(inputs["input_ids"],
            image_grid_thw=inputs.get("image_grid_thw"), attention_mask=inputs["attention_mask"])
        return positions

    def _chunk(self, inputs, positions, start, end, first_image, image_count=None):
        result = dict(input_ids=inputs["input_ids"][:, start:end],
            attention_mask=inputs["attention_mask"][:, :end], position_ids=positions[:, :, start:end],
            cache_position=torch.arange(start, end, device=self.wrapper.input_device))
        grids = inputs.get("image_grid_thw")
        if grids is not None:
            stop = len(grids) if image_count is None else first_image + image_count
            if stop > first_image:
                sizes = grids.prod(dim=-1)
                pixel_start = int(sizes[:first_image].sum().item())
                pixel_end = int(sizes[:stop].sum().item())
                result.update(image_grid_thw=grids[first_image:stop],
                              pixel_values=inputs["pixel_values"][pixel_start:pixel_end])
        return result

    def append(self, label, image):
        if self.failed:
            raise RuntimeError("The online cache failed; reset before starting another execution.")
        if len(self.images) >= self.max_frames:
            raise ValueError("Eight-frame execution limit reached; start a new execution.")
        images = self.images + [(label, image)]
        turn = {"role": "user", "images": images, "text": self.evaluation_text}
        inputs = self.wrapper.tokenize_conversation(
            self.reference_conversation + [self.wrapper.conversation_message(turn)], self.generation
        ).to(self.wrapper.input_device)
        self._validate(inputs)
        start = self.cached_ids.shape[1]
        if not torch.equal(inputs["input_ids"][:, :start], self.cached_ids):
            raise ValueError("Prompt prefix changed; cached history cannot be reused.")
        ends = (inputs["input_ids"][0] == self.vision_end).nonzero().flatten()
        image_count = self.reference_image_count + len(images)
        if len(ends) != image_count or len(inputs["image_grid_thw"]) != image_count:
            raise ValueError("Unexpected image-token layout; refusing an invalid cache continuation.")
        boundary = int(ends[-1].item()) + 1
        length = inputs["input_ids"].shape[1]
        if not start < boundary < length:
            raise ValueError("Expected new image tokens followed by the evaluation prompt.")
        positions = self._positions(inputs)
        captured = []
        try:
            with torch.inference_mode():
                output = self.wrapper.model(**self._chunk(inputs, positions, start, boundary,
                    image_count - 1, 1), past_key_values=self.cache, use_cache=True, logits_to_keep=1)
                self.cache = output.past_key_values
                def capture(_module, _args, hidden):
                    captured.append(hidden[0, -1].detach().float().cpu().clone())
                hook = self.wrapper._final_language_norm().register_forward_hook(capture)
                try:
                    self.wrapper.model(**self._chunk(inputs, positions, boundary, length, image_count, 0),
                        past_key_values=self.cache, use_cache=True, logits_to_keep=1)
                finally:
                    hook.remove()
                    self.cache.crop(boundary)
            if len(captured) != 1 or not torch.isfinite(captured[0]).all():
                raise RuntimeError("Invalid cached prompt-end feature.")
            self.images = images
            self.cached_ids = inputs["input_ids"][:, :boundary].clone()
            return captured[0]
        except Exception:
            self.failed = True
            raise

    def uncached_vector(self):
        return self.wrapper.extract_multiturn_cop_vector([
            self.reference_turn,
            {"role": "user", "images": self.images, "text": self.evaluation_text}],
            self.generation, reference_response=self.reference_response)
