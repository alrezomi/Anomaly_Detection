from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rynnbrain_vlm.lora import (
    adapter_training_bag_names, answer_loss, attach_lora, load_lora_adapter,
    resolve_bag_selection, supervised_example,
)
from rynnbrain_vlm.train_lora import training_settings


class LoraSelectionTests(unittest.TestCase):
    def test_explicit_labels_paths_and_separate_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("normal", "failure", "val_normal", "val_failure"):
                path = root / "nested" / name
                path.mkdir(parents=True)
                (path / "metadata.yaml").write_text("{}")
            settings = {"normal_bags": ["normal"], "failure_bags": ["nested/failure"],
                        "validation_normal_bags": [str(root / "nested/val_normal")],
                        "validation_failure_bags": ["val_failure"]}
            rows = resolve_bag_selection(settings, root, [])
            self.assertEqual([row["label"] for row in rows], ["normal", "fail", "normal", "fail"])
            self.assertEqual([row["split"] for row in rows], ["train", "train", "validation", "validation"])
            self.assertTrue(all(Path(row["bag_path"]).is_absolute() for row in rows))
            with self.assertRaisesRegex(ValueError, "more than once"):
                resolve_bag_selection({**settings, "validation_normal_bags": ["nested/normal"]}, root, [])
            with self.assertRaisesRegex(ValueError, "Reference"):
                resolve_bag_selection(settings, root, [str(root / "nested/normal")])

    def test_empty_missing_ambiguous_and_incomplete_splits_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "both normal and failure"):
                resolve_bag_selection({}, root, [])
            with self.assertRaises(FileNotFoundError):
                resolve_bag_selection({"normal_bags": ["missing"]}, root, [])
            for name in ("one/bag", "two/bag", "failure"):
                (root / name).mkdir(parents=True)
                (root / name / "metadata.yaml").write_text("{}")
            with self.assertRaisesRegex(ValueError, "Ambiguous"):
                resolve_bag_selection({"normal_bags": ["bag"]}, root, [])
            with self.assertRaisesRegex(ValueError, "validation"):
                resolve_bag_selection({"normal_bags": ["one/bag"], "failure_bags": ["failure"],
                                       "validation_failure_bags": ["two/bag"]}, root, [])

    def test_training_defaults_and_inference_adapter_guard(self):
        config = {"rynnbrain": {"model": {}, "output_dir": "/outputs/existing/rynnbrain", "reference_bags": ["reference"]}}
        self.assertEqual(Path(training_settings(config)["output_dir"]), Path("/outputs/existing/rynnbrain/lora_adapter"))
        config["rynnbrain"]["model"]["lora_adapter_path"] = "/saved/adapter"
        with self.assertRaisesRegex(ValueError, "null"):
            training_settings(config)

    def test_training_provenance_excludes_only_training_bags(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "training_manifest.json").write_text(json.dumps({"bags": [
                {"bag_name": "train", "split": "train"}, {"bag_name": "validation", "split": "validation"},
            ]}))
            self.assertEqual(adapter_training_bag_names(root), {"train"})


class LoraGradientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
            import peft
            from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
        except ImportError as error:
            raise unittest.SkipTest("LoRA gradient tests need requirements-rynnbrain.txt: " + str(error))
        cls.torch = torch
        cls.config_class = Qwen3VLConfig
        cls.model_class = Qwen3VLForConditionalGeneration
        torch.set_num_threads(1)

    def tiny_model(self):
        torch = self.torch
        torch.manual_seed(42)
        config = self.config_class(
            text_config={"vocab_size": 32, "hidden_size": 16, "intermediate_size": 32,
                         "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 2,
                         "head_dim": 8, "rope_scaling": {"rope_type": "default", "mrope_section": [2, 1, 1]},
                         "pad_token_id": 0, "eos_token_id": 2},
            vision_config={"depth": 1, "hidden_size": 16, "intermediate_size": 32,
                           "num_heads": 2, "out_hidden_size": 16, "patch_size": 16,
                           "spatial_merge_size": 2, "temporal_patch_size": 2,
                           "deepstack_visual_indexes": [0], "num_position_embeddings": 16},
            image_token_id=3, video_token_id=6, vision_start_token_id=4, vision_end_token_id=5,
        )
        config._name_or_path = "tiny-rynnbrain-test"
        return self.model_class(config)

    def sample(self):
        torch = self.torch
        class Tokenizer:
            eos_token_id = 2
            def encode(self, text, add_special_tokens=False):
                return [10, 11] if text == "Decision: success" else [10, 12]
        prefix = {
            "input_ids": torch.tensor([[4, 3, 5, 7]]), "attention_mask": torch.ones((1, 4), dtype=torch.long),
            "pixel_values": torch.randn(4, 3 * 2 * 16 * 16), "image_grid_thw": torch.tensor([[1, 2, 2]]),
        }
        return prefix, Tokenizer()

    def test_only_answer_tokens_are_supervised_and_images_are_not_truncated(self):
        prefix, tokenizer = self.sample()
        sample = supervised_example(prefix, tokenizer, "normal", 10)
        self.assertEqual(sample["labels"].tolist(), [[-100, -100, -100, -100, 10, 11, 2]])
        self.assertEqual(sample["input_ids"].tolist(), [[4, 3, 5, 7, 10, 11, 2]])
        self.assertIs(sample["pixel_values"], prefix["pixel_values"])
        with self.assertRaisesRegex(ValueError, "never truncated"):
            supervised_example(prefix, tokenizer, "fail", 5)

    def test_multimodal_backward_changes_only_lora_and_adapter_reloads(self):
        torch = self.torch
        base = self.tiny_model()
        model = attach_lora(base, {"rank": 2, "alpha": 4, "dropout": 0.0})
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
        prefix, tokenizer = self.sample()
        sample = supervised_example(prefix, tokenizer, "fail", 20)
        model.eval()
        with torch.no_grad():
            # Compare the memory-saving answer loss to the model's full masked
            # causal loss, so an off-by-one shift cannot pass the gradient test.
            torch.testing.assert_close(answer_loss(model, sample), model(**sample, use_cache=False).loss)
        model.train()
        for _ in range(2):
            optimizer.zero_grad()
            loss = answer_loss(model, sample)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            optimizer.step()
        changed = [name for name, p in model.named_parameters() if not torch.equal(before[name], p)]
        self.assertTrue(changed)
        self.assertTrue(all("lora_" in name and ".language_model." in name for name in changed))
        model.eval()
        with torch.no_grad():
            expected = answer_loss(model, sample)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model.save_pretrained(root, safe_serialization=True)
            (root / "training_manifest.json").write_text(json.dumps({"bags": [{"bag_name": "train", "split": "train"}]}))
            restored, digest, names = load_lora_adapter(self.tiny_model(), root, "tiny-rynnbrain-test")
            self.assertEqual(names, {"train"})
            self.assertEqual(len(digest), 64)
            self.assertFalse(any(p.requires_grad for p in restored.parameters()))
            restored.eval()
            with torch.no_grad():
                torch.testing.assert_close(answer_loss(restored, sample), expected)
                with restored.disable_adapter():
                    base_loss = answer_loss(restored, sample)
            self.assertNotAlmostEqual(float(expected), float(base_loss), places=5)
            with self.assertRaisesRegex(ValueError, "trained for"):
                load_lora_adapter(self.tiny_model(), root, "wrong-base")

    def check_preserved_device_map(self, device_map):
        from accelerate import dispatch_model
        from rynnbrain_vlm.model import _input_execution_device

        torch = self.torch
        adapted = attach_lora(self.tiny_model(), {"rank": 2, "alpha": 4, "dropout": 0.0})
        with torch.no_grad():
            for name, parameter in adapted.named_parameters():
                if "lora_B" in name:
                    parameter.fill_(0.03)
        adapted.eval()
        prefix, tokenizer = self.sample()
        sample = supervised_example(prefix, tokenizer, "fail", 20)
        with torch.no_grad():
            expected = answer_loss(adapted, sample)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapted.save_pretrained(root, safe_serialization=True)
            (root / "training_manifest.json").write_text(json.dumps({"bags": []}))
            base = dispatch_model(self.tiny_model(), device_map=device_map)
            with patch("peft.peft_model.infer_auto_device_map", side_effect=AssertionError("Must not recalculate placement")):
                restored, _, _ = load_lora_adapter(base, root, "tiny-rynnbrain-test")
            expected_map = {"base_model.model" + (f".{name}" if name else ""): device
                            for name, device in device_map.items()}
            self.assertEqual(restored.hf_device_map, expected_map)
            device = _input_execution_device(restored)
            inputs = {key: value.to(device) for key, value in sample.items()}
            with torch.no_grad():
                torch.testing.assert_close(answer_loss(restored, inputs).cpu(), expected, rtol=1e-4, atol=1e-5)
                restored.generate(
                    **{key: value.to(device) for key, value in prefix.items()},
                    max_new_tokens=1, do_sample=False, pad_token_id=0,
                )

    def test_real_lora_cached_reference_matches_single_bag_generation_and_vector(self):
        from PIL import Image
        from transformers import BatchFeature
        from peft.tuners.lora.layer import LoraLayer
        from rynnbrain_vlm.model import RynnBrainModel

        torch = self.torch
        wrapper = RynnBrainModel.__new__(RynnBrainModel)
        wrapper.input_device = torch.device("cpu")
        wrapper.max_image_size = 640
        wrapper.adapter_identity = "test-adapter"
        wrapper.model = attach_lora(self.tiny_model(), {"rank": 2, "alpha": 4, "dropout": 0.0}).eval()
        with torch.no_grad():
            for name, parameter in wrapper.model.named_parameters():
                if "lora_B" in name:
                    parameter.fill_(0.03)
        before = {name: value.detach().clone() for name, value in wrapper.model.named_parameters()}
        prefix, _ = self.sample()
        conversations = []

        class Processor:
            def apply_chat_template(self, conversation, **kwargs):
                conversations.append(conversation.copy())
                ids, image_count = [], 0
                for message in conversation:
                    ids.append(7)
                    for item in message["content"]:
                        if item["type"] == "image":
                            ids.extend([4, 3, 5])
                            image_count += 1
                        else:
                            ids.append(8 + sum(map(ord, item["text"])) % 24)
                return BatchFeature({
                    "input_ids": torch.tensor([ids]),
                    "attention_mask": torch.ones((1, len(ids)), dtype=torch.long),
                    "pixel_values": prefix["pixel_values"].repeat(image_count, 1),
                    "image_grid_thw": prefix["image_grid_thw"].repeat(image_count, 1),
                })

            def decode(self, tokens, **kwargs):
                return str(tokens.tolist())

        wrapper.processor = Processor()
        turns = [{"role": "user", "text": text, "images": [(text, Image.new("RGB", (2, 2)))]}
                 for text in ("Reference", "Execution")]
        layer = next(module for module in wrapper.model.modules() if isinstance(module, LoraLayer))
        adapter_states, inputs_seen = [], []
        generate = wrapper.model.generate

        def tracked_generate(**kwargs):
            adapter_states.append(not layer.disable_adapters)
            inputs_seen.append(kwargs["input_ids"].clone())
            return generate(**kwargs)

        generation = {"max_new_tokens": 1, "do_sample": False}
        with patch.object(wrapper.model, "generate", side_effect=tracked_generate):
            single = wrapper.generate_multiturn_with_cop_vector(turns, generation)
            cached = wrapper.generate_multiturn_with_cop_vector(turns, generation, reference_response=single.nominal_response)
        self.assertEqual(adapter_states, [False, True, True])
        self.assertEqual(single.evaluation_response, cached.evaluation_response)
        torch.testing.assert_close(single.cop_vector, cached.cop_vector, rtol=0, atol=0)
        torch.testing.assert_close(inputs_seen[1], inputs_seen[2], rtol=0, atol=0)
        self.assertEqual(conversations[-1][1]["content"][0]["text"], single.nominal_response)
        self.assertEqual(sum(item["type"] == "image" for message in conversations[-1] for item in message["content"]), 2)
        self.assertFalse(layer.disable_adapters)
        self.assertTrue(all(torch.equal(before[name], value) for name, value in wrapper.model.named_parameters()))
        with wrapper.model.disable_adapter():
            base = wrapper.generate_multiturn_with_cop_vector(turns, generation, reference_response=single.nominal_response)
        self.assertFalse(torch.allclose(base.cop_vector, cached.cop_vector))

    def test_adapter_keeps_root_and_module_maps_without_recalculating_memory(self):
        for device_map in ({"": "cpu"}, {"model": "cpu", "lm_head": "cpu"}):
            with self.subTest(device_map=device_map):
                self.check_preserved_device_map(device_map)

    def test_adapter_keeps_gpu_cpu_offload_map(self):
        if not self.torch.cuda.is_available():
            self.skipTest("Needs a CUDA GPU for mixed GPU/CPU offload verification")
        self.check_preserved_device_map({
            "model.visual": 0, "model.language_model.embed_tokens": 0,
            "model.language_model.layers.0": 0, "model.language_model.layers.1": "cpu",
            "model.language_model.norm": 0, "lm_head": 0,
        })


if __name__ == "__main__":
    unittest.main()
