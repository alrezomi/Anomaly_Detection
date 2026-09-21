from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class ModelInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
            from rynnbrain_vlm.model import RynnBrainModel, _input_execution_device
        except ImportError as error:
            raise unittest.SkipTest(str(error))
        cls.torch = torch
        cls.wrapper_class = RynnBrainModel
        cls.resolve_device = staticmethod(_input_execution_device)

    def embedding_model(self, device="cpu"):
        torch = self.torch

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(16, 4, device=device)

            def get_input_embeddings(self):
                return self.embedding

        return Model()

    def test_cpu_model_overrides_stale_cuda_config_and_accepts_tokens(self):
        model = self.embedding_model()
        with patch("rynnbrain_vlm.model.AutoProcessor.from_pretrained"), patch(
            "rynnbrain_vlm.model.AutoModelForImageTextToText.from_pretrained", return_value=model
        ):
            wrapper = self.wrapper_class({"model_id": "test", "input_device": "cuda"})
        self.assertEqual(wrapper.input_device, self.torch.device("cpu"))
        tokens = self.torch.tensor([[1, 2]]).to(wrapper.input_device)
        self.assertEqual(tuple(model.get_input_embeddings()(tokens).shape), (1, 2, 4))

    def test_offloaded_meta_embedding_uses_execution_hook_including_sequential_hooks(self):
        model = self.embedding_model("meta")
        model.embedding._hf_hook = SimpleNamespace(hooks=[
            SimpleNamespace(execution_device=None), SimpleNamespace(execution_device=1),
        ])
        self.assertEqual(self.resolve_device(model), self.torch.device("cuda:1"))
        del model.embedding._hf_hook
        model._hf_hook = SimpleNamespace(execution_device="cuda:0")
        self.assertEqual(self.resolve_device(model), self.torch.device("cuda:0"))
        del model._hf_hook
        with self.assertRaisesRegex(RuntimeError, "meta"):
            self.resolve_device(model)

    def test_real_accelerate_disk_offload_generation_on_cpu(self):
        try:
            from accelerate import disk_offload
            from transformers import GPT2Config, GPT2LMHeadModel
        except ImportError as error:
            self.skipTest(str(error))
        import tempfile
        model = GPT2LMHeadModel(GPT2Config(
            vocab_size=16, n_embd=8, n_layer=1, n_head=1,
            bos_token_id=1, eos_token_id=2, pad_token_id=0,
        )).eval()
        with tempfile.TemporaryDirectory() as directory:
            model = disk_offload(model, directory, execution_device=self.torch.device("cpu"))
            self.assertEqual(model.get_input_embeddings().weight.device.type, "meta")
            device = self.resolve_device(model)
            tokens = self.torch.tensor([[1, 3]]).to(device)
            with self.torch.inference_mode():
                output = model.generate(input_ids=tokens, attention_mask=self.torch.ones_like(tokens), max_new_tokens=1)
            self.assertEqual(output.shape[1], 3)

    def test_evidence_uses_base_model_without_decision_and_restores_adapter_on_error(self):
        torch = self.torch
        wrapper = self.wrapper_class.__new__(self.wrapper_class)
        wrapper.input_device = torch.device("cpu")
        wrapper.adapter_identity = "trained-adapter"
        wrapper.max_image_size = 640
        conversations = []

        class Inputs(dict):
            def to(self, device):
                return Inputs({key: value.to(device) for key, value in self.items()})

        class Processor:
            def apply_chat_template(self, conversation, **kwargs):
                conversations.append(conversation)
                return Inputs(input_ids=torch.tensor([[1, 2]]))

            def decode(self, tokens, **kwargs):
                return "Visual evidence: Object remains at the start."

        class Model:
            adapter_enabled = True
            fail = False

            @contextmanager
            def disable_adapter(self):
                self.adapter_enabled = False
                try:
                    yield
                finally:
                    self.adapter_enabled = True

            def generate(self, **kwargs):
                assert not self.adapter_enabled
                if self.fail:
                    raise RuntimeError("out of memory")
                return torch.tensor([[1, 2, 3]])

        wrapper.processor, wrapper.model = Processor(), Model()
        turns = [{"role": "user", "text": "Reference", "images": []},
                 {"role": "user", "text": "Decision: success", "images": []}]
        text = wrapper.generate_visual_evidence(turns, "Nominal description", "Describe visible motion", {})
        self.assertIn("Object remains", text)
        self.assertTrue(wrapper.model.adapter_enabled)
        self.assertNotIn("Decision: success", str(conversations))
        self.assertEqual(turns[1]["text"], "Decision: success")
        wrapper.model.fail = True
        with self.assertRaisesRegex(RuntimeError, "out of memory"):
            wrapper.generate_visual_evidence(turns, "Nominal description", "Describe visible motion", {})
        self.assertTrue(wrapper.model.adapter_enabled)

    def test_evidence_is_saved_separately_and_failure_keeps_decision_and_vector(self):
        import json
        from pathlib import Path
        import tempfile
        from unittest.mock import Mock
        import numpy as np
        from rynnbrain_vlm.run import evaluate_multiturn, write_multiturn_outputs

        vector = self.torch.tensor([1.0, 2.0])
        model = SimpleNamespace(
            adapter_training_bags=set(), adapter_identity="adapter", model_id="test",
            model=SimpleNamespace(config=SimpleNamespace(text_config=SimpleNamespace(hidden_size=2))),
            generate_multiturn_with_cop_vector=Mock(return_value=SimpleNamespace(
                nominal_response="Nominal description", evaluation_response="Decision: success", cop_vector=vector,
            )),
            generate_visual_evidence=Mock(return_value="Visual evidence: Object remains at the start."),
        )
        config = {"test_bag": "/data/failure_1.12", "output_dir": "/outputs/test", "camera_topics": ["camera"]}
        vlm = {"source": "rosbag", "input_modes": ["raw"], "reference_bags": ["/data/reference"]}
        with tempfile.TemporaryDirectory() as directory, patch(
            "rynnbrain_vlm.run._raw_inputs", return_value=([], [])
        ), patch("rynnbrain_vlm.run._save_inputs"), patch("builtins.print"):
            root = Path(directory)
            for enabled, failure in ((False, False), (True, False), (True, True)):
                with self.subTest(enabled=enabled, failure=failure):
                    model.generate_visual_evidence.reset_mock()
                    model.generate_visual_evidence.side_effect = RuntimeError("out of memory") if failure else None
                    rows, frames, records, task = evaluate_multiturn(
                        model, config, vlm, 8, {"explain_decision": enabled}, root,
                    )
                    write_multiturn_outputs(root, rows, frames, records, task)
                    self.assertEqual(rows[0]["decision"], "success")
                    self.assertEqual(rows[0]["response"], "Decision: success")
                    np.testing.assert_array_equal(np.load(rows[0]["cop_vector_path"]), vector.numpy())
                    saved = json.loads((root / "rynnbrain_responses_multiturn.json").read_text())["results"][0]
                    if not enabled:
                        model.generate_visual_evidence.assert_not_called()
                        self.assertIsNone(saved["visual_evidence"])
                    elif failure:
                        self.assertEqual(saved["visual_evidence_error"], "out of memory")
                        self.assertIsNone(saved["visual_evidence"])
                    else:
                        self.assertIn("Object remains", saved["visual_evidence"])
                        self.assertEqual(saved["visual_evidence_source"], "base_model_separate_pass")
                        self.assertIsNone(saved["visual_evidence_error"])

    def test_prepared_reference_skips_generation_and_preserves_execution_vector(self):
        torch = self.torch
        wrapper = self.wrapper_class.__new__(self.wrapper_class)
        wrapper.input_device = torch.device("cpu")
        wrapper.adapter_identity = None
        wrapper.max_image_size = 640
        norm = torch.nn.Identity()
        calls = []
        conversations = []

        class Inputs(dict):
            def to(self, device):
                return self

        class Processor:
            def apply_chat_template(self, conversation, **kwargs):
                conversations.append(list(conversation))
                return Inputs(input_ids=torch.tensor([[1, 2]]))

            def decode(self, *args, **kwargs):
                return "reference response" if len(calls) == 1 else "Decision: success"

        def generate(**kwargs):
            calls.append(kwargs)
            norm(torch.ones((1, 2, 4)))
            return torch.tensor([[1, 2, 3]])

        wrapper.processor = Processor()
        wrapper.model = SimpleNamespace(generate=generate, config=SimpleNamespace(text_config=SimpleNamespace(hidden_size=4)))
        turns = [{"role": "user", "text": "Reference", "images": []},
                 {"role": "user", "text": "Execution", "images": []}]
        with patch.object(wrapper, "_final_language_norm", return_value=norm):
            original = wrapper.generate_multiturn_with_cop_vector(turns, {})
            self.assertEqual(len(calls), 2)
            cached = wrapper.generate_multiturn_with_cop_vector(turns, {}, reference_response=original.nominal_response)
        self.assertEqual(len(calls), 3)
        self.assertEqual(cached.nominal_response, original.nominal_response)
        self.assertEqual(cached.evaluation_response, original.evaluation_response)
        torch.testing.assert_close(cached.cop_vector, original.cop_vector)
        self.assertEqual(conversations[-1][1]["content"][0]["text"], original.nominal_response)


if __name__ == "__main__":
    unittest.main()
