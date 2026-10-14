import unittest
from unittest.mock import patch


class OnlineCacheTests(unittest.TestCase):
    def test_real_qwen_lora_incremental_images_match_full_prefixes_and_reset(self):
        import torch
        from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration, BatchFeature
        from PIL import Image
        from rynnbrain_vlm.model import RynnBrainModel
        from rynnbrain_vlm.lora import attach_lora
        from rynnbrain_vlm.online_cache import CachedExecution
        from rynnbrain_vlm.cop_classifier import CoPLogisticClassifier
        from rynnbrain_vlm.cop_knn import CoPKNNDetector
        import numpy as np
        torch.set_num_threads(1)
        torch.manual_seed(15)
        config = Qwen3VLConfig(
            text_config=dict(vocab_size=64, hidden_size=16, intermediate_size=32,
                num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
                head_dim=8, rope_scaling={"rope_type": "default", "mrope_section": [2, 1, 1]}),
            vision_config=dict(depth=2, hidden_size=16, intermediate_size=32, num_heads=2,
                out_hidden_size=16, patch_size=16, spatial_merge_size=2, temporal_patch_size=2,
                deepstack_visual_indexes=[0, 1], num_position_embeddings=16),
            image_token_id=3, video_token_id=6, vision_start_token_id=4, vision_end_token_id=5)
        wrapper = RynnBrainModel.__new__(RynnBrainModel)
        wrapper.input_device = torch.device("cpu")
        wrapper.max_image_size = 640
        wrapper.adapter_identity = "test-adapter"
        wrapper.model = attach_lora(Qwen3VLForConditionalGeneration(config),
                                    dict(rank=2, alpha=4, dropout=0.)).eval()
        with torch.no_grad():
            for name, parameter in wrapper.model.named_parameters():
                if "lora_B" in name:
                    parameter.fill_(.05)

        class Processor:
            def apply_chat_template(self, messages, **kwargs):
                ids, pixels, grids = [], [], []
                for message in messages:
                    ids.append(7 if message["role"] == "user" else 8)
                    for item in message["content"]:
                        if item["type"] == "image":
                            value = item["image"].getpixel((0, 0))[0]
                            # Unequal spatial grids exercise multimodal positions and pixel slicing.
                            height = 4 if value % 2 else 2
                            ids.extend([4] + [3] * height + [5])
                            pixels.append(torch.full((height * 4, 1536), value / 255.))
                            grids.append([1, height, 4])
                        else:
                            ids.extend([10 + ord(c) % 50 for c in item["text"]])
                    ids.append(9)
                if kwargs["add_generation_prompt"]:
                    ids.append(8)
                return BatchFeature(dict(input_ids=torch.tensor([ids]),
                    attention_mask=torch.ones(1, len(ids), dtype=torch.long),
                    pixel_values=torch.cat(pixels), image_grid_thw=torch.tensor(grids)))
        wrapper.processor = Processor()
        reference = dict(role="user", images=[("ref", Image.new("RGB", (4, 4), (1, 0, 0)))], text="Reference")
        before = {name: value.clone() for name, value in wrapper.model.named_parameters()}
        engine = CachedExecution(wrapper, reference, "Normal", "Decide", {}, 8)
        pristine = engine.reference_cache.get_seq_length()
        visual = engine.base.visual
        first = None
        logistic = CoPLogisticClassifier(np.linspace(-1, 1, 16), .1, np.zeros(16), .5, "raw", "test", {}, {})
        knn = CoPKNNDetector(np.eye(16)[:3], 2, .8, "raw", "test", {}, {})
        with patch.object(wrapper.model, "generate", side_effect=AssertionError("No answer generation")):
            for step in range(8):
                with patch.object(visual, "forward", wraps=visual.forward) as vision:
                    vector = engine.append(f"Frame {step}", Image.new("RGB", (4, 4), (step + 2, 0, 0)), capture_visual=True)
                self.assertEqual(vision.call_count, 1)
                self.assertEqual(len(vision.call_args.kwargs["grid_thw"]), 1)
                self.assertEqual(engine.last_visual_descriptor.shape, (16,))
                self.assertAlmostEqual(float(np.linalg.norm(engine.last_visual_descriptor)), 1., places=6)
                # The descriptor comes from this image, independent of conversation history.
                from rynnbrain_vlm.visual_matching import _visual_descriptor
                from types import SimpleNamespace
                standalone = _visual_descriptor(SimpleNamespace(model=wrapper, cache=engine, vlm={}),
                                               Image.new("RGB", (4, 4), (step + 2, 0, 0)))
                np.testing.assert_allclose(engine.last_visual_descriptor, standalone, atol=1e-6)
                self.assertFalse(visual._forward_hooks)
                full = engine.uncached_vector()
                torch.testing.assert_close(vector, full, rtol=2e-4, atol=2e-5)
                for classifier, predict in ((logistic, logistic.predict_failure_probability),
                                            (knn, knn.predict_anomaly_score)):
                    cached_score, full_score = predict(vector.numpy()), predict(full.numpy())
                    self.assertAlmostEqual(cached_score, full_score, places=5)
                    self.assertEqual(classifier.predict_label(cached_score), classifier.predict_label(full_score))
                self.assertEqual(engine.cache.get_seq_length(), engine.cached_ids.shape[1])
                self.assertEqual(engine.reference_cache.get_seq_length(), pristine)
                if step == 0:
                    first = vector
            with self.assertRaisesRegex(ValueError, "limit"):
                engine.append("extra", Image.new("RGB", (4, 4)))
            engine.reset()
            self.assertIsNone(engine.last_visual_descriptor)
            repeated = engine.append("Frame 0", Image.new("RGB", (4, 4), (2, 0, 0)))
            torch.testing.assert_close(first, repeated)
            engine.evaluation_text = "A changed suffix is recomputed"
            vector = engine.append("Frame 1", Image.new("RGB", (4, 4), (3, 0, 0)))
            torch.testing.assert_close(vector, engine.uncached_vector(), rtol=2e-4, atol=2e-5)
        self.assertTrue(all(torch.equal(before[name], value) for name, value in wrapper.model.named_parameters()))


if __name__ == "__main__":
    unittest.main()
