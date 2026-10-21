"""Exercise real ViT forward passes without downloading pretrained weights."""
import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
from transformers import DINOv3ViTConfig, DINOv3ViTModel, AutoImageProcessor

from vision_dinov2.dinov2_features import DINOv2FeatureExtractor
from vision_dinov2.model_settings import model_settings
from vision_dinov2.nominal_cache import memory_signature


class DinoModelTests(unittest.TestCase):
    def test_build_and_replay_request_identical_v3_nominal_calibration(self):
        import run_rosbag_vision as launcher
        from vision_dinov2 import run_vision_test as runner
        from rynnbrain_vlm.replay_detectors import DinoReplayDetector
        config = dict(nominal_bags=["normal_a", "normal_b", "normal_c"],
                      camera_topics=["/camera"], dino_calibration_bags=[],
                      dino_model_name="facebook/dinov3-vits16plus-pretrain-lvd1689m",
                      dino_input_size=832)
        requested = []
        with tempfile.TemporaryDirectory() as folder:
            config["output_dir"] = folder
            path = Path(folder) / "config.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with patch.dict(runner.__dict__), \
                    patch("sys.argv", ["run_rosbag_vision.py", "--config", str(path), "--mode", "build-memory"]), \
                    patch.object(launcher, "resolve_bag_path"), \
                    patch.object(runner, "main", side_effect=lambda **_: requested.append(runner.nominal_cache_runtime_signature())):
                launcher.main()
            with patch("vision_dinov2.nominal_cache.load_nominal_cache", return_value=None) as load:
                with self.assertRaisesRegex(ValueError, "No compatible DINO calibration"):
                    DinoReplayDetector(config, lambda _: None)
                self.assertEqual(load.call_args.args[1], requested[0])
            self.assertEqual(requested[0]["calibration_sources"], [])
            self.assertEqual(requested[0]["model_name"], config["dino_model_name"])

    def test_v3_forward_removes_registers_and_preserves_grid(self):
        config = DINOv3ViTConfig(hidden_size=32, intermediate_size=64,
                                num_hidden_layers=2, num_attention_heads=4,
                                num_register_tokens=4, patch_size=16)
        config._attn_implementation = "eager"
        model = DINOv3ViTModel(config)
        # AutoImageProcessor resolves the Fast class in 4.57 and its renamed
        # equivalent in newer Transformers, just as a downloaded model does.
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "preprocessor_config.json").write_text(json.dumps({
                "image_processor_type": "DINOv3ViTImageProcessorFast"}), encoding="utf-8")
            processor = AutoImageProcessor.from_pretrained(folder)
        with patch("vision_dinov2.dinov2_features.AutoModel.from_pretrained", return_value=model), \
                patch("vision_dinov2.dinov2_features.AutoImageProcessor.from_pretrained",
                      return_value=processor):
            extractor = DINOv2FeatureExtractor("facebook/dinov3-vits16-pretrain-lvd1689m", 64, "cpu")
            pixels = np.random.default_rng(5).integers(0, 256, (71, 97, 3), dtype=np.uint8)
            features, attention, cls, grid = extractor.extract(Image.fromarray(pixels))
        self.assertEqual(grid, 4)
        self.assertEqual(features.shape, (16, 32))
        self.assertEqual(attention.shape, (16,))
        self.assertTrue(np.isfinite(attention).all())
        self.assertGreater(float(attention.max()), .9)
        np.testing.assert_allclose(np.linalg.norm(features, axis=1), 1, atol=1e-6)
        self.assertAlmostEqual(float(np.linalg.norm(cls)), 1, places=6)

    def test_model_and_resolution_are_validated_and_in_feature_signature(self):
        model = "facebook/dinov3-vits16plus-pretrain-lvd1689m"
        self.assertEqual(model_settings({"dino_model_name": model, "dino_input_size": 736}), (model, 736))
        with self.assertRaisesRegex(ValueError, "multiple of 16"):
            model_settings({"dino_model_name": model, "dino_input_size": 728})
        with self.assertRaises(ValueError):
            model_settings({"dino_model_name": "facebook/dinov3-convnext-tiny-pretrain-lvd1689m"})
        old = dict(model_name="facebook/dinov2-small", dino_input_size=728)
        new = dict(model_name=model, dino_input_size=736)
        self.assertNotEqual(memory_signature(old), memory_signature(new))


if __name__ == "__main__":
    unittest.main()
