import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pipeline_settings import load_pipeline_config, resolve_pipeline_config


class PipelineSettingsTests(unittest.TestCase):
    def config(self):
        return {"output_dir": "/outputs/v130", "camera_topics": ["top"], "rynnbrain": {
            "source": "rosbag", "input_modes": ["raw"], "num_frames": 8,
            "reference_bags": ["reference"], "model": {"lora_adapter_path": None},
            "lora": {"training": {"temporal_supervision": True,
                                  "normal_bags": ["n"], "failure_bags": ["f"]}},
            "cop_classifier": {"enabled": True, "method": "logistic",
                "training": {"normal_bags": ["n"], "failure_bags": ["f"]}}}}

    def test_defaults_follow_one_root_and_one_camera_without_mutation(self):
        config = self.config()
        before = copy.deepcopy(config)
        for folder in ("v130", "v131"):
            config["output_dir"] = f"/outputs/{folder}"
            resolved = resolve_pipeline_config(config)
            vlm = resolved["rynnbrain"]
            root = Path(config["output_dir"])
            self.assertEqual(Path(vlm["output_dir"]), root / "rynnbrain")
            self.assertEqual(Path(vlm["lora"]["training"]["output_dir"]), root / "rynnbrain/lora_temporal_adapter")
            self.assertEqual(Path(vlm["cop_classifier"]["training"]["input_dir"]), root / "rynnbrain_benchmark")
            self.assertEqual(Path(vlm["cop_classifier"]["model_paths"]["raw"]), root / "cop_classifier/raw_logistic.npz")
            self.assertEqual(vlm["camera_topics"], ["top"])
            self.assertEqual(vlm["memory_camera_topics"], ["top"])
            self.assertIsNone(vlm["model"]["lora_adapter_path"])
            self.assertEqual(vlm["lora"]["training"]["normal_bags"], ["n"])
        config["output_dir"] = before["output_dir"]
        self.assertEqual(config, before)
        self.assertEqual(resolve_pipeline_config(resolved), resolved)

    def test_explicit_paths_and_distinct_camera_overrides_win(self):
        config = self.config()
        vlm = config["rynnbrain"]
        vlm.update(output_dir="/existing/results", camera_topics=["flange"], memory_camera_topics=["reference_camera"])
        vlm["model"]["lora_adapter_path"] = "/existing/adapter"
        vlm["lora"]["training"]["output_dir"] = "/custom/new_adapter"
        classifier = vlm["cop_classifier"]
        classifier["model_paths"] = {"raw": "/existing/custom_name.npz"}
        classifier["training"]["input_dir"] = "/existing/vectors"
        self.assertEqual(resolve_pipeline_config(config), config)

    def test_classifier_method_and_temporal_flag_choose_distinct_default_files(self):
        config = self.config()
        config["rynnbrain"]["lora"]["training"]["temporal_supervision"] = False
        config["rynnbrain"]["cop_classifier"]["method"] = "knn"
        resolved = resolve_pipeline_config(config)["rynnbrain"]
        self.assertEqual(Path(resolved["lora"]["training"]["output_dir"]).name, "lora_adapter")
        self.assertEqual(Path(resolved["cop_classifier"]["model_paths"]["raw"]).name, "raw_knn.npz")

    def test_shared_loader_agrees_across_training_scoring_and_validation(self):
        from rynnbrain_vlm.cop_classifier import configured_training_settings
        from rynnbrain_vlm.run import _common_config
        from rynnbrain_vlm.train_lora import training_settings
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(self.config()))
            before = path.read_bytes()
            config, vlm, count, _ = _common_config(SimpleNamespace(config=path))
            classifier = configured_training_settings(path)
            lora = training_settings(load_pipeline_config(path))
            self.assertEqual(classifier["input_directory"], Path(vlm["cop_classifier"]["training"]["input_dir"]))
            self.assertEqual(classifier["output_file"], Path(vlm["cop_classifier"]["model_paths"]["raw"]))
            self.assertEqual(lora["temporal_profile"]["camera_topics"], ["top"])
            self.assertEqual(lora["temporal_profile"]["memory_camera_topics"], ["top"])
            self.assertEqual(lora["temporal_profile"]["num_frames"], count)
            self.assertEqual(path.read_bytes(), before)

    def test_dino_selection_preserves_heatmap_video_and_explicit_workflows(self):
        from run_benchmark import _should_run_dino, parse_arguments
        flags = SimpleNamespace(skip_dino=False, skip_vlm=False, with_dino=False)
        raw = {"source": "rosbag", "input_modes": ["raw"]}
        self.assertFalse(_should_run_dino(flags, raw))
        for vlm in ({"source": "generated_videos", "input_modes": ["raw"]},
                    {"source": "rosbag", "input_modes": ["heatmap"]},
                    {"source": "rosbag", "input_modes": ["raw", "raw_heatmap"]}):
            self.assertTrue(_should_run_dino(flags, vlm))
        flags.with_dino = True
        self.assertTrue(_should_run_dino(flags, raw))
        flags.with_dino, flags.skip_vlm = False, True
        self.assertTrue(_should_run_dino(flags, raw))
        flags.skip_dino = True
        self.assertFalse(_should_run_dino(flags, raw))
        with patch.object(sys, "argv", ["benchmark", "--with-dino", "--skip-dino"]), patch("sys.stderr"):
            with self.assertRaises(SystemExit):
                parse_arguments()


if __name__ == "__main__":
    unittest.main()
