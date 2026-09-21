from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import pandas as pd
from PIL import Image

from rynnbrain_vlm.cop_classifier import (
    configured_training_settings, prepare_training_vectors, train_from_saved_vectors,
)


class ClassifierPreparationTests(unittest.TestCase):
    def setUp(self):
        try:
            import torch
            from rynnbrain_vlm import run
        except ImportError as error:
            self.skipTest(str(error))
        self.torch, self.run = torch, run
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.names = ["normal_1", "normal_2", "failure_1", "failure_2"]
        for name in self.names + ["reference", "normal_test", "failure_test"]:
            bag = self.root / "data" / name
            bag.mkdir(parents=True)
            (bag / "metadata.yaml").write_text("{}")
        self.model_path = self.root / "classifier" / "raw.npz"
        self.config = {
            "camera_topics": ["camera"], "output_dir": str(self.root / "outputs"),
            "rynnbrain": {
                "source": "rosbag", "input_modes": ["raw"], "num_frames": 8,
                "reference_bags": [str(self.root / "data/reference")],
                "model": {"model_id": "test"}, "generation": {"do_sample": False},
                "cop_vectors": {"enabled": False},
                "cop_classifier": {"enabled": True, "model_paths": {"raw": str(self.model_path)},
                    "training": {"input_dir": str(self.root / "vectors"), "input_mode": "raw",
                        "normal_bags": [str(self.root / "data" / name) for name in self.names[:2]],
                        "failure_bags": [str(self.root / "data" / name) for name in self.names[2:]],
                    }},
            },
        }
        self.config_path = self.root / "config.json"
        self.config_path.write_text(json.dumps(self.config))
        self.settings = configured_training_settings(self.config_path)
        self.extractions = Counter()

        def generate(turns, generation, **kwargs):
            name = turns[1]["images"][0][0]
            self.extractions[name] += 1
            sign = -1 if name.startswith("failure") else 1
            return SimpleNamespace(nominal_response=kwargs.get("reference_response", "reference response"),
                evaluation_response="Decision: failure" if sign < 0 else "Decision: success",
                cop_vector=torch.tensor([sign, 0.1 if name.endswith("1") else 0.2, 0.3, 0.4]))
        self.model = SimpleNamespace(
            model_id="test", adapter_identity="adapter_v1", adapter_training_bags=set(self.names),
            model=SimpleNamespace(config=SimpleNamespace(text_config=SimpleNamespace(hidden_size=4))),
            generate_nominal=Mock(return_value="reference response"),
            generate_multiturn_with_cop_vector=Mock(side_effect=generate),
        )
        for target, options in (
            ("rynnbrain_vlm.run._raw_inputs", {"side_effect": lambda path, *args: ([(path.name, Image.new("RGB", (2, 2)))], [])}),
            ("rynnbrain_vlm.run._save_inputs", {}),
            ("rynnbrain_vlm.run._print_inputs", {}),
            ("builtins.print", {}),
        ):
            patcher = patch(target, **options)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_missing_stale_and_corrupt_vectors_refresh_only_selected_training_bags(self):
        paths, _ = prepare_training_vectors(self.config, self.settings, model=self.model)
        self.assertEqual(self.extractions, Counter(self.names))
        sentinel = self.root / "vectors/benchmark_clean.csv"
        sentinel.write_text("keep the benchmark unchanged")
        train_from_saved_vectors(**self.settings, metadata_paths=paths)
        self.assertTrue(self.model_path.is_file())
        self.assertEqual(sentinel.read_text(), "keep the benchmark unchanged")
        prepare_training_vectors(self.config, self.settings, model=self.model)
        self.assertEqual(self.extractions, Counter(self.names))
        paths[0].with_suffix(".npy").unlink()
        prepare_training_vectors(self.config, self.settings, model=self.model)
        self.assertEqual(self.extractions["normal_1"], 2)
        self.assertEqual(self.extractions["failure_1"], 1)
        self.model.adapter_identity = "adapter_v2"
        prepare_training_vectors(self.config, self.settings, model=self.model)
        self.assertEqual(self.extractions["normal_1"], 3)
        self.assertEqual(self.extractions["failure_1"], 2)
        prepare_training_vectors(self.config, self.settings, model=self.model, refresh=True)
        self.assertEqual(self.extractions["failure_1"], 3)

    def test_reference_overlap_and_stochastic_generation_are_rejected(self):
        self.config["rynnbrain"]["reference_bags"] = [self.settings["bag_names"][0]]
        with self.assertRaisesRegex(ValueError, "reference_bags"):
            prepare_training_vectors(self.config, self.settings, model=self.model)
        self.config["rynnbrain"]["generation"]["do_sample"] = True
        with self.assertRaisesRegex(ValueError, "do_sample"):
            prepare_training_vectors(self.config, self.settings, model=self.model)
        self.model.generate_nominal.assert_not_called()

    def test_benchmark_prepares_once_excludes_training_and_populates_reports(self):
        import run_benchmark as benchmark
        # Exercise real preparation, fitting, scoring, and CSV/JSON output; only
        # the expensive VLM/ROS reads and unrelated PCA/ROC rendering are mocked.
        output = self.root / "benchmark"
        all_bags = [self.root / "data" / name for name in self.names + ["normal_test", "failure_test"]]
        argv = ["benchmark", "--config", str(self.config_path), "--data-root", str(self.root / "data"),
                "--benchmark-dir", str(output)]
        # run_benchmark may have been imported by dependency-light selection tests.
        with patch("sys.argv", argv), patch.object(benchmark, "RynnBrainModel", return_value=self.model) as factory, \
             patch.object(benchmark, "discover_bags", return_value=all_bags), \
             patch.object(benchmark, "infer_bag_record", side_effect=lambda path, *args: {
                 "bag_name": path.name, "bag_path": str(path),
                 "label": "fail" if path.name.startswith("failure") else "normal"}), \
             patch.object(benchmark, "evaluate_multiturn", self.run.evaluate_multiturn), \
             patch.object(benchmark, "write_multiturn_outputs", self.run.write_multiturn_outputs), \
             patch.object(benchmark, "_run_dino_test", return_value=True) as dino, \
             patch.object(benchmark, "analyze_saved_vectors", return_value=None), \
             patch.object(benchmark, "write_roc_report", return_value={"metrics": []}):
            benchmark.main()
        factory.assert_called_once()
        self.assertEqual(dino.call_count, 2)
        self.assertEqual(self.extractions, Counter(self.names + ["normal_test", "failure_test"]))
        clean = pd.read_csv(output / "benchmark_clean.csv")
        self.assertEqual(set(clean.bag_name), {"normal_test", "failure_test"})
        self.assertTrue(clean.failure_probability.notna().all())
        statistics = json.loads((output / "benchmark_statistics.json").read_text())
        self.assertEqual(statistics["classifier"]["scored_rows"], 2)
        probability_report = statistics["failure_probability"]["modes"][0]
        self.assertEqual(probability_report["scored_rows"], 2)
        self.assertTrue(Path(probability_report["plot_file"]).is_file())
        self.assertEqual(json.loads(self.config_path.read_text()), self.config)

    def test_dino_only_benchmark_does_not_load_vlm_or_prepare_enabled_classifier(self):
        import run_benchmark as benchmark
        output = self.root / "dino_only"
        argv = ["benchmark", "--config", str(self.config_path), "--benchmark-dir", str(output), "--skip-vlm"]
        with patch("sys.argv", argv), patch.object(benchmark, "discover_bags", return_value=[]), \
             patch.object(benchmark, "RynnBrainModel", side_effect=AssertionError("VLM must not load")) as factory, \
             patch.object(benchmark, "write_roc_report", return_value={"metrics": []}):
            benchmark.main()
        factory.assert_not_called()
        self.assertFalse(self.model_path.exists())
        statistics = json.loads((output / "benchmark_statistics.json").read_text())
        self.assertEqual(statistics["failure_probability"]["modes"][0]["status"], "skipped")


if __name__ == "__main__":
    unittest.main()
