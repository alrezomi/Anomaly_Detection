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
                "cop_classifier": {"enabled": True, "plot_timeline": False, "model_paths": {"raw": str(self.model_path)},
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
            lora_adapter_path="/saved/adapter",
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

    def test_training_responses_are_saved_and_missing_or_stale_files_are_repaired(self):
        paths, _ = prepare_training_vectors(self.config, self.settings, model=self.model)
        for path in paths:
            directory = path.parent.parent
            metadata = json.loads(path.read_text())
            response = json.loads((directory / "rynnbrain_responses_multiturn.json").read_text())["results"][0]
            row = pd.read_csv(directory / "rynnbrain_results_multiturn.csv").iloc[0]
            self.assertTrue((directory / "selected_vlm_frames.csv").is_file())
            self.assertEqual(response["evaluation_id"], metadata["evaluation_id"])
            self.assertEqual(row.evaluation_id, metadata["evaluation_id"])
            self.assertEqual(response["sample_role"], "classifier_training")
            self.assertEqual(response["lora_adapter_sha256"], "adapter_v1")
            self.assertEqual(response["lora_adapter_path"], "/saved/adapter")
            self.assertEqual(response["turns"][0]["response"], "reference response")
            self.assertEqual(response["turns"][1]["response"], row.response)
        directory = paths[0].parent.parent
        for filename in ("rynnbrain_results_multiturn.csv", "rynnbrain_responses_multiturn.json", "selected_vlm_frames.csv"):
            (directory / filename).unlink()
            before = self.extractions.copy()
            prepare_training_vectors(self.config, self.settings, model=self.model)
            self.assertEqual(self.extractions - before, Counter({"normal_1": 1}))
        # A response left over from an older run must not validate a newer vector.
        response_path = directory / "rynnbrain_responses_multiturn.json"
        stale = json.loads(response_path.read_text())
        stale["results"][0]["evaluation_id"] = "older-run"
        response_path.write_text(json.dumps(stale))
        before = self.extractions.copy()
        prepare_training_vectors(self.config, self.settings, model=self.model)
        self.assertEqual(self.extractions - before, Counter({"normal_1": 1}))

    def test_preparation_preserves_other_modes_and_reuses_complete_responses(self):
        prepare_training_vectors(self.config, self.settings, model=self.model)
        heatmap_settings = {**self.settings, "input_mode": "heatmap"}
        with patch("rynnbrain_vlm.run._heatmap_inputs", return_value=([("heatmap", Image.new("RGB", (2, 2)))], [])):
            prepare_training_vectors(self.config, heatmap_settings, model=self.model)
            before = self.extractions.copy()
            prepare_training_vectors(self.config, self.settings, model=self.model)
            prepare_training_vectors(self.config, heatmap_settings, model=self.model)
        self.assertEqual(self.extractions, before)
        directory = self.root / "vectors/normal_1/rynnbrain_multiturn"
        responses = json.loads((directory / "rynnbrain_responses_multiturn.json").read_text())["results"]
        self.assertEqual({row["input_mode"] for row in responses}, {"raw", "heatmap"})
        self.assertEqual(set(pd.read_csv(directory / "rynnbrain_results_multiturn.csv").input_mode), {"raw", "heatmap"})

    def test_benchmark_prepares_once_excludes_training_and_populates_reports(self):
        self._check_benchmark_workflow()

    def test_knn_benchmark_uses_only_nominals_and_preserves_responses_and_scores(self):
        classifier = self.config["rynnbrain"]["cop_classifier"]
        classifier.update(method="knn", knn={"n_neighbors": 1, "threshold_quantile": 0.95})
        classifier["training"]["failure_bags"] = ["/missing/unused_failure_bag"]
        self.config_path.write_text(json.dumps(self.config))
        self.settings = configured_training_settings(self.config_path)
        self.model.adapter_training_bags = set(self.names[:2])
        self._check_benchmark_workflow(knn=True)
        # The failure list is ignored even when its bags are unavailable. Only
        # nominal training responses/vectors are prepared and later reused.
        self.assertFalse((self.root / "vectors/failure_1").exists())
        before = self.extractions.copy()
        paths, _ = prepare_training_vectors(self.config, self.settings, model=self.model)
        self.assertEqual(len(paths), 2)
        self.assertEqual(self.extractions, before)
        metadata = json.loads(self.settings["output_file"].with_suffix(".json").read_text())
        self.assertEqual(metadata["training_bags"], self.names[:2])
        for path in paths:
            saved = json.loads((path.parent.parent / "rynnbrain_responses_multiturn.json").read_text())["results"][0]
            self.assertEqual(saved["sample_role"], "classifier_training")
            self.assertEqual(saved["lora_adapter_sha256"], "adapter_v1")

    def _check_benchmark_workflow(self, knn=False):
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
        factory.assert_called_once_with(self.config["rynnbrain"]["model"])
        test_names = ["normal_test", "failure_test"] + (self.names[2:] if knn else [])
        self.assertEqual(dino.call_count, len(test_names))
        self.assertEqual(self.extractions, Counter(self.names + ["normal_test", "failure_test"]))
        clean = pd.read_csv(output / "benchmark_clean.csv")
        self.assertEqual(set(clean.bag_name), set(test_names))
        if knn:
            self.assertTrue(clean.failure_probability.isna().all())
            self.assertTrue(clean.anomaly_score.notna().all())
            self.assertTrue((clean.anomaly_threshold > 0).all())
            self.assertTrue(clean.classifier_decision.isin(["success", "failure"]).all())
        else:
            self.assertTrue(clean.failure_probability.notna().all())
        summary = pd.read_csv(output / "benchmark_summary.csv")
        self.assertTrue((summary.lora_adapter_sha256 == "adapter_v1").all())
        self.assertTrue((summary.response_source == "lora_adapted_model").all())
        for name in test_names:
            response_path = output / name / "rynnbrain_multiturn/rynnbrain_responses_multiturn.json"
            saved = json.loads(response_path.read_text())["results"][0]
            self.assertEqual(saved["sample_role"], "evaluation")
            self.assertEqual(saved["response"], summary.loc[summary.bag_name == name, "response"].iloc[0])
        statistics = json.loads((output / "benchmark_statistics.json").read_text())
        self.assertEqual(statistics["classifier"]["scored_rows"], len(test_names))
        score_report = statistics["anomaly_score" if knn else "failure_probability"]["modes"][0]
        self.assertEqual(score_report["scored_rows"], len(test_names))
        self.assertTrue(Path(score_report["plot_file"]).is_file())
        roc = statistics["anomaly_roc" if knn else "probability_roc"]
        self.assertEqual(roc["score_column"], "classifier_anomaly_score" if knn else "classifier_failure_probability")
        self.assertEqual(roc["metrics"][0]["status"], "created")
        self.assertTrue(all(Path(path).is_file() for path in roc["plot_files"]))
        if knn:
            self.assertEqual(statistics["probability_roc"]["metrics"][0]["status"], "skipped")
        self.assertEqual(json.loads(self.config_path.read_text()), self.config)

    def test_knn_benchmark_trains_progress_prefixes_and_writes_each_case_timeline(self):
        vlm = self.config["rynnbrain"]
        vlm["num_frames"] = 3
        vlm["cop_classifier"].update(method="knn", plot_timeline=True, knn={"n_neighbors": 1})
        self.model.adapter_training_bags = set(self.names[:2])
        self.model.extract_multiturn_cop_vector = Mock(side_effect=lambda turns, *args, **kwargs:
            self.torch.tensor([1.0, len(turns[1]["images"]) * 0.2, 0.3, 0.4]))
        self.config_path.write_text(json.dumps(self.config))
        self.settings = configured_training_settings(self.config_path)
        frames = [{"timestamp_sec": time, "sample_index": i} for i, time in enumerate((4.5, 9.2, 20.0))]
        def inputs(path, *args):
            return [(path.name, Image.new("RGB", (2, 2))) for _ in frames], frames
        with patch("rynnbrain_vlm.run._raw_inputs", side_effect=inputs):
            self._check_benchmark_workflow(knn=True)
            summary = pd.read_csv(self.root / "benchmark/benchmark_summary.csv")
            self.assertTrue((summary.knn_timeline_status == "created").all())
            self.assertEqual(self.model.extract_multiturn_cop_vector.call_count, 12)
            for _, row in summary.iterrows():
                timeline = pd.read_csv(row.knn_timeline_csv)
                self.assertTrue(Path(row.knn_timeline_plot).is_file())
                self.assertEqual(timeline.timestamp_sec.tolist(), [4.5, 9.2, 20.0])
                self.assertAlmostEqual(timeline.anomaly_score.iloc[-1], row.classifier_anomaly_score)
                self.assertAlmostEqual(timeline.threshold.iloc[-1], row.classifier_threshold)
            before = self.model.extract_multiturn_cop_vector.call_count
            prepare_training_vectors(self.config, self.settings, model=self.model)
            self.assertEqual(self.model.extract_multiturn_cop_vector.call_count, before)
            # An optional plotting failure must still save the complete VLM
            # response/vector and score. Disabling plotting clears stale graphs
            # and performs no prefix passes.
            bag_config = {**self.config, "test_bag": str(self.root / "data/normal_test")}
            bag_vlm = {**vlm, "cop_vectors": {"enabled": True}}
            output = self.root / "single"
            with patch("rynnbrain_vlm.cop_timeline.write_knn_timeline", side_effect=RuntimeError("plot unavailable")):
                rows, frames_out, responses, task = self.run.evaluate_multiturn(
                    self.model, bag_config, bag_vlm, 3, {}, output, reference_response="reference response")
                self.run.write_multiturn_outputs(output, rows, frames_out, responses, task)
            self.assertEqual(rows[0]["knn_timeline_error"], "plot unavailable")
            self.assertEqual(rows[0]["response"], "Decision: success")
            self.assertIsNotNone(rows[0]["classifier_anomaly_score"])
            self.assertTrue((output / "rynnbrain_responses_multiturn.json").is_file())
            self.assertTrue(Path(rows[0]["cop_vector_path"]).is_file())
            (output / "knn_timeline_raw.png").write_bytes(b"stale plot")
            bag_vlm["cop_classifier"] = {**vlm["cop_classifier"], "plot_timeline": False}
            rows, *_ = self.run.evaluate_multiturn(
                self.model, bag_config, bag_vlm, 3, {}, output, reference_response="reference response")
            self.assertEqual(rows[0]["knn_timeline_status"], "disabled")
            self.assertFalse((output / "knn_timeline_raw.png").exists())
            self.assertEqual(self.model.extract_multiturn_cop_vector.call_count, before)

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

    def test_benchmark_records_why_a_bag_has_no_response(self):
        import run_benchmark as benchmark
        self.config["rynnbrain"]["cop_classifier"]["enabled"] = False
        self.config_path.write_text(json.dumps(self.config))
        output = self.root / "failed_benchmark"
        bag = self.root / "data/normal_test"
        argv = ["benchmark", "--config", str(self.config_path), "--benchmark-dir", str(output), "--skip-dino"]
        with patch("sys.argv", argv), patch.object(benchmark, "RynnBrainModel", return_value=self.model), \
             patch.object(benchmark, "discover_bags", return_value=[bag]), \
             patch.object(benchmark, "infer_bag_record", return_value={"bag_name": bag.name, "bag_path": str(bag), "label": "normal"}), \
             patch.object(benchmark, "evaluate_multiturn", side_effect=RuntimeError("test generation failed")), \
             patch.object(benchmark, "analyze_saved_vectors", return_value=None), \
             patch.object(benchmark, "write_roc_report", return_value={"metrics": []}):
            benchmark.main()
        row = pd.read_csv(output / "benchmark_summary.csv").iloc[0]
        self.assertEqual(row.decision, "failed")
        self.assertEqual(row.evaluation_error, "test generation failed")
        self.assertTrue(pd.isna(row.response))


if __name__ == "__main__":
    unittest.main()
