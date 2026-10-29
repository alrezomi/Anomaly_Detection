"""Unattended orchestration checks; GPU training/inference are simulated."""

from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from pipeline_settings import load_pipeline_config
from rynnbrain_vlm.benchmark_training import (
    prepare_pipeline_run, run_lora_training, train_lora_enabled, resolve_trained_artifacts,
)


class BenchmarkTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.events = []
        self.names = ["train_normal_1", "train_normal_2", "train_fail_1", "train_fail_2"]
        self.bags = {}
        for name in self.names + ["reference", "test_normal", "test_fail"]:
            bag = self.root / "data/chosen" / name
            bag.mkdir(parents=True)
            (bag / "metadata.yaml").write_text("{}")
            self.bags[name] = str(bag)
        old = self.root / "old_adapter"
        old.mkdir()
        (old / "adapter_model.safetensors").write_bytes(b"keep old adapter")
        self.config_path = self.root / "config.json"
        self.config_path.write_text(json.dumps({
            "output_dir": str(self.root / "experiment"), "camera_topics": ["top"],
            "stage_topic": "/recording_stage", "stage_count": 3,
            "vlm_benchmark": {"train_lora": True, "decision_mode": "timeline", "folders": ["chosen"]},
            "rynnbrain": {
                "source": "rosbag", "prompt_variant": "B", "input_modes": ["raw"], "num_frames": 3,
                "task_description": "Pick the scissors and place them in the blue basket.",
                "reference_bags": [self.bags["reference"]], "model": {"model_id": "test", "lora_adapter_path": str(old)},
                "generation": {"do_sample": False}, "inference": {"feature_only": True},
                "lora": {"rank": 8, "training": {
                    "normal_bags": [self.bags[name] for name in self.names[:2]],
                    "failure_bags": [self.bags[name] for name in self.names[2:]],
                    "temporal_supervision": True, "epochs": 3}},
                "cop_classifier": {"enabled": False, "method": "logistic", "train_on_benchmark": False,
                    "model_paths": {"raw": str(self.root / "old_classifier.npz")},
                    "logistic_onset": {"enabled": True},
                    "training": {"threshold": .6, "regularization_c": 100.,
                        "normal_bags": [self.bags[name] for name in self.names[:2]],
                        "failure_bags": [self.bags[name] for name in self.names[2:]]}},
            }}))
        (self.root / "old_classifier.npz").write_bytes(b"keep old classifier")
        self.original = self.config_path.read_bytes()

    def arguments(self):
        return SimpleNamespace(config=self.config_path, skip_vlm=False, benchmark_dir=None,
                               data_root=self.root / "data")

    def test_flag_is_opt_in_and_rejects_strings(self):
        self.assertFalse(train_lora_enabled({}))
        self.assertFalse(train_lora_enabled({"vlm_benchmark": {"train_lora": False}}))
        with self.assertRaisesRegex(ValueError, "must be true or false"):
            train_lora_enabled({"vlm_benchmark": {"train_lora": "false"}})

    def _completed_artifacts(self, status="completed"):
        config = load_pipeline_config(self.config_path)
        arguments, saved, output = prepare_pipeline_run(self.arguments(), config)
        adapter = Path(saved["rynnbrain"]["model"]["lora_adapter_path"])
        adapter.mkdir()
        for name in ["adapter_config.json", "training_manifest.json"]:
            (adapter / name).write_text("{}")
        (adapter / "adapter_model.safetensors").write_bytes(b"trained adapter")
        classifier = Path(saved["rynnbrain"]["cop_classifier"]["model_paths"]["raw"])
        classifier.parent.mkdir()
        classifier.write_bytes(b"trained classifier")
        classifier.with_suffix(".json").write_text("{}")
        output.mkdir()
        (output / "benchmark_runtime.json").write_text(json.dumps({"status": status}))
        return arguments, saved

    def test_replay_finds_existing_completed_run_without_a_new_pointer_or_training(self):
        _, saved = self._completed_artifacts()
        current = load_pipeline_config(self.config_path)
        current["rynnbrain"]["inference"]["feature_only"] = False
        current["rynnbrain"]["cop_classifier"]["training"]["threshold"] = .7
        before = deepcopy(current)
        progress = Mock()
        resolved = resolve_trained_artifacts(current, progress)
        self.assertEqual(resolved["rynnbrain"]["model"]["lora_adapter_path"],
                         saved["rynnbrain"]["model"]["lora_adapter_path"])
        self.assertEqual(resolved["rynnbrain"]["cop_classifier"]["model_paths"],
                         saved["rynnbrain"]["cop_classifier"]["model_paths"])
        self.assertFalse(resolved["rynnbrain"]["inference"]["feature_only"])
        self.assertEqual(resolved["rynnbrain"]["cop_classifier"]["training"]["threshold"], .7)
        self.assertEqual(current, before)
        self.assertEqual(self.config_path.read_bytes(), self.original)
        self.assertIn("Using completed training run", progress.call_args_list[0].args[0])
        # Check that OnlineDetector uses the resolved path before loading a model.
        from rynnbrain_vlm.online_replay import OnlineDetector
        from rynnbrain_vlm import run  # Bind real imports before mocking the loader.
        with patch("rynnbrain_vlm.cop_classifier.load_classifier", side_effect=RuntimeError("loader reached")) as loader:
            with self.assertRaisesRegex(RuntimeError, "loader reached"):
                OnlineDetector(current)
        self.assertEqual(loader.call_args.args[0], Path(saved["rynnbrain"]["cop_classifier"]["model_paths"]["raw"]))

    def test_replay_ignores_failed_runs_and_honors_explicit_available_models(self):
        _, saved = self._completed_artifacts()
        self._completed_artifacts(status="failed")
        current = load_pipeline_config(self.config_path)
        self.assertEqual(resolve_trained_artifacts(current)["rynnbrain"]["model"]["lora_adapter_path"],
                         saved["rynnbrain"]["model"]["lora_adapter_path"])
        current["vlm_benchmark"]["train_lora"] = False
        explicit = Path(current["rynnbrain"]["cop_classifier"]["model_paths"]["raw"])
        explicit.write_bytes(b"explicit classifier")
        explicit.with_suffix(".json").write_text("{}")
        self.assertIs(resolve_trained_artifacts(current), current)

    def test_replay_without_completed_runs_keeps_config_and_missing_artifacts_raise(self):
        current = load_pipeline_config(self.config_path)
        self.assertIs(resolve_trained_artifacts(current), current)
        _, saved = self._completed_artifacts()
        Path(saved["rynnbrain"]["cop_classifier"]["model_paths"]["raw"]).unlink()
        with self.assertRaisesRegex(FileNotFoundError, "missing artifacts"):
            resolve_trained_artifacts(current)

    def test_disabled_training_preserves_null_or_explicit_adapter_with_missing_classifier(self):
        self._completed_artifacts()
        current = load_pipeline_config(self.config_path)
        current["vlm_benchmark"]["train_lora"] = False
        Path(current["rynnbrain"]["cop_classifier"]["model_paths"]["raw"]).unlink(missing_ok=True)
        for adapter in (None, str(self.root / "explicit_adapter")):
            with self.subTest(adapter=adapter):
                current["rynnbrain"]["model"]["lora_adapter_path"] = adapter
                before = deepcopy(current)
                progress = Mock()
                with patch.object(Path, "glob", side_effect=AssertionError("Must not discover runs")):
                    self.assertIs(resolve_trained_artifacts(current, progress), current)
                self.assertEqual(current, before)
                progress.assert_not_called()

    def test_normal_benchmark_keeps_configured_paths_when_training_disabled(self):
        import run_benchmark as benchmark
        from rynnbrain_vlm.cop_classifier import configured_training_settings
        _, saved = self._completed_artifacts()
        original = json.loads(self.original)
        original["vlm_benchmark"]["train_lora"] = False
        original["rynnbrain"]["cop_classifier"]["train_on_benchmark"] = False
        reuse = self.root / "reuse.json"
        reuse.write_text(json.dumps(original))
        expected = load_pipeline_config(reuse)
        def check(arguments, config, update):
            settings = configured_training_settings(arguments.config, "raw")
            self.assertEqual(settings["output_file"], Path(expected["rynnbrain"]["cop_classifier"]["model_paths"]["raw"]))
            self.assertEqual(config["rynnbrain"]["model"]["lora_adapter_path"],
                             original["rynnbrain"]["model"]["lora_adapter_path"])
        with patch.object(sys, "argv", ["benchmark", "--config", str(reuse)]), \
             patch.object(benchmark, "_run_benchmark", side_effect=check), \
             patch("rynnbrain_vlm.benchmark_training.subprocess.Popen") as training:
            benchmark.main()
        training.assert_not_called()

    def test_fresh_paths_preserve_old_models_prompt_training_and_user_config(self):
        config = load_pipeline_config(self.config_path)
        original = deepcopy(config)
        arguments, effective, directory = prepare_pipeline_run(self.arguments(), config)
        second, _, _ = prepare_pipeline_run(self.arguments(), config)
        self.assertNotEqual(arguments.benchmark_dir, second.benchmark_dir)
        self.assertEqual(config, original)
        self.assertEqual(directory, arguments.benchmark_dir / "timeline")
        self.assertEqual(self.config_path.read_bytes(), self.original)
        training = load_pipeline_config(arguments.lora_training_config)
        vlm = effective["rynnbrain"]
        self.assertIsNone(training["rynnbrain"]["model"]["lora_adapter_path"])
        self.assertEqual(vlm["model"]["lora_adapter_path"], training["rynnbrain"]["lora"]["training"]["output_dir"])
        self.assertTrue(vlm["lora"]["training"]["temporal_supervision"])
        self.assertEqual(vlm["lora"]["training"]["normal_bags"], original["rynnbrain"]["lora"]["training"]["normal_bags"])
        self.assertEqual(vlm["prompt_variant"], "B")
        self.assertEqual(vlm["task_description"], original["rynnbrain"]["task_description"])
        self.assertEqual(effective["vlm_benchmark"]["folders"], ["chosen"])
        self.assertFalse(effective["vlm_benchmark"]["train_lora"])
        for path in [vlm["model"]["lora_adapter_path"], vlm["cop_classifier"]["model_paths"]["raw"],
                     vlm["cop_classifier"]["training"]["input_dir"]]:
            self.assertTrue(Path(path).is_relative_to(arguments.benchmark_dir))
        self.assertEqual(load_pipeline_config(arguments.config), effective)
        self.assertTrue(vlm["cop_classifier"]["enabled"])
        self.assertTrue(vlm["cop_classifier"]["train_on_benchmark"])
        self.assertEqual(vlm["cop_classifier"]["training"]["threshold"], .6)
        self.assertTrue(vlm["cop_classifier"]["logistic_onset"]["enabled"])
        bad = self.arguments()
        bad.skip_vlm = True
        with self.assertRaisesRegex(ValueError, "remove --skip-vlm"):
            prepare_pipeline_run(bad, config)

    def _run_pipeline(self, exit_code=0):
        import run_benchmark as benchmark
        import pandas as pd

        def process(command, **kwargs):
            self.events.append("lora")
            self.assertEqual(command[1:4], ["-u", "-m", "rynnbrain_vlm.train_lora"])
            self.assertEqual(kwargs["env"]["STAGE_STARTUP_IGNORE_SEC"], "0.1")
            config = load_pipeline_config(command[command.index("--config") + 1])
            self.assertIsNone(config["rynnbrain"]["model"]["lora_adapter_path"])
            output = Path(config["rynnbrain"]["lora"]["training"]["output_dir"])
            if exit_code == 0:
                output.mkdir()
                (output / "adapter_config.json").write_text("{}")
                (output / "adapter_model.safetensors").write_bytes(b"new weights")
                (output / "training_summary.json").write_text(json.dumps({"saved_epoch": 3, "optimizer_steps": 5}))
                (output / "training_manifest.json").write_text(json.dumps({"bags": [
                    {"bag_name": name, "bag_path": self.bags[name], "split": "train"} for name in self.names]}))
            child = Mock()
            child.stdout = iter(["Epoch 3: loss=0.2\n"])
            child.returncode = exit_code
            child.wait.return_value = exit_code
            child.__enter__ = Mock(return_value=child)
            child.__exit__ = Mock(return_value=False)
            return child

        def model_factory(settings):
            self.events.append("model")
            self.assertTrue((Path(settings["lora_adapter_path"]) / "adapter_model.safetensors").is_file())
            return SimpleNamespace(temporal_profile={}, adapter_training_bags=set(self.names))

        def vectors(config, settings, **kwargs):
            self.events.append("vectors")
            self.assertTrue(Path(config["rynnbrain"]["model"]["lora_adapter_path"]).is_dir())
            return [], "reference response"

        def fit(**settings):
            self.events.append("classifier")
            path = settings["output_file"]
            path.parent.mkdir(parents=True)
            path.write_bytes(b"new classifier")
            path.with_suffix(".json").write_text(json.dumps({"training_bags": self.names}))
            return {"model_file": str(path)}

        def evaluate(model, bag_config, vlm, count, generation, output, **kwargs):
            self.events.append("test")
            name = Path(bag_config["test_bag"]).name
            failed = name == "test_fail"
            output.mkdir(parents=True, exist_ok=True)
            path = output / "logistic_timeline_raw.csv"
            pd.DataFrame({"timestamp_sec": [1., 2., 3.], "anomaly_score": [.1, .8 if failed else .2, .1],
                          "threshold": [.6] * 3}).to_csv(path, index=False)
            return ([{"input_mode": "raw", "decision": "not_generated", "confidence": None, "response": "",
                      "classifier_decision": "success", "classifier_failure_probability": .1, "classifier_threshold": .6,
                      "logistic_timeline_status": "created", "logistic_timeline_csv": str(path)}], [], [], "task")

        with ExitStack() as stack:
            stack.enter_context(patch.object(sys, "argv", ["benchmark", "--config", str(self.config_path),
                                                          "--data-root", str(self.root / "data")]))
            stack.enter_context(patch("rynnbrain_vlm.benchmark_training.subprocess.Popen", side_effect=process))
            factory = stack.enter_context(patch.object(benchmark, "RynnBrainModel", side_effect=model_factory))
            stack.enter_context(patch("rynnbrain_vlm.cop_classifier.prepare_training_vectors", side_effect=vectors))
            stack.enter_context(patch("rynnbrain_vlm.cop_classifier.train_from_saved_vectors", side_effect=fit))
            stack.enter_context(patch.object(benchmark, "infer_bag_record", side_effect=lambda path, *args:
                {"bag_name": path.name, "bag_path": str(path), "label": "fail" if path.name == "test_fail" else "normal"}))
            stack.enter_context(patch.object(benchmark, "evaluate_multiturn", side_effect=evaluate))
            stack.enter_context(patch.object(benchmark, "write_multiturn_outputs"))
            stack.enter_context(patch.object(benchmark, "_align_benchmark_timing"))
            stack.enter_context(patch.object(benchmark, "analyze_saved_vectors", return_value=None))
            try:
                benchmark.main()
            finally:
                if exit_code:
                    factory.assert_not_called()

    def test_prompt_a_whole_demo_uses_custom_root_and_ordinary_training_when_configured(self):
        from pipeline_settings import resolve_pipeline_config
        config = json.loads(self.original)
        config["rynnbrain"]["prompt_variant"] = "A"
        config["rynnbrain"]["lora"]["training"]["temporal_supervision"] = False
        config["vlm_benchmark"]["decision_mode"] = "whole_demo"
        arguments = self.arguments()
        arguments.benchmark_dir = self.root / "custom_reports"
        arguments, effective, output = prepare_pipeline_run(arguments, resolve_pipeline_config(config))
        self.assertTrue(output.is_relative_to(self.root / "custom_reports/pipeline_runs"))
        self.assertEqual(output.name, "whole_demo")
        self.assertEqual(Path(effective["rynnbrain"]["model"]["lora_adapter_path"]).name, "lora_adapter")
        self.assertFalse(effective["rynnbrain"]["lora"]["training"]["temporal_supervision"])
        self.assertEqual(effective["rynnbrain"]["prompt_variant"], "A")
        self.assertEqual(effective["rynnbrain"]["cop_classifier"]["model_paths"]["raw"],
                         str(arguments.benchmark_dir / "cop_classifier/raw_logistic.npz"))

    def test_complete_pipeline_orders_stages_and_saves_reusable_config_and_reports(self):
        self._run_pipeline()
        import pandas as pd
        self.assertEqual(self.events, ["lora", "model", "vectors", "classifier", "test", "test"])
        paths = list((self.root / "experiment").glob("**/pipeline_config.json"))
        self.assertEqual(len(paths), 1)
        effective = load_pipeline_config(paths[0])
        self.assertFalse(effective["vlm_benchmark"]["train_lora"])
        self.assertFalse(effective["rynnbrain"]["cop_classifier"]["train_on_benchmark"])
        output = paths[0].parent / "timeline"
        summary = pd.read_csv(output / "benchmark_summary.csv")
        self.assertEqual(set(summary.bag_name), {"test_normal", "test_fail"})
        self.assertEqual(set(summary.classifier_decision), {"success", "failure"})
        for file in ["benchmark_metrics.png", "benchmark_failure_probability_raw.png", "REPORTS.md"]:
            self.assertTrue((output / file).is_file())
        runtime = json.loads((output / "benchmark_runtime.json").read_text())
        self.assertEqual(runtime["status"], "completed")
        self.assertEqual(runtime["lora_training_status"], "completed")
        self.assertGreaterEqual(runtime["lora_training_sec"], 0.)
        self.assertIn("Epoch 3: loss=0.2", (output / "benchmark_run.log").read_text())
        self.assertEqual(self.config_path.read_bytes(), self.original)
        self.assertEqual((self.root / "old_classifier.npz").read_bytes(), b"keep old classifier")
        self.assertEqual((self.root / "old_adapter/adapter_model.safetensors").read_bytes(), b"keep old adapter")

    def test_failed_training_stops_before_classifier_or_benchmark_and_records_failure(self):
        with self.assertRaises(subprocess.CalledProcessError):
            self._run_pipeline(exit_code=2)
        self.assertEqual(self.events, ["lora"])
        runtime_path = next((self.root / "experiment").glob("**/benchmark_runtime.json"))
        runtime = json.loads(runtime_path.read_text())
        self.assertEqual(runtime["status"], "failed")
        self.assertEqual(runtime["lora_training_status"], "failed")
        self.assertFalse((runtime_path.parent / "benchmark_summary.csv").exists())

    def test_validation_bags_cannot_be_reused_for_classifier_training(self):
        config = load_pipeline_config(self.config_path)
        training = config['rynnbrain']['lora']['training']
        training['validation_normal_bags'] = [self.bags['test_normal']]
        training['validation_failure_bags'] = [self.bags['test_fail']]
        classifier = config['rynnbrain']['cop_classifier']['training']
        classifier['normal_bags'].append(self.bags['test_normal'])
        with self.assertRaisesRegex(ValueError, 'Remove LoRA validation bags'):
            prepare_pipeline_run(self.arguments(), config)
        classifier['normal_bags'].remove(self.bags['test_normal'])
        _, effective, _ = prepare_pipeline_run(self.arguments(), config)
        self.assertEqual(effective['rynnbrain']['lora']['training']['validation_normal_bags'],
                         [self.bags['test_normal']])

    def test_successful_exit_without_checkpoint_is_not_accepted(self):
        child = Mock()
        child.stdout = iter([])
        child.wait.return_value = 0
        child.__enter__ = Mock(return_value=child)
        child.__exit__ = Mock(return_value=False)
        update = Mock()
        with patch("rynnbrain_vlm.benchmark_training.subprocess.Popen", return_value=child):
            with self.assertRaisesRegex(RuntimeError, "without complete adapter artifacts"):
                run_lora_training(self.config_path, self.root / "data", self.root / "missing", update, .1)
        self.assertEqual(update.call_args.kwargs["lora_training_status"], "failed")


if __name__ == "__main__":
    unittest.main()
