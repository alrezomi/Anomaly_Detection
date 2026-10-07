import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from rynnbrain_vlm.temporal_lora import (training_profile, validate_profile, timestamped_images,
    temporal_annotation, prefix_samples, temporal_answer, evaluate_vlm_onset, adapter_profile)
from rynnbrain_vlm.train_lora import training_settings, validate_temporal_rows
from rynnbrain_vlm.prompts import evaluation_prompt_multiturn


class TemporalLoraTests(unittest.TestCase):
    def setUp(self):
        self.config = {"camera_topics": ["top", "side"], "output_dir": "outputs", "rynnbrain": {
            "source": "rosbag", "input_modes": ["raw"], "num_frames": 3,
            "camera_topics": ["top", "side"], "memory_camera_topics": ["top", "side"],
            "reference_bags": ["reference"], "model": {"model_id": "test"}, "output_dir": "outputs/vlm",
            "lora": {"training": {"temporal_supervision": True}}}}
        self.settings = training_settings(self.config)
        self.profile = self.settings["temporal_profile"]
        self.frames = [{"topic": topic, "sample_index": i, "timestamp_sec": time + offset}
                       for i, time in enumerate([1., 5., 9.]) for topic, offset in [("top", 0), ("side", .2)]]
        self.images = [("old label", Image.new("RGB", (4, 4))) for _ in self.frames]
        self.row = {"bag_name": "failure_a", "bag_path": "failure_a", "label": "fail", "split": "train"}
        self.annotation = {"status": "available", "time_sec": 5.1}

    def test_optional_defaults_and_new_adapter_directory_do_not_mutate_config(self):
        before = json.dumps(self.config, sort_keys=True)
        self.assertEqual(Path(self.settings["output_dir"]), Path("outputs/vlm/lora_temporal_adapter"))
        self.assertEqual(before, json.dumps(self.config, sort_keys=True))
        self.config["rynnbrain"]["lora"]["training"]["temporal_supervision"] = False
        legacy = training_settings(self.config)
        self.assertEqual(legacy["temporal_profile"], {})
        self.assertEqual(Path(legacy["output_dir"]), Path("outputs/vlm/lora_adapter"))

    def test_multiview_labels_use_actual_times_and_prefixes_never_include_future_images(self):
        images = timestamped_images(self.images, self.frames, ["top", "side"], 3)
        self.assertIn("camera top, time 1.000 s", images[0][0])
        self.assertIn("camera side, time 1.200 s", images[1][0])
        samples = prefix_samples(self.row, images, self.frames, self.profile, self.annotation)
        self.assertEqual([sample["label"] for sample in samples], ["normal", "fail", "fail"])
        self.assertEqual([len(sample["images"]) for sample in samples], [2, 4, 6])
        self.assertEqual([sample["cutoff_sec"] for sample in samples], [1.2, 5.2, 9.2])
        self.assertEqual(samples[1]["target_text"], "Decision: failure\nFailure onset (s): 5.100")
        self.assertEqual(samples[0]["target_text"], "Decision: success\nFailure onset (s): none")
        self.assertTrue(all("5.100" not in label for sample in samples for label, _ in sample["images"]))
        self.assertNotIn("5.100", evaluation_prompt_multiturn("pick", "raw", temporal=True))
        exact = prefix_samples(self.row, images, self.frames, self.profile, {**self.annotation, "time_sec": 5.2})
        self.assertEqual(exact[1]["label"], "fail")
        nominal = prefix_samples({**self.row, "label": "normal"}, images, self.frames, self.profile,
                                 {"status": "missing_marker", "time_sec": None})
        self.assertTrue(all(sample["label"] == "normal" for sample in nominal))

    def test_missing_cameras_invalid_times_and_changed_adapter_inputs_fail(self):
        for images, frames in ((self.images[:-1], self.frames[:-1]),
                               (self.images, [*self.frames[:-1], self.frames[0]])):
            with self.assertRaises(ValueError):
                timestamped_images(images, frames, ["top", "side"], 3)
        for field, value in (("camera_topics", ["top"]), ("memory_camera_topics", ["side", "top"]), ("source", "generated_videos")):
            with self.assertRaisesRegex(ValueError, "differ"):
                validate_profile(self.profile, self.config, {**self.config["rynnbrain"], field: value}, 3, ["raw"])
        with self.assertRaisesRegex(ValueError, "outside"):
            prefix_samples(self.row, self.images, self.frames, self.profile, {**self.annotation, "time_sec": 10.})

    def test_missing_or_conflicting_annotations_cannot_silently_train(self):
        with patch("rynnbrain_vlm.temporal_lora.read_failure_annotation", return_value={"status": "missing_marker"}):
            with self.assertRaisesRegex(ValueError, "no eligible"):
                temporal_annotation(self.row, self.config, .1)
        with patch("rynnbrain_vlm.temporal_lora.read_failure_annotation", return_value=self.annotation):
            with self.assertRaisesRegex(ValueError, "Nominal"):
                temporal_annotation({**self.row, "label": "normal"}, self.config, .1)

    def test_generated_onset_is_separate_from_threshold_timing_and_invalid_values_are_rejected(self):
        result = evaluate_vlm_onset("Decision: failure\nFailure onset (s): 5.500", "failure", self.frames, self.annotation, "fail")
        self.assertEqual(result["vlm_failure_onset_sec"], 5.5)
        self.assertAlmostEqual(result["vlm_failure_time_error_sec"], .4)
        self.assertNotIn("failure_time_error_sec", result)
        for value in ("nan", "inf", "-1", "20"):
            result = evaluate_vlm_onset(f"Failure onset (s): {value}", "failure", self.frames, self.annotation, "fail")
            self.assertEqual(result["vlm_failure_onset_status"], "invalid_time")
        none = evaluate_vlm_onset(temporal_answer("normal", None), "success", self.frames, self.annotation, "fail")
        self.assertEqual(none["vlm_failure_onset_status"], "not_detected")
        self.assertIsNone(none["vlm_failure_time_error_sec"])

    def test_validate_only_reads_both_camera_streams_and_stage_labels_without_gpu(self):
        from rynnbrain_vlm import run
        with patch.object(run, "_raw_inputs", return_value=(self.images, self.frames)) as reader, \
             patch("rynnbrain_vlm.temporal_lora.read_failure_annotation", return_value=self.annotation):
            validate_temporal_rows(self.config, self.settings, [self.row])
        self.assertEqual(reader.call_count, 2)
        self.assertTrue(all(call.args[1] == ["top", "side"] for call in reader.call_args_list))

    def test_single_evaluation_matches_training_prompt_and_returns_generated_time(self):
        import torch
        from rynnbrain_vlm import run
        from run_benchmark import _build_clean_report
        model = SimpleNamespace(model_id="test", temporal_profile=self.profile, adapter_identity="temporal-sha",
            adapter_training_bags=set(), lora_adapter_path="new_adapter",
            model=SimpleNamespace(config=SimpleNamespace(text_config=SimpleNamespace(hidden_size=4))),
            generate_multiturn_with_cop_vector=Mock(return_value=SimpleNamespace(nominal_response="reference",
                evaluation_response="Decision: failure\nFailure onset (s): 5.500", cop_vector=torch.tensor([1., 2., 3., 4.]))))
        config = {**self.config, "test_bag": "held_out"}
        vlm = {**config["rynnbrain"], "ground_truth_label": "fail", "cop_classifier": {"enabled": False}}
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(run, "_raw_inputs", return_value=(self.images, self.frames)), \
             patch.object(run, "read_failure_annotation", return_value=self.annotation), \
             patch.object(run, "_save_inputs"), patch.object(run, "_print_inputs"), patch.object(run, "_print_exchange"):
            rows, frames, responses, task = run.evaluate_multiturn(model, config, vlm, 3, {}, Path(directory))
            run.write_multiturn_outputs(Path(directory), rows, frames, responses, task)
            result = rows[0]
            self.assertAlmostEqual(result["vlm_failure_time_error_sec"], .4)
            self.assertIsNone(result["failure_time_error_sec"])
            turns = model.generate_multiturn_with_cop_vector.call_args.args[0]
            self.assertEqual(turns[1]["text"], evaluation_prompt_multiturn(task, "raw", temporal=True))
            self.assertEqual(len(turns[1]["images"]), 6)
            self.assertTrue(all("time " in label for turn in turns for label, _ in turn["images"]))
            self.assertNotIn("5.100", json.dumps([{**turn, "images": [label for label, _ in turn["images"]]} for turn in turns]))
            metadata = json.loads(Path(result["cop_metadata_path"]).read_text())
            self.assertEqual(metadata["comparison_signature"]["temporal_profile"], self.profile)
            self.assertEqual(_build_clean_report(rows).vlm_failure_onset_sec.iloc[0], 5.5)
            self.assertTrue((Path(directory) / "rynnbrain_responses_multiturn.json").is_file())


if __name__ == "__main__":
    unittest.main()
