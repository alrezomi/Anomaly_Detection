import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from rynnbrain_vlm.cop_analysis import save_cop_vector, discover_saved_vectors
from rynnbrain_vlm.cop_classifier import train_from_saved_vectors, load_classifier, configured_training_settings
from rynnbrain_vlm.cop_onset import (onset_model_path, train_onset_classifier, load_onset_classifier,
                                    select_annotated_records, onset_training_options)
from rynnbrain_vlm.cop_timeline import (_prefix_cache_paths, _load_prefix_cache, timeline_sampling,
                                       write_logistic_timeline)


class OnsetClassifierTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.signature = {"num_frames": 3, "hidden_size": 3, "source": "rosbag",
                          "turn1_prompt": "reference", "turn2_prompt": "execution",
                          "lora_adapter_sha256": "adapter-a"}
        self.path = self.root / "raw_logistic.npz"
        self.names = ["normal_1", "normal_2", "failure_1", "failure_2"]
        self.targets = [0, 0, 1, 1]
        for index, name in enumerate(self.names):
            before = np.array([-1., .1 * (index + 1), .3], dtype=np.float32)
            after = np.array([1., .1 * (index + 1), .3], dtype=np.float32)
            vectors = np.stack([before, after, after] if index > 1 else [before, before, before])
            save_cop_vector(self.root / name / "rynnbrain_multiturn", "raw", vectors[-1], {
                "bag_name": name, "test_bag": "/data/" + name, "evaluation_id": name + "-eval",
                "ground_truth_label": "fail" if index > 1 else "normal",
                "representation_id": "test", "comparison_signature": self.signature})
            record = next(row for row in discover_saved_vectors(self.root) if row.metadata["bag_name"] == name)
            cache, sidecar = _prefix_cache_paths(record)
            cache.parent.mkdir(parents=True)
            np.savez_compressed(cache, vectors=vectors)
            sampling = timeline_sampling({}, 3)
            sidecar.write_text(json.dumps({"protocol": sampling["protocol"], "sampling": sampling,
                "full_evaluation_id": name + "-eval", "comparison_signature": self.signature,
                "snapshots": [{"timestamp_sec": time} for time in [1., 5., 9.]]}))
        self.records = sorted(discover_saved_vectors(self.root), key=lambda row: self.names.index(row.metadata["bag_name"]))
        self.metadata = train_from_saved_vectors(self.root, self.path, bag_names=self.names, c_value=100)
        self.complete = load_classifier(self.path)
        self.options = {"topic": "/recording_stage", "startup_ignore_sec": .1}
        patcher = patch("rynnbrain_vlm.cop_onset.read_failure_annotation", side_effect=self.annotation)
        self.read_annotation = patcher.start()
        self.addCleanup(patcher.stop)

    @staticmethod
    def annotation(path, **kwargs):
        failure = Path(path).name.startswith("failure")
        return {"status": "available" if failure else "missing_marker", "time_sec": 5. if failure else None,
                "marker": "Error" if failure else None, "error": None}

    def train(self):
        return train_onset_classifier(self.records, self.targets, self.path, self.metadata, self.options)

    def test_training_labels_before_at_and_after_marker_and_grouped_validation(self):
        original = self.path.read_bytes(), self.path.with_suffix('.json').read_bytes()
        metadata = self.train()
        rows = pd.read_csv(metadata["cross_validation"]["predictions_csv"])
        self.assertEqual(metadata["training_prefix_count"], 12)
        for name in self.names:
            group = rows[rows.bag_name == name]
            self.assertEqual(group.failure_has_occurred.tolist(), [0, 1, 1] if name.startswith("failure") else [0, 0, 0])
            self.assertEqual(group.validation_fold.nunique(), 1)
        self.assertTrue(rows.out_of_fold_failure_probability.between(0, 1).all())
        self.assertEqual(rows.validation_fold.nunique(), 2)
        self.assertEqual(original, (self.path.read_bytes(), self.path.with_suffix('.json').read_bytes()))
        detector, path = load_onset_classifier(self.path, self.complete, bag_name="held_out")
        self.assertEqual(path, onset_model_path(self.path))
        self.assertLess(detector.predict_failure_probability([-1., .2, .3]), .5)
        self.assertGreater(detector.predict_failure_probability([1., .2, .3]), .5)
        # Only configured training bags are ever read for fitting.
        self.assertEqual({call.args[0] for call in self.read_annotation.call_args_list}, {"/data/" + name for name in self.names})

    def test_bad_annotations_are_excluded_and_insufficient_supervision_fails(self):
        def annotations(path, **kwargs):
            result = self.annotation(path)
            if Path(path).name == "failure_1":
                result.update(status="missing_marker", time_sec=None)
            if Path(path).name == "normal_1":
                result.update(status="available", time_sec=4.)
            return result
        self.read_annotation.side_effect = annotations
        selected, _, excluded = select_annotated_records(self.records, self.targets, self.options)
        self.assertEqual({row.metadata["bag_name"] for row in selected}, {"normal_2", "failure_2"})
        self.assertEqual(len(excluded), 2)
        with self.assertRaisesRegex(ValueError, "at least two"):
            self.train()
        self.assertFalse(onset_model_path(self.path).exists())
        self.assertTrue(self.path.is_file())

    def test_error_after_last_image_is_excluded_instead_of_labeling_it_positive(self):
        self.read_annotation.side_effect = lambda path, **kwargs: {**self.annotation(path),
            **({"time_sec": 10.} if Path(path).name == "failure_1" else {})}
        with self.assertRaisesRegex(ValueError, "error_after_last_sample"):
            self.train()

    def test_missing_timestamps_stale_vectors_and_stale_models_are_rejected(self):
        self.train()
        with self.assertRaisesRegex(ValueError, "held-out"):
            load_onset_classifier(self.path, self.complete, bag_name="failure_1")
        path = self.path.with_suffix('.json')
        content = json.loads(path.read_text())
        content["threshold"] = .6
        path.write_text(json.dumps(content))
        with self.assertRaisesRegex(ValueError, "stale"):
            load_onset_classifier(self.path, self.complete, bag_name="test")
        _, sidecar = _prefix_cache_paths(self.records[0])
        content = json.loads(sidecar.read_text())
        content["snapshots"][-1].pop("timestamp_sec")
        sidecar.write_text(json.dumps(content))
        with self.assertRaisesRegex(ValueError, "timestamps"):
            _load_prefix_cache(self.records[0], require_timestamps=True)
        with self.assertRaisesRegex(ValueError, "Missing/stale"):
            self.train()
        self.assertFalse(onset_model_path(self.path).exists())

    def test_timeline_uses_onset_classifier_for_all_points_without_reading_test_annotation(self):
        self.train()
        detector, _ = load_onset_classifier(self.path, self.complete, bag_name="test")
        self.read_annotation.reset_mock()
        vectors = [np.array([-1., .25, .3]), np.array([1., .25, .3]), np.array([1., .2, .3])]
        model = SimpleNamespace(extract_multiturn_cop_vector=Mock(side_effect=vectors[:2]))
        frames = [{"sample_index": i, "timestamp_sec": time} for i, time in enumerate([1., 5., 9.])]
        turns = [{"images": [], "text": "reference"}, {"images": ["a", "b", "c"], "text": "execution"}]
        report = write_logistic_timeline(model=model, classifier=self.complete, turns=turns, generation={},
            nominal_response="reference response", frame_metadata=frames, final_vector=vectors[-1],
            comparison_signature=self.signature, output_directory=self.root, mode="raw", bag_name="test",
            provenance={"evaluation_id": "test-eval"}, classifier_model_path=str(self.path),
            use_onset_classifier=True, source="rosbag")
        rows = pd.read_csv(report["csv"])
        np.testing.assert_allclose(rows.failure_probability, [detector.predict_failure_probability(vector) for vector in vectors])
        self.assertTrue((rows.threshold_source == "timestamp_supervised_prefix_classifier").all())
        self.assertEqual(report["first_alert"]["first_above_sec"], 5.)
        self.assertEqual(model.extract_multiturn_cop_vector.call_count, 2)
        self.assertEqual(model.extract_multiturn_cop_vector.call_args_list[0].args[0][1]["images"], ["a"])
        self.assertEqual(model.extract_multiturn_cop_vector.call_args_list[1].args[0][1]["images"], ["a", "b"])
        self.read_annotation.assert_not_called()

    def test_onset_missing_never_silently_falls_back_to_complete_classifier(self):
        with self.assertRaisesRegex(ValueError, "missing"):
            load_onset_classifier(self.path, self.complete, bag_name="test")

    def test_configuration_is_explicit_optional_and_requires_rosbag_raw(self):
        config = {"stage_topic": "/stages", "rynnbrain": {"source": "rosbag", "cop_classifier": {
            "method": "logistic", "logistic_onset": {"enabled": True}}}}
        self.assertEqual(onset_training_options(config, "raw")["topic"], "/stages")
        with self.assertRaisesRegex(ValueError, "rosbag"):
            onset_training_options(config, "heatmap")
        config["rynnbrain"]["source"] = "generated_videos"
        with self.assertRaisesRegex(ValueError, "rosbag"):
            onset_training_options(config, "raw")
        config["rynnbrain"]["cop_classifier"]["method"] = "knn"
        self.assertIsNone(onset_training_options(config, "raw"))
        self.assertIsNone(onset_training_options({}, "raw"))


if __name__ == "__main__":
    unittest.main()
