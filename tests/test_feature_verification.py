import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from rynnbrain_vlm.feature_verification import (
    compare_populations, run_verification, saved_baseline, score_comparison,
    signature_changes, vector_difference,
)
from rynnbrain_vlm.cop_classifier import CoPLogisticClassifier


class FeatureVerificationTests(unittest.TestCase):
    def classifier(self):
        return CoPLogisticClassifier(np.array([1., -1.]), 0., np.zeros(2), .5,
                                     "raw", "test", {"model_id": "model"}, {})

    def test_vector_and_probability_checks_detect_real_difference(self):
        vectors = {"generation": np.array([2., 1.]), "direct_before": np.array([1., 2.]),
                   "direct_after": np.array([2., 1.])}
        self.assertFalse(vector_difference(vectors["generation"], vectors["direct_before"])["within_tolerance"])
        comparison = score_comparison(self.classifier(), vectors, "raw", {"model_id": "model"})
        self.assertFalse(comparison["within_tolerance"])
        self.assertEqual(comparison["decisions"]["generation"], "failure")
        self.assertEqual(comparison["decisions"]["direct_before"], "success")
        with self.assertRaises(ValueError):
            score_comparison(self.classifier(), vectors, "raw", {"model_id": "other"})
        for bad in ([0., 0.], [float("nan"), 1.], [1.]):
            with self.assertRaises(ValueError):
                vector_difference([2., 1.], bad)

    def test_near_threshold_decision_change_is_not_called_a_match(self):
        vectors = {"generation": [1., 1.000001], "direct_before": [1.000001, 1.]}
        self.assertTrue(vector_difference(*vectors.values())["within_tolerance"])
        self.assertFalse(score_comparison(self.classifier(), vectors, "raw", {"model_id": "model"})["within_tolerance"])

    def test_knn_compares_distances_not_probabilities(self):
        from rynnbrain_vlm.cop_knn import CoPKNNDetector
        classifier = Mock(spec=CoPKNNDetector, threshold=.2)
        classifier.predict_anomaly_score.return_value = .1
        classifier.predict_label.return_value = "success"
        result = score_comparison(classifier, {"generation": [1., 2.], "direct_after": [1., 2.]}, "raw", {})
        self.assertEqual(result["score_kind"], "knn_distance")
        self.assertTrue(result["within_tolerance"])

    def test_baseline_reports_changed_adapter_and_population_without_modifying_files(self):
        old = {"model_config": {"lora_adapter_path": "v116"}, "lora_adapter_sha256": "old"}
        new = {"model_config": {"lora_adapter_path": "v120"}, "lora_adapter_sha256": "new"}
        self.assertEqual(set(signature_changes(old, new)), {"model_config.lora_adapter_path", "lora_adapter_sha256"})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a, b = root / "old", root / "new"
            location = a / "bag" / "rynnbrain_multiturn" / "cop_vectors"
            location.mkdir(parents=True)
            b.mkdir()
            metadata = location / "raw.json"
            metadata.write_text(json.dumps({"comparison_signature": old, "vector_file": "raw.npy"}))
            np.save(location / "raw.npy", [1., 2.])
            before = metadata.read_bytes()
            result = saved_baseline(a, "bag", "raw", new, np.array([1., 2.]))
            self.assertFalse(result["same_representation_settings"])
            self.assertEqual(metadata.read_bytes(), before)
            (a / "benchmark_summary.csv").write_text("bag_name\nbag\nold_bag\n")
            (b / "benchmark_summary.csv").write_text("bag_name\nbag\nbag\nnew_bag\n")
            population = compare_populations(a, b)
            self.assertEqual(population["baseline_bags"], 2)
            self.assertEqual(population["added_bags"], ["new_bag"])
            self.assertEqual(population["removed_bags"], ["old_bag"])

    def test_verification_uses_same_context_and_preserves_existing_outputs(self):
        import torch
        vector = torch.tensor([2., 1.])
        model = SimpleNamespace(adapter_training_bags=set(), temporal_profile={},
            generate_nominal=Mock(return_value="Reference description"),
            extract_multiturn_cop_vector=Mock(return_value=vector),
            generate_multiturn_with_cop_vector=Mock(return_value=SimpleNamespace(
                cop_vector=vector, evaluation_response="Decision: failure")))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            classifier_file = root / "classifier.npz"
            classifier_file.write_bytes(b"existing classifier")
            preserved = root / "rynnbrain_responses_multiturn.json"
            preserved.write_text("existing results")
            vlm = {"model": {}, "output_dir": str(root), "input_modes": ["raw"], "reference_bags": ["reference"],
                   "camera_topics": ["camera"], "cop_classifier": {"enabled": True,
                   "method": "logistic", "model_paths": {"raw": str(classifier_file)}}}
            config = {"test_bag": "/data/bag", "output_dir": str(root), "camera_topics": ["camera"], "rynnbrain": vlm}
            before = copy.deepcopy(config)
            with patch("rynnbrain_vlm.model.RynnBrainModel", return_value=model), \
                 patch("rynnbrain_vlm.cop_classifier.load_classifier", return_value=self.classifier()), \
                 patch("rynnbrain_vlm.run._execution_inputs", return_value=({"raw": ([("test", None)], [])}, [])), \
                 patch("rynnbrain_vlm.run._raw_inputs", return_value=([("reference", None)], [])), \
                 patch("rynnbrain_vlm.run.cop_comparison_signature", return_value={"model_id": "model"}):
                run_verification(config, vlm, 8, {"do_sample": False})
            report = json.loads(next(root.glob("feature_verification/bag/*/comparison.json")).read_text())
            self.assertEqual(report["results"][0]["status"], "match")
            self.assertEqual(preserved.read_text(), "existing results")
            self.assertEqual(classifier_file.read_bytes(), b"existing classifier")
            self.assertEqual(config, before)
            self.assertEqual(len(list(root.glob("feature_verification/bag/*/*.npy"))), 3)
            model.generate_nominal.assert_called_once()
            self.assertEqual(model.extract_multiturn_cop_vector.call_count, 2)
            generated = model.generate_multiturn_with_cop_vector.call_args
            for direct in model.extract_multiturn_cop_vector.call_args_list:
                self.assertEqual(direct, generated)

    def test_disabled_timeline_has_explicit_timing_status(self):
        from rynnbrain_vlm.failure_timing import evaluate_failure_timing
        result = evaluate_failure_timing({"time_sec": 12.}, {"status": "disabled"},
                                        source="rosbag", mode="raw", ground_truth="fail", method="logistic")
        self.assertEqual(result["failure_timing_status"], "timeline_disabled")
        self.assertIn("plot_timeline", result["failure_timing_error"])
        self.assertIsNone(result["failure_time_error_sec"])


if __name__ == "__main__":
    unittest.main()
