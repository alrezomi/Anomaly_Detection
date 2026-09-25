from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from rynnbrain_vlm.cop_analysis import save_cop_vector, write_anomaly_score_report
from rynnbrain_vlm.benchmark_roc import write_anomaly_roc_report
from rynnbrain_vlm.cop_classifier import (
    classifier_model_paths, configured_training_settings, load_classifier, train_from_saved_vectors,
)


class NominalKNNTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.signature = {"model_id": "tiny", "lora_adapter_sha256": "adapter-1"}
        self.vectors = np.array([[1, offset, 0, 0] for offset in (0.0, 0.1, 0.2, 0.4)])
        self.names = [f"normal_{i}" for i in range(4)]
        for name, vector in zip(self.names, self.vectors):
            self.save(name, "normal", vector)
        self.save("failure", "fail", [-1, 0, 0, 0])
        self.path = self.root / "raw_knn.npz"

    def save(self, name, label, vector):
        return save_cop_vector(self.root / name, "raw", vector,
                               {"bag_name": name, "test_bag": f"/data/{name}", "ground_truth_label": label,
                                "representation_id": "test", "comparison_signature": self.signature})

    def train(self, names=None, **options):
        return train_from_saved_vectors(self.root, self.path, method="knn", bag_names=self.names if names is None else names,
                                         knn_options={"n_neighbors": 2, "threshold_quantile": 0.95, **options})

    def test_full_vector_distance_and_leave_one_out_cutoff(self):
        metadata = self.train()
        detector = load_classifier(self.path)
        memory = self.vectors / np.linalg.norm(self.vectors, axis=1)[:, None]
        scores = []
        for index, vector in enumerate(memory):
            distances = sorted(np.linalg.norm(vector - other) for j, other in enumerate(memory) if index != j)
            scores.append(np.mean(distances[:2]))
        saved = pd.read_csv(self.path.with_name("raw_knn_training_predictions.csv"))
        np.testing.assert_allclose(saved.leave_one_out_anomaly_score, scores)
        self.assertGreater(min(scores), 0)
        self.assertEqual(metadata["training_class_counts"], {"normal": 4, "fail": 0})
        self.assertNotIn("failure", metadata["training_bags"])
        expected = np.nextafter(np.quantile(scores, 0.95, method="higher"), np.inf)
        self.assertAlmostEqual(detector.threshold, expected)
        query = np.array([1, 0.2, 0, 1.0])
        expected_distance = np.mean(sorted(np.linalg.norm(memory - query / np.linalg.norm(query), axis=1))[:2])
        self.assertAlmostEqual(detector.predict_anomaly_score(query), expected_distance)
        self.assertAlmostEqual(detector.predict_anomaly_score(query * 10), expected_distance)
        self.assertEqual(detector.predict_label(detector.predict_anomaly_score([-1, 0, 0, 0])), "failure")
        self.assertEqual(detector.predict_label(detector.predict_anomaly_score([1, 0.1, 0, 0])), "success")
        with self.assertRaisesRegex(ValueError, "representation"):
            detector.predict_anomaly_score(query, comparison_signature={})
        with self.assertRaises(ValueError):
            detector.predict_anomaly_score(query, input_mode="heatmap")
        for vector in ([0, 0, 0, 0], [1, 0], [np.nan, 0, 0, 0]):
            with self.assertRaises(ValueError):
                detector.predict_anomaly_score(vector)

    def test_failure_training_and_invalid_k_or_quantile_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "cannot use failure"):
            self.train(self.names + ["failure"])
        for k in (0, 4, 1.5, True, None, float("nan")):
            with self.subTest(k=k), self.assertRaises(ValueError):
                self.train(n_neighbors=k)
        for quantile in (0, 1, float("nan"), None):
            with self.subTest(quantile=quantile), self.assertRaises(ValueError):
                self.train(threshold_quantile=quantile)
        with self.assertRaisesRegex(ValueError, "explicit normal"):
            train_from_saved_vectors(self.root, self.path, method="knn")

    def test_refitting_refreshes_loaded_model_and_corrupt_memory_is_rejected(self):
        self.train()
        original = load_classifier(self.path)
        self.train(n_neighbors=1)
        self.assertEqual(original.n_neighbors, 2)
        refreshed = load_classifier(self.path)
        self.assertEqual(refreshed.n_neighbors, 1)
        np.savez_compressed(self.path, nominal_vectors=refreshed.nominal_vectors * 2,
                            n_neighbors=1, threshold=refreshed.threshold)
        load_classifier.cache_clear()
        with self.assertRaisesRegex(ValueError, "L2-normalized"):
            load_classifier(self.path)

    def test_identical_nominals_do_not_flag_exact_matches_and_failure_data_has_no_effect(self):
        for name in self.names:
            self.save(name, "normal", [1, 0, 0, 0])
        self.train()
        detector = load_classifier(self.path)
        self.assertGreater(detector.threshold, 0)
        self.assertEqual(detector.predict_label(detector.predict_anomaly_score([2, 0, 0, 0])), "success")
        self.save("failure", "fail", [0, 1, 0, 0])
        self.train()
        # The learned memory and cutoff must not depend on available failures.
        restored = load_classifier(self.path)
        np.testing.assert_array_equal(restored.nominal_vectors, detector.nominal_vectors)
        self.assertEqual(restored.threshold, detector.threshold)

    def test_configuration_keeps_logistic_path_and_uses_only_normal_selection(self):
        config = {"model_paths": {"raw": str(self.root / "raw_logistic.npz")}, "method": "knn",
                  "knn": {"n_neighbors": 2}, "training": {"normal_bags": self.names,
                  "failure_bags": ["missing_failure_bag"], "input_dir": str(self.root)}}
        path = self.root / "config.json"
        path.write_text(json.dumps({"rynnbrain": {"cop_classifier": config}}))
        settings = configured_training_settings(path)
        self.assertEqual(settings["bag_names"], self.names)
        self.assertEqual(settings["output_file"], self.path)
        self.assertEqual(set(settings["label_overrides"].values()), {"normal"})
        self.assertEqual(classifier_model_paths({**config, "method": "logistic"})["raw"], config["model_paths"]["raw"])
        sentinel = self.root / "raw_logistic.npz"
        sentinel.write_bytes(b"existing logistic model")
        train_from_saved_vectors(**settings)
        self.assertEqual(sentinel.read_bytes(), b"existing logistic model")
        with self.assertRaisesRegex(ValueError, "overwrite"):
            train_from_saved_vectors(self.root, sentinel, method="knn", bag_names=self.names)

    def test_distance_reports_do_not_clip_scores_above_one_or_report_probabilities(self):
        rows = [{"input_mode": "raw", "ground_truth_label": label,
                 "bag_path": "/data/Failure_1_grasp_miss/" + str(i),
                 "classifier_anomaly_score": score, "classifier_failure_probability": None,
                 "classifier_threshold": 1.2}
                for i, (label, score) in enumerate([("normal", 0.1), ("normal", 0.2), ("fail", 1.1), ("fail", 1.8)])]
        roc = write_anomaly_roc_report(rows, self.root)
        metric = roc["metrics"][0]
        self.assertEqual(roc["score_column"], "classifier_anomaly_score")
        self.assertEqual(metric["auroc"], 1.0)
        self.assertEqual(metric["recall_at_threshold"], 0.5)
        self.assertEqual(metric["invalid_anomaly_score_count"], 0)
        box = write_anomaly_score_report(rows, self.root)
        self.assertEqual(box["units"], "distance")
        self.assertAlmostEqual(box["modes"][0]["groups"][1]["median_score"], 1.45)
        self.assertTrue(Path(box["modes"][0]["plot_file"]).is_file())
        self.assertTrue(all("benchmark_anomaly_roc" in path for path in roc["plot_files"]))


if __name__ == "__main__":
    unittest.main()
