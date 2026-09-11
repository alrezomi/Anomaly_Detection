from __future__ import annotations

from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from rynnbrain_vlm.cop_analysis import save_cop_vector
from rynnbrain_vlm.cop_classifier import (
    fit_logistic_classifier,
    load_classifier,
    parse_arguments,
    train_from_saved_vectors,
    _resolved_training_settings,
)


class CoPClassifierTests(unittest.TestCase):
    signature = {
        "model_id": "test-model",
        "input_mode": "raw",
        "reference_bags": ["reference_01"],
    }

    def _save_training_vector(
        self, root: Path, bag_name: str, label: str, vector: list[float]
    ) -> None:
        save_cop_vector(
            root / bag_name / "rynnbrain_multiturn",
            "raw",
            vector,
            {
                "representation_id": "test-representation",
                "bag_name": bag_name,
                "test_bag": f"/data/{bag_name}",
                "ground_truth_label": label,
                "comparison_signature": self.signature,
            },
        )

    def test_fit_uses_full_feature_width_and_separates_classes(self) -> None:
        vectors = np.asarray(
            [
                [1.0, 0.2, 0.0, 0.1],
                [0.9, 0.1, 0.1, 0.0],
                [-1.0, -0.2, 0.0, -0.1],
                [-0.9, -0.1, -0.1, 0.0],
            ],
            dtype=np.float32,
        )
        weights, intercept, mean, diagnostics = fit_logistic_classifier(
            vectors, [0, 0, 1, 1], c_value=10.0
        )

        self.assertEqual(weights.shape, (4,))
        self.assertEqual(mean.shape, (4,))
        normalized = vectors / np.linalg.norm(vectors, axis=1)[:, None]
        probabilities = 1.0 / (1.0 + np.exp(-((normalized - mean) @ weights + intercept)))
        self.assertLess(float(probabilities[:2].max()), 0.5)
        self.assertGreater(float(probabilities[2:].min()), 0.5)
        self.assertGreater(diagnostics["effective_rank"], 0)
        self.assertEqual(diagnostics["class_weight"], "balanced")

    def test_train_save_load_and_validate_signature(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self._save_training_vector(root, "normal_01", "normal", [1.0, 0.2, 0.0, 0.1])
            self._save_training_vector(root, "normal_02", "normal", [0.9, 0.1, 0.1, 0.0])
            self._save_training_vector(root, "failure_01", "fail", [-1.0, -0.2, 0.0, -0.1])
            self._save_training_vector(root, "failure_02", "fail", [-0.9, -0.1, -0.1, 0.0])
            model_path = root / "classifier" / "raw_logistic.npz"

            metadata = train_from_saved_vectors(
                root, model_path, input_mode="raw", c_value=10.0
            )
            classifier = load_classifier(model_path)
            normal_probability = classifier.predict_failure_probability(
                [1.0, 0.15, 0.0, 0.05],
                input_mode="raw",
                comparison_signature=self.signature,
            )
            failure_probability = classifier.predict_failure_probability(
                [-1.0, -0.15, 0.0, -0.05],
                input_mode="raw",
                comparison_signature=self.signature,
            )

            self.assertLess(normal_probability, failure_probability)
            self.assertEqual(metadata["training_class_counts"], {"normal": 2, "fail": 2})
            self.assertEqual(metadata["class_weight"], "balanced")
            self.assertTrue(model_path.with_suffix(".json").is_file())
            self.assertTrue(
                model_path.with_name("raw_logistic_training_predictions.csv").is_file()
            )
            with self.assertRaisesRegex(ValueError, "different RynnBrain"):
                classifier.predict_failure_probability(
                    [1.0, 0.0, 0.0, 0.0],
                    comparison_signature={"model_id": "different"},
                )

    def test_training_rejects_too_few_examples_per_class(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self._save_training_vector(root, "normal_01", "normal", [1.0, 0.0, 0.0])
            self._save_training_vector(root, "failure_01", "fail", [-1.0, 0.0, 0.0])

            with self.assertRaisesRegex(ValueError, "at least two normal"):
                train_from_saved_vectors(root, root / "classifier.npz")

    def test_configured_training_labels_override_vector_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self._save_training_vector(root, "normal_01", "fail", [1.0, 0.2, 0.0])
            self._save_training_vector(root, "normal_02", "fail", [0.9, 0.1, 0.0])
            self._save_training_vector(root, "failure_01", "normal", [-1.0, -0.2, 0.0])
            self._save_training_vector(root, "failure_02", "normal", [-0.9, -0.1, 0.0])
            labels = {
                "normal_01": "normal",
                "normal_02": "normal",
                "failure_01": "fail",
                "failure_02": "fail",
            }

            metadata = train_from_saved_vectors(
                root,
                root / "classifier.npz",
                bag_names=list(labels),
                label_overrides=labels,
            )

            self.assertEqual(metadata["training_labels"], labels)

    def test_balanced_weighting_equalizes_uneven_class_totals(self) -> None:
        vectors = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.9, 0.1, 0.0],
                [0.8, 0.2, 0.0],
                [0.7, 0.3, 0.0],
                [-1.0, 0.0, 0.0],
                [-0.9, -0.1, 0.0],
            ]
        )
        _, _, _, diagnostics = fit_logistic_classifier(
            vectors, [0, 0, 0, 0, 1, 1], class_weight="balanced"
        )

        weights = diagnostics["class_weights"]
        self.assertAlmostEqual(weights["normal"] * 4, weights["fail"] * 2)

    def test_training_settings_are_loaded_from_config(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config_path = root / "pipeline_config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "rynnbrain": {
                            "cop_classifier": {
                                "model_paths": {"raw": "/outputs/classifier.npz"},
                                "training": {
                                    "input_dir": "/outputs/training_vectors",
                                    "input_mode": "raw",
                                    "normal_bags": ["normal_01"],
                                    "failure_bags": ["failure_01"],
                                    "class_weight": "balanced",
                                    "regularization_c": 2.0,
                                    "threshold": 0.4,
                                },
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            with patch.object(sys, "argv", ["cop_classifier", "--config", str(config_path)]):
                settings = _resolved_training_settings(parse_arguments())

            self.assertEqual(settings["bag_names"], ["normal_01", "failure_01"])
            self.assertEqual(
                settings["label_overrides"],
                {"normal_01": "normal", "failure_01": "fail"},
            )
            self.assertEqual(settings["class_weight"], "balanced")
            self.assertEqual(settings["c_value"], 2.0)
            self.assertEqual(settings["threshold"], 0.4)
            self.assertEqual(settings["output_file"], Path("/outputs/classifier.npz"))

    def test_config_refuses_an_accidental_all_bags_training_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = Path(temporary_directory) / "pipeline_config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "rynnbrain": {
                            "cop_classifier": {
                                "model_paths": {"raw": "/outputs/classifier.npz"},
                                "training": {
                                    "input_dir": "/outputs/training_vectors",
                                    "normal_bags": [],
                                    "failure_bags": [],
                                },
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            with patch.object(sys, "argv", ["cop_classifier", "--config", str(config_path)]):
                arguments = parse_arguments()

            with self.assertRaisesRegex(ValueError, "No classifier training bags"):
                _resolved_training_settings(arguments)


if __name__ == "__main__":
    unittest.main()
