from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from rynnbrain_vlm.cop_analysis import (
    analyze_saved_vectors,
    discover_saved_vectors,
    pca_2d,
    save_cop_vector,
    write_failure_probability_report,
)


class CoPAnalysisTests(unittest.TestCase):
    def test_probability_statistics_use_ground_truth_percentages_and_separate_modes(self):
        rows = [
            {"input_mode": "raw", "ground_truth_label": "normal", "decision": "failure", "classifier_failure_probability": value}
            for value in (0.1, 0.2, 0.3, 0.4)
        ] + [
            {"input_mode": "raw", "ground_truth_label": "fail", "decision": "success", "classifier_failure_probability": 0.9},
            {"input_mode": "heatmap", "ground_truth_label": "normal", "classifier_failure_probability": 0.8},
            {"input_mode": "raw", "ground_truth_label": "unknown", "classifier_failure_probability": 0.5},
            {"input_mode": "raw", "ground_truth_label": "fail", "classifier_failure_probability": None},
            {"input_mode": "raw", "ground_truth_label": "fail", "classifier_failure_probability": float("nan")},
            {"input_mode": "raw", "ground_truth_label": "fail", "classifier_failure_probability": 1.2},
        ]
        with tempfile.TemporaryDirectory() as directory:
            report = write_failure_probability_report(rows, Path(directory))
            modes = {mode["input_mode"]: mode for mode in report["modes"]}
            raw = modes["raw"]
            normal, failure = raw["groups"]
            self.assertEqual(raw["scored_rows"], 5)
            self.assertEqual(normal["count"], 4)
            self.assertEqual(normal["mean_percent"], 25.0)
            self.assertEqual(normal["median_percent"], 25.0)
            self.assertEqual(normal["q1_percent"], 17.5)
            self.assertEqual(normal["q3_percent"], 32.5)
            self.assertEqual(failure["median_percent"], 90.0)
            self.assertEqual(raw["excluded_rows"], {"unknown_label": 1, "missing_probability": 1, "invalid_probability": 2})
            self.assertEqual(modes["heatmap"]["groups"][0]["median_percent"], 80.0)
            self.assertEqual(modes["heatmap"]["groups"][1]["count"], 0)
            self.assertIsNone(modes["heatmap"]["groups"][1]["median_percent"])
            for mode in modes.values():
                self.assertTrue(Path(mode["plot_file"]).is_file())
            saved = pd.read_csv(report["summary_csv"])
            self.assertEqual(len(saved), 4)
            json.dumps(report, allow_nan=False)

    def test_empty_probability_report_removes_stale_plot_without_inventing_zero_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stale = root / "benchmark_failure_probability_raw.png"
            stale.write_bytes(b"old plot")
            report = write_failure_probability_report([], root, input_modes=["raw"])
            self.assertFalse(stale.exists())
            mode = report["modes"][0]
            self.assertEqual(mode["status"], "skipped")
            self.assertIsNone(mode["plot_file"])
            self.assertTrue(all(group["count"] == 0 and group["median_percent"] is None for group in mode["groups"]))

    def _save(
        self,
        root: Path,
        bag_name: str,
        label: str,
        vector: list[float],
        mode: str = "raw",
        signature: dict | None = None,
    ) -> None:
        save_cop_vector(
            root / bag_name / "rynnbrain_multiturn",
            mode,
            vector,
            {
                "representation_id": "test_representation",
                "bag_name": bag_name,
                "test_bag": f"/data/{bag_name}",
                "ground_truth_label": label,
                "comparison_signature": signature or {
                    "model_id": "test-model",
                    "input_mode": mode,
                },
            },
        )

    def test_saved_vectors_round_trip_and_create_labeled_pca(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self._save(root, "normal_01", "normal", [1.0, 0.1, 0.0, 0.0])
            self._save(root, "normal_02", "nominal", [0.9, 0.2, 0.0, 0.0])
            self._save(root, "failure_01", "fail", [-1.0, -0.1, 0.1, 0.0])
            self._save(root, "failure_02", "failure", [-0.8, -0.2, 0.0, 0.1])

            saved = discover_saved_vectors(root)
            self.assertEqual(len(saved), 4)
            self.assertTrue(all(item.vector.dtype == np.float32 for item in saved))

            summary = analyze_saved_vectors(root)
            mode_summary = summary["modes"][0]
            self.assertEqual(mode_summary["status"], "created")
            self.assertEqual(mode_summary["normal_count"], 2)
            self.assertEqual(mode_summary["failure_count"], 2)

            coordinates_path = Path(mode_summary["coordinates_csv"])
            plot_path = Path(mode_summary["plot_file"])
            archive_path = Path(mode_summary["vectors_archive"])
            self.assertTrue(coordinates_path.is_file())
            self.assertGreater(plot_path.stat().st_size, 0)
            self.assertTrue(archive_path.is_file())

            coordinates = pd.read_csv(coordinates_path)
            self.assertEqual(set(coordinates["label"]), {"normal", "fail"})
            self.assertEqual(len(coordinates), 4)
            with np.load(archive_path, allow_pickle=False) as archive:
                self.assertEqual(archive["vectors"].shape, (4, 4))
                self.assertEqual(set(archive["labels"].tolist()), {"normal", "fail"})
                self.assertEqual(archive["l2_norms"].shape, (4,))
            with np.load(mode_summary["pca_model_file"], allow_pickle=False) as model:
                self.assertEqual(model["components"].shape, (2, 4))
                self.assertEqual(model["feature_mean"].shape, (4,))

            normal_metadata = [
                item.metadata_path
                for item in saved
                if item.metadata["ground_truth_label"] in {"normal", "nominal"}
            ]
            skipped = analyze_saved_vectors(root, metadata_paths=normal_metadata)
            self.assertEqual(skipped["selection"], "explicit_vectors")
            self.assertEqual(skipped["modes"][0]["status"], "skipped")
            self.assertFalse(plot_path.exists())

    def test_analysis_skips_mode_without_both_classes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self._save(root, "normal_01", "normal", [1.0, 0.0, 0.0])
            self._save(root, "normal_02", "normal", [0.9, 0.1, 0.0])

            summary = analyze_saved_vectors(root)
            mode_summary = summary["modes"][0]
            self.assertEqual(mode_summary["status"], "skipped")
            self.assertIn("missing ground-truth", mode_summary["reason"])
            self.assertTrue(Path(summary["summary_file"]).is_file())

    def test_analysis_refuses_to_mix_different_settings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self._save(
                root,
                "normal_01",
                "normal",
                [1.0, 0.0, 0.0],
                signature={"model_id": "model-a", "input_mode": "raw"},
            )
            self._save(
                root,
                "failure_01",
                "fail",
                [-1.0, 0.0, 0.0],
                signature={"model_id": "model-b", "input_mode": "raw"},
            )

            summary = analyze_saved_vectors(root)
            mode_summary = summary["modes"][0]
            self.assertEqual(mode_summary["status"], "skipped")
            self.assertIn("different comparison settings", mode_summary["reason"])

    def test_pca_is_deterministic_and_rejects_identical_vectors(self) -> None:
        matrix = np.asarray(
            [[1.0, 0.0, 0.1], [0.8, 0.2, 0.0], [-1.0, 0.0, -0.1]],
            dtype=np.float32,
        )
        first_coordinates, first_components, first_ratio, first_mean = pca_2d(matrix)
        second_coordinates, second_components, second_ratio, second_mean = pca_2d(matrix)
        np.testing.assert_allclose(first_coordinates, second_coordinates)
        np.testing.assert_allclose(first_components, second_components)
        np.testing.assert_allclose(first_ratio, second_ratio)
        np.testing.assert_allclose(first_mean, second_mean)

        with self.assertRaisesRegex(ValueError, "identical vectors"):
            pca_2d(np.ones((3, 4), dtype=np.float32))

    def test_analysis_skips_identical_vectors_and_writes_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self._save(root, "normal_01", "normal", [1.0, 1.0, 1.0])
            self._save(root, "failure_01", "fail", [1.0, 1.0, 1.0])

            summary = analyze_saved_vectors(root)

            self.assertEqual(summary["modes"][0]["status"], "skipped")
            self.assertIn("identical vectors", summary["modes"][0]["reason"])
            self.assertTrue(Path(summary["summary_file"]).is_file())

    def test_metadata_is_plain_json_and_vector_is_not_embedded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            vector_path, metadata_path = save_cop_vector(
                root,
                "raw heatmap",
                [1.0, 2.0, 3.0],
                {
                    "ground_truth_label": "normal",
                    "comparison_signature": {"input_mode": "raw heatmap"},
                },
            )
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["vector_file"], vector_path.name)
            self.assertEqual(metadata["vector_dimension"], 3)
            self.assertNotIn("vector", metadata)


if __name__ == "__main__":
    unittest.main()
