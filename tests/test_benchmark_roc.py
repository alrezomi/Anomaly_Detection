from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from rynnbrain_vlm.benchmark_roc import failure_category, roc_curve, write_roc_report


class BenchmarkRocTests(unittest.TestCase):
    def _row(self, name, label, decision, category="Failure_1_grasp_miss", mode="raw"):
        folder = "Nominal/setting 1" if label == "normal" else category
        return {
            "bag_name": name,
            "bag_path": f"/data/{folder}/{name}",
            "ground_truth_label": label,
            "input_mode": mode,
            "decision": decision,
        }

    def test_categories_preserve_folder_names_and_support_both_path_styles(self):
        for path in (
            "/data/Failure_2_slip_at_start/failure_2.1",
            r"D:\data\Failure_2_slip_at_start\setting 1\failure_2.1",
        ):
            self.assertEqual(failure_category(path, "fail"), "Failure_2_slip_at_start")
            self.assertIsNone(failure_category(path, "normal"))
            self.assertIsNone(failure_category(path, "unknown"))
        self.assertEqual(failure_category("/data/bag_001", "fail"), "uncategorized_failure")

    def test_perfect_reversed_and_tied_scores(self):
        targets = np.array([0, 0, 1, 1])
        for scores, expected in (
            ([0.1, 0.2, 0.8, 0.9], 1.0),
            ([0.9, 0.8, 0.2, 0.1], 0.0),
            ([0.5, 0.5, 0.5, 0.5], 0.5),
            ([0.1, 0.5, 0.5, 0.9], 0.875),
        ):
            with self.subTest(scores=scores):
                fpr, tpr, thresholds, auroc = roc_curve(targets, np.array(scores))
                self.assertAlmostEqual(auroc, expected)
                self.assertEqual((fpr[0], tpr[0]), (0.0, 0.0))
                self.assertEqual((fpr[-1], tpr[-1]), (1.0, 1.0))
                self.assertTrue(np.isinf(thresholds[0]))
                self.assertTrue(np.all(np.diff(fpr) >= 0))
                self.assertTrue(np.all(np.diff(tpr) >= 0))

    def test_ties_are_independent_of_bag_order(self):
        targets = np.array([0, 1, 0, 1])
        scores = np.array([0.2, 0.8, 0.8, 0.2])
        first = roc_curve(targets, scores)
        second = roc_curve(targets[::-1], scores[::-1])
        for before, after in zip(first, second):
            np.testing.assert_array_equal(before, after)

    def test_invalid_curve_inputs_are_rejected(self):
        for targets, scores in (([], []), ([0], [0.2]), ([1], [0.8]),
                                ([0, 1], [0.1]), ([0, 2], [0.1, 0.8]),
                                ([0, 1], [np.nan, 0.8])):
            with self.subTest(targets=targets, scores=scores), self.assertRaises(ValueError):
                roc_curve(np.array(targets), np.array(scores))

    def test_categories_use_normal_controls_and_modes_stay_separate(self):
        rows = [
            self._row("normal_1", "normal", "success"),
            self._row("failure_1", "fail", "failure"),
            self._row("failure_2", "fail", "success", "Failure_2_slip_at_start"),
            self._row("normal_1", "normal", "failure", mode="raw_heatmap"),
            self._row("failure_1", "fail", "success", mode="raw_heatmap"),
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            report = write_roc_report(rows, root)
            metrics = {(item["input_mode"], item["failure_category"]): item for item in report["metrics"]}
            self.assertEqual(metrics["raw", "all_failures"]["auroc"], 0.75)
            self.assertEqual(metrics["raw", "Failure_1_grasp_miss"]["auroc"], 1.0)
            self.assertEqual(metrics["raw", "Failure_2_slip_at_start"]["auroc"], 0.5)
            self.assertEqual(metrics["raw_heatmap", "Failure_1_grasp_miss"]["auroc"], 0.0)
            self.assertEqual(metrics["raw", "Failure_1_grasp_miss"]["failure_count"], 1)
            self.assertEqual(metrics["raw", "Failure_2_slip_at_start"]["normal_count"], 1)
            for metric in metrics.values():
                self.assertEqual(metric["auroc"], metric["balanced_accuracy_decided"])
            self.assertEqual(len(report["plot_files"]), 2)
            for filename in report["plot_files"]:
                self.assertGreater(Path(filename).stat().st_size, 0)
            saved = pd.read_csv(root / "benchmark_roc" / "auroc.csv")
            self.assertEqual(len(saved), 5)
            points = pd.read_csv(root / "benchmark_roc" / "roc_points.csv")
            self.assertEqual(set(points["input_mode"]), {"raw", "raw_heatmap"})
            self.assertEqual(json.loads((root / "benchmark_roc" / "roc_summary.json").read_text()), report)

    def test_abstentions_unknown_labels_and_failed_runs_are_counted(self):
        rows = [self._row("normal", "normal", "success"), self._row("failure", "fail", "failure")]
        for index, decision in enumerate([None, np.nan, "uncertain", "not_parsed", "failed", "not_run"]):
            rows.append(self._row(f"bad_{index}", "fail", decision))
        rows.append(self._row("unknown", "unknown", "failure"))
        rows.append(self._row("failed_run", "fail", None, mode=None))
        with tempfile.TemporaryDirectory() as temporary_directory:
            report = write_roc_report(rows, Path(temporary_directory))
            metric = report["metrics"][0]
            self.assertEqual(metric["auroc"], 1.0)
            self.assertEqual(metric["excluded_decision_count"], 6)
            self.assertEqual(metric["failure_abstention_count"], 6)
            self.assertEqual(metric["normal_abstention_count"], 0)
            self.assertEqual(metric["uncertain_count"], 1)
            self.assertEqual(metric["unparsed_count"], 1)
            self.assertEqual(metric["other_excluded_decision_count"], 4)
            self.assertEqual(metric["accuracy"], 0.25)
            self.assertEqual(metric["decision_coverage"], 0.25)
            self.assertEqual(metric["unknown_label_count"], 1)
            self.assertEqual(report["rows_without_input_mode"], 1)

    def test_unavailable_decisions_or_classes_are_skipped(self):
        for rows in (
            [self._row("failure", "fail", "failure")],
            [self._row("normal", "normal", "success")],
            [self._row("normal", "normal", None), self._row("failure", "fail", None)],
            [],
        ):
            with self.subTest(rows=rows), tempfile.TemporaryDirectory() as temporary_directory:
                report = write_roc_report(rows, Path(temporary_directory))
                self.assertEqual(report["plot_files"], [])
                for metric in report["metrics"]:
                    self.assertIsNone(metric["auroc"])
                    self.assertEqual(metric["status"], "skipped")
                    self.assertTrue(metric["reason"])
                points = pd.read_csv(Path(temporary_directory) / "benchmark_roc" / "roc_points.csv")
                self.assertTrue(points.empty)

    def test_confusion_counts_and_rates_use_vlm_decisions_not_classifier(self):
        # Deliberately disagreeing classifier probabilities must have no effect.
        rows = [
            self._row("normal_1", "normal", "success"),
            self._row("normal_2", "normal", "failure"),
            self._row("normal_3", "normal", "uncertain"),
            self._row("failure_1", "fail", "failure"),
            self._row("failure_2", "fail", "failure"),
            self._row("failure_3", "fail", "success"),
        ]
        for row in rows:
            row["classifier_failure_probability"] = 1.0 if row["decision"] == "success" else 0.0
        with tempfile.TemporaryDirectory() as temporary_directory:
            report = write_roc_report(rows, Path(temporary_directory))
            metric = report["metrics"][0]
            self.assertEqual(report["score_column"], "decision")
            self.assertEqual(metric["true_positive"], 2)
            self.assertEqual(metric["false_positive"], 1)
            self.assertEqual(metric["true_negative"], 1)
            self.assertEqual(metric["false_negative"], 1)
            self.assertEqual(metric["accuracy"], 0.5)
            self.assertAlmostEqual(metric["decision_coverage"], 5 / 6)
            self.assertAlmostEqual(metric["failure_recall_decided"], 2 / 3)
            self.assertEqual(metric["normal_specificity_decided"], 0.5)
            self.assertEqual(metric["false_positive_rate_decided"], 0.5)
            self.assertAlmostEqual(metric["auroc"], (2 / 3 + 0.5) / 2)

    def test_constant_vlm_decisions_have_chance_binary_auroc(self):
        for decision in ("success", "failure"):
            rows = [self._row("normal", "normal", decision), self._row("failure", "fail", decision)]
            with self.subTest(decision=decision), tempfile.TemporaryDirectory() as temporary_directory:
                metric = write_roc_report(rows, Path(temporary_directory))["metrics"][0]
                self.assertEqual(metric["auroc"], 0.5)
                self.assertEqual(metric["accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()
