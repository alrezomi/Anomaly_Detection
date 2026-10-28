from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from rynnbrain_vlm.threshold_sweep import write_threshold_sweep_report


def row(name, label, score, *, mode="raw", protocol="timeline", category="Failure_1_grasp_miss"):
    return dict(bag_path=f"/data/{category if label == 'fail' else 'Nominal'}/{name}",
                ground_truth_label=label, input_mode=mode, classifier_failure_probability=score,
                classifier_threshold=.6, classifier_decision="success", decision="uncertain",
                benchmark_decision_mode=protocol)


class ThresholdSweepTests(unittest.TestCase):
    def test_sweeps_every_percent_reports_all_accuracy_ties_and_preserves_inputs(self):
        rows = [row("n1", "normal", .65), row("n2", "normal", .75),
                row("f1", "fail", .8), row("f2", "fail", .95)]
        before = deepcopy(rows)
        with tempfile.TemporaryDirectory() as directory:
            report = write_threshold_sweep_report(rows, Path(directory))
            overall = report["comparisons"][0]
            self.assertEqual(overall["status"], "created")
            self.assertEqual(overall["best_accuracy"], 1.)
            self.assertEqual(overall["best_accuracy_thresholds"], [p / 100 for p in range(76, 81)])
            data = pd.read_csv(report["csv"])
            data = data[data.failure_category == "all_failures"]
            self.assertEqual(list(data.threshold_percent), list(range(60, 91)))
            at_75 = data[data.threshold_percent == 75].iloc[0]
            self.assertEqual((at_75.true_positive, at_75.false_positive, at_75.true_negative, at_75.false_negative), (2, 1, 1, 0))
            self.assertEqual(at_75.accuracy, .75)
            self.assertEqual(at_75.false_positive_rate, .5)
            at_80 = data[data.threshold_percent == 80].iloc[0]
            self.assertEqual(at_80.true_positive, 2)  # Equality is an alarm.
            self.assertEqual(at_80.recall, 1.)
            at_81 = data[data.threshold_percent == 81].iloc[0]
            self.assertEqual(at_81.false_negative, 1)
            self.assertEqual(at_81.recall, .5)
            self.assertEqual(at_81.precision, 1.)
            self.assertEqual(list(data[data.is_best_accuracy].threshold_percent), list(range(76, 81)))
            self.assertEqual(json.loads((Path(directory) / "benchmark_threshold_sweep/threshold_summary.json").read_text()), report)
            self.assertTrue(all(Path(path).stat().st_size > 0 for path in report["plot_files"]))
        self.assertEqual(rows, before)

    def test_modes_protocols_and_failure_categories_are_not_pooled(self):
        rows = [row("n", "normal", .65), row("f1", "fail", .8),
                row("f2", "fail", .2, category="Failure_2_slip"),
                row("n", "normal", .9, mode="heatmap"), row("f1", "fail", .1, mode="heatmap"),
                row("n", "normal", .1, protocol="whole_demo"), row("f1", "fail", .95, protocol="whole_demo")]
        with tempfile.TemporaryDirectory() as directory:
            report = write_threshold_sweep_report(rows, Path(directory))
            comparisons = {(c["input_mode"], c["benchmark_decision_mode"], c["failure_category"]): c
                           for c in report["comparisons"]}
            self.assertEqual(comparisons["raw", "timeline", "all_failures"]["failure_count"], 2)
            self.assertEqual(comparisons["raw", "timeline", "Failure_1_grasp_miss"]["failure_count"], 1)
            self.assertEqual(comparisons["raw", "timeline", "Failure_2_slip"]["normal_count"], 1)
            self.assertEqual(comparisons["raw", "whole_demo", "all_failures"]["best_accuracy"], 1.)
            self.assertEqual(comparisons["heatmap", "timeline", "all_failures"]["best_accuracy"], 0.)
            self.assertEqual(len(report["plot_files"]), 3)

    def test_invalid_scores_unknown_labels_and_single_class_do_not_produce_a_best_threshold(self):
        rows = [row("n", "normal", .4)]
        rows += [row(f"invalid_{i}", "fail", value) for i, value in enumerate([None, "", "bad", float("inf"), -.1, 1.1])]
        rows += [row("unknown", "unknown", .9)]
        with tempfile.TemporaryDirectory() as directory:
            report = write_threshold_sweep_report(rows, Path(directory))
            overall = report["comparisons"][0]
            self.assertEqual(overall["status"], "skipped")
            self.assertEqual(overall["excluded_score_count"], 6)
            self.assertEqual(overall["unknown_label_count"], 1)
            self.assertEqual(overall["score_coverage"], 1 / 7)
            self.assertIsNone(overall["best_accuracy"])
            self.assertEqual(overall["best_accuracy_thresholds"], [])
            self.assertTrue(all(Path(path).exists() for path in report["plot_files"]))

    def test_empty_report_and_rescoring_replace_results_without_loading_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [row("n", "normal", .1, protocol="whole_demo"), row("f", "fail", .95, protocol="whole_demo")]
            write_threshold_sweep_report(rows, root)
            report = write_threshold_sweep_report([], root, input_modes=["raw"])
            self.assertEqual(report["comparisons"][0]["status"], "skipped")
            self.assertEqual(report["comparisons"][0]["scored_bags"], 0)
            data = pd.read_csv(report["csv"])
            self.assertTrue(data.accuracy.isna().all())
            self.assertFalse(data.is_best_accuracy.any())
            self.assertEqual(report["comparisons"][0]["best_accuracy_thresholds"], [])
