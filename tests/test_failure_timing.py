import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from rynnbrain_vlm.failure_timing import (
    read_failure_annotation, evaluate_failure_timing, attach_failure_timing, write_failure_timing_report,
)


class FailureTimingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.annotation = {"status": "available", "time_sec": 5., "marker": "Error", "marker_count": 1, "error": None}

    def timeline(self, scores=(.1, .2, .9), times=(1., 4., 8.), status="created"):
        path = self.root / "timeline.csv"
        pd.DataFrame({"timestamp_sec": times, "anomaly_score": scores, "threshold": [.5] * len(times)}).to_csv(path, index=False)
        return {"csv": str(path), "plot": str(self.root / "timeline.png"), "status": status}

    def evaluate(self, timeline=None, annotation=None, **kwargs):
        return evaluate_failure_timing(annotation or self.annotation, timeline or self.timeline(),
            **{"source": "rosbag", "mode": "raw", "ground_truth": "fail", "method": "knn", **kwargs})

    def test_annotation_uses_first_error_after_startup_and_all_messages(self):
        events = pd.DataFrame({"time": [9., .02, 2., 3., 4., 5., 10.],
                               "time_ns": [900, 2, 200, 300, 400, 500, 1000],
                               "stage": ["FAIL", "Error", "Error", "1", "2", "3", "3"]})
        with patch("rosbag_io.read_stage_events", return_value=events) as read:
            annotation = read_failure_annotation(Path("bag"), "/custom_stage", .1)
        read.assert_called_once_with(Path("bag"), "/custom_stage")
        self.assertEqual(annotation["time_sec"], 2.)
        self.assertEqual(annotation["time_ns"], 200)
        self.assertEqual(annotation["marker_count"], 2)
        # This marker is earlier than the last three messages but still defines the first recorded failure.
        self.assertEqual(annotation["marker"], "Error")

    def test_missing_topic_empty_events_and_bad_times_are_explicit(self):
        with patch("rosbag_io.read_stage_events", side_effect=ValueError("missing topic")):
            missing = read_failure_annotation("bag")
        self.assertEqual(missing["status"], "unavailable")
        self.assertIn("missing topic", missing["error"])
        for events in (pd.DataFrame(), pd.DataFrame({"time": [1, 2, 3], "stage": ["1", "2", "3"]})):
            with patch("rosbag_io.read_stage_events", return_value=events):
                self.assertEqual(read_failure_annotation("bag")["status"], "missing_marker")
        with patch("rosbag_io.read_stage_events", return_value=pd.DataFrame({"time": [np.nan], "stage": ["Error"]})):
            self.assertEqual(read_failure_annotation("bag")["status"], "unavailable")

    def test_signed_absolute_and_exact_timing_use_observed_samples_without_interpolation(self):
        late = self.evaluate()
        self.assertEqual(late["failure_time_error_sec"], 3.)
        self.assertEqual(late["failure_time_absolute_error_sec"], 3.)
        self.assertEqual(late["predicted_failure_time_sec"], 8.)
        early = self.evaluate(self.timeline(scores=(.1, .8, .9)), method="logistic")
        self.assertEqual(early["failure_time_error_sec"], -1.)
        self.assertEqual(early["failure_time_absolute_error_sec"], 1.)
        exact = self.evaluate(self.timeline(scores=(.1, .5, .9), times=(1, 5, 8)))
        self.assertEqual(exact["failure_time_error_sec"], 0.)
        self.assertEqual(exact["failure_timing_status"], "compared")

    def test_missing_detections_and_incomplete_timelines_are_not_zero_error(self):
        missed = self.evaluate(self.timeline(scores=(.1, .2, .3)))
        self.assertEqual(missed["failure_timing_status"], "not_detected")
        self.assertIsNone(missed["predicted_failure_time_sec"])
        self.assertIsNone(missed["failure_time_error_sec"])
        partial = self.evaluate(self.timeline(scores=(.1, np.nan, .9), status="partial"))
        self.assertEqual(partial["predicted_failure_time_sec"], 8.)
        self.assertEqual(partial["failure_timing_status"], "partial_timeline")
        self.assertIsNone(partial["failure_time_error_sec"])

    def test_annotation_outside_window_label_conflicts_and_unaligned_clocks(self):
        for time in (0., 10.):
            result = self.evaluate(annotation={**self.annotation, "time_sec": time})
            self.assertEqual(result["failure_timing_status"], "annotation_outside_window")
            self.assertIsNone(result["failure_time_error_sec"])
        self.assertEqual(self.evaluate(ground_truth="normal")["failure_timing_status"], "label_conflict")
        for source, mode in (("generated_videos", "raw"), ("rosbag", "heatmap"), ("rosbag", "raw_heatmap")):
            result = self.evaluate(source=source, mode=mode)
            self.assertEqual(result["failure_timing_status"], "unaligned_timebase")
            self.assertIsNone(result["predicted_failure_time_sec"])
            self.assertIsNone(result["failure_time_error_sec"])

    def test_nominal_false_alert_and_missing_failure_annotation(self):
        annotation = {"status": "missing_marker", "time_sec": None}
        self.assertEqual(self.evaluate(annotation=annotation)["failure_timing_status"], "missing_annotation")
        false = self.evaluate(annotation=annotation, ground_truth="normal")
        self.assertEqual(false["failure_timing_status"], "false_alert")
        self.assertIsNone(false["failure_time_error_sec"])
        nominal = self.evaluate(self.timeline(scores=(.1, .2, .3)), annotation=annotation, ground_truth="normal")
        self.assertEqual(nominal["failure_timing_status"], "normal_no_alert")

    def test_disabled_and_invalid_timeline_do_not_produce_timing_errors(self):
        disabled = self.evaluate({"status": "disabled"})
        self.assertEqual(disabled["failure_timing_status"], "timeline_disabled")
        self.assertIsNone(disabled["failure_time_error_sec"])
        invalid = self.evaluate(self.timeline(times=(1, 4, 3)))
        self.assertEqual(invalid["failure_timing_status"], "invalid_timeline")

    def test_aggregate_mean_absolute_error_does_not_cancel_early_and_late_alerts(self):
        late = self.evaluate()
        early = self.evaluate(self.timeline(scores=(.8, .9, .9)))  # -4 s
        missed = self.evaluate(self.timeline(scores=(.1, .2, .3)))
        rows = [{"bag_name": name, "input_mode": "raw", "ground_truth_label": "fail", **result}
                for name, result in (("late", late), ("early", early), ("missed", missed))]
        rows.append({"bag_name": "pipeline_failed", "input_mode": None})
        # A nominal/unknown row must not enter the failure-only mean, even if
        # importing an older report with a numeric timing result for that row.
        rows += [{"bag_name": label, "input_mode": "raw", "ground_truth_label": label,
                  **{**late, "failure_time_error_sec": 100.}} for label in ("normal", "unknown")]
        report = write_failure_timing_report(rows, self.root, input_modes=["raw", "heatmap"], method="knn")
        raw, heatmap = report["modes"]
        self.assertEqual(raw["compared_bags"], 2)
        self.assertEqual(raw["evaluated_bags"], 6)
        self.assertEqual(raw["failure_bags"], 3)
        self.assertEqual(raw["mean_signed_error_sec"], -.5)
        self.assertEqual(raw["mean_absolute_error_sec"], 3.5)
        self.assertEqual(raw["status_counts"]["not_detected"], 1)
        self.assertEqual(raw["status_counts"]["not_evaluated"], 1)
        self.assertEqual(heatmap["compared_bags"], 0)
        self.assertIsNone(heatmap["mean_absolute_error_sec"])
        self.assertTrue(Path(raw["plot"]).is_file())
        self.assertEqual(len(pd.read_csv(raw["csv"])), 6)
        saved = json.loads((self.root / "benchmark_failure_timing/failure_timing_summary.json").read_text())
        self.assertEqual(saved["modes"][0]["mean_absolute_error_sec"], 3.5)

    def test_empty_report_has_no_invented_zero_mean(self):
        report = write_failure_timing_report([], self.root, input_modes=["raw"], method="logistic")
        self.assertEqual(report["modes"][0]["evaluated_bags"], 0)
        self.assertIsNone(report["modes"][0]["mean_signed_error_sec"])

    def test_overlay_keeps_predictions_and_scores_unchanged(self):
        timeline = self.timeline()
        before = Path(timeline["csv"]).read_bytes()
        result = attach_failure_timing(self.annotation, timeline, source="rosbag", mode="raw",
                                      ground_truth="fail", method="logistic", bag_name="test")
        self.assertEqual(result["failure_time_error_sec"], 3.)
        self.assertEqual(Path(timeline["csv"]).read_bytes(), before)
        self.assertTrue(Path(timeline["plot"]).is_file())
