from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from rynnbrain_vlm.cop_classifier import CoPLogisticClassifier
from rynnbrain_vlm.cop_timeline import write_logistic_timeline, _plot_timeline


class LogisticTimelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.signature = {"num_frames": 3, "lora_adapter_sha256": "adapter-a"}
        self.classifier = CoPLogisticClassifier(np.array([4., 0, 0]), 0., np.zeros(3),
                                                .7, "raw", "test", self.signature, {})
        self.frames = [{"sample_index": index // 2, "timestamp_sec": time} for index, time in
                       enumerate([1., 1.2, 5., 5.2, 9., 9.2])]
        self.images = [(str(i), object()) for i in range(6)]
        self.turns = [{"images": [("reference", object())], "text": "reference"},
                      {"images": self.images, "text": "execution"}]
        self.vectors = np.array([[-1., .1, 0], [.5, .2, .1], [1., .1, 0]], dtype=np.float32)
        self.model = SimpleNamespace(extract_multiturn_cop_vector=Mock(side_effect=list(self.vectors[:2])))

    def write(self, signature=None):
        return write_logistic_timeline(model=self.model, classifier=self.classifier, turns=self.turns,
            generation={"max_new_tokens": 280}, nominal_response="reference answer",
            frame_metadata=self.frames, final_vector=self.vectors[-1],
            comparison_signature=signature or self.signature, output_directory=self.directory,
            mode="raw", bag_name="held_out", provenance={"evaluation_id": "eval-a", "lora_adapter_sha256": "adapter-a"},
            classifier_model_path="/outputs/raw_logistic.npz")

    def test_prefixes_use_only_available_images_and_final_score_is_unchanged(self):
        report = self.write()
        rows = pd.read_csv(report["csv"])
        self.assertEqual(report["status"], "created")
        self.assertTrue(Path(report["plot"]).is_file())
        self.assertEqual(rows.timestamp_sec.tolist(), [1.2, 5.2, 9.2])
        self.assertEqual(rows.image_count.tolist(), [2, 4, 6])
        self.assertEqual(self.model.extract_multiturn_cop_vector.call_count, 2)
        for step, call in enumerate(self.model.extract_multiturn_cop_vector.call_args_list):
            self.assertEqual(call.args[0][1]["images"], self.images[:2 * (step + 1)])
            self.assertIs(call.args[0][0], self.turns[0])
            self.assertEqual(call.kwargs["reference_response"], "reference answer")
        for row in rows.itertuples():
            self.assertLessEqual(max(json.loads(row.image_timestamps_sec)), row.timestamp_sec)
        expected = [self.classifier.predict_failure_probability(vector) for vector in self.vectors]
        np.testing.assert_allclose(rows.failure_probability, expected)
        np.testing.assert_allclose(rows.anomaly_score, rows.failure_probability)
        np.testing.assert_allclose(rows.failure_percent, np.array(expected) * 100)
        self.assertTrue((rows.threshold == .7).all())
        self.assertTrue((rows.threshold_percent == 70).all())
        self.assertEqual(rows.above_threshold.tolist(), [False, True, True])
        self.assertEqual(rows.vector_source.iloc[-1], "existing_full_execution")
        self.assertTrue((rows.evaluation_id == "eval-a").all())
        self.assertTrue((rows.lora_adapter_sha256 == "adapter-a").all())
        self.assertEqual(report["first_alert"]["last_below_sec"], 1.2)
        self.assertEqual(report["first_alert"]["first_above_sec"], 5.2)
        self.assertEqual(len(self.turns[1]["images"]), 6)

    def test_failed_prefixes_leave_gaps_and_preserve_the_complete_probability(self):
        self.model.extract_multiturn_cop_vector.side_effect = [RuntimeError("OOM"), np.array([np.nan, 0, 0])]
        report = self.write()
        rows = pd.read_csv(report["csv"])
        self.assertEqual(report["status"], "partial")
        self.assertEqual(report["failed_point_count"], 2)
        self.assertTrue(rows.failure_probability.iloc[:2].isna().all())
        self.assertIn("OOM", rows.error.iloc[0])
        self.assertAlmostEqual(rows.failure_probability.iloc[-1], self.classifier.predict_failure_probability(self.vectors[-1]))
        self.assertIsNone(report["first_alert"]["last_below_sec"])

    def test_incompatible_adapter_is_rejected_before_prefix_inference(self):
        with self.assertRaisesRegex(ValueError, "different RynnBrain"):
            self.write({**self.signature, "lora_adapter_sha256": "adapter-b"})
        self.model.extract_multiturn_cop_vector.assert_not_called()
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_plot_uses_percent_for_both_scores_and_threshold(self):
        import matplotlib.pyplot as plt
        data = pd.DataFrame({"timestamp_sec": [1, 2, 3], "anomaly_score": [.1, .75, .9], "threshold": [.7] * 3})
        def inspect(*args, **kwargs):
            axis = plt.gcf().axes[0]
            np.testing.assert_allclose(axis.lines[0].get_ydata(), [10, 75, 90])
            np.testing.assert_allclose(axis.lines[1].get_ydata(), [70, 70, 70])
            self.assertEqual(axis.get_ylabel(), "Failure probability (%)")
            self.assertTrue(any("not calibrated" in text.get_text() for text in plt.gcf().texts))
        with patch("matplotlib.figure.Figure.savefig", side_effect=inspect) as save:
            _plot_timeline(data, self.directory / "preview.png", "example", "raw", probability=True)
        save.assert_called_once()


if __name__ == "__main__":
    unittest.main()
