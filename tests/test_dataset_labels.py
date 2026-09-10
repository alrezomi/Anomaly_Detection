from __future__ import annotations

from pathlib import Path
from types import ModuleType
import sys
import unittest
from unittest.mock import patch

import pandas as pd

rosbag_module = ModuleType("rosbag_io")
rosbag_module.list_bag_topics = lambda *args, **kwargs: None
rosbag_module.read_stage_events = lambda *args, **kwargs: None
sys.modules.setdefault("rosbag_io", rosbag_module)

import build_dataset_manifest


class DatasetLabelTests(unittest.TestCase):
    @staticmethod
    def _topics() -> pd.DataFrame:
        return pd.DataFrame(
            [{"topic": "/recording_stage", "message_count": 5}]
        )

    def _infer(self, events: pd.DataFrame) -> dict:
        with (
            patch.object(build_dataset_manifest, "list_bag_topics", return_value=self._topics()),
            patch.object(build_dataset_manifest, "read_stage_events", return_value=events),
        ):
            return build_dataset_manifest.infer_bag_record(
                Path("bag_01"), "/recording_stage", 0.1
            )

    def test_early_failure_is_ignored_when_not_in_final_three(self) -> None:
        events = pd.DataFrame(
            {
                "time_ns": [1, 2, 3, 4, 5],
                "time": [1.0, 2.0, 3.0, 4.0, 5.0],
                "stage": ["Error", "setup", "1", "2", "3"],
            }
        )

        record = self._infer(events)

        self.assertEqual(record["label"], "normal")
        self.assertEqual(record["considered_message_count"], 3)
        self.assertIn("Error", record["ignored_earlier_markers"])
        self.assertNotIn("Error", record["used_stage_markers"])

    def test_failure_in_final_three_labels_bag_as_fail(self) -> None:
        events = pd.DataFrame(
            {
                "time_ns": [30, 10, 20, 40],
                "time": [3.0, 1.0, 2.0, 4.0],
                "stage": ["Anomaly", "old error", "2", "3"],
            }
        )

        record = self._infer(events)

        self.assertEqual(record["label"], "fail")
        self.assertIn("Anomaly", record["failure_markers"])
        self.assertNotIn("old error", record["used_stage_markers"])

    def test_fewer_than_three_usable_messages_are_still_classified(self) -> None:
        events = pd.DataFrame(
            {
                "time_ns": [1, 2, 3],
                "time": [0.01, 1.0, 2.0],
                "stage": ["Error", "2", "3"],
            }
        )

        record = self._infer(events)

        self.assertEqual(record["label"], "normal")
        self.assertEqual(record["considered_message_count"], 2)
        self.assertIn("Error", record["ignored_startup_markers"])

    def test_empty_stage_topic_remains_unknown(self) -> None:
        events = pd.DataFrame(columns=["time_ns", "time", "stage"])

        record = self._infer(events)

        self.assertEqual(record["label"], "unknown")
        self.assertEqual(record["used_stage_markers"], "<no stage messages>")


if __name__ == "__main__":
    unittest.main()
