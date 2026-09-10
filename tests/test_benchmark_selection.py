from __future__ import annotations

from types import ModuleType
import sys
import unittest
from unittest.mock import patch


# Keep these selection tests independent of ROS and the GPU model dependencies.
rosbag_module = ModuleType("rosbag_io")
rosbag_module.list_bag_topics = lambda *args, **kwargs: None
rosbag_module.read_stage_events = lambda *args, **kwargs: None
sys.modules.setdefault("rosbag_io", rosbag_module)

model_module = ModuleType("rynnbrain_vlm.model")
model_module.RynnBrainModel = object
sys.modules.setdefault("rynnbrain_vlm.model", model_module)

run_module = ModuleType("rynnbrain_vlm.run")
run_module.evaluate_multiturn = lambda *args, **kwargs: None
run_module.write_multiturn_outputs = lambda *args, **kwargs: None
sys.modules.setdefault("rynnbrain_vlm.run", run_module)

from run_benchmark import _select_named_records, parse_arguments


class BenchmarkSelectionTests(unittest.TestCase):
    def test_manual_cli_options_are_repeatable(self) -> None:
        argv = [
            "run_benchmark.py",
            "--normal-bag",
            "normal_01",
            "--failure-bag",
            "failure_01",
            "--failure-bag",
            "failure_02",
        ]
        with patch.object(sys, "argv", argv):
            arguments = parse_arguments()

        self.assertEqual(arguments.normal_bag, ["normal_01"])
        self.assertEqual(arguments.failure_bag, ["failure_01", "failure_02"])

    def test_manual_labels_override_recorded_labels(self) -> None:
        records = [
            {"bag_name": "normal_01", "label": "fail"},
            {"bag_name": "failure_01", "label": "unknown"},
        ]
        selected = _select_named_records(
            records,
            ["normal_01", "failure_01"],
            set(),
            {"normal_01": "normal", "failure_01": "fail"},
        )

        self.assertEqual([record["label"] for record in selected], ["normal", "fail"])
        self.assertEqual(
            [record["recorded_label"] for record in selected],
            ["fail", "unknown"],
        )
        self.assertTrue(all(record["label_source"] == "manual_cli" for record in selected))

    def test_same_bag_cannot_receive_two_cli_roles(self) -> None:
        with self.assertRaisesRegex(ValueError, "appear only once"):
            _select_named_records(
                [{"bag_name": "bag_01", "label": "unknown"}],
                ["bag_01", "bag_01"],
                set(),
                {"bag_01": "fail"},
            )


if __name__ == "__main__":
    unittest.main()
