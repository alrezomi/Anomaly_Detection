from __future__ import annotations

from types import ModuleType
import sys
import unittest
from unittest.mock import patch


# Keep these selection tests independent of ROS and the GPU model dependencies.
_stub_names = ("rosbag_io", "rynnbrain_vlm.model", "rynnbrain_vlm.run")
_previous_modules = {name: sys.modules.get(name) for name in _stub_names}
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

from run_benchmark import (
    _build_clean_report,
    _excluded_bag_names,
    _report_statistics,
    _select_named_records,
    parse_arguments,
)

# Do not leak dependency stubs into the real inference tests during discovery.
for _name, _previous in _previous_modules.items():
    if _previous is None:
        sys.modules.pop(_name, None)
    else:
        sys.modules[_name] = _previous


class BenchmarkSelectionTests(unittest.TestCase):
    def test_lora_training_bags_stay_excluded_with_nominal_override(self) -> None:
        config = {
            "nominal_bags": ["/data/dino_memory"],
            "rynnbrain": {"reference_bags": ["/data/reference"], "model": {"lora_adapter_path": "/saved/adapter"}},
        }
        with patch("rynnbrain_vlm.lora.adapter_training_bag_names", return_value={"trained_bag"}):
            self.assertEqual(_excluded_bag_names(config, include_nominal_bags=True), {"reference", "trained_bag"})
            with self.assertRaisesRegex(ValueError, "LoRA training"):
                _select_named_records([], ["trained_bag"], _excluded_bag_names(config, True))

    def test_clean_report_contains_only_decision_fields(self) -> None:
        report = _build_clean_report(
            [
                {
                    "bag_name": "bag_01",
                    "ground_truth_label": "normal",
                    "decision": "success",
                    "decision_correct": True,
                    "classifier_failure_probability": 0.12,
                    "response": "unneeded",
                }
            ]
        )

        self.assertEqual(
            list(report.columns),
            [
                "bag_name",
                "label",
                "model_decision",
                "correct",
                "failure_probability",
            ],
        )
        self.assertEqual(
            report.iloc[0].tolist(), ["bag_01", "normal", "success", True, 0.12]
        )

    def test_report_statistics_counts_scored_and_unscored_rows(self) -> None:
        statistics = _report_statistics(
            [
                {
                    "bag_name": "normal_01",
                    "ground_truth_label": "normal",
                    "decision": "success",
                    "decision_correct": True,
                    "input_mode": "raw",
                },
                {
                    "bag_name": "fail_01",
                    "ground_truth_label": "fail",
                    "decision": "success",
                    "decision_correct": False,
                    "input_mode": "raw",
                },
                {
                    "bag_name": "unknown_01",
                    "ground_truth_label": "unknown",
                    "decision": "failure",
                    "decision_correct": None,
                    "input_mode": "raw_heatmap",
                },
            ]
        )

        self.assertEqual(statistics["total_rows"], 3)
        self.assertEqual(statistics["scored_rows"], 2)
        self.assertEqual(statistics["correct_rows"], 1)
        self.assertEqual(statistics["incorrect_rows"], 1)
        self.assertEqual(statistics["unscored_rows"], 1)
        self.assertEqual(statistics["accuracy"], 0.5)
        self.assertEqual(statistics["by_label"]["fail"]["incorrect_rows"], 1)
        self.assertEqual(statistics["by_input_mode"]["raw_heatmap"]["scored_rows"], 0)

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
