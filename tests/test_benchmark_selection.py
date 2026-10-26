from __future__ import annotations

from types import ModuleType
from pathlib import Path
import tempfile
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
    _discover_benchmark_bags,
    _benchmark_folder,
    _align_benchmark_timing,
    parse_arguments,
    _run_benchmark,
)

# Do not leak dependency stubs into the real inference tests during discovery.
for _name, _previous in _previous_modules.items():
    if _previous is None:
        sys.modules.pop(_name, None)
    else:
        sys.modules[_name] = _previous


class BenchmarkSelectionTests(unittest.TestCase):
    def test_timeline_benchmark_writes_alarm_metrics_and_preserves_final_score(self):
        import json
        import pandas as pd
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bag = root/'data'/'selected'/'normal_01'
            bag.mkdir(parents=True)
            (bag/'metadata.yaml').write_text('')
            timeline = root/'prefix.csv'
            pd.DataFrame(dict(timestamp_sec=[1.,2.,3.], anomaly_score=[.1,.7,.01],
                              threshold=[.5]*3)).to_csv(timeline,index=False)
            original = dict(input_mode='raw', decision='not_generated', confidence=None, response='',
                            classifier_decision='success', classifier_failure_probability=.01,
                            classifier_threshold=.5, inference_mode='feature_only',
                            logistic_timeline_status='created', logistic_timeline_csv=str(timeline))
            config = dict(rynnbrain=dict(source='rosbag', input_modes=['raw'], num_frames=3,
                                         model={}, cop_vectors={'enabled':False},
                                         cop_classifier=dict(enabled=True, train_on_benchmark=False,
                                                             method='logistic', plot_timeline=False)),
                          vlm_benchmark=dict(folders=['selected'],decision_mode='timeline'))
            with patch.object(sys,'argv',['benchmark','--data-root',str(root/'data'),
                                          '--benchmark-dir',str(root/'reports')]):
                args = parse_arguments()
            with patch('run_benchmark._load_reused_classifiers', return_value=({},None)), \
                 patch('run_benchmark._excluded_bag_names', return_value=set()), \
                 patch('run_benchmark.infer_bag_record', return_value=dict(bag_path=str(bag),bag_name=bag.name,label='normal')), \
                 patch('run_benchmark.RynnBrainModel', return_value=object()), \
                 patch('run_benchmark.evaluate_multiturn', return_value=([original],[],[],'task')) as evaluate, \
                 patch('run_benchmark._align_benchmark_timing'), \
                 patch('run_benchmark.write_multiturn_outputs'):
                _run_benchmark(args, config, lambda **kwargs: None)
            self.assertTrue(evaluate.call_args.args[2]['cop_classifier']['plot_timeline'])
            output = root/'reports'/'timeline'
            summary = pd.read_csv(output/'benchmark_summary.csv').iloc[0]
            self.assertEqual(summary.classifier_decision,'failure')
            self.assertEqual(summary.whole_demo_classifier_decision,'success')
            self.assertEqual(summary.classifier_failure_probability,.7)
            self.assertEqual(summary.timeline_first_alarm_time_sec,2.)
            metrics = pd.read_csv(output/'benchmark_metrics.csv').iloc[0]
            self.assertEqual(metrics.decision_source,'classifier_timeline')
            self.assertEqual(metrics.false_positive_rate,1.)
            self.assertEqual(metrics.accuracy,0.)
            stats = json.loads((output/'benchmark_statistics.json').read_text())
            self.assertEqual(stats['classifier']['accuracy'],0.)
            self.assertEqual(stats['accuracy'],0.)
            self.assertEqual(stats['decision_source'],'classifier_timeline')
            clean = pd.read_csv(output/'benchmark_clean.csv').iloc[0]
            self.assertEqual(clean.classifier_decision,'failure')
            self.assertEqual(clean.whole_demo_classifier_decision,'success')
            self.assertTrue((output/'benchmark_protocol.json').is_file())
            self.assertFalse((root/'reports'/'benchmark_summary.csv').exists())
            self.assertEqual(original['classifier_decision'],'success')

    def test_benchmark_timing_ignores_pre_stage_one_error_without_rescoring(self):
        import pandas as pd
        events = pd.DataFrame(dict(time=[.5,1.,2.5,3.,4.], stage=['Error','1','Fail','2','3']))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'timeline.csv'
            pd.DataFrame(dict(timestamp_sec=[1.,2.,3.], anomaly_score=[.1,.8,.9], threshold=[.5]*3)).to_csv(path,index=False)
            rows = [dict(input_mode='raw', decision='success', classifier_decision='failure',
                         logistic_timeline_status='created', logistic_timeline_csv=str(path))]
            with patch('rosbag_io.read_stage_events', return_value=events):
                _align_benchmark_timing(rows, Path('bag'), '/stage',
                    {'source':'rosbag','cop_classifier':{'method':'logistic'}}, 'fail')
            self.assertEqual(rows[0]['recorded_failure_time_sec'],2.5)
            self.assertEqual(rows[0]['predicted_failure_time_sec'],2.)
            self.assertEqual(rows[0]['failure_time_error_sec'],-.5)
            self.assertEqual(rows[0]['decision'],'success')
            self.assertEqual(rows[0]['classifier_decision'],'failure')

    def test_optional_folders_scan_only_selection_and_omission_scans_all(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ['selected/a', 'selected/nested/b', 'other/c']:
                (root/name).mkdir(parents=True)
                (root/name/'metadata.yaml').write_text('')
            bags, folders = _discover_benchmark_bags(root, ['selected'])
            self.assertEqual({bag.name for bag in bags}, {'a', 'b'})
            self.assertEqual(folders, ['selected'])
            self.assertEqual(_benchmark_folder(bags[0], root, folders), 'selected')
            for omitted in [None, []]:
                all_bags, names = _discover_benchmark_bags(root, omitted)
                self.assertEqual({bag.name for bag in all_bags}, {'a', 'b', 'c'})
                self.assertEqual(names, [])
            direct, _ = _discover_benchmark_bags(root, ['selected/a'])
            self.assertEqual([bag.name for bag in direct], ['a'])
            shallow, _ = _discover_benchmark_bags(root, ['selected'], recursive=False)
            self.assertEqual([bag.name for bag in shallow], ['a'])
            for invalid in ['selected', [''], ['.'], ['selected', 'selected/nested'], ['../']]:
                with self.assertRaises(ValueError):
                    _discover_benchmark_bags(root, invalid)

    def test_folder_cli_is_repeatable(self):
        with patch.object(sys, 'argv', ['benchmark', '--folder', 'a', '--folder', 'b']):
            self.assertEqual(parse_arguments().folder, ['a', 'b'])

    def test_knn_ignores_old_failure_selection_but_preserves_lora_exclusions(self) -> None:
        config = {
            "nominal_bags": ["/data/dino_memory"],
            "rynnbrain": {"reference_bags": ["/data/reference"], "model": {"lora_adapter_path": "/saved/adapter"},
                "cop_classifier": {"enabled": True, "method": "knn", "training": {
                    "normal_bags": ["/data/normal_train"], "failure_bags": ["/data/failure_cop_old"]}}},
        }
        with patch("rynnbrain_vlm.lora.adapter_training_bag_names", return_value={"failure_lora_train"}):
            self.assertEqual(_excluded_bag_names(config, include_nominal_bags=True),
                             {"reference", "normal_train", "failure_lora_train"})

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
