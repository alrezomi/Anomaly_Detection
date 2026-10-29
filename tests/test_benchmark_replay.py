from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from PIL import Image

from rynnbrain_vlm.benchmark_replay import (_choose_threshold, _scan_bag, run_threshold_search,
    threshold_search_enabled, validate_threshold_search)


class FakeDetector:
    def __init__(self):
        self.model = SimpleNamespace(adapter_identity="saved-adapter")
        self.classifier = SimpleNamespace(threshold=.7)
        self.path = Path("saved_classifier.npz")
        self.signature = dict(turn1_prompt="Saved reference", turn2_prompt="Saved evaluation")
        self.response = "Nominal reference response"
        self.calls = []

    def reset(self, bag):
        self.bag, self.steps = Path(bag).name, 0

    def append(self, image, timestamp, verify=False):
        assert not verify
        score = image.getpixel((0, 0))[0] / 255
        self.calls.append((self.bag, timestamp, score))
        self.steps += 1
        return dict(timestamp_sec=timestamp, anomaly_score=score, threshold=.7,
            above_threshold=score >= .7, cached_feature_sec=.003, classifier_sec=.001,
            score_kind="failure_probability", verification=None), np.array([score, self.steps, 1.]), None


def packets(specs):
    return [(i, float(t), Image.new("RGB", (4, 4), (red, 0, 0))) for i, (t, red) in enumerate(specs)]


class BenchmarkReplayTests(unittest.TestCase):
    def config(self, root):
        return dict(output_dir=str(root), camera_topics=["top"], vlm_benchmark={"threshold_search": True, "decision_mode": "timeline"},
            rynnbrain=dict(source="rosbag", input_modes=["raw"], num_frames=8, camera_topics=["top"],
                model={"model_id": "test-model", "lora_adapter_path": "/saved/adapter"},
                inference={"feature_only": True}, sampling_start_sec=0., sampling_end_sec=None,
                online_replay={"interval_mode": "adaptive", "max_observations": None, "stop_on_alarm": True},
                cop_classifier={"enabled": True, "method": "logistic", "train_on_benchmark": False,
                                "training": {"threshold": .7}}, cop_vectors={"enabled": True}))

    def fixtures(self):
        return dict(normal=packets([(0, 26), (10, 191), (13, 26), (23, 26), (33, 26), (40, 26)]),
                    failure=packets([(0, 26), (10, 128), (15, 204), (18, 242), (21, 230), (24, 26), (34, 26)]))

    def test_search_is_opt_in_and_rejects_unsupported_inputs(self):
        self.assertFalse(threshold_search_enabled({}))
        with self.assertRaises(ValueError):
            threshold_search_enabled({"threshold_search": "true"})
        config = self.config(Path("test"))
        validate_threshold_search(config, "timeline", False, False)
        for mode, dino, skip in [("whole_demo", False, False), ("timeline", True, False), ("timeline", False, True)]:
            with self.assertRaises(ValueError):
                validate_threshold_search(config, mode, dino, skip)
        config["rynnbrain"]["inference"]["feature_only"] = False
        with self.assertRaisesRegex(ValueError, "feature_only"):
            validate_threshold_search(config, "timeline", False, False)

    def test_scan_passes_alarms_and_eight_observations_without_waiting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.config(root)
            config["rynnbrain"]["online_replay"] = dict(interval_mode="constant", interval_sec=3.,
                                                          max_observations=None, stop_on_alarm=True)
            detector = FakeDetector()
            source = packets([(i * 3, 230) for i in range(12)])
            scan = _scan_bag(detector, config, Path("test"), root / "scan.npz", source)
            self.assertTrue(scan["complete"])
            self.assertEqual(len(scan["rows"]), 12)
            self.assertEqual([call[1] for call in detector.calls], list(np.arange(12) * 3.))
            with np.load(root / "scan.npz", allow_pickle=False) as saved:
                self.assertEqual(saved["vectors"].shape, (12, 3))
                self.assertIn("image_11", saved)

    def test_scan_error_is_not_a_complete_negative(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            detector = FakeDetector()
            detector.append = Mock(side_effect=RuntimeError("GPU failure"))
            scan = _scan_bag(detector, self.config(root), Path("test"), root / "scan.npz", packets([(0, 26)]))
            self.assertFalse(scan["complete"])
            self.assertIn("GPU failure", scan["error"])
            self.assertEqual(scan["rows"], [])
        with self.assertRaisesRegex(ValueError, "both nominal and failure"):
            _choose_threshold({"comparisons": []})

    def test_selects_once_exports_one_threshold_and_preserves_all_reports_and_timings(self):
        import run_benchmark as benchmark
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.config(root)
            config["rynnbrain"]["output_dir"] = str(root / "vlm")
            config["vlm_benchmark"]["folders"] = ["selected"]
            data = root / "data" / "selected"
            records = {}
            for name, label in [("normal", "normal"), ("failure", "fail")]:
                bag = data / name
                bag.mkdir(parents=True)
                (bag / "metadata.yaml").write_text("")
                records[name] = dict(bag_path=str(bag), bag_name=name, label=label)
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config))
            original = config_path.read_bytes()
            detector = FakeDetector()
            fixtures = self.fixtures()
            times = [0.]
            def measurement_clock():
                times[0] += .005
                return times[0]
            def annotation(bag, topic):
                return dict(status="available" if Path(bag).name == "failure" else "missing_marker",
                            time_sec=16. if Path(bag).name == "failure" else None, error=None)
            with patch.object(sys, "argv", ["benchmark", "--config", str(config_path), "--data-root", str(root / "data")]), \
                 patch("run_benchmark.RynnBrainModel", return_value=detector.model) as model_loader, \
                 patch("run_benchmark._load_reused_classifiers", return_value=({}, None)), \
                 patch("run_benchmark._excluded_bag_names", return_value=set()), \
                 patch("run_benchmark.infer_bag_record", side_effect=lambda bag, *a: records[bag.name]), \
                 patch("rynnbrain_vlm.online_replay.OnlineDetector", return_value=detector) as detector_loader, \
                 patch("rosbag_io.iter_rosbag_image_frames", side_effect=lambda source, **kw: iter(fixtures[source.bag_path.name])), \
                 patch("rynnbrain_vlm.visual_matching.read_corrected_failure_annotation", side_effect=annotation), \
                 patch("rynnbrain_vlm.benchmark_replay.clock", side_effect=measurement_clock), \
                 patch("run_benchmark.evaluate_multiturn", side_effect=AssertionError("Must not run whole-demo inference")), \
                 patch("run_benchmark._validate_reused_classifiers", side_effect=AssertionError("GUI detector validates its own inputs")):
                benchmark.main()
            model_loader.assert_called_once()
            self.assertIs(detector_loader.call_args.kwargs["model"], detector.model)
            self.assertEqual(config_path.read_bytes(), original)
            self.assertEqual(len(detector.calls), 12)  # 5 nominal + 7 failure; no per-threshold or export inference.
            timeline = root / "vlm_benchmark" / "timeline"
            output = timeline / "best_threshold"
            choice = json.loads((output / "selected_threshold.json").read_text())
            self.assertEqual(choice["threshold"], .75)
            self.assertEqual(choice["accuracy"], 1.)
            self.assertEqual(choice["tied_thresholds"], [i / 100 for i in range(75, 95)])
            rows = pd.read_csv(output / "benchmark_summary.csv").set_index("bag_name")
            self.assertEqual(rows.loc["failure", "classifier_failure_probability"], .8)
            self.assertAlmostEqual(rows.loc["failure", "threshold_selection_peak_score"], 242 / 255)
            self.assertEqual(rows.loc["failure", "predicted_failure_time_sec"], 15.)
            self.assertEqual(rows.loc["failure", "failure_time_error_sec"], -1.)
            self.assertEqual(rows.loc["failure", "stop_reason"], "alarm_threshold")
            prefix = pd.read_csv(output / "failure/rynnbrain_multiturn/logistic_timeline_raw.csv")
            self.assertEqual(list(prefix.timestamp_sec), [0., 10., 15.])
            self.assertTrue(prefix.above_threshold.iloc[-1])
            full = pd.read_csv(timeline / "threshold_selection/failure.csv")
            self.assertEqual(len(full), 7)
            self.assertEqual(set(full.threshold), {.75})
            runtime = json.loads((output / "benchmark_frame_runtime.json").read_text())
            self.assertEqual(runtime["observations"], 8)
            self.assertAlmostEqual(runtime["average_processing_sec_per_frame"], .005)
            clean = pd.read_csv(output / "benchmark_clean.csv")
            self.assertAlmostEqual(clean.runtime_frame_sec_mean.iloc[0], .005)
            stats = json.loads((output / "benchmark_statistics.json").read_text())
            self.assertEqual(stats["accuracy"], 1.)
            self.assertEqual(stats["probability_roc"]["score_column"], "threshold_selection_peak_score")
            self.assertEqual(stats["probability_roc"]["metrics"][0]["auroc"], 1.)
            effective = json.loads((output / "pipeline_config.json").read_text())
            self.assertEqual(effective["rynnbrain"]["cop_classifier"]["training"]["threshold"], .75)
            self.assertFalse(effective["vlm_benchmark"]["threshold_search"])
            for path in ["benchmark_bag_selection.csv", "benchmark_metrics.png", "benchmark_failure_probability_raw.png", "benchmark_frame_runtime.png",
                         "benchmark_probability_roc/roc_raw.png", "benchmark_probability_roc/auroc_raw.png",
                         "benchmark_failure_timing/failure_timing_raw_logistic.png", "cop_pca/cop_pca_raw.png", "REPORTS.md",
                         "normal/rynnbrain_multiturn/rynnbrain_results_multiturn.csv",
                         "failure/rynnbrain_multiturn/rynnbrain_responses_multiturn.json",
                         "failure/rynnbrain_multiturn/logistic_timeline_raw.png"]:
                self.assertTrue((output / path).is_file(), path)
            self.assertFalse((timeline / "normal").exists())
            self.assertFalse((output / "threshold_60").exists())
            # Re-select on saved observations and export every graph without model/ROS-image inference.
            from rynnbrain_vlm.benchmark_replay import refresh_saved_threshold_reports
            original_output = (output / 'benchmark_summary.csv').read_bytes()
            original_scan_csv = (timeline / 'threshold_selection/failure.csv').read_bytes()
            with patch('run_benchmark.RynnBrainModel', side_effect=AssertionError('No model loading')), \
                 patch('rynnbrain_vlm.online_replay.OnlineDetector', side_effect=AssertionError('No detector loading')), \
                 patch('rosbag_io.iter_rosbag_image_frames', side_effect=AssertionError('No image sampling')), \
                 patch('rynnbrain_vlm.visual_matching.read_corrected_failure_annotation', side_effect=annotation):
                refreshed, new_root, _ = refresh_saved_threshold_reports(timeline)
            self.assertEqual(new_root, timeline / 'best_threshold_0_100')
            self.assertEqual((output / 'benchmark_summary.csv').read_bytes(), original_output)
            self.assertEqual((timeline / 'threshold_selection/failure.csv').read_bytes(), original_scan_csv)
            self.assertEqual(len(detector.calls), 12)
            self.assertEqual(len(refreshed), 2)
            self.assertTrue((new_root / 'REPORTS.md').is_file())
            self.assertTrue((new_root / 'benchmark_failure_probability_points.csv').is_file())
            thresholds = pd.read_csv(new_root / 'benchmark_threshold_sweep/threshold_metrics.csv')
            self.assertEqual(set(thresholds.threshold_percent), set(range(101)))
