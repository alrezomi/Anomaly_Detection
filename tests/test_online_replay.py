import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from http.server import ThreadingHTTPServer
from urllib.request import urlopen, Request
from urllib.error import HTTPError

import numpy as np
from PIL import Image

from rynnbrain_vlm.online_replay import (ReplayFeed, SequentialReplayFeed, gmr_sample_frames, ReplayApplication, OnlineDetector, make_handler,
                                       finish_verification, verification_failure_message,
                                       signature_differences, validate_online_classifier, replay_bags)


class OnlineReplayTests(unittest.TestCase):
    def test_gui_ts_selection_is_separate_from_pipeline_config(self):
        config = {'rynnbrain': {'num_frames': 8}}
        app = ReplayApplication(config, [Path('heldout')], 10., False)
        self.assertEqual(app.snapshot()['gmr_signal'], 'pose')
        app.update(status='ready')
        app.start(0, detector_type='gmr', gmr_signal='pose_gripper')
        self.assertEqual(app.snapshot()['gmr_signal'], 'pose_gripper')
        self.assertEqual(app.jobs.get_nowait(), 0)
        self.assertEqual(config, {'rynnbrain': {'num_frames': 8}})
        app.update(status='completed')
        with self.assertRaisesRegex(ValueError, 'Select TCP pose'):
            app.start(0, detector_type='gmr', gmr_signal=[])

    def test_sequential_feed_keeps_every_sample_when_inference_falls_behind(self):
        now = [0.]
        stop = Mock()
        stop.is_set.return_value = False
        def wait(duration):
            now[0] += duration
            return False
        stop.wait.side_effect = wait
        image = Image.new('RGB', (2, 2))
        published = Mock()
        feed = SequentialReplayFeed(((i, i * .5, image) for i in range(12)), published, stop, clock=lambda: now[0])
        seen = []
        while (packet := feed.next_frame(1000.)) is not None:
            seen.append(packet[0])
            now[0] += 1.  # Inference takes longer than the sampling period.
        self.assertEqual(seen, list(range(12)))
        self.assertEqual(published.call_count, 12)
        self.assertGreater(now[0], 5.5)
        stop.is_set.return_value = True
        self.assertIsNone(feed.next_frame(0.))

    def test_gmr_signal_clock_does_not_drop_samples_or_select_future_images(self):
        image = Image.new('RGB', (2, 2))
        frames = iter([(0, .5, image), (1, 1.5, image)])
        times = np.linspace(0., 2., 21)
        packets = list(gmr_sample_frames(frames, times))
        self.assertEqual(len(packets), 21)
        np.testing.assert_array_equal([p[1] for p in packets], times)
        self.assertIsNone(packets[0][2])
        self.assertEqual(packets[-1][3], 1.5)
        self.assertTrue(all(p[3] is None or p[3] <= p[1] for p in packets))
        self.assertEqual(len(list(gmr_sample_frames(iter([]), times))), 21)

    def test_non_vlm_runs_all_native_samples_despite_vlm_limit_interval_and_window(self):
        import pandas as pd
        for kind in ['dino', 'gmr']:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                config = {'output_dir': directory, 'rynnbrain': {'camera_topics': ['top'],
                          'num_frames': 8, 'sampling_start_sec': 10., 'sampling_end_sec': 11.}}
                app = ReplayApplication(config, [Path('test')], 10., False)
                app.update(detector_type=kind, note='native sampling')
                times = np.arange(12) * .001
                def append(image, timestamp, verify):
                    self.assertFalse(verify)
                    return dict(timestamp_sec=timestamp, anomaly_score=timestamp * 2, threshold=timestamp + .1,
                                above_threshold=False, cached_feature_sec=.001, verification=None), np.ones(2), None
                app.detector = SimpleNamespace(reset=Mock(), append=append, frame_count=None,
                    path=Path('existing'), score_source='test', signature={}, knn=True, sample_fps=2.,
                    results=pd.DataFrame({'time': times}), start_sec=0., end_sec=float(times[-1]))
                image = Image.new('RGB', (4, 4))
                # GMR must also finish scoring after the camera ends.
                frames = [(i, float(t), image) for i, t in enumerate(times[:3] if kind == 'gmr' else times)]
                with patch('rosbag_io.iter_rosbag_image_frames', return_value=iter(frames)) as reader, \
                     patch('rynnbrain_vlm.online_replay.save_plot'), \
                     patch('rynnbrain_vlm.visual_matching.read_corrected_failure_annotation', return_value={}):
                    app.run_bag(Path('test'))
                state = app.snapshot()
                self.assertEqual(state['status'], 'completed', state.get('error'))
                self.assertEqual(len(state['rows']), 12)
                self.assertIsNone(state['max_frames'])
                self.assertEqual(reader.call_args.kwargs['start_sec'], 0.)
                if kind == 'dino':
                    self.assertEqual(reader.call_args.kwargs['sample_fps'], 2.)
                else:
                    self.assertEqual(state['video_time_sec'], times[2])
                    self.assertEqual(state['total_samples'], 12)
                saved_path = next(Path(directory).glob('online_replay/*/test/online_results.json'))
                saved = json.loads(saved_path.read_text())
                self.assertIsNone(saved['interval_sec'])
                np.testing.assert_array_equal([r['timestamp_sec'] for r in saved['rows']], times)
                np.testing.assert_allclose([r['threshold'] for r in saved['rows']], times + .1)
                import csv
                with saved_path.with_name('online_scores.csv').open() as stream:
                    self.assertEqual(len(list(csv.DictReader(stream))), 12)

    def test_bag_discovery_keeps_default_first_deduplicates_and_honors_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("normal/a", "failure/b", "reference/c"):
                bag = root / name
                bag.mkdir(parents=True)
                (bag / "metadata.yaml").write_text("rosbag2_bagfile_information: {}")
            (root / "not_a_bag").mkdir()
            selected = root / "failure/b"
            config = {"test_bag": str(selected)}
            bags = replay_bags(config, data_root=root)
            self.assertEqual(bags[0], selected.resolve())
            self.assertEqual(len(bags), 3)
            self.assertEqual(set(bags), {root.resolve() / p for p in ("normal/a", "failure/b", "reference/c")})
            self.assertEqual(replay_bags(config, [selected, selected], root), [selected.resolve()])
            self.assertEqual(config, {"test_bag": str(selected)})
            with self.assertRaisesRegex(ValueError, "one bag-path string"):
                replay_bags({"test_bag": [str(selected)]}, data_root=root)

    def test_discovered_training_bags_disabled_but_dino_only_bags_remain_available(self):
        app = ReplayApplication({"nominal_bags": ["dino_only"]},
                                [Path("reference"), Path("train"), Path("dino_only"), Path("heldout")], 10., False)
        detector = SimpleNamespace(excluded={"reference", "train"}, knn=True, frame_count=8, score_source="test")
        # Initialize without entering the job loop, just as a shutdown during startup would do.
        app.shutdown.set()
        with patch("rynnbrain_vlm.online_replay.OnlineDetector", return_value=detector), \
             patch("rynnbrain_vlm.online_replay.synchronized_time", side_effect=time.monotonic):
            app._worker()
        self.assertEqual(app.snapshot()["excluded_bag_indices"], [0, 1])
        with self.assertRaisesRegex(ValueError, "training/reference"):
            app.start(1)
        app.start(2)
        self.assertEqual(app.jobs.get_nowait(), 2)

    def test_compatibility_reports_exact_fields_and_still_rejects_mismatch(self):
        saved = {"model_config": {"lora_adapter_path": "/v102/adapter"},
                 "enable_thinking": True, "test_camera_topics": ["top"], "legacy": None}
        current = {"model_config": {"lora_adapter_path": "/v120/adapter"},
                   "enable_thinking": False, "test_camera_topics": ["top"], "new": 8}
        classifier = SimpleNamespace(comparison_signature=saved, representation_id="norm")
        validate_online_classifier(classifier, saved, "norm", "raw_knn.npz")
        with self.assertRaises(ValueError) as error:
            validate_online_classifier(classifier, current, "norm2", "raw_knn.npz")
        message = str(error.exception)
        self.assertIn('comparison_signature.model_config.lora_adapter_path: saved="/v102/adapter"; current="/v120/adapter"', message)
        self.assertIn("enable_thinking: saved=true; current=false", message)
        self.assertIn("legacy: saved=null; current=<missing>", message)
        self.assertIn("new: saved=<missing>; current=8", message)
        self.assertIn('representation_id: saved="norm"; current="norm2"', message)
        self.assertNotIn("test_camera_topics", message)
        self.assertEqual(signature_differences(saved, saved), [])

    def test_online_knn_uses_memory_and_threshold_for_current_prefix(self):
        from rynnbrain_vlm.cop_knn import CoPKNNDetector
        from rynnbrain_vlm.cop_timeline import PrefixDetector, timeline_sampling
        signature = {"num_frames": 8}
        complete = CoPKNNDetector(np.array([[-1., 0.], [0., -1.]]), 1, .9, "raw", "norm", signature, {})
        memories = np.repeat(complete.nominal_vectors[None], 8, axis=0)
        memories[0] = np.array([[1., 0.], [0., 1.]])
        thresholds = np.linspace(.2, .9, 8)
        detector = OnlineDetector.__new__(OnlineDetector)
        detector.knn, detector.complete, detector.signature = True, complete, signature
        detector.prefix = PrefixDetector(memories, thresholds, timeline_sampling({}, 8))
        first_score, first_threshold = detector.score(np.array([1., 0.]), 0)
        self.assertEqual((first_score, first_threshold), (0., .2))
        final_score, final_threshold = detector.score(np.array([1., 0.]), 7)
        self.assertAlmostEqual(final_score, np.sqrt(2))
        self.assertEqual(final_threshold, .9)

    def test_report_identifies_feature_only_failure_without_relaxing_limits(self):
        report = finish_verification(dict(relative_l2_difference=.003925737583476273,
            relative_l2_tolerance=.001, within_tolerance=False),
            .0024194448931056773, .0023110176619342117, .5)
        self.assertEqual(report["failed_checks"], ["feature_vector"])
        self.assertTrue(report["score_within_tolerance"])
        self.assertTrue(report["decisions_match"])
        self.assertFalse(report["within_tolerance"])
        message = verification_failure_message(report)
        self.assertIn("0.393%", message)
        self.assertIn("0.100%", message)
        near = finish_verification(dict(within_tolerance=True), .5000001, .4999999, .5)
        self.assertEqual(near["failed_checks"], ["threshold_decision"])

    def test_diagnostics_continue_all_eight_using_uncached_results_on_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            app = ReplayApplication({"output_dir": directory, "rynnbrain": {"camera_topics": ["top"]}},
                                    [Path("test")], 1., False, diagnose=True)
            self.assertTrue(app.verify)
            def append(_image, timestamp, verify):
                self.assertTrue(verify)
                index = int(timestamp)
                # Include feature-only failures, decision flips and passing checks.
                cached_score = .8 if index % 2 else .2
                baseline_score = .2
                check = finish_verification(dict(within_tolerance=index == 0,
                    relative_l2_difference=0. if index == 0 else .004, relative_l2_tolerance=.001),
                    cached_score, baseline_score, .5)
                return dict(timestamp_sec=timestamp, anomaly_score=cached_score, threshold=.5,
                    above_threshold=cached_score >= .5, cached_feature_sec=.001,
                    peak_gpu_allocated_bytes={}, verification=check), np.ones(2), np.array([2., 3.])
            app.detector = SimpleNamespace(reset=Mock(), append=append, frame_count=8, path=Path("existing.npz"),
                                           score_source="test", signature={}, knn=False)
            image = Image.new("RGB", (4, 4))
            packets = [(i, float(i), image) for i in range(8)]
            with patch("rosbag_io.iter_rosbag_image_frames", return_value=iter([])), \
                 patch.object(ReplayFeed, "run"), patch.object(ReplayFeed, "next_frame", side_effect=packets), \
                 patch("rynnbrain_vlm.online_replay.save_plot"):
                app.run_bag(Path("test"))
            self.assertEqual(app.snapshot()["status"], "completed")
            self.assertEqual(app.snapshot()["cache_verification_status"], "failed")
            path = next(Path(directory).glob("online_replay/*/test/online_results.json"))
            result = json.loads(path.read_text())
            self.assertEqual(len(result["rows"]), 8)
            self.assertEqual(result["cache_verification_status"], "failed")
            self.assertEqual(result["rows"][0]["score_computation"], "cached")
            for row in result["rows"][1:]:
                self.assertEqual(row["score_computation"], "uncached_verification_fallback")
                self.assertEqual(row["anomaly_score"], .2)
                self.assertFalse(row["above_threshold"])
                np.testing.assert_array_equal(np.load(path.parent / row["vector_file"]), [2., 3.])
                np.testing.assert_array_equal(np.load(path.parent / row["cached_vector_file"]), [1., 1.])
            # Same schema across passed and failed rows; no interrupted CSV write.
            import csv
            with (path.parent / "online_scores.csv").open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 8)

    def test_web_interface_state_video_and_controls(self):
        app = ReplayApplication({}, [Path("test")], 10., False)
        app.detector = Mock()
        app.update(status="ready")
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        address = f"http://127.0.0.1:{server.server_port}"
        try:
            with urlopen(address) as response:
                self.assertIn(b"Execution monitor", response.read())
            with urlopen(address + "/state") as response:
                self.assertEqual(json.load(response)["status"], "ready")
            app.publish(Image.new("RGB", (10, 10)), 1.)
            with urlopen(address + "/frame") as response:
                self.assertEqual(response.headers["Content-Type"], "image/jpeg")
                self.assertEqual(response.read()[:2], b"\xff\xd8")
            for endpoint, attribute in (("/observed-frame", "observed_jpeg"), ("/matched-frame", "matched_jpeg")):
                with urlopen(address + endpoint) as response:
                    self.assertEqual(response.status, 204)
                setattr(app, attribute, app.jpeg)
                with urlopen(address + endpoint) as response:
                    self.assertEqual(response.read(), app.jpeg)
            request = Request(address + "/start", data=b'{"bag_index":0,"interval_sec":5,"mode":"diagnostic","match_preview":false}', headers={"Origin": address})
            with urlopen(request) as response:
                self.assertEqual(response.status, 200)
            self.assertEqual(app.jobs.get_nowait(), 0)
            self.assertEqual(app.interval, 5.)
            self.assertTrue(app.diagnose)
            self.assertTrue(app.verify)
            self.assertFalse(app.snapshot()['match_preview'])
            self.assertIsNone(app.matched_jpeg)
            with urlopen(Request(address + "/stop", data=b"{}")):
                self.assertTrue(app.stop.is_set())
            with self.assertRaises(HTTPError) as error:
                urlopen(Request(address + "/start", data=b"{}", headers={"Origin": "http://elsewhere"}))
            self.assertEqual(error.exception.code, 403)
            error.exception.close()
            with self.assertRaises(HTTPError) as error:
                urlopen(Request(address + "/start", data=b"[]"))
            self.assertEqual(error.exception.code, 400)
            error.exception.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=1)

    def test_page_and_state_respond_while_model_initialization_is_blocked(self):
        app = ReplayApplication({}, [Path("test")], 10., False)
        entered, release = threading.Event(), threading.Event()
        def initialize(_config, progress):
            progress("Preparing reusable reference cache")
            entered.set()
            release.wait(3)
            return SimpleNamespace(score_source="test", knn=False, frame_count=8)
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        address = f"http://127.0.0.1:{server.server_port}"
        try:
            with patch("rynnbrain_vlm.online_replay.OnlineDetector", side_effect=initialize), \
                 patch("rynnbrain_vlm.online_replay.synchronized_time", side_effect=time.monotonic):
                app.worker.start()
                self.assertTrue(entered.wait(1))
                with urlopen(address, timeout=1) as response:
                    self.assertIn(b"Execution monitor", response.read())
                with urlopen(address + "/state", timeout=1) as response:
                    state = json.load(response)
                    self.assertEqual(state["status"], "loading")
                    self.assertIn("reference cache", state["initialization_phase"])
                release.set()
                app.shutdown.set()
                app.worker.join(timeout=2)
                self.assertEqual(app.snapshot()["status"], "ready")
        finally:
            release.set()
            app.shutdown.set()
            server.shutdown()
            server.server_close()
            thread.join(timeout=1)

    def test_browser_options_apply_only_between_runs_without_changing_config(self):
        config = {"rynnbrain": {"num_frames": 8}}
        app = ReplayApplication(config, [Path("test")], 10., False)
        app.detector = Mock()
        app.update(status="ready")
        with self.assertRaisesRegex(ValueError, "preview"):
            app.start(0, match_preview="false")
        for interval, mode in [(0, "cached"), (float("nan"), "cached"), (True, "cached"),
                               ("10", "cached"), (10, "unknown")]:
            with self.assertRaises(ValueError):
                app.start(0, interval, mode)
            self.assertEqual(app.interval, 10.)
            self.assertFalse(app.verify)
            self.assertTrue(app.jobs.empty())
        for mode, verify, diagnose in [("diagnostic", True, True), ("strict", True, False),
                                       ("cached", False, False)]:
            app.start(0, 5., mode)
            self.assertEqual(app.jobs.get_nowait(), 0)
            self.assertEqual((app.verify, app.diagnose), (verify, diagnose))
            with self.assertRaises(ValueError):
                app.start(0, 20., "strict")
            self.assertEqual(app.interval, 5.)
            self.assertEqual(app.snapshot()["replay_mode"], mode)
            app.update(status="completed")
        self.assertEqual(config, {"rynnbrain": {"num_frames": 8}})

    def test_replay_follows_timestamps_and_mailbox_keeps_only_latest(self):
        now = [0.]
        deadlines = []
        class Stop:
            def is_set(self):
                return False
            def wait(self, duration):
                now[0] += duration
                deadlines.append(now[0])
                return False
        image = Image.new("RGB", (2, 2))
        publish = Mock()
        feed = ReplayFeed(iter([(0, 1., image), (1, 2., image), (2, 3., image)]), publish, Stop(), clock=lambda: now[0])
        feed.run()
        self.assertEqual(deadlines, [1., 2., 3.])
        self.assertEqual(publish.call_count, 3)
        self.assertEqual(feed.next_frame(0.)[:2], (2, 3.))
        self.assertIsNone(feed.next_frame(3.1))

    def test_wait_can_be_cancelled_and_reader_errors_are_visible(self):
        stop = threading.Event()
        feed = ReplayFeed(iter([]), Mock(), stop)
        result = []
        thread = threading.Thread(target=lambda: result.append(feed.next_frame(0)))
        thread.start()
        stop.set()
        thread.join(timeout=1)
        self.assertEqual(result, [None])
        def broken():
            raise ValueError("bad camera topic")
            yield
        feed = ReplayFeed(broken(), Mock(), threading.Event())
        feed.run()
        with self.assertRaisesRegex(RuntimeError, "bad camera"):
            feed.next_frame(0)

    def test_reset_rejects_training_bags_before_modifying_cache(self):
        detector = OnlineDetector.__new__(OnlineDetector)
        detector.excluded = {"training"}
        detector.cache = Mock()
        with self.assertRaisesRegex(ValueError, "held-out"):
            detector.reset(Path("/data/training"))
        detector.cache.reset.assert_not_called()
        detector.reset(Path("/data/test"))
        detector.cache.reset.assert_called_once()

    def test_replay_saves_incrementally_and_new_executions_do_not_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            config = {"output_dir": directory, "rynnbrain": {"camera_topics": ["top"]}}
            app = ReplayApplication(config, [Path("test")], .01, True)
            count = [0]
            def reset(_bag):
                count[0] = 0
            def append(_image, timestamp, _verify):
                index = count[0]
                count[0] += 1
                return dict(sample_index=index, timestamp_sec=timestamp, anomaly_score=.2,
                    threshold=.5, above_threshold=False, cached_feature_sec=.001,
                    classifier_sec=.0001, peak_gpu_allocated_bytes={}, score_kind="failure_probability",
                    verification={"within_tolerance": True}), np.array([1., 2.]), np.array([1., 2.])
            app.detector = SimpleNamespace(reset=reset, append=append, frame_count=2,
                path=Path("existing.npz"), score_source="test", signature={}, knn=False)
            frames = lambda *a, **kw: iter([(i, i * .02, Image.new("RGB", (4, 4))) for i in range(5)])
            with patch("rosbag_io.iter_rosbag_image_frames", side_effect=frames), \
                 patch("rynnbrain_vlm.online_replay.save_plot") as plot:
                for _ in range(2):
                    app.stop.clear()
                    app.run_bag(Path("test"))
                    self.assertEqual(app.snapshot()["status"], "completed")
                self.assertEqual(plot.call_count, 2)
            results = list(Path(directory).glob("online_replay/*/test/online_results.json"))
            self.assertEqual(len(results), 2)
            for path in results:
                saved = json.loads(path.read_text())
                self.assertEqual(saved["status"], "completed")
                self.assertEqual(len(saved["rows"]), 2)
                self.assertTrue((path.parent / "online_scores.csv").is_file())
                self.assertTrue((path.parent / "frame_02_uncached.npy").is_file())

    def test_verification_mismatch_stops_and_keeps_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            config = {"output_dir": directory, "rynnbrain": {"camera_topics": ["top"]}}
            app = ReplayApplication(config, [Path("test")], 1., True)
            row = dict(timestamp_sec=0., anomaly_score=.2, threshold=.5, cached_feature_sec=.001,
                       peak_gpu_allocated_bytes={}, verification={"within_tolerance": False})
            app.detector = SimpleNamespace(reset=Mock(), append=Mock(return_value=(row, np.ones(2), np.zeros(2))),
                frame_count=8, path=Path("existing.npz"), score_source="test", signature={}, knn=False)
            with patch("rosbag_io.iter_rosbag_image_frames", return_value=iter([(0, 0., Image.new("RGB", (4, 4)))])), \
                 patch("rynnbrain_vlm.online_replay.save_plot"):
                app.run_bag(Path("test"))
            self.assertEqual(app.snapshot()["status"], "error")
            app.detector.append.assert_called_once()
            saved = json.loads(next(Path(directory).glob("online_replay/*/test/online_results.json")).read_text())
            self.assertEqual(saved["status"], "error")
            self.assertIn("verification failed", saved["error"])

    def test_matching_preview_is_saved_without_changing_classifier_score(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            thumbnail = root / 'nominal.jpg'
            Image.new('RGB', (4, 4), 'green').save(thumbnail)
            config = {'output_dir': directory, 'rynnbrain': {'camera_topics': ['top']}}
            app = ReplayApplication(config, [Path('test')], 1., False)
            match = {'status': 'matched', 'stage': 1, 'best': {'image_file': str(thumbnail)}}
            def append(image, timestamp, verify):
                return dict(timestamp_sec=timestamp, anomaly_score=.2, threshold=.5,
                            above_threshold=False, cached_feature_sec=.001, verification=None,
                            visual_matching=match), np.ones(2), None
            detector = SimpleNamespace(reset=Mock(), append=append, frame_count=1,
                path=Path('existing.npz'), score_source='test', signature={}, knn=False,
                cache=SimpleNamespace(last_visual_descriptor=np.array([1., 0.])))
            def prepare(bag, progress, stop):
                detector.capture_visual = True
                return {'status': 'ready'}
            detector.prepare_matching = prepare
            app.detector = detector
            with patch('rosbag_io.iter_rosbag_image_frames', return_value=iter([(0, 0., Image.new('RGB', (4, 4)))])), \
                 patch('rynnbrain_vlm.online_replay.save_plot'), \
                 patch('rynnbrain_vlm.visual_matching.read_corrected_failure_annotation', return_value={'status': 'available', 'time_sec': 0.}):
                app.run_bag(Path('test'))
            saved_path = next(root.glob('online_replay/*/test/online_results.json'))
            saved = json.loads(saved_path.read_text())
            self.assertEqual(saved['status'], 'completed')
            self.assertEqual(saved['rows'][0]['anomaly_score'], .2)
            self.assertFalse(saved['rows'][0]['above_threshold'])
            self.assertEqual(app.matched_jpeg, thumbnail.read_bytes())
            self.assertTrue((saved_path.parent / 'frame_01_nominal.jpg').is_file())
            np.testing.assert_array_equal(np.load(saved_path.parent / 'frame_01_visual.npy'), [1., 0.])

    def test_missing_stages_disable_preview_but_not_scoring(self):
        with tempfile.TemporaryDirectory() as directory:
            app = ReplayApplication({'output_dir': directory, 'rynnbrain': {'camera_topics': ['top']}},
                                    [Path('test')], 1., False)
            row = dict(timestamp_sec=0., anomaly_score=.2, threshold=.5, cached_feature_sec=.001,
                       verification=None)
            app.detector = SimpleNamespace(reset=Mock(), prepare_matching=Mock(side_effect=ValueError('Missing stage')),
                append=Mock(return_value=(row, np.ones(2), None)), frame_count=1,
                path=Path('existing.npz'), score_source='test', signature={}, knn=False)
            with patch('rosbag_io.iter_rosbag_image_frames', return_value=iter([(0, 0., Image.new('RGB', (4, 4)))])), \
                 patch('rynnbrain_vlm.online_replay.save_plot'):
                app.run_bag(Path('test'))
            self.assertEqual(app.snapshot()['status'], 'completed')
            self.assertFalse(app.detector.capture_visual)
            self.assertIn('Missing stage', app.snapshot()['matching_status'])
            app.detector.append.assert_called_once()

    def test_start_rejects_overlap_and_unknown_bags(self):
        app = ReplayApplication({}, [Path("test")], 10., False)
        app.detector = Mock()
        with self.assertRaisesRegex(ValueError, "initialization"):
            app.start(0)
        app.update(status="ready")
        with self.assertRaisesRegex(ValueError, "listed bag"):
            app.start(-1)
        app.start(0)
        with self.assertRaisesRegex(ValueError, "current execution"):
            app.start(0)

    def test_detector_switch_recovers_from_vlm_failure_and_uses_own_exclusions(self):
        app = ReplayApplication({}, [Path('vlm_training'), Path('dino_training')], 10., False)
        app.update(status='error', excluded_bag_indices=[0])
        # The selected detector can be started even if the VLM failed to load.
        app.start(0, mode='strict', detector_type='dino')
        self.assertEqual(app.jobs.get_nowait(), 0)
        self.assertFalse(app.verify)
        dino = SimpleNamespace(excluded={'dino_training'}, note='dino', score_source='dino',
                               score_kind='dino_distance', knn=True, frame_count=8, catalogue=lambda: [])
        with patch('rynnbrain_vlm.dino_cameras.DinoCameraReplay', return_value=dino), \
             patch('rynnbrain_vlm.online_replay.OnlineDetector', side_effect=AssertionError('No VLM load')):
            app.switch_detector('dino')
        self.assertIs(app.detector, dino)
        self.assertEqual(app.snapshot()['excluded_bag_indices'], [1])
        self.assertEqual(app.snapshot()['detector_type'], 'dino')
        self.assertEqual(app.snapshot()['score_kind'], 'dino_distance')
        app.update(status='completed')
        with self.assertRaisesRegex(ValueError, 'training/reference'):
            app.start(1, detector_type='dino')


if __name__ == "__main__":
    unittest.main()
