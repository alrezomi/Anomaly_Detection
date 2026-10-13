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

from rynnbrain_vlm.online_replay import (ReplayFeed, ReplayApplication, OnlineDetector, make_handler,
                                       finish_verification, verification_failure_message)


class OnlineReplayTests(unittest.TestCase):
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
            request = Request(address + "/start", data=b'{"bag_index":0,"interval_sec":5,"mode":"diagnostic"}', headers={"Origin": address})
            with urlopen(request) as response:
                self.assertEqual(response.status, 200)
            self.assertEqual(app.jobs.get_nowait(), 0)
            self.assertEqual(app.interval, 5.)
            self.assertTrue(app.diagnose)
            self.assertTrue(app.verify)
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


if __name__ == "__main__":
    unittest.main()
