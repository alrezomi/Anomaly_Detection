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

from rynnbrain_vlm.online_replay import ReplayFeed, ReplayApplication, OnlineDetector, make_handler


class OnlineReplayTests(unittest.TestCase):
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
            request = Request(address + "/start", data=b'{"bag_index":0}', headers={"Origin": address})
            with urlopen(request) as response:
                self.assertEqual(response.status, 200)
            self.assertEqual(app.jobs.get_nowait(), 0)
            with urlopen(Request(address + "/stop", data=b"{}")):
                self.assertTrue(app.stop.is_set())
            with self.assertRaises(HTTPError) as error:
                urlopen(Request(address + "/start", data=b"{}", headers={"Origin": "http://elsewhere"}))
            self.assertEqual(error.exception.code, 403)
            with self.assertRaises(HTTPError) as error:
                urlopen(Request(address + "/start", data=b"[]"))
            self.assertEqual(error.exception.code, 400)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=1)

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
