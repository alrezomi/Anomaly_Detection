import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from rynnbrain_vlm.replay_policy import replay_settings, observation_interval
from rynnbrain_vlm.online_replay import ReplayApplication


class ReplayPolicyTests(unittest.TestCase):
    def test_defaults_keep_constant_sampling_eight_observations_and_no_alarm_stop(self):
        self.assertEqual(replay_settings({}), dict(interval_mode="constant", max_observations=8, stop_on_alarm=False))
        self.assertEqual(replay_settings({"rynnbrain": {"num_frames": 4}})["max_observations"], 4)
        self.assertEqual(observation_interval(replay_settings({}), None, 12.), 12.)

    def test_adaptive_boundaries_and_invalid_scores(self):
        policy = replay_settings({"rynnbrain": {"online_replay": {"interval_mode": "adaptive", "max_observations": None}}})
        self.assertIsNone(policy["max_observations"])
        for score, interval in [(0., 10.), (.29999, 10.), (.3, 7.), (.49999, 7.),
                                (.5, 5.), (.59999, 5.), (.6, 3.), (1., 3.)]:
            with self.subTest(score=score):
                self.assertEqual(observation_interval(policy, score, 20.), interval)
        for score in (None, float("nan"), float("inf"), -.1, 1.1):
            with self.subTest(score=score), self.assertRaisesRegex(ValueError, "failure score"):
                observation_interval(policy, score, 10.)

    def test_reject_invalid_policy_values(self):
        for options in (None, {"interval_mode": "unknown"}, {"max_observations": True},
                        {"max_observations": 0}, {"max_observations": 2.5}, {"stop_on_alarm": "true"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                replay_settings({"rynnbrain": {"online_replay": options}})

    def run_replay(self, options, scores, threshold=.9, interval=10.):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        config = {"output_dir": temporary.name, "rynnbrain": {"camera_topics": ["top"],
                  "num_frames": 8, "inference": {"feature_only": True}, "online_replay": options}}
        before = json.dumps(config, sort_keys=True)
        app = ReplayApplication(config, [Path("heldout")], interval, False)
        app.update(match_preview=False)
        policy = replay_settings(config)
        observed = []
        def append(image, timestamp, verify):
            score = scores[len(observed)]
            observed.append(timestamp)
            return dict(timestamp_sec=timestamp, anomaly_score=score, threshold=threshold,
                        above_threshold=score >= threshold, cached_feature_sec=.001,
                        score_kind="failure_probability", verification=None), np.ones(2), None
        app.detector = SimpleNamespace(reset=Mock(), append=append, frame_count=8,
            observation_limit=policy["max_observations"], path=Path("existing.npz"),
            score_source="onset_prefix_classifier", signature={"num_frames": 8}, knn=False)
        image = Image.new("RGB", (4, 4))
        class Feed:
            def __init__(self, frames, publish, stop):
                self.origin = time.monotonic()
                self.requests = []
                self.index = 0
            def run(self):
                pass
            def next_frame(self, earliest):
                self.requests.append(earliest)
                if self.index >= len(scores):
                    return None
                self.index += 1
                return self.index, earliest, image
        feeds = []
        def feed_factory(*args):
            feed = Feed(*args)
            feeds.append(feed)
            return feed
        with patch("rosbag_io.iter_rosbag_image_frames", return_value=iter([])), \
             patch("rynnbrain_vlm.online_replay.ReplayFeed", side_effect=feed_factory), \
             patch("rynnbrain_vlm.online_replay.save_plot") as plot, \
             patch("rynnbrain_vlm.visual_matching.read_corrected_failure_annotation", return_value={}):
            app.run_bag(Path("heldout"))
        plot.assert_called_once()
        self.assertEqual(json.dumps(config, sort_keys=True), before)
        self.assertTrue(app.stop.is_set())
        result = next(Path(temporary.name).glob("online_replay/*/heldout/online_results.json"))
        self.assertTrue(result.with_name("online_scores.csv").is_file())
        return app.snapshot(), json.loads(result.read_text()), feeds[0], result.parent

    def test_adaptive_replay_passes_eight_and_stops_at_configured_threshold(self):
        options = dict(interval_mode="adaptive", max_observations=None, stop_on_alarm=True)
        scores = [.1, .3, .5, .6, .2, .2, .2, .2, .2, .9, .95]
        state, saved, feed, output = self.run_replay(options, scores)
        self.assertEqual(state["status"], "stopped", state.get("error"))
        self.assertEqual(state["termination_reason"], "alarm_threshold")
        self.assertEqual(len(saved["rows"]), 10)
        self.assertEqual(feed.requests, [0., 10., 17., 22., 25., 35., 45., 55., 65., 75.])
        self.assertEqual(saved["alarm_timestamp_sec"], 75.)
        self.assertEqual(saved["rows"][-1]["anomaly_score"], .9)
        self.assertIsNone(saved["rows"][-1]["next_interval_sec"])
        self.assertIsNone(saved["max_frames"])
        self.assertEqual(saved["comparison_signature"]["num_frames"], 8)
        self.assertTrue((output / "frame_10.npy").is_file())
        self.assertFalse((output / "frame_11.npy").exists())

    def test_constant_unlimited_replay_reaches_end_and_ignores_alarms_when_disabled(self):
        state, saved, feed, _ = self.run_replay(dict(max_observations=None), [.95] * 12, interval=4.)
        self.assertEqual(state["status"], "completed", state.get("error"))
        self.assertEqual(len(saved["rows"]), 12)
        self.assertEqual(saved["termination_reason"], "end_of_recording")
        self.assertEqual(feed.requests, list(np.arange(13) * 4.))
        self.assertTrue(all(row["next_interval_sec"] == 4. for row in saved["rows"]))

    def test_unchanged_defaults_limit_to_eight_even_when_scores_cross_threshold(self):
        state, saved, feed, _ = self.run_replay({}, [.95] * 12)
        self.assertEqual(state["status"], "completed", state.get("error"))
        self.assertEqual(len(saved["rows"]), 8)
        self.assertEqual(saved["termination_reason"], "observation_limit")
        self.assertEqual(len(feed.requests), 8)
