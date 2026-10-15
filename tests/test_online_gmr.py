import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from time_series_gmr_scripts.online_gmr import (
    CausalSamples, GmrOnlineDetector, GmrOnlineFeed, StageClock,
    TrailingAlarm, estimate_phase, signal_values,
)


def execution(offset=0., end=2.3, tail=0.):
    events = [(0., 'stage', '3'), (.1, 'stage', '2'), (.3, 'stage', '1'),
              (.8, 'stage', 'Error'), (1.3, 'stage', '2'), (end, 'stage', '3')]
    events += [(round(float(t), 9), 'joints', {'joint_position': np.sin(t) + offset + (tail if t > 1.4 else 0.),
                                    'joint_effort': 999.}) for t in np.arange(0., 3., .05)]
    return iter(sorted(events, key=lambda e: e[0]))


class OnlineGmrTests(unittest.TestCase):
    def test_stage_filter_end_and_failures(self):
        sampler = CausalSamples(['joints'])
        packets = list(sampler.packets(execution()))
        rows = [p for _, kind, p in packets if kind == 'sample']
        self.assertEqual(sampler.clock.boundaries, {1: .3, 2: 1.3, 3: 2.3})
        self.assertEqual(len(rows), 20)
        self.assertTrue(all(.3 <= r['time'] < 2.3 for r in rows))
        self.assertTrue(all(r['stage'] == (1 if r['time'] < 1.3 - 1e-8 else 2) for r in rows))
        self.assertEqual(sampler.clock.failures, [{'time': .8, 'label': 'Error'}])
        self.assertNotIn('joint_effort', sampler.columns)
        clock = StageClock()
        for t, label in enumerate(['2', '3', '1', '1', '2', '1', '2', '3']):
            clock.observe(t, label)
        self.assertEqual(clock.boundaries, {1: 2, 2: 4, 3: 7})

    def test_future_tail_and_end_cannot_change_prefix(self):
        def prefix(events):
            return [row for t, k, row in CausalSamples(['joints']).packets(events)
                    if k == 'sample' and t < 1.4]
        self.assertEqual(prefix(execution()), prefix(execution(end=2.8, tail=100.)))
        self.assertAlmostEqual(estimate_phase(1, .5, [1., 2.]), .25)
        self.assertAlmostEqual(estimate_phase(2, 1., [1., 2.]), .75)
        self.assertEqual(estimate_phase(1, 100., [1., 2.]), .5)

    def test_missing_markers_are_errors_and_early_end_is_valid(self):
        with self.assertRaisesRegex(ValueError, 'No stage 1'):
            list(CausalSamples(['gripper']).packets(iter([(0., 'stage', '3')])))
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            list(CausalSamples(['gripper']).packets(iter([(0., 'stage', '1')])))
        samples = CausalSamples(['gripper'])
        list(samples.packets(iter([(0., 'stage', '1'), (.5, 'stage', '3')])))
        self.assertEqual(samples.clock.stage, 3)
        with self.assertRaisesRegex(ValueError, 'Nominal'):
            samples.clock.durations()
        clock = StageClock()
        for t, label in [(0., '1'), (1., '2'), (1., '3')]:
            clock.observe(t, label)
        with self.assertRaisesRegex(ValueError, 'positive duration'):
            clock.durations()

    def test_stop_closes_stream_while_waiting_for_start(self):
        stop = threading.Event()
        closed = []
        def stream():
            try:
                yield 0., 'stage', '2'
                stop.set()
                yield .1, 'stage', '3'
            finally:
                closed.append(True)
        self.assertEqual(list(CausalSamples(['joints'], stop).packets(stream())), [])
        self.assertEqual(closed, [True])

    def test_combined_channels_hold_past_values_and_reject_frame_changes(self):
        pose = dict(position_x=1., position_y=0., position_z=0.,
                    orientation_x=0., orientation_y=0., orientation_z=0., orientation_w=1., pose_frame='base')
        events = [(0., 'stage', '1'), (0., 'pose', pose), (0., 'gripper', {'gripper_position': .1}),
                  (.25, 'gripper', {'gripper_position': .9}), (.5, 'stage', '2'), (.6, 'stage', '3')]
        rows = [r for _, k, r in CausalSamples(['pose', 'gripper']).packets(iter(events)) if k == 'sample']
        self.assertEqual([r['gripper_position'] for r in rows], [.1, .1, .1, .9, .9, .9])
        negative_q = {**pose, 'orientation_w': -1.}
        self.assertEqual(signal_values('pose', pose), signal_values('pose', negative_q))
        with self.assertRaisesRegex(ValueError, 'reference frame'):
            list(CausalSamples(['pose']).packets(iter([(0., 'stage', '1'), (0., 'pose', pose),
                 (.2, 'pose', {**pose, 'pose_frame': 'other'})])))
        with self.assertRaisesRegex(ValueError, 'stale'):
            list(CausalSamples(['pose']).packets(iter([(0., 'stage', '1'), (0., 'pose', pose),
                 (2., 'stage', '3')])) )

    def test_trailing_filter_cannot_anticipate_future_alarm(self):
        def scores(sequence):
            alarm = TrailingAlarm(3, 3, 2)
            return [alarm.append(value) for value in sequence]
        self.assertEqual(scores([.1, .1, .1, 20., 20.])[:3], scores([.1, .1, .1]))
        self.assertEqual(scores([.1, 20., 20.])[-1], (20., True))

    def test_training_never_reads_test_signals_and_cache_reuses_nominal_model(self):
        with tempfile.TemporaryDirectory() as directory:
            config = dict(output_dir=directory, nominal_bags=['nominal_a', 'nominal_b'], camera_topics=[])
            def events(bag, *_args):
                self.assertNotEqual(str(bag), 'test', 'Preparation must not read test observations')
                return execution(offset=.02 if bag == 'nominal_b' else 0.)
            with patch('rynnbrain_vlm.replay_detectors.gmr_signal_topics', return_value={'joints': '/joint_states'}), \
                 patch('rynnbrain_vlm.visual_matching._source_identity', return_value=[]), \
                 patch('rosbag_io.iter_ts_replay_events', side_effect=events) as reader:
                detector = GmrOnlineDetector(config, Mock())
                detector.signal_mode = 'joints'
                detector.reset('test')
                prepared = detector.prepare_replay(Mock(), threading.Event())
                self.assertFalse(prepared['offline'])
                self.assertEqual(reader.call_count, 2)
                detector.pending = dict(stage=1, stage_elapsed=.4, joint_position=.65)
                original = detector.append(None, .7)[0]['anomaly_score']
                restored = GmrOnlineDetector(config, Mock())
                restored.signal_mode = 'joints'
                restored.reset('test')
                restored.prepare_replay(Mock(), threading.Event())
                self.assertEqual(reader.call_count, 2)
                restored.pending = detector.pending
                self.assertAlmostEqual(restored.append(None, .7)[0]['anomaly_score'], original, places=12)

    def test_rosbag_reader_decodes_lab_gripper_schema_and_preserves_clock(self):
        from rosbags.rosbag2 import Writer
        from rosbag_io import ROS2_TYPESTORE as store, ROBOTIQ_INPUT_TYPE, iter_ts_replay_events
        with tempfile.TemporaryDirectory() as directory:
            bag = Path(directory) / 'bag'
            with Writer(bag, version=9) as writer:
                stage = writer.add_connection('/stage', 'std_msgs/msg/String', typestore=store)
                grip = writer.add_connection('/grip', ROBOTIQ_INPUT_TYPE, typestore=store)
                events = [(0., stage, store.types['std_msgs/msg/String']('3')),
                          (.1, stage, store.types['std_msgs/msg/String']('1')),
                          (.2, grip, store.types[ROBOTIQ_INPUT_TYPE](1, 1, 1, 0, 0, 255, 128, 0)),
                          (.5, stage, store.types['std_msgs/msg/String']('2')),
                          (.8, stage, store.types['std_msgs/msg/String']('3'))]
                for t, connection, msg in events:
                    writer.write(connection, 1_000_000_000 + int(t * 1e9), store.serialize_cdr(msg, connection.msgtype))
            sampler = CausalSamples(['gripper'])
            packets = list(sampler.packets(iter_ts_replay_events(bag, {'gripper': '/grip'}, '/stage')))
            rows = [r for _, k, r in packets if k == 'sample']
            self.assertEqual(len(rows), 6)
            self.assertTrue(all(r['gripper_position'] == 128 / 255 for r in rows))
            self.assertEqual(sampler.clock.boundaries, {1: .1, 2: .5, 3: .8})

    def test_gui_scores_progressively_without_offline_test_annotation(self):
        from rynnbrain_vlm.online_replay import ReplayApplication
        with tempfile.TemporaryDirectory() as directory:
            config = dict(output_dir=directory, rynnbrain={'camera_topics': ['top'], 'num_frames': 8})
            app = ReplayApplication(config, [Path('test')], 10., False)
            app.update(detector_type='gmr')
            detector = GmrOnlineDetector(config, Mock())
            app.detector = detector
            now = [0.]
            def make_feed(publish, stop, update):
                sampler = detector.sampler = CausalSamples(['joints'])
                # Run faster than wall time while preserving the recorded ordering.
                feed = GmrOnlineFeed(detector, sampler.packets(execution()), publish, stop, update,
                                     clock=lambda: now[0])
                stop.wait = lambda delay: now.__setitem__(0, now[0] + delay) or False
                return feed
            def score(_image, stamp, _verify):
                prior_rows = app.snapshot()['rows']
                if prior_rows:
                    self.assertAlmostEqual(prior_rows[-1]['timestamp_sec'], stamp - .1)
                return dict(timestamp_sec=stamp, anomaly_score=.2, threshold=1.,
                    above_threshold=False, cached_feature_sec=.001, verification=None,
                    score_kind='gmr_epsilon', alarm_filtered=False, signal_time_sec=stamp), np.ones(2), None
            with patch.object(detector, 'prepare_replay', return_value={}), \
                 patch.object(detector, 'create_feed', side_effect=make_feed), \
                 patch.object(detector, 'append', side_effect=score), \
                 patch('rynnbrain_vlm.visual_matching.read_corrected_failure_annotation') as offline_annotation, \
                 patch('rynnbrain_vlm.online_replay.save_plot'):
                app.run_bag(Path('test'))
            offline_annotation.assert_not_called()
            result = app.snapshot()
            self.assertEqual(result['status'], 'completed', result.get('error'))
            self.assertEqual(len(result['rows']), 20)
            self.assertEqual(result['gmr_stage'], 3)
            self.assertTrue(all(r['score_computation'] == 'causal_ts' for r in result['rows']))
            saved = json.loads(next(Path(directory).glob('online_replay/*/test/online_results.json')).read_text())
            self.assertEqual(saved['stage_boundaries_sec'], {'1': .3, '2': 1.3, '3': 2.3})
            self.assertEqual(len(saved['rows']), 20)


if __name__ == '__main__':
    unittest.main()
