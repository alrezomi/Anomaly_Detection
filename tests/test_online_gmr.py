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
    TrailingAlarm, estimate_phase, signal_values, channel_diagnostics,
)
from time_series_gmr_scripts.anomaly_detection import GMRAnomalyDetector


def execution(offset=0., end=2.3, tail=0.):
    events = [(0., 'stage', '3'), (.1, 'stage', '2'), (.3, 'stage', '1'),
              (.8, 'stage', 'Error'), (1.3, 'stage', '2'), (end, 'stage', '3')]
    events += [(round(float(t), 9), 'joints', {'joint_position': np.sin(t) + offset + (tail if t > 1.4 else 0.),
                                    'joint_effort': 999.}) for t in np.arange(0., 3., .05)]
    return iter(sorted(events, key=lambda e: e[0]))


class OnlineGmrTests(unittest.TestCase):
    @staticmethod
    def correlated_model():
        model = GMRAnomalyDetector(['phase'], ['a_position', 'b_position'], n_components=1)
        model.scaler.mean_ = np.array([0., 10., -20.])
        model.scaler.scale_ = np.array([1., 2., 3.])
        model.gmm.weights_ = np.array([1.])
        model.gmm.means_ = np.array([[0., 0., 0.]])
        model.gmm.covariances_ = np.array([[[1., 0., 0.], [0., 1., .8], [0., .8, 1.]]])
        return model

    def test_channel_envelope_is_in_original_units_and_is_not_independent_alarm(self):
        model = self.correlated_model()
        scaled = np.array([.25, 1., -1.])
        distance, component = model._score_scaled_point(scaled)
        channels = channel_diagnostics(model, scaled, 2.)
        self.assertGreater(distance, 2.)  # Opposite deviations violate the learned correlation.
        self.assertEqual(channels['a_position']['observed'], 12.)
        self.assertEqual(channels['b_position']['observed'], -23.)
        self.assertEqual(channels['a_position']['expected'], 10.)
        self.assertEqual(channels['b_position']['expected'], -20.)
        for channel in channels.values():
            self.assertLess(channel['lower'], channel['observed'])
            self.assertGreater(channel['upper'], channel['observed'])
            self.assertAlmostEqual(channel['standardized_deviation'], 1., places=4)
        self.assertAlmostEqual(channels['b_position']['upper'] + 20.,
                               (channels['a_position']['upper'] - 10.) * 1.5)
        # The projected endpoint lies on the joint ellipsoid when other channels
        # move along their correlation, not when other channels are held fixed.
        _, covariance = model._conditional_output_distribution(scaled[:1], component)
        covariance += np.eye(2) * model.reg_covar
        endpoint = 2. * covariance[:, 0] / np.sqrt(covariance[0, 0])
        self.assertAlmostEqual(float(endpoint @ np.linalg.pinv(covariance) @ endpoint), 4.)

    def test_each_sensor_has_its_own_threshold_and_alarm(self):
        import pandas as pd
        from time_series_gmr_scripts import run_time_series_test as settings
        force = GMRAnomalyDetector(['phase'], ['force_z'], n_components=1)
        force.scaler.mean_, force.scaler.scale_ = np.zeros(2), np.ones(2)
        force.gmm.weights_, force.gmm.means_, force.gmm.covariances_ = np.array([1.]), np.zeros((1, 2)), np.eye(2)[None]
        detector = GmrOnlineDetector({'output_dir': '.', 'nominal_bags': []}, Mock())
        detector.reset('test')
        detector.sensors = {'position': ['a_position', 'b_position'], 'force': ['force_z']}
        detector.models = {'position': self.correlated_model(), 'force': force}
        detector.durations = [1., 1.]
        detector.thresholds = {
            'position': pd.DataFrame({'phase_center': [0., 1.], 'threshold_mahalanobis': [1., 3.]}),
            'force': pd.DataFrame({'phase_center': [0., 1.], 'threshold_mahalanobis': [2., 2.]})}
        legacy_alarm = TrailingAlarm(settings.SCORE_SMOOTH_WINDOW, settings.MAJORITY_WINDOW, settings.MAJORITY_VOTES)
        for i in range(20):
            t = i / 10
            stage = 1 if t < 1 else 2
            elapsed = t if stage == 1 else t - 1
            detector.pending = dict(stage=stage, stage_elapsed=elapsed, a_position=12., b_position=-23., force_z=0.)
            row, _, _ = detector.append(None, t)
            position = row['sensors']['position']
            smoothed, alarm = legacy_alarm.append(position['distance'] / position['threshold'])
            # Position keeps the existing per-sensor smoothing and voting; force is nominal.
            self.assertEqual(position['epsilon_smooth'], smoothed)
            self.assertEqual(row['alarm_filtered'], alarm)
            self.assertEqual(row['alarm_sources'], ['position'] if alarm else [])
            self.assertFalse(row['sensors']['force']['alarm'])
            self.assertAlmostEqual(row['sensors']['force']['distance'], 0.)
            self.assertEqual(row['score_sensor'], 'position')
            self.assertEqual(row['anomaly_score'], position['epsilon_raw'])
            self.assertEqual(row['threshold'], 1.)
            self.assertEqual(row['score_kind'], 'gmr_sensor_ratio')
            self.assertEqual(set(row['channels']), {'a_position', 'b_position', 'force_z'})
        self.assertTrue(row['alarm_filtered'])

    def test_force_only_alarms_above_its_safety_limit(self):
        import pandas as pd
        detector = GmrOnlineDetector({'output_dir': '.', 'nominal_bags': []}, Mock())
        detector.reset('test')
        detector.sensors = {'position': ['a_position', 'b_position']}
        detector.models = {'position': self.correlated_model()}
        detector.thresholds = {'position': pd.DataFrame({'phase_center': [0., 1.], 'threshold_mahalanobis': [100., 100.]})}
        detector.durations, detector.force_limit = [1., 1.], 50.
        for fz, alarm in [(-30., False), (-60., True)]:
            detector.pending = dict(stage=1, stage_elapsed=.5, a_position=10., b_position=-20.,
                                    force_x=0., force_y=0., force_z=fz)
            row, _, _ = detector.append(None, .5)
            self.assertEqual(row['sensors']['force']['distance'], abs(fz))
            self.assertEqual(row['sensors']['force']['alarm'], alarm)
            self.assertEqual(row['alarm_sources'], ['force'] if alarm else [])
            self.assertIsNone(row['channels']['force_z']['expected'])

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
        self.assertEqual(set(signal_values('pose', pose)), {'position_x', 'position_y', 'position_z'})
        with self.assertRaisesRegex(ValueError, 'reference frame'):
            list(CausalSamples(['pose']).packets(iter([(0., 'stage', '1'), (0., 'pose', pose),
                 (.2, 'pose', {**pose, 'pose_frame': 'other'})])))
        with self.assertRaisesRegex(ValueError, 'stale'):
            list(CausalSamples(['pose']).packets(iter([(0., 'stage', '1'), (0., 'pose', pose),
                 (2., 'stage', '3')])) )

    def test_force_is_tick_averaged_and_zeroed_at_stage_one(self):
        def wrench(fz):
            return dict(force_x=0., force_y=0., force_z=fz, torque_x=9., torque_y=9., torque_z=9.)
        pose = dict(position_x=0., position_y=0., position_z=0., pose_frame='base')
        events = [(0., 'force', wrench(100.)), (.01, 'stage', '1'), (.01, 'pose', pose),
                  (.04, 'force', wrench(10.)), (.08, 'force', wrench(12.)),
                  (.14, 'force', wrench(15.)), (.18, 'force', wrench(17.)), (.25, 'stage', '2'), (.3, 'stage', '3')]
        rows = [r for _, k, r in CausalSamples(['pose', 'force']).packets(iter(events)) if k == 'sample']
        # Pre-stage force is ignored. Ticks at .11 and .21 average 10/12 then 15/17;
        # the first averaged tick is the zero.
        self.assertEqual([r['force_z'] for r in rows], [0., 5.])
        self.assertNotIn('torque_x', rows[0])

    def test_release_is_first_opening_after_grasp_and_waits_past_stage_three(self):
        def run(opens):
            pose = lambda t: dict(position_x=t, position_y=0., position_z=0., pose_frame='base')
            events = [(0., 'stage', '1'), (.5, 'stage', '2'), (1., 'stage', '3')]
            events += [(round(t, 9), 'pose', pose(t)) for t in np.arange(0., 1.6, .05)]
            events += [(.02, 'gripper', {'gripper_position': .01}), (.2, 'gripper', {'gripper_position': .9}),
                       (1.1, 'gripper', {'gripper_position': .9})]
            if opens:
                events += [(1.32, 'gripper', {'gripper_position': .05})]
            sampler = CausalSamples(['pose'], release=True)
            packets = list(sampler.packets(iter(sorted(events, key=lambda e: e[0]))))
            return sampler, packets
        sampler, packets = run(True)
        samples = [p for _, k, p in packets if k == 'sample']
        self.assertTrue(all(r['time'] < 1. for r in samples))
        # The gripper is shown with each sample but is not a GMR channel.
        self.assertNotIn('gripper_position', sampler.columns)
        self.assertEqual(samples[-1]['gripper_position'], .9)
        self.assertEqual([k for _, k, _ in packets][-1], 'release')
        self.assertEqual(sampler.release['stage'], 3)
        self.assertAlmostEqual(sampler.release['position_x'], 1.3)
        sampler, packets = run(False)
        self.assertIsNone(sampler.release)
        self.assertEqual(sampler.clock.stage, 3)

    def test_scoring_continues_for_the_window_after_stage_three(self):
        pose = lambda t: dict(position_x=t, position_y=0., position_z=0., pose_frame='base')
        events = [(0., 'stage', '1'), (.5, 'stage', '2'), (1., 'stage', '3')]
        events += [(round(t, 9), 'pose', pose(t)) for t in np.arange(0., 3., .05)]
        sampler = CausalSamples(['pose'], post_sec=.5)
        samples = [p for _, k, p in sampler.packets(iter(sorted(events, key=lambda e: e[0]))) if k == 'sample']
        after = [r for r in samples if r['stage'] == 3]
        self.assertEqual(len(after), 6)  # Ticks at 1.0 ... 1.5 s, then sampling stops.
        self.assertAlmostEqual(after[-1]['stage_elapsed'], .5)
        self.assertAlmostEqual(estimate_phase(3, .25, [1., 2., .5]), 5 / 6)

    def test_release_threshold_is_calibrated_on_held_out_nominal_releases(self):
        rng = np.random.default_rng(0)
        names = [f'nominal_{i}' for i in range(12)]
        detector = GmrOnlineDetector({'output_dir': '.', 'nominal_bags': names}, Mock())
        detector.reset('test')
        nominal = np.array([.37, -.68, -.03])
        releases = [dict(zip(['position_x', 'position_y', 'position_z'], nominal + rng.normal(0, .005, 3)))
                    for _ in names]
        detector.calibrate_release(releases, Mock())
        inside = detector.check_release(dict(time=1., stage=3, **releases[0]))
        self.assertFalse(inside['alarm'])
        beside = dict(zip(['position_x', 'position_y', 'position_z'], nominal + [0., .2, 0.]))
        self.assertTrue(detector.check_release(dict(time=2., stage=3, **beside))['alarm'])
        self.assertTrue(detector.check_release(None)['alarm'])

    def test_gmr_blends_overlapping_components(self):
        model = GMRAnomalyDetector(['phase'], ['a_position'], n_components=2)
        model.gmm.weights_ = np.array([.5, .5])
        model.gmm.means_ = np.array([[-1., 0.], [1., 10.]])
        model.gmm.covariances_ = np.array([np.eye(2), np.eye(2)])
        mean, covariance, _ = model._predict_output_distribution(np.array([0.]))
        self.assertAlmostEqual(mean[0], 5.)
        self.assertAlmostEqual(covariance[0, 0], 1. + 25. + model.reg_covar)
        # Midway between components, the blended prediction does not jump to either one.
        self.assertAlmostEqual(model._score_scaled_point(np.array([0., 5.]))[0], 0.)

    def test_pose_force_uses_force_only_when_every_bag_recorded_it(self):
        def events(bag, topics, *_args):
            pose = lambda t: dict(position_x=np.sin(t) + (.02 if bag == 'nominal_b' else 0.), position_y=t,
                                  position_z=0., pose_frame='base')
            stream = [(0., 'stage', '1'), (1., 'stage', '2'), (2., 'stage', '3')]
            stream += [(round(t, 9), 'pose', pose(t)) for t in np.arange(0., 2.5, .05)]
            if 'force' in topics:
                stream += [(round(t, 9), 'force', dict(force_x=0., force_y=0., force_z=np.cos(t) + .1 * (bag == 'nominal_b')))
                           for t in np.arange(0., 2.5, .01)]
            return iter(sorted(stream, key=lambda e: e[0]))
        for test_topics, columns in [({'pose': '/pose', 'force': '/wrench'}, ['force_x', 'force_y', 'force_z']), ({'pose': '/pose'}, [])]:
            with tempfile.TemporaryDirectory() as directory, \
                 patch('rynnbrain_vlm.replay_detectors.gmr_signal_topics',
                       side_effect=lambda bag, mode: test_topics if bag == 'test' else {'pose': '/pose', 'force': '/wrench'}), \
                 patch('rynnbrain_vlm.visual_matching._source_identity', return_value=[]), \
                 patch('rosbag_io.iter_ts_replay_events', side_effect=events):
                detector = GmrOnlineDetector(dict(output_dir=directory, nominal_bags=['nominal_a', 'nominal_b']), Mock())
                detector.reset('test')
                prepared = detector.prepare_replay(Mock(), threading.Event())
                self.assertEqual(detector.columns, sorted(['position_x', 'position_y', 'position_z', *columns]))
                # Force is a safety limit, not a GMR sensor.
                self.assertEqual(list(detector.sensors), ['position'])
                self.assertEqual(detector.force_limit is not None, bool(columns))
                self.assertEqual('force' in prepared['topics'], bool(columns))
                self.assertEqual('force not recorded' in prepared['signal_description'], not columns)

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
