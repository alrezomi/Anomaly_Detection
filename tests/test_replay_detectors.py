from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from PIL import Image

from rynnbrain_vlm.replay_detectors import DinoReplayDetector, GmrReplayDetector, wrench_topic, synchronized_wrench
from rynnbrain_vlm.visual_matching import RecordedStages
from time_series_gmr_scripts.ts_data import preprocess_replay_signals
from rosbag_io import _numeric_message_values
from types import SimpleNamespace


class ReplayDetectorTests(unittest.TestCase):
    @staticmethod
    def pose_signal(offset=0.):
        t = np.linspace(0., 12., 90)
        data = pd.DataFrame(dict(time=t, position_x=np.sin(t)+offset, position_y=np.cos(t),
            position_z=t*.01, orientation_x=0., orientation_y=0.,
            orientation_z=np.sin(t*.05), orientation_w=np.cos(t*.05)))
        data.attrs['frame_id'] = 'base'
        return data

    def test_stage_resampling_quaternion_sign_invariance_and_gripper_hold(self):
        pose = self.pose_signal()
        flipped = pose.copy()
        flipped.loc[::2, ['orientation_x', 'orientation_y', 'orientation_z', 'orientation_w']] *= -1
        grip = pd.DataFrame({'time': [0., 5., 12.], 'gripper_position': [0., 1., 1.]})
        stages = RecordedStages((1., 4., 8.), ((6., 'Error'),))
        actual = preprocess_replay_signals({'pose': flipped, 'gripper': grip}, stages, 31)
        expected = preprocess_replay_signals({'pose': pose, 'gripper': grip}, RecordedStages((1., 4., 8.)), 31)
        pd.testing.assert_frame_equal(actual, expected)
        self.assertEqual(actual.attrs['pose_frame'], 'base')
        np.testing.assert_allclose(actual.time.iloc[[0, 10, 20, 30]], [1., 4., 8., 12.])
        self.assertTrue((actual.loc[actual.time < 5., 'gripper_position'] == 0).all())
        self.assertTrue((actual.loc[actual.time >= 5., 'gripper_position'] == 1).all())
        grip['gripper_position'] = [0., np.nan, 1.]
        with self.assertRaisesRegex(ValueError, 'non-finite'):
            preprocess_replay_signals({'gripper': grip}, stages)

    def test_joint_channels_are_named_and_effort_is_not_used_as_force(self):
        def message(names, positions):
            return SimpleNamespace(name=names, position=positions, velocity=[], effort=[10., 20.])
        first = _numeric_message_values(message(['a', 'b'], [1., 2.]), 'sensor_msgs/msg/JointState')
        second = _numeric_message_values(message(['b', 'a'], [2., 1.]), 'sensor_msgs/msg/JointState')
        data = pd.DataFrame([dict(time=0., **first), dict(time=3., **second)])
        result = preprocess_replay_signals({'joints': data}, RecordedStages((0., 1., 2.)), 10)
        self.assertEqual(list(result), ['time', 'phase', 'a_position', 'b_position'])
        np.testing.assert_allclose(result.a_position, 1.)
        np.testing.assert_allclose(result.b_position, 2.)
        data.loc[1, 'a_position'] = np.nan
        with self.assertRaisesRegex(ValueError, 'non-finite'):
            preprocess_replay_signals({'joints': data}, RecordedStages((0., 1., 2.)))

    def test_gripper_reads_actual_position_not_command_echo(self):
        typename = 'robotiq_hand_e_gripper_interfaces/msg/Robotiq2FGripperRobotInput'
        values = _numeric_message_values(SimpleNamespace(gPO=128, gPR=255), typename)
        self.assertEqual(values, {'gripper_position': 128/255})
        with self.assertRaisesRegex(ValueError, 'actual-position'):
            _numeric_message_values(SimpleNamespace(gPR=255), typename)

    def test_custom_gripper_cdr_is_read_when_bag_embeds_its_definition(self):
        from rosbags.rosbag2 import Writer
        from rosbags.typesys import Stores, get_typestore, get_types_from_msg
        from rosbag_io import read_numeric_topic
        typename = 'robotiq_hand_e_gripper_interfaces/msg/Robotiq2FGripperRobotInput'
        # A different embedded schema must override the lab-schema fallback.
        store = get_typestore(Stores.ROS2_HUMBLE)
        store.register(get_types_from_msg('uint8 g_pr\nuint8 g_po\n', typename))
        with tempfile.TemporaryDirectory() as directory:
            bag = Path(directory) / 'bag'
            with Writer(bag, version=9) as writer:
                connection = writer.add_connection('/Robotiq2FGripperRobotInputMsg', typename, typestore=store)
                for i, position in enumerate([0, 128, 255]):
                    message = store.types[typename](g_pr=255, g_po=position)
                    writer.write(connection, 1_000_000_000 + i*500_000_000,
                                 store.serialize_cdr(message, typename))
            result = read_numeric_topic(bag, '/Robotiq2FGripperRobotInputMsg')
            np.testing.assert_allclose(result.gripper_position, [0., 128/255, 1.])
            np.testing.assert_allclose(result.time, [0., .5, 1.])

    def test_gripper_lab_schema_decodes_sqlite_bag_without_message_definitions(self):
        import sqlite3
        from contextlib import closing
        from rosbags.rosbag2 import Writer
        from rosbag_io import read_numeric_topic, ROS2_TYPESTORE, ROBOTIQ_INPUT_TYPE, open_bag
        with tempfile.TemporaryDirectory() as directory:
            bag = Path(directory) / 'bag'
            with Writer(bag, version=9) as writer:
                connection = writer.add_connection('/Robotiq2FGripperRobotInputMsg', ROBOTIQ_INPUT_TYPE,
                                                   typestore=ROS2_TYPESTORE)
                for i, position in enumerate([12, 128, 240]):
                    message = ROS2_TYPESTORE.types[ROBOTIQ_INPUT_TYPE](
                        g_act=1, g_gto=1, g_sta=3, g_obj=2, g_flt=0,
                        g_pr=255, g_po=position, g_cu=30)
                    writer.write(connection, 1_000_000_000 + i*500_000_000,
                                 ROS2_TYPESTORE.serialize_cdr(message, ROBOTIQ_INPUT_TYPE))
            # Reproduce older bags' lack of an embedded definition.
            with closing(sqlite3.connect(next(bag.glob('*.db3')))) as database:
                database.execute('DELETE FROM message_definitions')
                database.commit()
            with open_bag(bag) as reader:
                self.assertFalse(reader.connections[0].msgdef.data)
            result = read_numeric_topic(bag, '/Robotiq2FGripperRobotInputMsg')
            np.testing.assert_allclose(result.gripper_position, np.array([12, 128, 240])/255.)
            np.testing.assert_allclose(result.time, [0., .5, 1.])

    def test_pose_gmr_trains_without_wrench_and_rejects_other_reference_frame(self):
        from time_series_gmr_scripts import run_time_series_test as settings
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bags = [root / name for name in ['n1', 'n2', 'test']]
            for bag in bags:
                bag.mkdir()
                (bag / 'metadata.yaml').write_text('test')
            topics = pd.DataFrame({'topic': ['/tcp_pose_broadcaster/pose'],
                                   'message_type': ['geometry_msgs/msg/PoseStamped'], 'message_count': [90]})
            def read(bag, topic):
                return self.pose_signal(bags.index(Path(bag)) * .01)
            config = {'output_dir': str(root / 'out'), 'nominal_bags': [str(p) for p in bags[:2]]}
            with patch('rosbag_io.list_bag_topics', return_value=topics), \
                 patch('rosbag_io.read_numeric_topic', side_effect=read), \
                 patch('rynnbrain_vlm.visual_matching.read_recorded_stages', return_value=RecordedStages((1., 4., 8.))), \
                 patch.object(settings, 'RESAMPLE_LENGTH', 32):
                detector = GmrReplayDetector(config, Mock())
                detector.signal_mode = 'pose'
                detector.reset(bags[-1])
                metadata = detector.prepare_replay(Mock(), threading.Event())
                self.assertEqual(metadata['signal_mode'], 'pose')
                self.assertEqual(len(metadata['output_columns']), 9)
                self.assertEqual(metadata['pose_frame'], 'base')
                self.assertEqual(len(detector.results), 32)
                self.assertTrue(np.isfinite(detector.results.epsilon_smooth).all())
                self.assertFalse(any('force' in name for name in detector.model.output_columns))
                with patch('time_series_gmr_scripts.anomaly_detection.GMRAnomalyDetector.fit', side_effect=AssertionError('cache must be reused')):
                    detector.prepare_replay(Mock(), threading.Event())
                different = self.pose_signal()
                different.attrs['frame_id'] = 'tool'
                with patch('rosbag_io.read_numeric_topic', side_effect=lambda bag, topic: different if Path(bag) == bags[-1] else read(bag, topic)):
                    with self.assertRaisesRegex(ValueError, 'reference frame differs'):
                        detector.prepare_replay(Mock(), threading.Event())

    def test_dino_matches_existing_score_and_returns_selected_frame_and_heatmap(self):
        from vision_dinov2.anomaly_detection import compare_frame_to_nominal_memory
        detector = DinoReplayDetector.__new__(DinoReplayDetector)
        features = np.tile([1., 0.], (4, 1))
        attention = np.ones(4)
        cls = np.array([1., 0.])
        detector.extractor = Mock()
        detector.extractor.extract.return_value = (features, attention, cls, 2)
        detector.signature = dict(top_k_cls=1, attention_power=1.5, min_weight=.1)
        detector.stages = RecordedStages((0., 10., 20.))
        detector.step = 0
        detector.memory = [dict(features=features + delta, attention=attention, cls=cls,
            stage=stage, progress=.5, video_id=0, frame_id=stage, timestamp_sec=11.) for stage, delta in [(1, 0.), (2, .2)]]
        detector.thresholds = pd.DataFrame({'stage_center': [0., 1.], 'adaptive_threshold': [.1, .1]})
        detector._thumbnail = Mock(return_value=('normal', Image.new('RGB', (8, 8))))
        with patch('rynnbrain_vlm.online_replay.synchronized_time', side_effect=[1., 1.1]):
            row, vector, baseline = detector.append(Image.new('RGB', (8, 8)), 12.)
        expected = compare_frame_to_nominal_memory(features, attention, cls, detector.memory,
            top_k_cls=1, attention_power=1.5, min_weight=.1, required_stage=2)[0]
        self.assertAlmostEqual(row['anomaly_score'], expected)
        self.assertEqual(row['visual_matching']['best']['frame_id'], 2)
        self.assertTrue(row['above_threshold'])
        self.assertEqual(detector.preview_image.size, (8, 8))
        np.testing.assert_array_equal(vector, cls)
        self.assertIsNone(baseline)
        detector.extractor.extract.assert_called_once()

    def test_wrench_topic_never_substitutes_pose_or_arbitrary_topic(self):
        with patch('rosbag_io.list_bag_topics', return_value=pd.DataFrame({
                'topic': ['/pose'], 'message_type': ['geometry_msgs/msg/PoseStamped']})):
            with self.assertRaisesRegex(ValueError, 'no supported topic'):
                wrench_topic('test')
        with patch('rosbag_io.list_bag_topics', return_value=pd.DataFrame({
                'topic': ['/force'], 'message_type': ['geometry_msgs/msg/WrenchStamped']})):
            self.assertEqual(wrench_topic('test'), '/force')

    @staticmethod
    def raw_signal(offset=0.):
        t = np.linspace(4., 10., 80)
        values = {'time': t, 'time_ns': (t * 1e9).astype('int64')}
        for i, name in enumerate(['force_x', 'force_y', 'force_z', 'torque_x', 'torque_y', 'torque_z']):
            values[name] = np.sin(t * (1 + i / 5)) + offset + (t > 5.) * .8
        return pd.DataFrame(values)

    def test_gmr_restores_bag_clock_after_legacy_synchronization(self):
        from time_series_gmr_scripts import run_time_series_test as settings
        from time_series_gmr_scripts.ts_data import sync_and_preprocess_ts_dataframe
        raw = self.raw_signal()
        with patch('rosbag_io.read_numeric_topic', return_value=raw):
            actual = synchronized_wrench('bag', '/force', settings)
        expected, info, _ = sync_and_preprocess_ts_dataframe(raw, resample_length=settings.RESAMPLE_LENGTH,
            baseline_fraction=settings.BASELINE_FRACTION, threshold_std_factor=settings.THRESHOLD_STD_FACTOR,
            min_force_change=settings.MIN_FORCE_CHANGE, padding_seconds=settings.PADDING_SECONDS)
        trimmed = raw if info['used_full_trajectory'] else raw.loc[raw.time.between(info['start_time'], info['end_time'])]
        expected['time'] += trimmed.time.iloc[0]
        pd.testing.assert_frame_equal(actual, expected)

    def test_real_gmr_offline_pipeline_caches_scores_and_looks_up_past_sample(self):
        from time_series_gmr_scripts import run_time_series_test as settings
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bags = [root / name for name in ['n1', 'n2', 'test']]
            for bag in bags:
                bag.mkdir()
                (bag / 'metadata.yaml').write_text('test')
            config = {'output_dir': str(root / 'out'), 'nominal_bags': [str(p) for p in bags[:2]],
                      'rynnbrain': {'num_frames': 8}}
            def read(bag, topic):
                return self.raw_signal(bags.index(Path(bag)) * .05)
            with patch('rynnbrain_vlm.replay_detectors.wrench_topic', return_value='/force'), \
                 patch('rosbag_io.read_numeric_topic', side_effect=read), \
                 patch.object(settings, 'RESAMPLE_LENGTH', 32):
                detector = GmrReplayDetector(config, Mock())
                detector.reset(bags[-1])
                metadata = detector.prepare_replay(Mock(), threading.Event())
                self.assertTrue(metadata['offline'])
                timestamp = float(detector.results.time.iloc[5]) + .001
                row, _, _ = detector.append(None, timestamp)
                self.assertEqual(row['signal_time_sec'], float(detector.results.time.iloc[5]))
                self.assertEqual(row['anomaly_score'], float(detector.results.epsilon_smooth.iloc[5]))
                self.assertEqual(row['alarm_filtered'], bool(detector.results.alarm_filtered.iloc[5]))
                self.assertIn('OFFLINE', detector.note)
                with self.assertRaises(ValueError):
                    detector.append(None, detector.start_sec - 1.)
                second = GmrReplayDetector(config, Mock())
                second.reset(bags[-1])
                with patch('time_series_gmr_scripts.anomaly_detection.GMRAnomalyDetector.fit', side_effect=AssertionError('Do not retrain cached results')):
                    second.prepare_replay(Mock(), threading.Event())
                np.testing.assert_allclose(second.results.epsilon_smooth, detector.results.epsilon_smooth)


if __name__ == '__main__':
    unittest.main()
