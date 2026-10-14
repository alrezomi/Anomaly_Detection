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


class ReplayDetectorTests(unittest.TestCase):
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
