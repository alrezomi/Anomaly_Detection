from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from PIL import Image

from vision_dinov2.local_scoring import (fit_local_threshold, local_decision,
    region_distances, score_matched_frame, LOCAL_METHOD)
from vision_dinov2.threshold import build_sequence_threshold, scaled_threshold
from vision_dinov2.nominal_cache import save_nominal_cache, save_nominal_calibration, load_nominal_cache


class LocalDinoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.maps, rows = [], []
        for run in range(26):
            for step in range(8):
                patches = np.full((14, 14), .01 + run * .00002 + step * .00001)
                # A large region of normal robot movement varies strongly.
                patches[:6, 7:] += .03 + .02 * np.sin(run + step) ** 2
                cls.maps.append(patches.reshape(-1))
                rows.append(dict(held_out_video_id=run, matched_progress=step / 7, score=float(patches.mean())))
        cls.rows = pd.DataFrame(rows)
        cls.table, cls.calibration = fit_local_threshold(cls.rows, cls.maps, n_bins=6)
        cls.profile = cls.table.attrs['local_profile']

    def test_small_local_mismatch_is_not_diluted_by_background(self):
        patches = self.maps[4].reshape(14, 14).copy()
        patches[8:11, 1:4] += .025
        result = local_decision(patches.reshape(-1), 4 / 7, self.profile)
        self.assertGreater(result['score'], result['threshold'])
        self.assertEqual((result['region_x'], result['region_y']), (1, 8))
        old, _ = build_sequence_threshold(self.rows, n_bins=6)
        global_limit = np.interp(4 / 7, old.stage_center, old.adaptive_threshold)
        self.assertLess(float(patches.mean()), global_limit)

    def test_normal_motion_and_nominal_limit_runs_are_not_flagged_in_fixture(self):
        limit = self.calibration[self.calibration.calibration_role == 'limit']
        self.assertFalse(limit.above_threshold.any())
        result = local_decision(self.maps[4], 4 / 7, self.profile)
        self.assertLessEqual(result['score'], result['threshold'])

    def test_spatial_location_matters_even_when_global_mean_is_unchanged(self):
        patches = self.maps[4].reshape(14, 14).copy()
        original = patches.copy()
        patches[8:11, 1:4], patches[1:4, 9:12] = original[1:4, 9:12], original[8:11, 1:4]
        self.assertAlmostEqual(patches.mean(), original.mean())
        self.assertGreater(local_decision(patches.reshape(-1), 4 / 7, self.profile)['epsilon'], 1.)

    def test_limit_runs_cannot_change_location_baseline(self):
        altered = [item.copy() for item in self.maps]
        for index in self.calibration.index[self.calibration.calibration_role == 'limit']:
            altered[index][:9] += .1
        table, _ = fit_local_threshold(self.rows, altered, n_bins=6)
        np.testing.assert_allclose(table.attrs['local_profile']['mean'], self.profile['mean'])
        np.testing.assert_allclose(table.attrs['local_profile']['scale'], self.profile['scale'])
        self.assertGreater(table.bandwidth.iloc[0], self.table.bandwidth.iloc[0])

    def test_multiplier_is_identical_for_replay_and_scaled_batch_table(self):
        patches = self.maps[4]
        batch = score_matched_frame(.02, patches, 4 / 7, scaled_threshold(self.table, 1.2))
        replay = score_matched_frame(.02, patches, 4 / 7, self.table, 1.2)
        for field in ['score', 'threshold', 'base_threshold', 'epsilon', 'region_x', 'region_y']:
            self.assertAlmostEqual(batch[field], replay[field])
        original = score_matched_frame(.02, patches, 4 / 7, self.table)
        self.assertAlmostEqual(replay['threshold'], 1.2 * original['threshold'])
        self.assertEqual(replay['score'], original['score'])

    def test_small_patch_artifact_is_averaged_and_invalid_maps_fail(self):
        patches = np.zeros((14, 14))
        patches[7, 7] = .09
        regions, grid, size = region_distances(patches.reshape(-1))
        self.assertAlmostEqual(regions.max(), .01)
        self.assertEqual((grid, size), (14, 3))
        for values in [np.ones(10), np.full(196, np.nan), -np.ones(196)]:
            with self.assertRaises(ValueError):
                region_distances(values)
        with self.assertRaisesRegex(ValueError, 'grid'):
            local_decision(np.ones(16), .5, self.profile)

    def test_cache_profile_round_trip_and_missing_profile_is_rejected(self):
        item = dict(features=np.ones((196, 2)), attention=np.ones(196), cls=np.array([1., 0.]),
                    video_id=0, frame_id=0, timestamp_sec=0., progress=0., stage=1, video_path='memory')
        signature = dict(score_mode='local', calibration_method=LOCAL_METHOD)
        with tempfile.TemporaryDirectory() as folder:
            save_nominal_cache(folder, [item], 14, self.table, self.calibration, signature)
            loaded = load_nominal_cache(folder, signature)
            for key in self.profile:
                np.testing.assert_allclose(loaded[2].attrs['local_profile'][key], self.profile[key])
            self.assertIsNone(load_nominal_cache(folder, dict(score_mode='mean')))
            (Path(folder) / 'local_threshold.npz').unlink()
            self.assertIsNone(load_nominal_cache(folder, signature))
            save_nominal_calibration(folder, self.table, self.calibration, signature)
            self.assertIsNotNone(load_nominal_cache(folder, signature))

    def test_zero_attention_cannot_hide_local_difference_and_batch_replay_agree(self):
        from vision_dinov2.anomaly_detection import run_video_test
        from vision_dinov2.visualization import create_anomaly_heatmap_video
        from rynnbrain_vlm.replay_detectors import DinoReplayDetector
        patches = self.maps[4].reshape(14, 14).copy()
        patches[8:11, 1:4] += .025
        patches = patches.reshape(-1)
        best = dict(progress=4 / 7, video_id=0, frame_id=0, timestamp_sec=1., stage=1,
                    _progress_locked=True)
        match = (float(patches.mean()), patches, np.zeros(196), np.zeros(196), best, .01)
        image = Image.new('RGB', (140, 140))
        extractor = Mock()
        extractor.extract.return_value = (np.zeros((196, 2)), np.zeros(196), np.ones(2), 14)
        detector = DinoReplayDetector.__new__(DinoReplayDetector)
        detector.extractor, detector.thresholds, detector.memory = extractor, self.table, [best]
        detector.signature = dict(top_k_cls=5, attention_power=1.5, min_weight=.1)
        detector.stages, detector.step, detector.threshold_scale = None, 0, 1.
        detector._thumbnail = Mock(return_value=('memory', image))
        with patch('vision_dinov2.anomaly_detection.compare_frame_to_nominal_memory', return_value=match), \
                patch('vision_dinov2.anomaly_detection.iter_source_frames', return_value=iter([(0, 1., image)])):
            replay, _, _ = detector.append(image, 1.)
            batch = run_video_test(extractor, 'test', [best], self.table).iloc[0]
        with tempfile.TemporaryDirectory() as folder, \
                patch('vision_dinov2.visualization.compare_frame_to_nominal_memory', return_value=match), \
                patch('vision_dinov2.visualization.iter_source_frames', return_value=iter([(0, 1., image)])), \
                patch('vision_dinov2.visualization.cv2.VideoWriter'):
            video = create_anomaly_heatmap_video(extractor, 'test', [best], Path(folder) / 'video.mp4',
                                                 self.table).iloc[0]
        self.assertTrue(replay['above_threshold'])
        self.assertTrue(batch.alarm)
        self.assertAlmostEqual(replay['anomaly_score'], batch.score)
        self.assertAlmostEqual(replay['threshold'], batch.threshold)
        self.assertEqual(replay['region_x'], batch.region_x)
        self.assertAlmostEqual(video.score, batch.score)
        self.assertAlmostEqual(video.threshold, batch.threshold)
        self.assertEqual(video.alarm, batch.alarm)
        # A selected-region box is visible even when attention weights are zero.
        self.assertIsNotNone(detector.preview_image)


if __name__ == '__main__':
    unittest.main()
