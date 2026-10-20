"""Attention-scored progress calibration shared by build, replay and export."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from PIL import Image

from vision_dinov2.anomaly_detection import compare_frame_to_nominal_memory, run_video_test
from vision_dinov2.threshold import (build_notebook_threshold, calibrate_notebook_leave_one_video_out,
                                    score_calibration_memory, scaled_threshold, NOTEBOOK_CALIBRATION_METHOD)
from vision_dinov2.nominal_cache import save_nominal_cache, load_nominal_cache, memory_signature


class NotebookTests(unittest.TestCase):
    def test_extractor_still_l2_normalizes_patch_and_cls_features(self):
        import torch
        from vision_dinov2.dinov2_features import DINOv2FeatureExtractor
        extractor = DINOv2FeatureExtractor.__new__(DINOv2FeatureExtractor)
        extractor.input_size, extractor.patch_size, extractor.device = 28, 14, 'cpu'
        extractor.processor = Mock(return_value={'pixel_values': torch.zeros((1, 3, 28, 28))})
        hidden = torch.tensor([[[3., 4.], [6., 8.], [5., 12.], [8., 15.], [7., 24.]]])
        extractor.model = Mock(return_value=Mock(last_hidden_state=hidden,
            attentions=[torch.ones((1, 1, 5, 5)) / 5]))
        features, _, cls, grid = extractor.extract(Image.new('RGB', (56, 56)))
        self.assertEqual(grid, 2)
        np.testing.assert_allclose(np.linalg.norm(features, axis=-1), 1., atol=1e-6)
        self.assertAlmostEqual(float(np.linalg.norm(cls)), 1.)

    @staticmethod
    def item(seed, run=0, stage=1):
        rng = np.random.default_rng(seed)
        features = rng.normal(size=(16, 6)).astype(np.float32)
        features /= np.linalg.norm(features, axis=-1, keepdims=True)
        return dict(features=features, attention=rng.random(16), cls=np.array([1., 0.]),
                    video_id=run, frame_id=seed, stage=stage, progress=.5, timestamp_sec=float(seed),
                    video_path='memory', _progress_locked=True)

    def match(self, query, memory, **kwargs):
        return compare_frame_to_nominal_memory(query['features'], query['attention'], query['cls'],
                                               memory, notebook_scoring=True, min_weight=.1, **kwargs)

    def test_attention_formula_and_cls_shortlist(self):
        from vision_dinov2.anomaly_detection import make_attention_weights
        query = self.item(10)
        memory = [self.item(1), self.item(2)]
        memory[1]['features'] = query['features'].copy()
        memory[1]['cls'] = -query['cls']
        result = self.match(query, memory, top_k_cls=1)
        self.assertIs(result[4], memory[0])  # No exhaustive patch search.
        raw = np.sqrt(np.mean((query['features'] - memory[0]['features']) ** 2, axis=-1))
        weights = make_attention_weights(query['attention'], memory[0]['attention'], min_weight=.1)
        self.assertAlmostEqual(result[0], float(np.sum(raw * weights) / weights.sum()))
        query['attention'][:] = 0
        memory[0]['attention'][:] = 0
        self.assertNotAlmostEqual(result[0], self.match(query, memory, top_k_cls=1)[0])

    def test_progress_threshold_changes_and_runtime_crossings(self):
        from vision_dinov2.anomaly_detection import threshold_for_progress
        rows = pd.DataFrame(dict(score=[.1, .2, .3, .7, .8, .9],
            matched_progress=[.1]*3 + [.9]*3, held_out_video_id=[0]*6))
        table, _ = build_notebook_threshold(rows, n_bins=2, smooth_window=1)
        low, high = table.adaptive_threshold
        self.assertAlmostEqual(low, np.quantile([.1, .2, .3], .95)*1.1)
        self.assertAlmostEqual(high, np.quantile([.7, .8, .9], .95)*1.1)
        self.assertGreater(.5, threshold_for_progress(.1, table))
        self.assertLess(.5, threshold_for_progress(.9, table))

    def test_stage_restriction_and_no_spatial_substitution(self):
        query = self.item(2)
        same = self.item(2, stage=2)
        shifted = self.item(2, stage=1)
        shifted['features'] = np.roll(shifted['features'], 1, axis=0)
        score = self.match(query, [same, shifted], required_stage=1)[0]
        self.assertGreater(score, 0.)
        self.assertEqual(self.match(query, [same, shifted])[0], 0.)
        with self.assertRaisesRegex(ValueError, 'stage 3'):
            self.match(query, [same, shifted], required_stage=3)
        with self.assertRaisesRegex(ValueError, 'cannot be combined'):
            self.match(query, [same], local_scoring=True)

    def test_threshold_formula_and_complete_recording_exclusion(self):
        memory = [self.item(seed, run=run) for run in range(3) for seed in (run + 1, run + 10)]
        table, rows = calibrate_notebook_leave_one_video_out(memory)
        expected = []
        for query in memory:
            references = [x for x in memory if x['video_id'] != query['video_id']]
            expected.append(self.match(query, references)[0])
        np.testing.assert_allclose(rows.score, expected)
        self.assertTrue((rows.matched_video_id != rows.held_out_video_id).all())
        self.assertAlmostEqual(table.adaptive_threshold.iloc[0], np.quantile(expected, .95) * 1.1)
        self.assertEqual(np.ptp(table.adaptive_threshold), 0.)
        self.assertEqual(table.attrs['score_mode'], 'notebook')
        self.assertTrue((table.method == NOTEBOOK_CALIBRATION_METHOD).all())
        with self.assertRaisesRegex(ValueError, 'two nominal'):
            calibrate_notebook_leave_one_video_out(memory[:2])
        with self.assertRaisesRegex(ValueError, 'finite'):
            build_notebook_threshold(pd.DataFrame(dict(score=[np.nan], held_out_video_id=[0], matched_progress=[.5])))

    def test_build_replay_cache_agreement_and_old_profile_rejection(self):
        from rosbag_io import RosbagImageSource
        from vision_dinov2 import run_vision_test as runner
        from rynnbrain_vlm.replay_detectors import DinoReplayDetector
        for independent, configured_mode in ((False, 'notebook'), (True, 'notebook'), (False, 'mean'), (True, 'mean')):
            with self.subTest(independent=independent, mode=configured_mode), tempfile.TemporaryDirectory() as folder:
                source = RosbagImageSource(Path(folder) / 'memory_bag', '/camera')
                calibration = [str(Path(folder) / 'calibration_bag')] if independent else []
                with patch.multiple(runner, SCORE_MODE=configured_mode, NOMINAL_VIDEO_PATHS=[source],
                        CALIBRATION_VIDEO_PATHS=[RosbagImageSource(x, '/camera') for x in calibration]):
                    signature = runner.nominal_cache_signature()
                table, rows = build_notebook_threshold(pd.DataFrame(dict(score=[.1, .2], held_out_video_id=[0, 1], matched_progress=[.5,.5])))
                directory = Path(folder) / 'camera'
                save_nominal_cache(directory, [self.item(1)], 4, table, rows, signature)
                self.assertIsNone(load_nominal_cache(directory, dict(signature, score_mode='local')))
                self.assertIsNone(load_nominal_cache(directory, dict(signature, calibration_method='notebook_patch_rms_frame_quantile_v1')))
                self.assertIsNotNone(load_nominal_cache(directory, memory_signature(signature)))
                config = dict(nominal_cache_dir=folder, dino_score_mode=configured_mode,
                              dino_calibration_bags=calibration, rynnbrain={'camera_topics': ['/camera']})
                with patch('vision_dinov2.dinov2_features.DINOv2FeatureExtractor'):
                    detector = DinoReplayDetector(config, lambda _: None)
                self.assertEqual(detector.thresholds.attrs['score_mode'], 'notebook')
                self.assertEqual(detector.signature['threshold_quantile'], .95)
                self.assertEqual(len(detector.excluded), 1 + len(calibration))

    def test_real_matcher_agrees_in_calibration_replay_batch_and_video(self):
        from vision_dinov2.visualization import create_anomaly_heatmap_video
        from rynnbrain_vlm.replay_detectors import DinoReplayDetector
        query, memory = self.item(3), [self.item(1), self.item(2)]
        image = Image.new('RGB', (56, 56))
        extractor = Mock()
        extractor.extract.return_value = query['features'], query['attention'], query['cls'], 4
        table, _ = build_notebook_threshold(pd.DataFrame(dict(score=[.05, .1], held_out_video_id=[0, 1], matched_progress=[.5,.5])))
        detector = DinoReplayDetector.__new__(DinoReplayDetector)
        detector.extractor, detector.thresholds, detector.memory = extractor, table, memory
        detector.signature = dict(top_k_cls=5, attention_power=1.5, min_weight=.1)
        detector.stages, detector.step, detector.threshold_scale = None, 0, .9
        detector._thumbnail = Mock(return_value=('memory', image))
        replay, _, _ = detector.append(image, 3.)
        calibration = score_calibration_memory([query], memory, 0, notebook_scoring=True)
        with patch('vision_dinov2.anomaly_detection.iter_source_frames', return_value=iter([(0, 3., image)])):
            batch = run_video_test(extractor, 'test', memory, scaled_threshold(table, .9), min_weight=.1).iloc[0]
        with tempfile.TemporaryDirectory() as folder, \
                patch('vision_dinov2.visualization.iter_source_frames', return_value=iter([(0, 3., image)])), \
                patch('vision_dinov2.visualization.cv2.VideoWriter'):
            video = create_anomaly_heatmap_video(extractor, 'test', memory, Path(folder) / 'test.mp4',
                                                 scaled_threshold(table, .9), min_weight=.1).iloc[0]
        for score in [calibration[0]['score'], batch.score, video.score]:
            self.assertAlmostEqual(replay['anomaly_score'], score)
        for bound in [batch.threshold, video.threshold]:
            self.assertAlmostEqual(replay['threshold'], bound)
        self.assertEqual(replay['above_threshold'], batch.alarm)
        self.assertEqual(replay['above_threshold'], video.alarm)
        self.assertEqual(replay['score_mode'], 'notebook')
        self.assertIsNotNone(detector.preview_image)

    def test_build_reuses_features_and_calibrates_both_supported_splits(self):
        from vision_dinov2 import run_vision_test as runner
        memory = [self.item(1, run=0), self.item(2, run=1)]
        old_table = pd.DataFrame(dict(stage_center=[.5], adaptive_threshold=[.1]))
        for paths in ([], ['separate_nominal']):
            with self.subTest(paths=paths), tempfile.TemporaryDirectory() as folder, \
                    patch.multiple(runner, SCORE_MODE='notebook', OUTPUT_DIRECTORY=Path(folder),
                        NOMINAL_CACHE_DIRECTORY=Path(folder), NOMINAL_VIDEO_PATHS=['memory_a', 'memory_b'],
                        CALIBRATION_VIDEO_PATHS=paths, SETTINGS_JSON=None, REBUILD_NOMINAL_CACHE=False), \
                    patch.object(runner, 'check_input_paths'), patch.object(runner, 'print_experiment_summary'), \
                    patch.object(runner, 'DINOv2FeatureExtractor'), \
                    patch.object(runner, 'load_nominal_cache', return_value=None), \
                    patch.object(runner, 'load_feature_cache', side_effect=lambda path, sig: (memory, 4) if Path(path) == Path(folder) else None), \
                    patch.object(runner, 'build_nominal_video_memory', return_value=([self.item(3)], 4)) as extract, \
                    patch.object(runner, 'save_nominal_cache') as save_memory, \
                    patch.object(runner, 'save_nominal_calibration') as save, \
                    patch.object(runner, 'plot_adaptive_calibration'):
                runner.main('build-memory')
                save_memory.assert_not_called()
                self.assertEqual(extract.call_count, len(paths))
                table, rows, signature = save.call_args.args[1:]
                self.assertEqual(table.attrs['score_mode'], 'notebook')
                self.assertEqual(signature['calibration_method'], NOTEBOOK_CALIBRATION_METHOD)
                self.assertAlmostEqual(table.adaptive_threshold.iloc[0], np.quantile(rows.score, .95) * 1.1)
                if not paths:
                    self.assertTrue((rows.matched_video_id != rows.held_out_video_id).all())


if __name__ == '__main__':
    unittest.main()
