from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from vision_dinov2.nominal_cache import (save_nominal_cache, load_nominal_cache,
    save_nominal_calibration, save_feature_cache, load_feature_cache)


class FeatureCacheTests(unittest.TestCase):
    @staticmethod
    def item(value=1.):
        return dict(features=np.ones((4, 2), np.float32) * value, attention=np.ones(4),
                    cls=np.array([1., 0.]), video_id=0, frame_id=0, timestamp_sec=0.,
                    progress=.5, stage=1, video_path='memory', _progress_locked=True)

    @staticmethod
    def signature():
        return dict(sources=[dict(bag_path='/data/memory', topic='/top', stage_topic='/stage', stage_count=3)],
                    model_name='dino', dino_input_size=728, sample_fps=2., start_sec=None,
                    score_mode='local', calibration_method='local_v1', top_k_cls=5,
                    attention_power=1.5, min_weight=.1, threshold_quantile=.9,
                    stage_constrained_matching=True)

    def test_scoring_updates_reuse_features_but_not_old_thresholds(self):
        sig = self.signature()
        table = pd.DataFrame(dict(stage_center=[.5], adaptive_threshold=[.1]))
        with tempfile.TemporaryDirectory() as folder:
            # Use mean here so the fixture does not require a local profile file.
            sig['score_mode'] = 'mean'
            save_nominal_cache(folder, [self.item()], 2, table, table, sig)
            changed = dict(sig, top_k_cls=8, attention_power=2., min_weight=.2,
                           score_mode='notebook', threshold_quantile=.95, stage_constrained_matching=False)
            self.assertIsNone(load_nominal_cache(folder, changed))
            memory, grid = load_feature_cache(folder, changed)
            self.assertEqual(grid, 2)
            np.testing.assert_array_equal(memory[0]['features'], self.item()['features'])
            archive = Path(folder) / 'nominal_memory.npz'
            before = archive.read_bytes()
            save_nominal_calibration(folder, table, table, changed)
            self.assertEqual(before, archive.read_bytes())
            self.assertIsNotNone(load_nominal_cache(folder, changed))

    def test_extraction_changes_invalidate_features(self):
        sig = self.signature()
        with tempfile.TemporaryDirectory() as folder:
            save_feature_cache(folder, [self.item()], 2, sig)
            changes = [dict(dino_input_size=896), dict(model_name='other'), dict(sample_fps=3.),
                       dict(start_sec=1.), dict(sources=[dict(bag_path='/data/other', topic='/top')]),
                       dict(sources=[dict(bag_path='/data/memory', topic='/flange')]),
                       dict(sources=[dict(bag_path='/data/memory', topic='/top', stage_topic='/other')])]
            for change in changes:
                with self.subTest(change=change):
                    self.assertIsNone(load_feature_cache(folder, dict(sig, **change)))

    def test_legacy_migration_and_incomplete_calibration_do_not_lose_features(self):
        sig = dict(self.signature(), score_mode='mean')
        table = pd.DataFrame(dict(stage_center=[.5], adaptive_threshold=[.1]))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            save_nominal_cache(folder, [self.item()], 2, table, table, sig)
            (root / 'feature_signature.json').unlink()  # Simulate pre-update cache.
            (root / 'calibration.csv').unlink()
            self.assertIsNone(load_nominal_cache(folder, sig))
            self.assertIsNotNone(load_feature_cache(folder, sig))
            with patch.object(pd.DataFrame, 'to_csv', side_effect=RuntimeError('interrupted')):
                with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                    save_nominal_calibration(folder, table, table, sig)
            self.assertFalse((root / 'signature.json').exists())
            self.assertIsNotNone(load_feature_cache(folder, sig))
            self.assertIsNone(load_nominal_cache(folder, sig))

    def test_recalibration_reuses_query_features_and_does_not_load_model(self):
        from vision_dinov2 import run_vision_test as runner
        with tempfile.TemporaryDirectory() as folder, patch.multiple(runner,
                OUTPUT_DIRECTORY=Path(folder), NOMINAL_CACHE_DIRECTORY=Path(folder),
                NOMINAL_VIDEO_PATHS=['memory'], CALIBRATION_VIDEO_PATHS=['calibration'],
                SCORE_MODE='notebook', SETTINGS_JSON=None, REBUILD_NOMINAL_CACHE=False), \
                patch.object(runner, 'check_input_paths'), patch.object(runner, 'print_experiment_summary'), \
                patch.object(runner, 'plot_adaptive_calibration'), \
                patch.object(runner, 'DINOv2FeatureExtractor') as model, \
                patch.object(runner, 'build_nominal_video_memory', return_value=([self.item(.2)], 2)) as extract:
            save_feature_cache(folder, [self.item()], 2, runner.nominal_cache_signature())
            runner.main('build-memory')
            self.assertEqual(extract.call_count, 1)  # Only the calibration bag.
            self.assertEqual(model.call_count, 1)
            cached = list((Path(folder) / 'calibration_features').rglob('nominal_memory.npz'))
            self.assertEqual(len(cached), 1)
            before = cached[0].read_bytes()
            extract.reset_mock(); model.reset_mock()
            # Matcher changes require rescoring, not extraction of either set.
            with patch.object(runner, 'ATTENTION_POWER', 2.):
                runner.main('build-memory')
                extract.assert_not_called(); model.assert_not_called()
                self.assertEqual(before, cached[0].read_bytes())
                runner.main('build-memory')  # Fully compatible cache also skips model.
                extract.assert_not_called(); model.assert_not_called()
            # Explicit force-rebuild still does what the user requested.
            with patch.object(runner, 'REBUILD_NOMINAL_CACHE', True):
                runner.main('build-memory')
            self.assertEqual(extract.call_count, 2)


if __name__ == '__main__':
    unittest.main()
