from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from vision_dinov2.threshold import (build_sequence_threshold, scaled_threshold,
    score_calibration_memory, validate_threshold_scale, SEQUENCE_CALIBRATION_METHOD)
from vision_dinov2.nominal_cache import (save_nominal_cache, load_nominal_cache,
    save_nominal_calibration, memory_signature)


class DinoThresholdTests(unittest.TestCase):
    @staticmethod
    def scores(runs=26):
        rows = []
        for run in range(runs):
            for index, progress in enumerate(np.linspace(0, 1, 101)):
                # Every nominal run has a short, harmless transient.
                score = .02 + .02 * progress + run * .0001 + (.04 if index == 47 else 0)
                rows.append(dict(held_out_video_id=run, matched_progress=progress, score=score))
        return pd.DataFrame(rows)

    def test_sequence_maxima_cover_short_nominal_transients(self):
        data = self.scores()
        threshold, rows = build_sequence_threshold(data)
        limit = rows[rows.calibration_role == 'limit']
        self.assertFalse(limit.above_threshold.any())
        self.assertEqual(set(rows.groupby('held_out_video_id').calibration_role.nunique()), {1})
        self.assertEqual(threshold.limit_runs.iloc[0], 13)
        self.assertTrue((threshold.method == SEQUENCE_CALIBRATION_METHOD).all())
        self.assertGreater(np.ptp(threshold.adaptive_threshold), 0)
        # The maximum of every limit run is within the selected bandwidth.
        self.assertGreaterEqual(threshold.bandwidth.iloc[0],
            limit.groupby('held_out_video_id').standardized_deviation.max().max())

    def test_limit_subset_cannot_change_baseline_shape_and_smoothing_precedes_maxima(self):
        data = self.scores()
        original, rows = build_sequence_threshold(data)
        changed = data.copy()
        changed.loc[rows.calibration_role == 'limit', 'score'] += .5
        updated, _ = build_sequence_threshold(changed)
        np.testing.assert_allclose(original.baseline_mean, updated.baseline_mean)
        np.testing.assert_allclose(original.baseline_scale, updated.baseline_scale)
        self.assertGreater(updated.bandwidth.iloc[0], original.bandwidth.iloc[0])
        self.assertTrue((updated.adaptive_threshold >= original.adaptive_threshold).all())

    def test_long_bag_does_not_dominate_curve(self):
        data = self.scores()
        original, rows = build_sequence_threshold(data)
        shape_id = rows.loc[rows.calibration_role == 'shape', 'held_out_video_id'].iloc[0]
        repeated = pd.concat([data] + [data[data.held_out_video_id == shape_id]] * 8, ignore_index=True)
        updated, _ = build_sequence_threshold(repeated)
        np.testing.assert_allclose(original.adaptive_threshold, updated.adaptive_threshold)

    def test_degenerate_scores_are_finite_and_invalid_data_is_rejected(self):
        data = self.scores()
        data['score'] = 0.
        data['matched_progress'] = .5
        table, _ = build_sequence_threshold(data)
        self.assertTrue(np.isfinite(table.adaptive_threshold).all())
        self.assertTrue((table.adaptive_threshold > 0).all())
        with self.assertRaisesRegex(ValueError, 'Too few'):
            build_sequence_threshold(self.scores(10))
        data.loc[0, 'score'] = np.nan
        with self.assertRaisesRegex(ValueError, 'finite'):
            build_sequence_threshold(data)
        for value in [0, -1, np.nan, np.inf, True]:
            with self.assertRaises(ValueError):
                validate_threshold_scale(value)

    def test_multiplier_changes_limit_only_and_preserves_base_table(self):
        table, _ = build_sequence_threshold(self.scores())
        before = table.copy(deep=True)
        scaled = scaled_threshold(table, 1.2)
        np.testing.assert_allclose(scaled.adaptive_threshold, 1.2 * before.adaptive_threshold)
        pd.testing.assert_frame_equal(table, before)

    @staticmethod
    def item(delta=0., stage=1):
        return dict(features=np.ones((4, 2), dtype=np.float32) * delta,
                    attention=np.ones(4), cls=np.array([1., 0.]), stage=stage,
                    frame_id=0, video_id=0, timestamp_sec=1., progress=.5,
                    video_path='memory', _progress_locked=True)

    def test_calibration_uses_runtime_matcher_and_stage_restriction(self):
        from vision_dinov2.anomaly_detection import compare_frame_to_nominal_memory
        query = self.item(.2, 2)
        references = [self.item(.2, 1), self.item(0., 2)]
        rows = score_calibration_memory([query], references, 7, stage_constrained_matching=True)
        expected = compare_frame_to_nominal_memory(query['features'], query['attention'], query['cls'],
            references, top_k_cls=5, attention_power=1.5, min_weight=.10, required_stage=2)[0]
        self.assertAlmostEqual(rows[0]['score'], expected)
        self.assertGreater(rows[0]['score'], 0.)
        self.assertEqual(rows[0]['held_out_video_id'], 7)
        self.assertEqual(len(references), 2)

    def test_cache_recalibration_preserves_feature_archive_and_rejects_wrong_matcher(self):
        table, rows = build_sequence_threshold(self.scores())
        old = dict(sources=[{'bag_path': 'memory'}], top_k_cls=5, dino_input_size=728,
                   threshold_quantile=.90)
        new = dict(old, calibration_method=SEQUENCE_CALIBRATION_METHOD,
                   calibration_sources=[{'bag_path': 'calibration'}])
        with tempfile.TemporaryDirectory() as directory:
            save_nominal_cache(directory, [self.item()], 2, table, rows, old)
            archive = Path(directory) / 'nominal_memory.npz'
            before = archive.read_bytes(), archive.stat().st_mtime_ns
            self.assertIsNone(load_nominal_cache(directory, new))
            self.assertIsNotNone(load_nominal_cache(directory, memory_signature(new)))
            save_nominal_calibration(directory, table, rows, new)
            self.assertEqual(before, (archive.read_bytes(), archive.stat().st_mtime_ns))
            self.assertIsNotNone(load_nominal_cache(directory, new))
            self.assertIsNone(load_nominal_cache(directory, dict(new, top_k_cls=3)))

    def test_build_reuses_memory_and_never_adds_calibration_queries(self):
        from vision_dinov2 import run_vision_test as runner
        memory = [self.item()]
        old_table = pd.DataFrame(dict(stage_center=[.5], adaptive_threshold=[.1]))
        calibration_paths = [f'calibration_{i}' for i in range(26)]
        source_rows = self.scores()
        def score(queries, references, run_id, **kwargs):
            self.assertIs(references, memory)
            return source_rows[source_rows.held_out_video_id == run_id].to_dict('records')
        with tempfile.TemporaryDirectory() as directory, patch.multiple(runner,
                OUTPUT_DIRECTORY=Path(directory), NOMINAL_CACHE_DIRECTORY=Path(directory),
                NOMINAL_VIDEO_PATHS=['memory'], CALIBRATION_VIDEO_PATHS=calibration_paths,
                SETTINGS_JSON=None, REBUILD_NOMINAL_CACHE=False), \
                patch.object(runner, 'check_input_paths'), patch.object(runner, 'print_experiment_summary'), \
                patch.object(runner, 'DINOv2FeatureExtractor'), \
                patch.object(runner, 'load_nominal_cache', side_effect=[None, (memory, 2, old_table, pd.DataFrame())]), \
                patch.object(runner, 'build_nominal_video_memory', return_value=([self.item(.2)], 2)) as extract, \
                patch.object(runner, 'score_calibration_memory', side_effect=score), \
                patch.object(runner, 'save_nominal_cache') as save_memory, \
                patch.object(runner, 'save_nominal_calibration') as save_calibration, \
                patch.object(runner, 'plot_adaptive_calibration'):
            runner.main('build-memory')
            self.assertEqual(extract.call_count, 26)
            self.assertEqual([call.kwargs['nominal_video_paths'] if 'nominal_video_paths' in call.kwargs
                              else call.args[1] for call in extract.call_args_list], [[p] for p in calibration_paths])
            save_memory.assert_not_called()
            save_calibration.assert_called_once()

    def test_gui_loads_build_signature_and_excludes_all_calibration_bags(self):
        from rosbag_io import RosbagImageSource
        from vision_dinov2 import run_vision_test as runner
        from rynnbrain_vlm.replay_detectors import DinoReplayDetector
        topic = '/camera'
        table, rows = build_sequence_threshold(self.scores())
        with tempfile.TemporaryDirectory() as directory:
            memory_path = str(Path(directory) / 'nominal_memory_bag')
            calibration_paths = [str(Path(directory) / f'calibration_{i}') for i in range(26)]
            sources = [RosbagImageSource(path, topic) for path in calibration_paths]
            with patch.multiple(runner, NOMINAL_VIDEO_PATHS=[RosbagImageSource(memory_path, topic)],
                    CALIBRATION_VIDEO_PATHS=sources):
                signature = runner.nominal_cache_signature()
            save_nominal_cache(Path(directory) / 'camera', [self.item()], 2, table, rows, signature)
            config = dict(nominal_cache_dir=directory, rynnbrain={'camera_topics': [topic]},
                          dino_calibration_bags=calibration_paths, dino_threshold_scale=1.2)
            with patch('vision_dinov2.dinov2_features.DINOv2FeatureExtractor'):
                detector = DinoReplayDetector(config, lambda message: None)
                self.assertEqual(detector.threshold_scale, 1.2)
                self.assertEqual(detector.signature['calibration_method'], SEQUENCE_CALIBRATION_METHOD)
                self.assertEqual(len(detector.excluded), 27)
                pd.testing.assert_frame_equal(detector.thresholds, table, check_exact=False)
                with self.assertRaisesRegex(ValueError, 'calibration'):
                    detector.reset(calibration_paths[0])
                config['dino_calibration_bags'] = calibration_paths[1:]
                with self.assertRaisesRegex(ValueError, 'No compatible DINO memory'):
                    DinoReplayDetector(config, lambda message: None)


if __name__ == '__main__':
    unittest.main()
