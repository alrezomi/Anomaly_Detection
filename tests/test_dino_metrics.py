"""Metric arithmetic, held-out fitting, and independent decision limits."""
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from vision_dinov2.metrics import validate_metric, mahalanobis_patches, local_residual_vectors
from vision_dinov2.patch_normality import fit_patch_profile, patch_evidence, top_five_percent
from vision_dinov2.threshold import (calibrate_attention_leave_one_video_out, score_calibration_memory,
    score_matched_frame, scaled_threshold, ATTENTION_CALIBRATION_METHOD)
from vision_dinov2.nominal_cache import save_nominal_cache, load_nominal_cache, load_feature_cache


def memory():
    rng = np.random.default_rng(21)
    base = rng.normal(size=(16, 6))
    result = []
    for run in range(4):
        for stage in (1, 2):
            features = base + rng.normal(scale=.02, size=base.shape) + stage * .1
            features /= np.linalg.norm(features, axis=1, keepdims=True)
            result.append(dict(features=features.astype(np.float32), attention=np.arange(16) / 16,
                cls=np.array([1., .1 * stage]), stage=stage, progress=.25 if stage == 1 else .75,
                video_id=run, frame_id=stage, timestamp_sec=float(stage), video_path=f'run_{run}',
                _progress_locked=True))
    return result


class MetricTests(unittest.TestCase):
    def test_five_percent_not_max_or_whole_frame(self):
        values = np.arange(100, dtype=float)
        self.assertEqual(top_five_percent(values), 97.)
        self.assertEqual(top_five_percent(np.arange(21.)), 19.5)

    def test_three_euclidean_scores_and_no_pooling(self):
        profile = dict(centers=np.array([.5]), mean=np.ones((1, 100)), scale=np.full((1, 100), 2.))
        raw = np.ones(100)
        raw[50:55] = 9.
        scores, diagnostics = {}, {}
        primary, heatmap = patch_evidence(raw, np.zeros(100), .5, profile, scores=scores, diagnostics=diagnostics)
        self.assertEqual(scores, dict(baseline=8., scaled=4., weighted=2.))
        self.assertEqual(primary, 2.)
        self.assertEqual(np.count_nonzero(heatmap), 5)  # no spatial spreading
        np.testing.assert_array_equal(heatmap, diagnostics['weighted'])

    def test_mahalanobis_known_diagonal_covariance_and_attention(self):
        query = np.tile([2., 3.], (4, 1))
        reference = np.zeros_like(query)
        profile = dict(metric=np.asarray('mahalanobis'), centers=np.array([.5]),
            channels=np.array([0, 1]), vector_mean=np.zeros((1, 4, 2)),
            vector_variance=np.tile([4., 9.], (1, 4, 1)), mean=np.zeros((1, 4)), scale=np.ones((1, 4)))
        distances = mahalanobis_patches(query, reference, .5, profile)
        np.testing.assert_allclose(distances, np.sqrt(2))
        args = dict(query_features=query, reference_features=reference)
        high, _ = patch_evidence(np.zeros(4), np.ones(4), .5, profile, **args)
        low, _ = patch_evidence(np.zeros(4), np.full(4, .1), .5, profile, **args)
        self.assertAlmostEqual(high, np.sqrt(2))
        self.assertAlmostEqual(low, .55 * high)
        with self.assertRaisesRegex(ValueError, 'requires'):
            patch_evidence(np.zeros(4), np.ones(4), .5, profile)

    def test_local_vectors_do_not_wrap_and_keep_reverse_missing_content(self):
        query, reference = np.zeros((16, 2)), np.zeros((16, 2))
        reference[0] = 5
        forward, reverse = local_residual_vectors(query, reference, np.arange(2))
        np.testing.assert_array_equal(forward, 0)
        np.testing.assert_array_equal(reverse[0], -5)
        query[:] = 5
        query[3] = 0
        reference[:] = 5
        reference[0] = 0
        forward, _ = local_residual_vectors(query, reference, np.arange(2))
        np.testing.assert_array_equal(forward[3], -5)  # opposite edge isn't a neighbor

    def test_metrics_reject_unknown_values(self):
        for value in ['cosine', 'ecludian', None]:
            with self.assertRaises(ValueError):
                validate_metric(value)

    def test_both_metrics_crossfit_cache_and_runtime_agree(self):
        from vision_dinov2.anomaly_detection import compare_frame_to_nominal_memory, run_video_test
        from rynnbrain_vlm.replay_detectors import DinoReplayDetector
        from vision_dinov2.visualization import create_anomaly_heatmap_video
        data = memory()
        query = dict(data[0], features=data[0]['features'] + .03)
        image = Image.new('RGB', (56, 56))
        for metric in ['euclidean', 'mahalanobis']:
            with self.subTest(metric=metric):
                table, rows = calibrate_attention_leave_one_video_out(data, metric=metric, n_bins=2,
                                                                      stage_constrained_matching=True)
                variants = ['mahalanobis'] if metric == 'mahalanobis' else ['baseline', 'scaled', 'weighted']
                for name in variants:
                    self.assertIn(f'threshold_{name}', table)
                    self.assertTrue(np.isfinite(rows[f'score_{name}']).all())
                self.assertTrue((rows.matched_video_id != rows.held_out_video_id).all())
                signature = dict(calibration_method=ATTENTION_CALIBRATION_METHOD, dino_metric=metric)
                with tempfile.TemporaryDirectory() as directory:
                    save_nominal_cache(directory, data, 4, table, rows, signature)
                    loaded = load_nominal_cache(directory, signature)
                    self.assertIsNotNone(loaded)
                    changed = dict(signature, dino_metric='euclidean' if metric == 'mahalanobis' else 'mahalanobis')
                    self.assertIsNone(load_nominal_cache(directory, changed))
                    self.assertIsNotNone(load_feature_cache(directory, changed))
                    table = loaded[2]
                result = compare_frame_to_nominal_memory(query['features'], query['attention'], query['cls'], data,
                                                        min_weight=.1, required_stage=1)
                raw, patches, _, weights, best, _ = result
                decision = score_matched_frame(raw, best['progress'], table, 1.2, patches=patches,
                    attention_weights=weights, query_features=query['features'], reference_features=best['features'])
                scaled = score_matched_frame(raw, best['progress'], scaled_threshold(table, 1.2), patches=patches,
                    attention_weights=weights, query_features=query['features'], reference_features=best['features'])
                self.assertEqual(decision['score_variants'], scaled['score_variants'])
                calibration = score_calibration_memory([query], data, 42, patch_profile=table.attrs['patch_profile'],
                                                        stage_constrained_matching=True)[0]
                for name in variants:
                    self.assertAlmostEqual(decision['score_variants'][name]['score'], calibration[f'score_{name}'])
                extractor = Mock()
                extractor.extract.return_value = query['features'], query['attention'], query['cls'], 4
                detector = DinoReplayDetector.__new__(DinoReplayDetector)
                detector.extractor, detector.memory, detector.thresholds = extractor, data, table
                detector.signature = dict(top_k_cls=5, attention_power=1.5, min_weight=.1)
                detector.stages, detector.step, detector.threshold_scale = Mock(), 0, 1.2
                detector.stages.at.return_value = 1
                detector._thumbnail = Mock(return_value=('memory', image))
                replay, _, _ = detector.append(image, 1.)
                self.assertEqual(replay['score_variants'], decision['score_variants'])
                # Restrict test memory to stage 1, equivalent to replay's filter.
                refs = [item for item in data if item['stage'] == 1]
                with patch('vision_dinov2.anomaly_detection.iter_source_frames', return_value=iter([(0, 1., image)])):
                    batch = run_video_test(extractor, 'test', refs, scaled_threshold(table, 1.2), min_weight=.1).iloc[0]
                with tempfile.TemporaryDirectory() as directory, \
                        patch('vision_dinov2.visualization.iter_source_frames', return_value=iter([(0, 1., image)])), \
                        patch('vision_dinov2.visualization.cv2.VideoWriter'):
                    video = create_anomaly_heatmap_video(extractor, 'test', refs, directory+'/test.mp4',
                                                        scaled_threshold(table, 1.2), min_weight=.1).iloc[0]
                self.assertAlmostEqual(batch.score, replay['anomaly_score'])
                self.assertAlmostEqual(video.score, replay['anomaly_score'])

    def test_each_variant_has_an_independent_threshold_and_alarm(self):
        from vision_dinov2.threshold import build_attention_threshold
        import pandas as pd
        data = pd.DataFrame([dict(held_out_video_id=i, matched_progress=.5, score=2.,
                                 score_baseline=.1, score_scaled=4., score_weighted=2.) for i in range(4)])
        table, _ = build_attention_threshold(data, n_bins=1)
        self.assertAlmostEqual(table.threshold_scaled.iloc[0], 40 * table.threshold_baseline.iloc[0])
        profile = dict(centers=np.array([.5]), mean=np.zeros((1, 4)), scale=np.ones((1, 4)))
        table.attrs['patch_profile'] = profile
        result = score_matched_frame(0, .5, table, patches=np.ones(4), attention_weights=np.ones(4))
        self.assertTrue(result['score_variants']['baseline']['above_threshold'])
        self.assertFalse(result['score_variants']['scaled']['above_threshold'])
