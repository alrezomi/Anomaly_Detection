"""Behavioral regressions for nuisance variation and localized evidence."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from vision_dinov2.patch_normality import fit_patch_profile, patch_evidence, validate_patch_profile
from vision_dinov2.threshold import build_attention_threshold, calibrate_attention_leave_one_video_out
from vision_dinov2.nominal_cache import save_nominal_cache, load_nominal_cache, load_feature_cache


class PatchNormalityTests(unittest.TestCase):
    @staticmethod
    def memory():
        memory = []
        for run in range(6):
            for frame, progress in enumerate([.15, .5, .85]):
                # Top half models an arm whose appearance varies across normal
                # runs; bottom half models a stable table/object area.
                angle = np.full((32, 32), run * .0005)
                angle[:16] = run ** 2 * .025 + frame * .1
                features = np.stack([np.cos(angle), np.sin(angle)], axis=-1).reshape(-1, 2)
                memory.append(dict(features=features, attention=np.ones(1024),
                    cls=np.array([np.cos(progress), np.sin(progress)]),
                    progress=progress, stage=frame + 1, video_id=run, frame_id=frame,
                    timestamp_sec=float(frame), video_path=f"normal_{run}", _progress_locked=True))
        return memory

    def test_learned_variation_suppresses_normal_arm_but_preserves_small_stable_change(self):
        profile = fit_patch_profile(self.memory(), n_bins=3, stage_constrained_matching=True)
        mean, scale = profile["mean"][1], profile["scale"][1]
        self.assertGreater(mean[:512].mean(), 20 * mean[512:].mean())
        self.assertGreater(scale[:512].mean(), scale[512:].mean())
        weights = np.ones(1024)
        weights[512:] = .1  # The target is NOT an attention hotspot.
        normal = mean.copy()
        normal[:512] += .5 * scale[:512]
        changed = normal.reshape(32, 32).copy()
        changed[23:27, 10:14] += (8 * scale.reshape(32, 32))[23:27, 10:14]
        normal_score, _ = patch_evidence(normal, weights, .5, profile)
        failure_score, heatmap = patch_evidence(changed.ravel(), weights, .5, profile)
        self.assertGreater(failure_score, 4 * normal_score)
        y, x = divmod(int(heatmap.argmax()), 32)
        self.assertTrue(23 <= y < 27 and 10 <= x < 14)
        # No permanent arm ignore mask: a sufficiently unusual arm change counts.
        changed_arm = normal.copy()
        changed_arm[:512] += 8 * scale[:512]
        self.assertGreater(patch_evidence(changed_arm, weights, .5, profile)[0], failure_score)

    def test_final_evidence_uses_soft_attention_without_redistribution(self):
        profile = fit_patch_profile(self.memory(), n_bins=3)
        distances = profile["mean"][1].copy().reshape(32, 32)
        distances[20:24, 10:14] += 5 * profile["scale"][1].reshape(32, 32)[20:24, 10:14]
        score, heatmap = patch_evidence(distances.ravel(), np.ones(1024), .5, profile)
        low_score, low_map = patch_evidence(distances.ravel(), np.full(1024, .1), .5, profile)
        self.assertGreater(score, 0)
        self.assertAlmostEqual(low_score, .55 * score)
        np.testing.assert_allclose(low_map, .55 * heatmap)
        self.assertAlmostEqual(score, np.sort(heatmap)[-11:].mean())
        self.assertEqual(heatmap.reshape(32, 32)[0, 0], 0)

    def test_previous_calibration_is_rejected_but_features_reusable(self):
        from vision_dinov2.threshold import ATTENTION_CALIBRATION_METHOD
        memory = self.memory()
        table, rows = build_attention_threshold(pd.DataFrame(dict(score=[1., 2.],
            matched_progress=[.5, .5], held_out_video_id=[0, 1])))
        table.attrs["patch_profile"] = fit_patch_profile(memory, n_bins=3)
        old = dict(calibration_method="matched_patch_normality_progress_v3_no_attention", score_mode="attention")
        new = dict(old, calibration_method=ATTENTION_CALIBRATION_METHOD)
        with tempfile.TemporaryDirectory() as folder:
            save_nominal_cache(folder, memory, 32, table, rows, old)
            self.assertIsNone(load_nominal_cache(folder, new))
            self.assertIsNotNone(load_feature_cache(folder, new))

    def test_profile_excludes_self_matches_and_equal_weights_recordings(self):
        from vision_dinov2.anomaly_detection import compare_frame_to_nominal_memory
        memory = self.memory()
        def match(features, attention, cls, references, **kwargs):
            query = next(item for item in memory if item["features"] is features)
            self.assertNotIn(query["video_id"], {item["video_id"] for item in references})
            return compare_frame_to_nominal_memory(features, attention, cls, references, **kwargs)
        with patch("vision_dinov2.anomaly_detection.compare_frame_to_nominal_memory", side_effect=match):
            profile = fit_patch_profile(memory, n_bins=3)
        # Duplicate all frames of a single run; its moments and contribution stay unchanged.
        repeated = fit_patch_profile(memory + [x for x in memory if x["video_id"] == 0] * 5, n_bins=3,
                                     top_k_cls=100)
        original = fit_patch_profile(memory, n_bins=3, top_k_cls=100)
        np.testing.assert_allclose(repeated["mean"], original["mean"], atol=1e-7)
        np.testing.assert_allclose(repeated["scale"], original["scale"], atol=1e-7)
        self.assertTrue(np.isfinite(profile["scale"]).all())

    def test_sparse_and_constant_profiles_have_finite_nonzero_scales(self):
        memory = self.memory()
        for item in memory:
            item["features"] = np.tile([1., 0.], (1024, 1))
            item["progress"] = .5
        profile = fit_patch_profile(memory)
        validate_patch_profile(profile, 1024)
        self.assertTrue((profile["scale"] > 0).all())
        score, heatmap = patch_evidence(np.zeros(1024), np.ones(1024), 0., profile)
        self.assertEqual(score, 0.)
        self.assertTrue(np.isfinite(heatmap).all())
        with self.assertRaisesRegex(ValueError, "Invalid"):
            validate_patch_profile(dict(profile, scale=profile["scale"] * 0), 1024)
        with self.assertRaises(ValueError):
            patch_evidence(np.zeros(16), np.ones(16), .5, profile)

    def test_threshold_does_not_erase_nominal_peak_or_overweight_long_recording(self):
        rows = pd.DataFrame([dict(held_out_video_id=run, matched_progress=p, score=score + run * .01)
                             for run in range(6)
                             for p, score in [(.1, 1.), (.5, 5.), (.9, 1.)] for _ in range(3)])
        unsmoothed, _ = build_attention_threshold(rows, n_bins=3, smooth_window=1)
        table, _ = build_attention_threshold(rows, n_bins=3)
        self.assertGreaterEqual(table.adaptive_threshold.iloc[1], unsmoothed.adaptive_threshold.iloc[1])
        repeated = pd.concat([rows] + [rows[rows.held_out_video_id == 0]] * 10)
        repeated_table, _ = build_attention_threshold(repeated, n_bins=3)
        np.testing.assert_allclose(table.adaptive_threshold, repeated_table.adaptive_threshold)
        self.assertTrue((table.run_count == 6).all())

    def test_outer_calibration_run_is_excluded_from_patch_profile(self):
        from vision_dinov2.threshold import score_calibration_memory
        actual_fit = fit_patch_profile
        def fit(memory, **kwargs):
            profile = actual_fit(memory, **kwargs)
            profile["_test_ids"] = {item["video_id"] for item in memory}
            return profile
        def score(queries, references, run_id, **kwargs):
            self.assertNotIn(run_id, kwargs["patch_profile"]["_test_ids"])
            self.assertNotIn(run_id, {item["video_id"] for item in references})
            return score_calibration_memory(queries, references, run_id, **kwargs)
        with patch("vision_dinov2.patch_normality.fit_patch_profile", side_effect=fit), \
                patch("vision_dinov2.threshold.score_calibration_memory", side_effect=score):
            table, rows = calibrate_attention_leave_one_video_out(self.memory(), n_bins=3)
        self.assertEqual(len(rows), 18)
        self.assertEqual(table.attrs["patch_profile"]["training_runs"], 6)

    def test_profile_is_required_for_new_cache_and_features_survive_missing_profile(self):
        from vision_dinov2.threshold import ATTENTION_CALIBRATION_METHOD
        memory = self.memory()
        table, rows = build_attention_threshold(pd.DataFrame(dict(score=[1., 2.],
            matched_progress=[.5, .5], held_out_video_id=[0, 1])))
        profile = fit_patch_profile(memory, n_bins=3)
        table.attrs["patch_profile"] = profile
        signature = dict(calibration_method=ATTENTION_CALIBRATION_METHOD, score_mode="attention")
        with tempfile.TemporaryDirectory() as folder:
            save_nominal_cache(folder, memory, 32, table, rows, signature)
            loaded = load_nominal_cache(folder, signature)
            np.testing.assert_array_equal(loaded[2].attrs["patch_profile"]["mean"], profile["mean"])
            (Path(folder) / "patch_normality.npz").unlink()
            self.assertIsNone(load_nominal_cache(folder, signature))
            self.assertIsNotNone(load_feature_cache(folder, signature))


if __name__ == "__main__":
    unittest.main()
