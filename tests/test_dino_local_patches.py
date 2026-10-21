"""Spatial tolerance must suppress displacement without losing missing content."""
import unittest
from unittest.mock import patch

import numpy as np

from vision_dinov2.anomaly_detection import (symmetric_local_patch_distance,
    compare_frame_to_nominal_memory, make_attention_weights)


class LocalPatchTests(unittest.TestCase):
    @staticmethod
    def item(features, stage=1, cls=None, attention=None):
        return dict(features=features.reshape(-1, 2), stage=stage,
                    cls=np.array([1., 0.]) if cls is None else cls,
                    attention=np.ones(64) if attention is None else attention)

    def test_all_patches_use_one_frame_and_local_search_runs_once(self):
        query, left, right = self.scene(), self.scene(), self.scene()
        query[2, 2] = left[2, 2] = [0, 1]
        query[5, 5] = right[5, 5] = [0, 1]
        self.assertGreater(self.distance(query, left).max(), 0)
        self.assertGreater(self.distance(query, right).max(), 0)
        memory = [self.item(left), self.item(right)]
        with patch('vision_dinov2.anomaly_detection.symmetric_local_patch_distance',
                   wraps=symmetric_local_patch_distance) as local:
            result = compare_frame_to_nominal_memory(query.reshape(-1, 2), np.ones(64),
                                                     np.array([1., 0.]), memory, required_stage=1)
            self.assertEqual(local.call_count, 1)
        np.testing.assert_array_equal(result[1], self.distance(query, result[4]['features'].reshape(8, 8, 2)).ravel())
        self.assertGreater(result[1].max(), 0)
        # A novel feature absent from every local candidate must survive.
        query[6, 1] = [-1, 0]
        result = compare_frame_to_nominal_memory(query.reshape(-1, 2), np.ones(64),
                                                 np.array([1., 0.]), memory, required_stage=1)
        self.assertGreater(result[1].reshape(8, 8)[6, 1], 0)

    def test_patch_search_respects_cls_shortlist_and_stage(self):
        query = self.scene()
        query[3, 3] = [0, 1]
        memory = [self.item(self.scene()) for _ in range(5)]
        memory += [self.item(query, cls=np.array([-1., 0.])), self.item(query, stage=2)]
        args = (query.reshape(-1, 2), np.ones(64), np.array([1., 0.]), memory)
        result = compare_frame_to_nominal_memory(*args, required_stage=1)
        self.assertGreater(result[1][27], 0)
        result = compare_frame_to_nominal_memory(*args, required_stage=1, top_k_cls=6)
        np.testing.assert_array_equal(result[1], 0)
        with self.assertRaisesRegex(ValueError, 'stage 3'):
            compare_frame_to_nominal_memory(*args, required_stage=3)

    @staticmethod
    def scene():
        return np.tile(np.array([1., 0.], dtype=np.float32), (8, 8, 1))

    def distance(self, query, reference):
        return symmetric_local_patch_distance(query.reshape(-1, 2), reference.reshape(-1, 2)).reshape(8, 8)

    def test_one_patch_translation_is_tolerated_but_far_move_is_not(self):
        reference = self.scene()
        reference[2:4, 2:4] = [0, 1]
        shifted = self.scene()
        shifted[3:5, 3:5] = [0, 1]
        self.assertGreater(np.linalg.norm(shifted - reference), 0)
        np.testing.assert_array_equal(self.distance(shifted, reference), 0)
        far = self.scene()
        far[5:7, 5:7] = [0, 1]
        residual = self.distance(far, reference)
        self.assertGreater(residual[2, 2], 0)  # Missing expected object.
        self.assertGreater(residual[6, 6], 0)  # Object in the wrong place.

    def test_missing_and_extra_object_both_survive_background_matches(self):
        background = self.scene()
        object_scene = self.scene()
        object_scene[3, 3] = [0, 1]
        missing = self.distance(background, object_scene)
        extra = self.distance(object_scene, background)
        np.testing.assert_array_equal(missing, extra)
        self.assertEqual(missing[3, 3], 1)
        self.assertEqual(np.count_nonzero(missing), 1)
        np.testing.assert_array_equal(self.distance(object_scene, object_scene), 0)

    def test_border_does_not_wrap_to_opposite_edge(self):
        reference, query = self.scene(), self.scene()
        reference[0, 7], query[0, 0] = [0, 1], [0, 1]
        residual = self.distance(query, reference)
        self.assertEqual(residual[0, 0], 1)
        self.assertEqual(residual[0, 7], 1)

    def test_matches_explicit_bidirectional_search_and_is_bounded_by_aligned_distance(self):
        rng = np.random.default_rng(9)
        query, reference = rng.normal(size=(2, 4, 4, 5)).astype(np.float32)
        expected = np.empty((4, 4))
        for y in range(4):
            for x in range(4):
                ys, xs = slice(max(0, y - 1), y + 2), slice(max(0, x - 1), x + 2)
                forward = np.mean((query[y, x] - reference[ys, xs]) ** 2, axis=-1).min()
                reverse = np.mean((reference[y, x] - query[ys, xs]) ** 2, axis=-1).min()
                expected[y, x] = np.sqrt(max(forward, reverse))
        actual = symmetric_local_patch_distance(query.reshape(16, 5), reference.reshape(16, 5))
        np.testing.assert_allclose(actual, expected.ravel(), atol=1e-7)
        aligned = np.sqrt(np.mean((query - reference) ** 2, axis=-1)).ravel()
        self.assertTrue((actual <= aligned + 1e-7).all())

    def test_invalid_features_fail_explicitly(self):
        for query, reference in [(np.zeros((3, 2)), np.zeros((3, 2))),
                                  (np.zeros((4, 2)), np.zeros((4, 3))),
                                  (np.full((4, 2), np.nan), np.zeros((4, 2)))]:
            with self.assertRaises(ValueError):
                symmetric_local_patch_distance(query, reference)


if __name__ == "__main__":
    unittest.main()
