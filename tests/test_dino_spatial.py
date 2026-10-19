import unittest
from unittest.mock import patch

import numpy as np

from vision_dinov2.spatial_scoring import spatial_attention_distances
from vision_dinov2.anomaly_detection import compare_frame_to_nominal_memory
from vision_dinov2.threshold import score_calibration_memory


class SpatialDinoTests(unittest.TestCase):
    def scene(self, column=None):
        features = np.tile([1., 0.], (7, 7, 1))
        if column is not None:
            features[3, column] = [0., 1.]
        return features.reshape(-1, 2)

    def distance(self, query, nominal):
        return spatial_attention_distances(query, nominal, np.ones(49), np.ones(49), min_weight=1.)

    def test_identical_small_shift_large_shift_and_missing_object(self):
        reference = self.scene(2)
        same = self.distance(reference, reference)
        nearby = self.distance(self.scene(3), reference)
        distant = self.distance(self.scene(5), reference)
        missing = self.distance(self.scene(), reference)
        self.assertEqual(same.max(), 0.)
        self.assertGreater(nearby.max(), 0.)  # displacement is not free
        self.assertLess(nearby.max(), distant.max())
        self.assertGreater(missing.max(), nearby.max())
        np.testing.assert_allclose(missing, self.distance(reference, self.scene()))
        # Border searches must not wrap to the opposite image edge.
        self.assertGreater(self.distance(self.scene(0), self.scene(6)).max(), nearby.max())

    def test_reference_attention_preserves_missing_object_and_soft_floor(self):
        q, r = self.scene(), self.scene(2)
        attention = np.zeros(49)
        attention[3 * 7 + 2] = 1.
        focused = spatial_attention_distances(q, r, np.zeros(49), attention)
        low = spatial_attention_distances(q, r, np.zeros(49), np.zeros(49))
        self.assertGreater(focused.max(), low.max())
        self.assertGreater(low.max(), 0.)
        np.testing.assert_allclose(focused, spatial_attention_distances(r, q, attention, np.zeros(49)))

    def test_selected_frame_unchanged_and_calibration_uses_same_spatial_scores(self):
        def item(column, stage):
            return dict(features=self.scene(column), attention=np.linspace(0, 1, 49), cls=np.array([1., 0.]),
                        stage=stage, video_id=0, frame_id=0, timestamp_sec=0., progress=.5)
        query, reference = item(3, 2), item(2, 2)
        candidates = [item(3, 1), reference]
        args = (query['features'], query['attention'], query['cls'], candidates)
        old = compare_frame_to_nominal_memory(*args, required_stage=2)
        new = compare_frame_to_nominal_memory(*args, required_stage=2, local_scoring=True)
        self.assertIs(old[4], new[4])
        np.testing.assert_array_equal(new[2], old[1])  # raw display map
        maps = []
        score_calibration_memory([query], candidates, 1, stage_constrained_matching=True,
                                  patch_maps=maps, min_weight=.15)
        np.testing.assert_allclose(maps[0], new[1])

    def test_invalid_grids_and_nonfinite_features_rejected(self):
        with self.assertRaises(ValueError):
            self.distance(np.ones((5, 2)), np.ones((5, 2)))
        with self.assertRaises(ValueError):
            self.distance(np.full((49, 2), np.nan), self.scene())
