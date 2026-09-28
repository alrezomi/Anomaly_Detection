from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from rynnbrain_vlm.cop_knn import CoPKNNDetector, fit_nominal_memory
from rynnbrain_vlm.cop_neighbors import write_knn_neighbors


class NeighborPlotTests(unittest.TestCase):
    def test_distances_ranks_and_highlights_match_classifier_without_changing_memory(self):
        # Test ordering is deliberately different from training order; ties remain stable.
        memory, threshold, _, _ = fit_nominal_memory([[0, 1, 0], [1, 0, 0], [1, 0, 0], [-1, 0, 0]], n_neighbors=2)
        original = memory.copy()
        classifier = CoPKNNDetector(memory, 2, threshold, "raw", "test", {"adapter": "a"},
                                    {"training_bags": ["far", "near_a", "near_b", "furthest"]})
        query = np.array([1, .1, 0])
        with tempfile.TemporaryDirectory() as temporary:
            report = write_knn_neighbors(classifier, query, comparison_signature={"adapter": "a"},
                directory=Path(temporary), mode="raw", bag_name="test", provenance={"evaluation_id": "eval"})
            rows = pd.read_csv(report["csv"])
            self.assertEqual(rows.nominal_bag.tolist(), ["near_a", "near_b", "far", "furthest"])
            self.assertEqual(rows.used_for_score.tolist(), [True, True, False, False])
            self.assertAlmostEqual(rows.loc[rows.used_for_score, "distance"].mean(), classifier.predict_anomaly_score(query))
            self.assertTrue((rows.evaluation_id == "eval").all())
            self.assertGreater(Path(report["plot"]).stat().st_size, 1000)
        np.testing.assert_array_equal(memory, original)

    def test_mismatched_features_are_rejected_before_writing(self):
        memory, threshold, _, _ = fit_nominal_memory([[0, 1], [1, 0]], n_neighbors=1)
        classifier = CoPKNNDetector(memory, 1, threshold, "raw", "test", {"adapter": "a"},
                                    {"training_bags": ["a", "b"]})
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "different RynnBrain"):
                write_knn_neighbors(classifier, [1, 0], comparison_signature={"adapter": "b"},
                    directory=Path(temporary), mode="raw", bag_name="test", provenance={})
            self.assertEqual(list(Path(temporary).iterdir()), [])
