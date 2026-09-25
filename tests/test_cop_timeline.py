from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from rynnbrain_vlm.cop_analysis import save_cop_vector, discover_saved_vectors
from rynnbrain_vlm.cop_classifier import train_from_saved_vectors, load_classifier
from rynnbrain_vlm.cop_timeline import (
    selected_frame_prefixes, prepare_nominal_prefixes, train_prefix_detector,
    load_prefix_detector, write_knn_timeline, prefix_model_path,
)


class PrefixTimelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.signature = {"num_frames": 3, "hidden_size": 4, "turn1_prompt": "reference",
                          "turn2_prompt": "evaluate", "lora_adapter_sha256": "adapter-1"}
        self.vectors = {
            f"normal_{i}": np.asarray([[1, offset, 0, 0], [0, 1, 2 * offset, 0], [0, 0, 1, 3 * offset]], dtype=np.float32)
            for i, offset in enumerate((0, 0.1, 0.2, 0.3))
        }
        for name, vectors in self.vectors.items():
            save_cop_vector(self.root / name / "rynnbrain_multiturn", "raw", vectors[-1], {
                "bag_name": name, "test_bag": f"/data/{name}", "evaluation_id": name + "-eval",
                "ground_truth_label": "normal", "representation_id": "test",
                "comparison_signature": self.signature,
            })
        self.records = discover_saved_vectors(self.root)
        self.path = self.root / "raw_knn.npz"
        self.metadata = train_from_saved_vectors(self.root, self.path, method="knn",
            bag_names=list(self.vectors), knn_options={"n_neighbors": 2, "threshold_quantile": 0.95})
        self.classifier = load_classifier(self.path)
        self.frames = [{"timestamp_sec": value, "sample_index": i} for i, value in enumerate((2.5, 6.2, 11.4))]
        self.images = [(f"image-{i}", object()) for i in range(3)]
        self.turns = [{"role": "user", "images": [], "text": "reference"},
                      {"role": "user", "images": self.images, "text": "evaluate"}]
        self.query = np.asarray([[1, 0.15, 0, 0], [0, 1, 0.3, 0], [0, 0, 1, 0.45]], dtype=np.float32)

    def prepare(self):
        def inputs(config, vlm, count, modes):
            name = Path(config["test_bag"]).name
            return {"raw": ([(name, object()) for _ in range(3)], self.frames)}, self.frames
        def extract(turns, generation, **kwargs):
            images = turns[1]["images"]
            self.assertEqual(kwargs["reference_response"], "reference response")
            return self.vectors[images[0][0]][len(images) - 1]
        self.training_model = SimpleNamespace(extract_multiturn_cop_vector=Mock(side_effect=extract))
        try:
            from rynnbrain_vlm import run
        except ImportError as error:
            self.skipTest(str(error))
        with patch.object(run, "_execution_inputs", side_effect=inputs):
            prepare_nominal_prefixes(self.training_model, {}, {}, self.records, [], "reference response", {})
        train_prefix_detector(self.records, self.path, self.metadata)

    def write(self, model=None):
        if model is None:
            model = SimpleNamespace(extract_multiturn_cop_vector=Mock(
                side_effect=lambda turns, *args, **kwargs: self.query[len(turns[1]["images"]) - 1]))
        self.test_model = model
        with patch("builtins.print"):
            return write_knn_timeline(model=model, classifier=self.classifier, turns=self.turns,
                generation={}, nominal_response="reference response", frame_metadata=self.frames,
                final_vector=self.query[-1], comparison_signature=self.signature,
                output_directory=self.root, mode="raw", bag_name="held_out",
                provenance={"evaluation_id": "test-eval", "lora_adapter_sha256": "adapter-1"},
                classifier_model_path=str(self.path))

    def test_prefixes_use_actual_times_and_group_cameras_without_future_images(self):
        frames = [{"timestamp_sec": time, "sample_index": step} for step, time in
                  [(0, 1.0), (0, 1.1), (1, 2.7), (1, 2.8), (2, 4.0), (2, 4.1)]]
        snapshots = selected_frame_prefixes(frames, 6, 3)
        self.assertEqual([row["timestamp_sec"] for row in snapshots], [1.1, 2.8, 4.1])
        self.assertEqual([row["image_indices"] for row in snapshots], [[0, 1], [0, 1, 2, 3], list(range(6))])
        for row in snapshots:
            self.assertTrue(all(frames[i]["timestamp_sec"] <= row["timestamp_sec"] for i in row["image_indices"]))
        for metadata, count, steps in (([], 3, 3), (frames, 5, 3), (frames, 6, 4),
                ([{"timestamp_sec": float("nan")}], 1, 1), ([{"timestamp_sec": 2}, {"timestamp_sec": 1}], 2, 2)):
            with self.assertRaises(ValueError):
                selected_frame_prefixes(metadata, count, steps)

    def test_shared_loader_preserves_modes_image_order_and_heatmap_timestamps(self):
        from rynnbrain_vlm.run import _execution_inputs
        raw = [("raw-1", object()), ("raw-2", object())]
        heatmap = [("heat-1", object()), ("heat-2", object())]
        raw_times = [{"timestamp_sec": 1, "sample_index": 0}, {"timestamp_sec": 4, "sample_index": 1}]
        heat_times = [{"timestamp_sec": 1.2, "sample_index": 0}, {"timestamp_sec": 4.2, "sample_index": 1}]
        config = {"camera_topics": ["cam"], "output_dir": str(self.root), "test_bag": "/data/test"}
        with patch("rynnbrain_vlm.run._raw_inputs", return_value=(raw, raw_times)), \
             patch("rynnbrain_vlm.run._heatmap_inputs", return_value=(heatmap, heat_times)):
            modes, frames = _execution_inputs(config, {"source": "rosbag"}, 2, ["raw", "heatmap", "raw_heatmap"])
        self.assertEqual(modes["raw"], (raw, raw_times))
        self.assertEqual(modes["heatmap"], (heatmap, heat_times))
        self.assertEqual([row[0] for row in modes["raw_heatmap"][0]], ["raw-1", "heat-1", "raw-2", "heat-2"])
        self.assertEqual([row["timestamp_sec"] for row in modes["raw_heatmap"][1]], [1, 1.2, 4, 4.2])
        self.assertEqual(frames, raw_times)

    def test_nominal_prefix_fit_excludes_self_and_matches_full_detector_at_last_point(self):
        self.prepare()
        self.assertEqual(self.training_model.extract_multiturn_cop_vector.call_count, 8)
        detector = load_prefix_detector(self.path, self.classifier)
        for step in range(3):
            memory = np.stack([self.vectors[name][step] for name in self.vectors]).astype(float)
            memory /= np.linalg.norm(memory, axis=1)[:, None]
            distances = np.linalg.norm(memory[:, None, :] - memory[None, :, :], axis=2)
            np.fill_diagonal(distances, np.inf)
            loo = np.sort(distances, axis=1)[:, :2].mean(axis=1)
            np.testing.assert_allclose(detector.memories[step], memory)
            self.assertAlmostEqual(detector.thresholds[step], np.nextafter(np.quantile(loo, .95, method="higher"), np.inf))
        self.assertEqual(detector.thresholds[-1], self.classifier.threshold)
        # A healthy start is close to nominal starts, though far from nominal finishes.
        start = detector.at_step(0, self.classifier)
        self.assertLess(start.predict_anomaly_score(self.query[0]), start.threshold)
        self.assertGreater(self.classifier.predict_anomaly_score(self.query[0]), self.classifier.threshold)

    def test_cache_reuse_refreshes_only_stale_nominal_prefixes(self):
        self.prepare()
        self.training_model.extract_multiturn_cop_vector.reset_mock()
        prepare_nominal_prefixes(self.training_model, {}, {}, self.records, [], "reference response", {})
        self.training_model.extract_multiturn_cop_vector.assert_not_called()
        record = self.records[0]
        record.metadata["evaluation_id"] += "-new"
        images = [(record.metadata["bag_name"], object()) for _ in range(3)]
        with patch("rynnbrain_vlm.run._execution_inputs", return_value=({"raw": (images, self.frames)}, self.frames)):
            prepare_nominal_prefixes(self.training_model, {}, {}, self.records, [], "reference response", {})
        self.assertEqual(self.training_model.extract_multiturn_cop_vector.call_count, 2)

    def test_timeline_preserves_final_result_and_records_matching_thresholds_and_identity(self):
        self.prepare()
        report = self.write()
        self.assertEqual(report["status"], "created")
        rows = pd.read_csv(report["csv"])
        self.assertEqual(rows.timestamp_sec.tolist(), [2.5, 6.2, 11.4])
        self.assertEqual(rows.image_count.tolist(), [1, 2, 3])
        self.assertTrue((rows.evaluation_id == "test-eval").all())
        self.assertTrue((rows.lora_adapter_sha256 == "adapter-1").all())
        self.assertEqual(self.test_model.extract_multiturn_cop_vector.call_count, 2)
        self.assertEqual(len(self.turns[1]["images"]), 3)
        self.assertAlmostEqual(rows.anomaly_score.iloc[-1], self.classifier.predict_anomaly_score(self.query[-1]))
        self.assertAlmostEqual(rows.threshold.iloc[-1], self.classifier.threshold)
        self.assertGreater(rows.threshold.nunique(), 1)
        self.assertEqual(rows.vector_source.iloc[-1], "existing_full_execution")
        self.assertTrue(Path(report["plot"]).is_file())

    def test_snapshot_failure_is_a_gap_and_does_not_discard_final_point(self):
        self.prepare()
        model = SimpleNamespace(extract_multiturn_cop_vector=Mock(side_effect=[RuntimeError("OOM"), self.query[1]]))
        report = self.write(model)
        rows = pd.read_csv(report["csv"])
        self.assertEqual(report["status"], "partial")
        self.assertEqual(report["failed_point_count"], 1)
        self.assertTrue(pd.isna(rows.anomaly_score.iloc[0]))
        self.assertEqual(rows.error.iloc[0], "OOM")
        self.assertTrue(np.isfinite(rows.anomaly_score.iloc[-1]))

    def test_stale_prefix_model_cannot_be_used_with_a_refitted_full_classifier(self):
        self.prepare()
        sidecar = self.path.with_suffix(".json")
        metadata = json.loads(sidecar.read_text())
        metadata["comparison_signature"]["lora_adapter_sha256"] = "changed-adapter"
        sidecar.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "stale"):
            load_prefix_detector(self.path, self.classifier)
        self.assertTrue(prefix_model_path(self.path).exists())


if __name__ == "__main__":
    unittest.main()
