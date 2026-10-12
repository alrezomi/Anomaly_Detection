import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


class ClassifierReuseTests(unittest.TestCase):
    def test_missing_saved_file_fails_before_model_loading_without_training(self):
        import run_benchmark as benchmark
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            config = {"output_dir": directory, "camera_topics": ["top"], "rynnbrain": {
                "source": "rosbag", "model": {"model_id": "test"},
                "cop_classifier": {"enabled": True, "train_on_benchmark": False,
                                   "model_paths": {"raw": str(Path(directory) / "missing.npz")}}}}
            path.write_text(json.dumps(config))
            with patch("sys.argv", ["benchmark", "--config", str(path)]), \
                 patch.object(benchmark, "RynnBrainModel") as model, \
                 patch("rynnbrain_vlm.cop_classifier.train_from_saved_vectors") as train:
                with self.assertRaisesRegex(FileNotFoundError, "will not train a replacement"):
                    benchmark.main()
            model.assert_not_called()
            train.assert_not_called()

    def test_reuse_checks_adapter_and_camera_settings_before_evaluation(self):
        import run_benchmark as benchmark
        from rynnbrain_vlm.run import cop_comparison_signature
        from rynnbrain_vlm.model import COP_REPRESENTATION_ID
        model = SimpleNamespace(model_id="test", adapter_identity="old-adapter",
            model=SimpleNamespace(config=SimpleNamespace(text_config=SimpleNamespace(hidden_size=4))))
        config = {"camera_topics": ["top"]}
        vlm = {"source": "rosbag", "reference_bags": ["reference"], "model": {"model_id": "test"}}
        signature = cop_comparison_signature(model, config, vlm, "raw", "saved reference", 8, {})
        classifiers = {"raw": SimpleNamespace(representation_id=COP_REPRESENTATION_ID, comparison_signature=signature)}
        benchmark._validate_reused_classifiers(classifiers, model, config, vlm, "saved reference", 8, {})
        for changed_model, changed_config in ((SimpleNamespace(**{**vars(model), "adapter_identity": "new-adapter"}), config),
                                              (model, {"camera_topics": ["top", "flange"]})):
            with self.assertRaisesRegex(ValueError, "incompatible"):
                benchmark._validate_reused_classifiers(classifiers, changed_model, changed_config, vlm, "saved reference", 8, {})

    def test_wrong_classifier_type_and_mixed_references_are_rejected(self):
        import run_benchmark as benchmark
        from rynnbrain_vlm.cop_classifier import CoPLogisticClassifier
        classifier = Mock(spec=CoPLogisticClassifier, input_mode="raw", comparison_signature={"nominal_response": "saved"})
        vlm = {"input_modes": ["raw"], "cop_classifier": {"method": "knn", "model_paths": {"raw": "x.npz"}}}
        with patch("rynnbrain_vlm.cop_classifier.load_classifier", return_value=classifier):
            with self.assertRaisesRegex(ValueError, "method/input mode"):
                benchmark._load_reused_classifiers(vlm)
        vlm["cop_classifier"].update(method="logistic", model_paths={"raw": "x.npz", "heatmap": "y.npz"})
        vlm["input_modes"] = ["raw", "heatmap"]
        heatmap = copy.copy(classifier)
        heatmap.input_mode, heatmap.comparison_signature = "heatmap", {"nominal_response": "different"}
        with patch("rynnbrain_vlm.cop_classifier.load_classifier", side_effect=[classifier, heatmap]):
            with self.assertRaisesRegex(ValueError, "same nominal reference"):
                benchmark._load_reused_classifiers(vlm)


if __name__ == "__main__":
    unittest.main()
