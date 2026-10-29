import json
from pathlib import Path
import sys
import tempfile
from threading import Event
from time import perf_counter
import unittest
from unittest.mock import patch

import pandas as pd

from rynnbrain_vlm.runtime import _Tee, clock, feature_only_enabled, record_run


class RuntimeTests(unittest.TestCase):
    def test_blocked_terminal_write_or_flush_does_not_block_run_or_lose_file_log(self):
        for blocked_operation in ("write", "flush"):
            with self.subTest(operation=blocked_operation), tempfile.TemporaryDirectory() as directory:
                release, entered = Event(), Event()

                class Terminal:
                    def write(self, text):
                        if blocked_operation == "write":
                            entered.set()
                            if not release.wait(3):
                                raise RuntimeError("Test terminal remained blocked")

                    def flush(self):
                        if blocked_operation == "flush":
                            entered.set()
                            if not release.wait(3):
                                raise RuntimeError("Test terminal remained blocked")

                try:
                    started = perf_counter()
                    with patch.object(sys, "stdout", Terminal()):
                        with record_run(directory, "test") as update:
                            self.assertTrue(entered.wait(1))
                            # Exceed the terminal queue while its first operation is blocked.
                            for index in range(400):
                                print(f"observation {index}", flush=True)
                            update(phase="evaluation", completed_bags=1)
                    self.assertLess(perf_counter() - started, 2)
                    root = Path(directory)
                    state = json.loads((root / "test_runtime.json").read_text())
                    self.assertEqual(state["status"], "completed")
                    self.assertEqual(state["completed_bags"], 1)
                    log = (root / "test_run.log").read_text()
                    for index in range(400):
                        self.assertIn(f"observation {index}\n", log)
                    self.assertIn("Run completed", log)
                finally:
                    release.set()

    def test_broken_terminal_preserves_file_and_healthy_terminal_drains_on_close(self):
        import io

        class BrokenTerminal:
            def write(self, text):
                raise BrokenPipeError("terminal disconnected")

        for terminal in (io.StringIO(), BrokenTerminal()):
            stream = io.StringIO()
            tee = _Tee(terminal, stream)
            tee.write("first\n")
            tee.flush()
            tee.write("second\n")
            tee.close()
            self.assertEqual(stream.getvalue(), "first\nsecond\n")
            if isinstance(terminal, io.StringIO):
                self.assertEqual(terminal.getvalue(), stream.getvalue())

    def test_clock_waits_for_all_initialized_cuda_devices(self):
        import torch
        with patch.object(torch.cuda, "is_initialized", return_value=True), \
             patch.object(torch.cuda, "device_count", return_value=2), \
             patch.object(torch.cuda, "synchronize") as sync:
            self.assertGreater(clock(), 0)
        self.assertEqual([call.args for call in sync.call_args_list], [(0,), (1,)])
        with patch.object(torch.cuda, "is_initialized", return_value=False), \
             patch.object(torch.cuda, "synchronize") as sync:
            clock()
            sync.assert_not_called()

    def test_feature_mode_requires_explicit_boolean(self):
        from run_benchmark import _decision_correct, _report_statistics
        self.assertFalse(feature_only_enabled({}))
        self.assertTrue(feature_only_enabled({"inference": {"feature_only": True}}))
        with self.assertRaises(ValueError):
            feature_only_enabled({"inference": {"feature_only": "false"}})
        self.assertIsNone(_decision_correct("fail", "not_generated"))
        statistics = _report_statistics([{"ground_truth_label": "fail", "decision": "not_generated",
                                          "decision_correct": None, "input_mode": "raw"}])
        self.assertEqual(statistics["scored_rows"], 0)

    def test_run_log_and_status_survive_error_and_append_next_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "interrupted bag"):
                with record_run(root, "test") as update:
                    print("first bag complete")
                    update(phase="evaluation", completed_bags=1)
                    raise RuntimeError("interrupted bag")
            status = json.loads((root / "test_runtime.json").read_text())
            self.assertEqual(status["status"], "failed")
            self.assertEqual(status["completed_bags"], 1)
            self.assertIn("interrupted bag", (root / "test_run.log").read_text())
            with record_run(root, "test") as update:
                update(completed_bags=2)
            status = json.loads((root / "test_runtime.json").read_text())
            self.assertEqual(status["status"], "completed")
            self.assertIn("first bag complete", (root / "test_run.log").read_text())
            self.assertGreaterEqual(status["elapsed_sec"], 0)

    def test_benchmark_checkpoints_first_bag_when_second_bag_aborts(self):
        import run_benchmark as benchmark
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.json"
            config_path.write_text(json.dumps({"camera_topics": [], "output_dir": str(root),
                "rynnbrain": {"output_dir": str(root / "rynnbrain"), "cop_classifier": {"enabled": False}}}))
            paths = [root / "bag_one", root / "bag_two"]
            with patch.object(sys, "argv", ["benchmark", "--config", str(config_path), "--data-root", str(root), "--skip-vlm", "--skip-dino"]), \
                 patch.object(benchmark, "discover_bags", return_value=paths), \
                 patch.object(benchmark, "infer_bag_record", side_effect=lambda path, *a: {
                     "bag_name": path.name, "bag_path": str(path), "label": "normal"}), \
                 patch.object(benchmark, "read_failure_annotation", return_value={"status": "missing_marker"}), \
                 patch.object(benchmark, "_dino_summary", side_effect=[{}, RuntimeError("second bag stopped")]):
                with self.assertRaisesRegex(RuntimeError, "second bag stopped"):
                    benchmark.main()
            saved = root / "rynnbrain_benchmark" / "whole_demo"
            self.assertEqual(pd.read_csv(saved / "benchmark_summary.csv").bag_name.tolist(), ["bag_one"])
            self.assertEqual(pd.read_csv(saved / "benchmark_clean.csv").bag_name.tolist(), ["bag_one"])
            state = json.loads((saved / "benchmark_runtime.json").read_text())
            self.assertEqual(state["status"], "failed")
            self.assertEqual(state["completed_bags"], 1)
            self.assertEqual(state["current_bag"], "bag_two")
            self.assertEqual(state["selection_total_bags"], 2)
            self.assertEqual(state["selection_completed_bags"], 2)
            self.assertIn("second bag stopped", (saved / "benchmark_run.log").read_text())


if __name__ == "__main__":
    unittest.main()
