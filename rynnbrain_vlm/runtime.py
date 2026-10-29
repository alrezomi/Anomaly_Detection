"""Wall-clock measurements and durable status for long offline runs."""

from contextlib import contextmanager, redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import json
from pathlib import Path
from queue import Empty, Full, Queue
import sys
from threading import Event, Thread
from time import perf_counter
import traceback


RUNTIME_FIELDS = ("runtime_input_sec", "runtime_reference_sec", "runtime_model_sec",
                  "runtime_classifier_sec", "runtime_diagnostics_sec", "runtime_bag_sec")


def clock():
    """Complete outstanding CUDA work before reading a wall clock, including offload."""
    import torch
    if torch.cuda.is_initialized():
        for device in range(torch.cuda.device_count()):
            torch.cuda.synchronize(device)
    return perf_counter()


def feature_only_enabled(vlm):
    value = vlm.get("inference", {}).get("feature_only", False)
    if not isinstance(value, bool):
        raise ValueError("rynnbrain.inference.feature_only must be true or false.")
    return value


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


class _Tee:
    """Keep the file durable without letting a stalled terminal stop inference."""

    def __init__(self, terminal, stream):
        self.terminal, self.stream = terminal, stream
        self._pending = Queue(maxsize=256)
        self._closing = Event()
        self._terminal_failed = Event()
        self._worker = Thread(target=self._mirror, name="benchmark-terminal-log", daemon=True)
        self._worker.start()

    def _mirror(self):
        while True:
            try:
                text = self._pending.get(timeout=.05)
            except Empty:
                if self._closing.is_set():
                    return
                continue
            if text is None:
                return
            try:
                self.terminal.write(text)
                self.terminal.flush()
            except (OSError, ValueError):
                self._terminal_failed.set()
                return

    def write(self, text):
        self.stream.write(text)
        self.stream.flush()
        if not self._terminal_failed.is_set():
            try:
                self._pending.put_nowait(text)
            except Full:
                pass  # Only terminal copies may be dropped; the file contains everything.
        return len(text)

    def flush(self):
        self.stream.flush()

    def close(self):
        self._closing.set()
        try:
            self._pending.put_nowait(None)
        except Full:
            pass
        # A disconnected or flow-controlled terminal must not delay run completion.
        self._worker.join(timeout=.2)

    def isatty(self):
        return False


@contextmanager
def record_run(directory, name):
    """Flush progress/logs immediately; a hard kill leaves the last running status."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    state = {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
             "phase": "starting", "completed_bags": 0}
    status_path = directory / f"{name}_runtime.json"

    def update(**values):
        state.update(values)
        state["elapsed_sec"] = perf_counter() - start
        write_json(status_path, state)

    with (directory / f"{name}_run.log").open("a", encoding="utf-8", buffering=1) as stream:
        stdout, stderr = _Tee(sys.stdout, stream), _Tee(sys.stderr, stream)
        try:
            with redirect_stdout(stdout), redirect_stderr(stderr):
                update()
                print(f"\nRun started: {state['started_at_utc']}", flush=True)
                try:
                    yield update
                except BaseException as error:
                    traceback.print_exc()
                    update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                           error=f"{type(error).__name__}: {error}")
                    raise
                else:
                    update(status="completed", phase="finished")
                    print(f"Run completed in {state['elapsed_sec']:.3f} s", flush=True)
        finally:
            stdout.close()
            stderr.close()
