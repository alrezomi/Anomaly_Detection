"""Safe on-disk storage for reusable DINOv2 nominal features."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from .anomaly_detection import NominalItem


def save_nominal_cache(
    cache_directory: str | Path,
    nominal_memory: list[NominalItem],
    grid_size: int,
    adaptive_threshold_df: pd.DataFrame,
    calibration_df: pd.DataFrame,
    signature: dict[str, Any],
) -> None:
    """Save numeric memory arrays, calibration tables, and their signature."""
    directory = Path(cache_directory)
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        directory / "nominal_memory.npz",
        features=np.stack([item["features"] for item in nominal_memory]),
        attention=np.stack([item["attention"] for item in nominal_memory]),
        cls=np.stack([item["cls"] for item in nominal_memory]),
        video_id=np.asarray([item["video_id"] for item in nominal_memory]),
        frame_id=np.asarray([item["frame_id"] for item in nominal_memory]),
        timestamp_sec=np.asarray(
            [item["timestamp_sec"] for item in nominal_memory], dtype=np.float64
        ),
        progress=np.asarray(
            [item["progress"] for item in nominal_memory], dtype=np.float64
        ),
        stage=np.asarray([item.get("stage", 1) for item in nominal_memory]),
        video_path=np.asarray(
            [str(item["video_path"]) for item in nominal_memory], dtype=str
        ),
        grid_size=np.asarray(grid_size),
    )
    adaptive_threshold_df.to_csv(directory / "adaptive_threshold.csv", index=False)
    calibration_df.to_csv(directory / "calibration.csv", index=False)
    (directory / "signature.json").write_text(
        json.dumps(signature, indent=2, sort_keys=True), encoding="utf-8"
    )


def load_nominal_cache(
    cache_directory: str | Path,
    expected_signature: dict[str, Any],
) -> tuple[list[NominalItem], int, pd.DataFrame, pd.DataFrame] | None:
    """Load a matching cache, or return None when absent or incompatible."""
    directory = Path(cache_directory)
    required = [
        directory / "nominal_memory.npz",
        directory / "adaptive_threshold.csv",
        directory / "calibration.csv",
        directory / "signature.json",
    ]
    if not all(path.is_file() for path in required):
        return None

    saved_signature = json.loads(required[3].read_text(encoding="utf-8"))
    if any(saved_signature.get(key) != value for key, value in expected_signature.items()):
        print(f"Nominal cache settings changed; rebuilding: {directory}")
        return None

    cache_size_gib = required[0].stat().st_size / (1024**3)
    print(
        f"Loading nominal memory once ({cache_size_gib:.2f} GiB compressed): "
        f"{required[0]}"
    )
    started = perf_counter()
    with np.load(required[0], allow_pickle=False) as arrays:
        # Read every compressed member exactly once. Indexing arrays["features"]
        # inside the item loop would decompress the multi-GiB tensor repeatedly.
        features = arrays["features"]
        attention = arrays["attention"]
        cls = arrays["cls"]
        video_id = arrays["video_id"]
        frame_id = arrays["frame_id"]
        timestamp_sec = arrays["timestamp_sec"]
        progress = arrays["progress"]
        stage = arrays["stage"]
        video_path = arrays["video_path"]
        grid_size = int(arrays["grid_size"])
        item_count = len(video_id)
        memory: list[NominalItem] = []
        for index in range(item_count):
            memory.append(
                {
                    "video_id": int(video_id[index]),
                    "video_path": str(video_path[index]),
                    "frame_id": int(frame_id[index]),
                    "timestamp_sec": float(timestamp_sec[index]),
                    "features": features[index],
                    "attention": attention[index],
                    "cls": cls[index],
                    "progress": float(progress[index]),
                    "stage": int(stage[index]),
                    "stage_source": "nominal_cache",
                    "_progress_locked": True,
                }
            )
    elapsed = perf_counter() - started
    print(f"Nominal memory loaded in {elapsed:.1f}s")

    threshold = pd.read_csv(required[1])
    calibration = pd.read_csv(required[2])
    print(f"Loaded reusable nominal cache: {directory}")
    print(f"Cached nominal frames: {len(memory)}")
    return memory, grid_size, threshold, calibration
