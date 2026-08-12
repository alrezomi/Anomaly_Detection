"""Safe on-disk storage for reusable DINOv2 nominal features."""

from __future__ import annotations

import json
from pathlib import Path
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

    with np.load(required[0], allow_pickle=False) as arrays:
        item_count = len(arrays["video_id"])
        memory: list[NominalItem] = []
        for index in range(item_count):
            memory.append(
                {
                    "video_id": int(arrays["video_id"][index]),
                    "video_path": str(arrays["video_path"][index]),
                    "frame_id": int(arrays["frame_id"][index]),
                    "timestamp_sec": float(arrays["timestamp_sec"][index]),
                    "features": arrays["features"][index].copy(),
                    "attention": arrays["attention"][index].copy(),
                    "cls": arrays["cls"][index].copy(),
                    "progress": float(arrays["progress"][index]),
                    "stage": int(arrays["stage"][index]),
                    "stage_source": "nominal_cache",
                    "_progress_locked": True,
                }
            )
        grid_size = int(arrays["grid_size"])

    threshold = pd.read_csv(required[1])
    calibration = pd.read_csv(required[2])
    print(f"Loaded reusable nominal cache: {directory}")
    print(f"Cached nominal frames: {len(memory)}")
    return memory, grid_size, threshold, calibration
