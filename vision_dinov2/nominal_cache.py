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
    save_feature_cache(cache_directory, nominal_memory, grid_size, signature)
    save_nominal_calibration(cache_directory, adaptive_threshold_df, calibration_df, signature)


def save_feature_cache(cache_directory, nominal_memory, grid_size, signature):
    """Save extraction results independently of thresholds and scoring settings."""
    directory = Path(cache_directory)
    directory.mkdir(parents=True, exist_ok=True)
    # Invalidate both signatures before replacing features. An interrupted write
    # must not pair new features with an old calibration or legacy signature.
    for name in ('feature_signature.json', 'signature.json'):
        (directory / name).unlink(missing_ok=True)
    temporary = directory / 'nominal_memory.tmp.npz'
    np.savez_compressed(
        temporary,
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
    temporary.replace(directory / 'nominal_memory.npz')
    marker = directory / 'feature_signature.json.tmp'
    marker.write_text(json.dumps(memory_signature(signature), indent=2, sort_keys=True), encoding='utf-8')
    marker.replace(directory / 'feature_signature.json')


def save_nominal_calibration(directory, threshold, calibration, signature):
    """Replace calibration without recompressing the large feature memory.

    Invalidate the signature first so interrupted writes cannot look compatible.
    """
    directory = Path(directory)
    marker = directory / "signature.json"
    # Promote legacy extraction metadata before invalidating its only signature.
    feature_marker = directory / 'feature_signature.json'
    if not feature_marker.is_file() and marker.is_file() and (directory / 'nominal_memory.npz').is_file():
        temporary = directory / 'feature_signature.json.tmp'
        temporary.write_text(json.dumps(memory_signature(json.loads(marker.read_text(encoding='utf-8'))),
                                         indent=2, sort_keys=True), encoding='utf-8')
        temporary.replace(feature_marker)
    marker.unlink(missing_ok=True)
    if "patch_profile" in threshold.attrs:
        from .patch_normality import validate_patch_profile
        validate_patch_profile(threshold.attrs["patch_profile"])
        temporary = directory / "patch_normality.tmp.npz"
        np.savez_compressed(temporary, **threshold.attrs["patch_profile"])
        temporary.replace(directory / "patch_normality.npz")
    for name, table in [("adaptive_threshold.csv", threshold), ("calibration.csv", calibration)]:
        temporary = directory / (name + ".tmp")
        table.to_csv(temporary, index=False)
        temporary.replace(directory / name)
    temporary = directory / "signature.json.tmp"
    temporary.write_text(json.dumps(signature, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(marker)


def memory_signature(signature):
    """Calibration changes do not invalidate the extracted/matched memory."""
    return {key: value for key, value in signature.items()
            if not key.startswith("threshold_") and key not in {
                "min_points_per_bin", "calibration_sources", "calibration_method", "score_mode",
                "top_k_cls", "attention_power", "min_weight", "stage_constrained_matching", "dino_metric"}}


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
    differences = [key for key, value in expected_signature.items()
                   if saved_signature.get(key, 'mean' if key == 'score_mode' else None) != value]
    if differences:
        print(f"Saved DINO calibration is incompatible: {directory}")
        for key in differences:
            print(f"  {key}: saved={saved_signature.get(key)!r}; requested={expected_signature[key]!r}")
        print('Recalibrate with vision-memory; compatible extracted features can be reused.')
        return None

    profile = None
    from .threshold import ATTENTION_CALIBRATION_METHOD
    if saved_signature.get("calibration_method") == ATTENTION_CALIBRATION_METHOD:
        from .patch_normality import validate_patch_profile
        profile_path = directory / "patch_normality.npz"
        if not profile_path.is_file():
            print("Missing DINO patch-normality profile; rebuild vision-memory.")
            return None
        try:
            with np.load(profile_path, allow_pickle=False) as arrays:
                profile = {key: arrays[key] for key in arrays.files}
            validate_patch_profile(profile)
        except (OSError, ValueError) as error:
            print(f"Invalid DINO patch-normality profile: {error}. Rebuild vision-memory.")
            return None
    memory, grid_size = _read_feature_arrays(required[0])
    threshold = pd.read_csv(required[1])
    if profile is not None:
        metric = str(profile.get('metric', 'euclidean'))
        from .threshold import SCORE_VARIANTS
        variants = ('weighted',) if metric == 'mahalanobis' else SCORE_VARIANTS
        if metric != saved_signature.get('dino_metric', 'euclidean') or any(
                f'threshold_{name}' not in threshold or
                not np.isfinite(threshold[f'threshold_{name}']).all() or
                (threshold[f'threshold_{name}'] < 0).any() for name in variants):
            print('Missing or invalid DINO metric thresholds; rebuild vision-memory.')
            return None
    threshold.attrs['score_mode'] = saved_signature.get('score_mode', 'mean')
    if profile is not None:
        validate_patch_profile(profile, grid_size ** 2)
        threshold.attrs["patch_profile"] = profile
    calibration = pd.read_csv(required[2])
    if profile is not None and str(profile.get('metric', 'euclidean')) == 'mahalanobis':
        # v7's primary Mahalanobis score/threshold are unchanged. Ignore the
        # accidentally added secondary comparison in already calibrated caches.
        obsolete = ['score_pooled', 'threshold_pooled', 'alarm_pooled', 'base_threshold_pooled']
        threshold.drop(columns=obsolete, errors='ignore', inplace=True)
        calibration.drop(columns=obsolete, errors='ignore', inplace=True)
    print(f"Loaded reusable nominal cache: {directory}")
    print(f"Cached nominal frames: {len(memory)}")
    return memory, grid_size, threshold, calibration


def _read_feature_arrays(archive):
    cache_size_gib = archive.stat().st_size / (1024**3)
    print(
        f"Loading nominal memory once ({cache_size_gib:.2f} GiB compressed): "
        f"{archive}"
    )
    started = perf_counter()
    with np.load(archive, allow_pickle=False) as arrays:
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

    return memory, grid_size


def load_feature_cache(cache_directory, signature):
    """Reuse raw features even if calibration is missing or incompatible.

    Existing caches migrate read-only via their old full signature. A changed
    source, model, resolution, sampling or stage assignment still invalidates.
    """
    directory = Path(cache_directory)
    archive = directory / 'nominal_memory.npz'
    marker = directory / 'feature_signature.json'
    if not marker.is_file():
        marker = directory / 'signature.json'
    if not archive.is_file() or not marker.is_file():
        return None
    saved = memory_signature(json.loads(marker.read_text(encoding='utf-8')))
    if saved != memory_signature(signature):
        return None
    return _read_feature_arrays(archive)
