"""CLS-assisted and attention-weighted DINOv2 anomaly detection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .dinov2_features import DINOv2FeatureExtractor, iter_video_frames


NominalItem = dict[str, Any]


def build_nominal_video_memory(
    extractor: DINOv2FeatureExtractor,
    nominal_video_paths: list[str | Path],
    sample_fps: float = 2.0,
    start_sec: float | None = None,
    end_sec: float | None = None,
    max_frames_per_video: int | None = None,
) -> tuple[list[NominalItem], int]:
    """
    Build nominal memory from successful videos.

    Each sampled frame stores its patch features, attention values, and CLS token.
    """
    if not nominal_video_paths:
        raise ValueError("At least one nominal video path is required.")

    nominal_memory: list[NominalItem] = []
    common_grid_size: int | None = None

    for video_id, video_path in enumerate(nominal_video_paths):
        print("\n" + "=" * 90)
        print("Building nominal memory from:")
        print(video_path)
        print("=" * 90)

        frames_found = 0

        for frame_id, timestamp_sec, frame in iter_video_frames(
            video_path=video_path,
            sample_fps=sample_fps,
            start_sec=start_sec,
            end_sec=end_sec,
            max_frames=max_frames_per_video,
        ):
            (
                patch_features,
                attention,
                cls_feature,
                grid_size,
            ) = extractor.extract(frame)

            if common_grid_size is None:
                common_grid_size = grid_size
            elif grid_size != common_grid_size:
                raise ValueError(
                    "All nominal frames must have the same patch-grid size."
                )

            nominal_memory.append(
                {
                    "video_id": video_id,
                    "video_path": str(video_path),
                    "frame_id": frame_id,
                    "timestamp_sec": timestamp_sec,
                    "frame": frame,
                    "features": patch_features,
                    "attention": attention,
                    "cls": cls_feature,
                }
            )

            frames_found += 1
            print(
                f"Stored nominal video {video_id}, "
                f"frame {frame_id}, t={timestamp_sec:.2f}s"
            )

        if frames_found == 0:
            raise ValueError(
                f"No frames were read from nominal video: {video_path}"
            )

    if common_grid_size is None:
        raise ValueError("Nominal memory could not be created.")

    add_progress_to_nominal_memory(nominal_memory)

    print("\nNominal memory size:", len(nominal_memory))
    print("Patch grid size:", common_grid_size)

    return nominal_memory, common_grid_size


def add_progress_to_nominal_memory(
    nominal_memory: list[NominalItem],
) -> list[NominalItem]:
    """
    Add normalized execution progress to each nominal frame.

    Progress is calculated independently for each nominal video:
        progress = sampled_frame_id / maximum_sampled_frame_id
    """
    if not nominal_memory:
        raise ValueError("Nominal memory is empty.")

    max_frame_by_video: dict[int, int] = {}

    for item in nominal_memory:
        video_id = int(item["video_id"])
        frame_id = int(item["frame_id"])
        max_frame_by_video[video_id] = max(
            frame_id,
            max_frame_by_video.get(video_id, frame_id),
        )

    for item in nominal_memory:
        video_id = int(item["video_id"])
        max_frame_id = max_frame_by_video[video_id]

        item["progress"] = (
            float(item["frame_id"]) / float(max_frame_id)
            if max_frame_id > 0
            else 0.0
        )

    return nominal_memory


def compute_cls_distance(
    test_cls: np.ndarray,
    nominal_cls: np.ndarray,
) -> float:
    """
    Compute cosine distance between normalized CLS tokens.

    Lower values indicate more globally similar frames.
    """
    similarity = float(np.dot(test_cls, nominal_cls))
    return 1.0 - similarity


def select_top_k_nominal_by_cls(
    test_cls: np.ndarray,
    nominal_memory: list[NominalItem],
    top_k_cls: int = 5,
) -> tuple[list[NominalItem], np.ndarray]:
    """Select the globally closest nominal frames using CLS distance."""
    if not nominal_memory:
        raise ValueError("Nominal memory is empty.")

    if top_k_cls < 1:
        raise ValueError("top_k_cls must be at least 1.")

    distances = np.asarray(
        [
            compute_cls_distance(test_cls, item["cls"])
            for item in nominal_memory
        ],
        dtype=np.float32,
    )

    selected_count = min(top_k_cls, len(nominal_memory))
    selected_indices = np.argsort(distances)[:selected_count]
    selected_memory = [nominal_memory[index] for index in selected_indices]

    return selected_memory, distances[selected_indices]


def make_attention_weights(
    test_attention: np.ndarray,
    nominal_attention: np.ndarray,
    attention_power: float = 1.5,
    min_weight: float = 0.15,
) -> np.ndarray:
    """
    Combine test and nominal attention and convert it into soft weights.

    The maximum of the two maps is used so that an expected object region
    remains important even when the object is missing from the test frame.
    """
    if test_attention.shape != nominal_attention.shape:
        raise ValueError(
            "Test and nominal attention arrays must have the same shape."
        )

    if attention_power <= 0:
        raise ValueError("attention_power must be greater than zero.")

    if not 0.0 <= min_weight <= 1.0:
        raise ValueError("min_weight must be between 0 and 1.")

    combined_attention = np.maximum(
        test_attention,
        nominal_attention,
    )

    combined_attention = combined_attention - combined_attention.min()
    combined_attention = combined_attention / (
        combined_attention.max() + 1e-8
    )

    powered_attention = combined_attention ** attention_power
    return min_weight + (1.0 - min_weight) * powered_attention


def compute_patch_distance_attention_weighted(
    test_features: np.ndarray,
    nominal_features: np.ndarray,
    test_attention: np.ndarray,
    nominal_attention: np.ndarray,
    attention_power: float = 1.5,
    min_weight: float = 0.15,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate local patch differences and one weighted frame score.
    """
    if test_features.shape != nominal_features.shape:
        raise ValueError(
            "Test and nominal patch-feature arrays must have the same shape."
        )

    difference = test_features - nominal_features
    patch_scores = np.sqrt(np.mean(difference ** 2, axis=-1))

    weights = make_attention_weights(
        test_attention=test_attention,
        nominal_attention=nominal_attention,
        attention_power=attention_power,
        min_weight=min_weight,
    )

    frame_score = float(
        np.sum(patch_scores * weights) / np.sum(weights)
    )
    weighted_patch_scores = patch_scores * weights

    return frame_score, patch_scores, weighted_patch_scores, weights


def compare_frame_to_nominal_memory(
    test_features: np.ndarray,
    test_attention: np.ndarray,
    test_cls: np.ndarray,
    nominal_memory: list[NominalItem],
    top_k_cls: int = 5,
    attention_power: float = 1.5,
    min_weight: float = 0.15,
) -> tuple[
    float,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    NominalItem,
    float,
]:
    """
    Compare one test frame with nominal memory.

    First, CLS tokens select the globally closest nominal frames.
    Second, attention-weighted patch differences select the best local match.
    """
    selected_memory, cls_distances = select_top_k_nominal_by_cls(
        test_cls=test_cls,
        nominal_memory=nominal_memory,
        top_k_cls=top_k_cls,
    )

    best_result: tuple[
        float,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        NominalItem,
        float,
    ] | None = None

    for item, cls_distance in zip(selected_memory, cls_distances):
        (
            frame_score,
            patch_scores,
            weighted_patch_scores,
            weights,
        ) = compute_patch_distance_attention_weighted(
            test_features=test_features,
            nominal_features=item["features"],
            test_attention=test_attention,
            nominal_attention=item["attention"],
            attention_power=attention_power,
            min_weight=min_weight,
        )

        if best_result is None or frame_score < best_result[0]:
            best_result = (
                frame_score,
                patch_scores,
                weighted_patch_scores,
                weights,
                item,
                float(cls_distance),
            )

    if best_result is None:
        raise RuntimeError("No nominal comparison was completed.")

    return best_result


def threshold_for_progress(
    matched_progress: float,
    adaptive_threshold_df: pd.DataFrame,
) -> float:
    """Interpolate the adaptive threshold at one matched nominal progress."""
    required_columns = {"stage_center", "adaptive_threshold"}

    if not required_columns.issubset(adaptive_threshold_df.columns):
        raise ValueError(
            "adaptive_threshold_df must contain "
            "'stage_center' and 'adaptive_threshold'."
        )

    return float(
        np.interp(
            matched_progress,
            adaptive_threshold_df["stage_center"].to_numpy(),
            adaptive_threshold_df["adaptive_threshold"].to_numpy(),
        )
    )


def run_video_test(
    extractor: DINOv2FeatureExtractor,
    test_video_path: str | Path,
    nominal_memory: list[NominalItem],
    adaptive_threshold_df: pd.DataFrame | None = None,
    fixed_threshold: float | None = None,
    sample_fps: float = 2.0,
    start_sec: float | None = None,
    end_sec: float | None = None,
    max_frames: int | None = None,
    top_k_cls: int = 5,
    attention_power: float = 1.5,
    min_weight: float = 0.15,
) -> pd.DataFrame:
    """
    Process a test video sequentially and return frame-level results.

    Adaptive thresholding is used when adaptive_threshold_df is provided.
    Otherwise, fixed_threshold is used.
    """
    add_progress_to_nominal_memory(nominal_memory)

    rows: list[dict[str, Any]] = []

    for frame_id, timestamp_sec, frame in iter_video_frames(
        video_path=test_video_path,
        sample_fps=sample_fps,
        start_sec=start_sec,
        end_sec=end_sec,
        max_frames=max_frames,
    ):
        (
            test_features,
            test_attention,
            test_cls,
            _grid_size,
        ) = extractor.extract(frame)

        (
            score,
            _patch_scores,
            _weighted_patch_scores,
            _weights,
            best_match,
            cls_distance,
        ) = compare_frame_to_nominal_memory(
            test_features=test_features,
            test_attention=test_attention,
            test_cls=test_cls,
            nominal_memory=nominal_memory,
            top_k_cls=top_k_cls,
            attention_power=attention_power,
            min_weight=min_weight,
        )

        matched_progress = float(best_match["progress"])

        if adaptive_threshold_df is not None:
            current_threshold = threshold_for_progress(
                matched_progress,
                adaptive_threshold_df,
            )
            threshold_mode = "adaptive"
        else:
            current_threshold = fixed_threshold
            threshold_mode = "fixed"

        if current_threshold is None:
            epsilon = np.nan
            alarm = False
        else:
            epsilon = score / max(float(current_threshold), 1e-8)
            alarm = bool(epsilon > 1.0)

        row = {
            "frame_id": frame_id,
            "timestamp_sec": timestamp_sec,
            "score": score,
            "threshold": current_threshold,
            "epsilon": epsilon,
            "alarm": alarm,
            "threshold_mode": threshold_mode,
            "cls_distance": cls_distance,
            "matched_video_id": int(best_match["video_id"]),
            "matched_frame_id": int(best_match["frame_id"]),
            "matched_timestamp_sec": float(best_match["timestamp_sec"]),
            "matched_progress": matched_progress,
        }
        rows.append(row)

        threshold_text = (
            "None"
            if current_threshold is None
            else f"{current_threshold:.5f}"
        )
        epsilon_text = "None" if np.isnan(epsilon) else f"{epsilon:.3f}"

        print(
            f"frame={frame_id:03d}, "
            f"time={timestamp_sec:.2f}s, "
            f"score={score:.5f}, "
            f"threshold={threshold_text}, "
            f"epsilon={epsilon_text}, "
            f"alarm={alarm}, "
            f"CLS distance={cls_distance:.5f}, "
            f"match=video {best_match['video_id']} "
            f"frame {best_match['frame_id']}, "
            f"progress={matched_progress:.3f}"
        )

    if not rows:
        raise ValueError(
            "No test frames were processed. Check the test-video path "
            "and the selected start/end times."
        )

    return pd.DataFrame(rows)
