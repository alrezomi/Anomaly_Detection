"""Fixed and adaptive threshold calibration for the DINOv2 detector."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .anomaly_detection import (
    NominalItem,
    add_progress_to_nominal_memory,
    compare_frame_to_nominal_memory,
)


def calibrate_fixed_threshold_leave_one_video_out(
    nominal_memory: list[NominalItem],
    quantile: float = 0.95,
    margin: float = 1.10,
    top_k_cls: int = 5,
    attention_power: float = 1.5,
    min_weight: float = 0.15,
) -> tuple[float, np.ndarray]:
    """
    Calibrate one global threshold using leave-one-video-out nominal scores.
    """
    video_ids = sorted(
        {int(item["video_id"]) for item in nominal_memory}
    )

    if len(video_ids) < 2:
        raise ValueError(
            "At least two nominal videos are required for calibration."
        )

    calibration_scores: list[float] = []

    for held_out_video_id in video_ids:
        query_items = [
            item
            for item in nominal_memory
            if int(item["video_id"]) == held_out_video_id
        ]
        reference_memory = [
            item
            for item in nominal_memory
            if int(item["video_id"]) != held_out_video_id
        ]

        for query_item in query_items:
            score, *_ = compare_frame_to_nominal_memory(
                test_features=query_item["features"],
                test_attention=query_item["attention"],
                test_cls=query_item["cls"],
                nominal_memory=reference_memory,
                top_k_cls=top_k_cls,
                attention_power=attention_power,
                min_weight=min_weight,
            )
            calibration_scores.append(float(score))

    scores_array = np.asarray(calibration_scores, dtype=np.float32)
    threshold = float(np.quantile(scores_array, quantile) * margin)

    print("\nFixed-threshold calibration summary:")
    print("Minimum score:", float(scores_array.min()))
    print("Mean score:", float(scores_array.mean()))
    print("Maximum score:", float(scores_array.max()))
    print(f"{quantile:.3f} quantile:", float(np.quantile(scores_array, quantile)))
    print("Fixed threshold:", threshold)

    return threshold, scores_array


def build_adaptive_visual_threshold_from_scores(
    calibration_df: pd.DataFrame,
    n_bins: int = 30,
    quantile: float = 0.95,
    margin: float = 1.10,
    min_points_per_bin: int = 3,
    smooth_window: int | None = 3,
    threshold_floor_quantile: float | None = 0.20,
) -> pd.DataFrame:
    """
    Build a stage-dependent threshold from held-out nominal scores.

    calibration_df must contain:
    - score
    - matched_progress
    """
    required_columns = {"score", "matched_progress"}

    if not required_columns.issubset(calibration_df.columns):
        raise ValueError(
            "calibration_df must contain 'score' and 'matched_progress'."
        )

    if calibration_df.empty:
        raise ValueError("calibration_df is empty.")

    if n_bins < 1:
        raise ValueError("n_bins must be at least 1.")

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])

    scores = calibration_df["score"].to_numpy()
    global_threshold = float(np.quantile(scores, quantile) * margin)

    threshold_floor = (
        float(np.quantile(scores, threshold_floor_quantile))
        if threshold_floor_quantile is not None
        else 0.0
    )

    threshold_values: list[float] = []
    counts: list[int] = []

    for bin_index in range(n_bins):
        left = bins[bin_index]
        right = bins[bin_index + 1]

        if bin_index == n_bins - 1:
            mask = (
                (calibration_df["matched_progress"] >= left)
                & (calibration_df["matched_progress"] <= right)
            )
        else:
            mask = (
                (calibration_df["matched_progress"] >= left)
                & (calibration_df["matched_progress"] < right)
            )

        bin_scores = calibration_df.loc[mask, "score"].to_numpy()
        counts.append(len(bin_scores))

        if len(bin_scores) >= min_points_per_bin:
            bin_threshold = float(
                np.quantile(bin_scores, quantile) * margin
            )
        else:
            bin_threshold = np.nan

        threshold_values.append(bin_threshold)

    threshold_df = pd.DataFrame(
        {
            "stage_center": centers,
            "adaptive_threshold": threshold_values,
            "count": counts,
        }
    )

    threshold_df["adaptive_threshold"] = (
        threshold_df["adaptive_threshold"]
        .interpolate()
        .bfill()
        .ffill()
        .fillna(global_threshold)
    )

    threshold_df["adaptive_threshold"] = np.maximum(
        threshold_df["adaptive_threshold"].to_numpy(),
        threshold_floor,
    )

    if smooth_window is not None and smooth_window > 1:
        threshold_df["adaptive_threshold"] = (
            threshold_df["adaptive_threshold"]
            .rolling(
                window=smooth_window,
                center=True,
                min_periods=1,
            )
            .median()
        )

    return threshold_df


def calibrate_adaptive_threshold_leave_one_video_out(
    nominal_memory: list[NominalItem],
    quantile: float = 0.95,
    margin: float = 1.10,
    top_k_cls: int = 5,
    attention_power: float = 1.5,
    min_weight: float = 0.15,
    n_bins: int = 30,
    min_points_per_bin: int = 3,
    smooth_window: int | None = 3,
    threshold_floor_quantile: float | None = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calibrate an adaptive threshold using leave-one-video-out validation.

    Each held-out nominal frame is matched only against frames belonging
    to the other nominal videos.
    """
    add_progress_to_nominal_memory(nominal_memory)

    video_ids = sorted(
        {int(item["video_id"]) for item in nominal_memory}
    )

    if len(video_ids) < 2:
        raise ValueError(
            "At least two nominal videos are required for calibration."
        )

    rows: list[dict[str, float | int]] = []

    for held_out_video_id in video_ids:
        print(
            f"Adaptive calibration: holding out nominal video "
            f"{held_out_video_id}"
        )

        query_items = [
            item
            for item in nominal_memory
            if int(item["video_id"]) == held_out_video_id
        ]
        reference_memory = [
            item
            for item in nominal_memory
            if int(item["video_id"]) != held_out_video_id
        ]

        for query_item in query_items:
            (
                score,
                _patch_scores,
                _weighted_patch_scores,
                _weights,
                matched_item,
                cls_distance,
            ) = compare_frame_to_nominal_memory(
                test_features=query_item["features"],
                test_attention=query_item["attention"],
                test_cls=query_item["cls"],
                nominal_memory=reference_memory,
                top_k_cls=top_k_cls,
                attention_power=attention_power,
                min_weight=min_weight,
            )

            rows.append(
                {
                    "held_out_video_id": held_out_video_id,
                    "query_frame_id": int(query_item["frame_id"]),
                    "query_progress": float(query_item["progress"]),
                    "matched_video_id": int(matched_item["video_id"]),
                    "matched_frame_id": int(matched_item["frame_id"]),
                    "matched_progress": float(matched_item["progress"]),
                    "matched_timestamp_sec": float(
                        matched_item["timestamp_sec"]
                    ),
                    "score": float(score),
                    "cls_distance": float(cls_distance),
                }
            )

    calibration_df = pd.DataFrame(rows)

    adaptive_threshold_df = build_adaptive_visual_threshold_from_scores(
        calibration_df=calibration_df,
        n_bins=n_bins,
        quantile=quantile,
        margin=margin,
        min_points_per_bin=min_points_per_bin,
        smooth_window=smooth_window,
        threshold_floor_quantile=threshold_floor_quantile,
    )

    scores = calibration_df["score"].to_numpy()

    print("\nAdaptive-threshold calibration summary:")
    print("Minimum score:", float(scores.min()))
    print("Mean score:", float(scores.mean()))
    print("Maximum score:", float(scores.max()))
    print(f"{quantile:.3f} quantile:", float(np.quantile(scores, quantile)))
    print("Equivalent global threshold:", float(np.quantile(scores, quantile) * margin))

    return adaptive_threshold_df, calibration_df
