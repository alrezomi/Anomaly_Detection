"""Progress-bin calibration and shared decisions for attention-weighted DINO."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .anomaly_detection import compare_frame_to_nominal_memory


ATTENTION_CALIBRATION_METHOD = "attention_progress_quantile_v1"
THRESHOLD_QUANTILE = 0.95
THRESHOLD_MARGIN = 1.10


def build_attention_threshold(calibration_df, n_bins=30, min_points_per_bin=2,
                             smooth_window=3, threshold_floor_quantile=.20):
    """Attention-weighted held-out frame percentiles at matched nominal progress."""
    rows = calibration_df.copy()
    if rows.empty or not {'score', 'held_out_video_id', 'matched_progress'}.issubset(rows):
        raise ValueError('DINO attention calibration requires held-out scores, run IDs and matched_progress.')
    scores = rows.score.to_numpy(float)
    if not np.isfinite(scores).all() or (scores < 0).any():
        raise ValueError('Calibration scores must be finite and nonnegative.')
    table = build_adaptive_visual_threshold_from_scores(rows, n_bins=n_bins,
        quantile=THRESHOLD_QUANTILE, margin=THRESHOLD_MARGIN, min_points_per_bin=min_points_per_bin,
        smooth_window=smooth_window, threshold_floor_quantile=threshold_floor_quantile)
    table['method'] = ATTENTION_CALIBRATION_METHOD
    table.attrs['score_mode'] = 'attention'
    rows['adaptive_threshold'] = np.interp(rows.matched_progress,
        table.stage_center, table.adaptive_threshold)
    rows['above_threshold'] = scores > rows.adaptive_threshold
    rows['calibration_role'] = 'progress_frame_quantile'
    print(f'DINO attention calibration: {len(rows)} frames, {n_bins} progress bins, '
          f'{THRESHOLD_QUANTILE:g} quantile x {THRESHOLD_MARGIN:g}, with interpolation, floor and smoothing.')
    return table, rows


def calibrate_attention_leave_one_video_out(nominal_memory, stage_constrained_matching=False,
        top_k_cls=5, attention_power=1.5, min_weight=.10, **threshold_settings):
    ids = sorted({item['video_id'] for item in nominal_memory})
    if len(ids) < 2:
        raise ValueError('DINO attention leave-one-recording-out calibration needs at least two nominal bags.')
    rows = []
    for run_id in ids:
        print(f'DINO attention calibration: holding out complete nominal run {run_id}', flush=True)
        queries = [item for item in nominal_memory if item['video_id'] == run_id]
        references = [item for item in nominal_memory if item['video_id'] != run_id]
        rows.extend(score_calibration_memory(queries, references, run_id,
                    stage_constrained_matching=stage_constrained_matching,
                    top_k_cls=top_k_cls, attention_power=attention_power, min_weight=min_weight))
    return build_attention_threshold(pd.DataFrame(rows), **threshold_settings)


def validate_threshold_scale(value) -> float:
    if isinstance(value, bool):
        raise ValueError("dino_threshold_scale must be a finite positive number.")
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("dino_threshold_scale must be a finite positive number.")
    return value


def scaled_threshold(table: pd.DataFrame, scale: float) -> pd.DataFrame:
    """Scale only the decision limit; preserve scores and the saved calibration."""
    result = table.copy()
    result["base_threshold"] = result["adaptive_threshold"]
    result["threshold_scale"] = validate_threshold_scale(scale)
    result["adaptive_threshold"] *= result["threshold_scale"]
    result.attrs['threshold_scale'] = float(scale)
    return result


def score_matched_frame(score, progress, threshold_table, threshold_scale=1.0):
    """Apply the same decision rule in replay, batch CSVs and exported videos.

    A batch table may already be scaled; replay passes its multiplier directly.
    The saved unscaled bound remains available in both cases.
    """
    from .anomaly_detection import threshold_for_progress
    scale = validate_threshold_scale(threshold_scale)
    bound = threshold_for_progress(progress, threshold_table)
    base = (float(np.interp(progress, threshold_table.stage_center,
                            threshold_table.base_threshold))
            if "base_threshold" in threshold_table else bound)
    return dict(score=float(score), threshold=bound * scale,
                base_threshold=base, score_mode="attention")


def score_calibration_memory(query_memory, reference_memory, run_id, *, top_k_cls=5,
                             attention_power=1.5, min_weight=.10, stage_constrained_matching=False):
    """Score a separate successful recording using exactly the runtime matcher."""
    rows = []
    for query in query_memory:
        score, _, _, _, match, distance = compare_frame_to_nominal_memory(
            query["features"], query["attention"], query["cls"], reference_memory,
            top_k_cls=top_k_cls, attention_power=attention_power, min_weight=min_weight,
            required_stage=int(query["stage"]) if stage_constrained_matching else None)
        rows.append(dict(held_out_video_id=run_id, query_frame_id=int(query["frame_id"]),
            query_bag=str(getattr(query.get('source'), 'bag_path', query.get('video_path', ''))),
            query_progress=float(query["progress"]), matched_progress=float(match["progress"]),
            matched_video_id=int(match["video_id"]), matched_frame_id=int(match["frame_id"]),
            matched_timestamp_sec=float(match["timestamp_sec"]), score=float(score),
            cls_distance=float(distance), query_stage=int(query.get("stage", -1)),
            matched_stage=int(match.get("stage", -1))))
    return rows

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
    scores = calibration_df["score"].to_numpy(dtype=float)
    progress = calibration_df["matched_progress"].to_numpy(dtype=float)
    if not np.isfinite(scores).all() or (scores < 0).any():
        raise ValueError("Calibration scores must be finite and nonnegative.")
    if not np.isfinite(progress).all() or ((progress < 0) | (progress > 1)).any():
        raise ValueError("Calibration progress must be finite and in [0, 1].")
    if not 0 < quantile < 1 or not np.isfinite(margin) or margin < 1:
        raise ValueError("Invalid calibration quantile or margin.")
    if min_points_per_bin < 1:
        raise ValueError("min_points_per_bin must be positive.")
    if threshold_floor_quantile is not None and not 0 <= threshold_floor_quantile <= 1:
        raise ValueError("Invalid threshold floor quantile.")

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])

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
