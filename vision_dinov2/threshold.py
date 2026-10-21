"""Progress-bin calibration and shared decisions for attention-weighted DINO."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .anomaly_detection import compare_frame_to_nominal_memory


ATTENTION_CALIBRATION_METHOD = "matched_patch_normality_progress_v4_symmetric_local"
THRESHOLD_QUANTILE = 0.95
THRESHOLD_MARGIN = 1.10


def build_attention_threshold(calibration_df, n_bins=30, min_points_per_bin=2,
                             smooth_window=3, threshold_floor_quantile=.20):
    """Equal-recording progress bounds for the final, normality-adjusted score."""
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
    rows['calibration_role'] = 'independent_progress_calibration'
    print(f'DINO attention calibration: {len(rows)} frames, {n_bins} progress bins, '
          f'per-recording {THRESHOLD_QUANTILE:g} quantiles, then {THRESHOLD_QUANTILE:g} across runs x '
          f'{THRESHOLD_MARGIN:g}; smoothing never lowers populated-bin bounds.')
    return table, rows


def calibrate_attention_leave_one_video_out(nominal_memory, stage_constrained_matching=False,
        top_k_cls=5, attention_power=1.5, min_weight=.10, **threshold_settings):
    from .patch_normality import fit_patch_profile
    ids = sorted({item['video_id'] for item in nominal_memory})
    if len(ids) < 3:
        raise ValueError('Without separate calibration bags, patch normality needs at least three nominal bags.')
    matcher = dict(stage_constrained_matching=stage_constrained_matching,
                   top_k_cls=top_k_cls, attention_power=attention_power, min_weight=min_weight)
    rows = []
    for run_id in ids:
        print(f'DINO attention calibration: holding out complete nominal run {run_id}', flush=True)
        queries = [item for item in nominal_memory if item['video_id'] == run_id]
        references = [item for item in nominal_memory if item['video_id'] != run_id]
        # The held-out execution must not influence either the matcher memory
        # or the variability model used to score its calibration frames.
        profile = fit_patch_profile(references, n_bins=threshold_settings.get("n_bins", 30), **matcher)
        rows.extend(score_calibration_memory(queries, references, run_id,
                    patch_profile=profile, **matcher))
    table, rows = build_attention_threshold(pd.DataFrame(rows), **threshold_settings)
    table.attrs["patch_profile"] = fit_patch_profile(nominal_memory,
        n_bins=threshold_settings.get("n_bins", 30), **matcher)
    rows["calibration_role"] = "cross_fitted_progress_calibration"
    return table, rows


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


def score_matched_frame(score, progress, threshold_table, threshold_scale=1.0,
                        *, patches=None, attention_weights=None):
    """Apply the same decision rule in replay, batch CSVs and exported videos.

    A batch table may already be scaled; replay passes its multiplier directly.
    The saved unscaled bound remains available in both cases.
    """
    from .anomaly_detection import threshold_for_progress
    raw_score = float(score)
    heatmap = None
    if "patch_profile" in threshold_table.attrs:
        from .patch_normality import patch_evidence
        score, heatmap = patch_evidence(patches, attention_weights, progress, threshold_table.attrs["patch_profile"])
    scale = validate_threshold_scale(threshold_scale)
    bound = threshold_for_progress(progress, threshold_table)
    base = (float(np.interp(progress, threshold_table.stage_center,
                            threshold_table.base_threshold))
            if "base_threshold" in threshold_table else bound)
    return dict(score=float(score), threshold=bound * scale,
                base_threshold=base, score_mode="attention", raw_global_score=raw_score,
                heatmap=heatmap)


def score_calibration_memory(query_memory, reference_memory, run_id, *, top_k_cls=5,
                             attention_power=1.5, min_weight=.10, stage_constrained_matching=False,
                             patch_profile=None):
    """Score a separate successful recording using exactly the runtime matcher."""
    rows = []
    for query in query_memory:
        score, patches, _, weights, match, distance = compare_frame_to_nominal_memory(
            query["features"], query["attention"], query["cls"], reference_memory,
            top_k_cls=top_k_cls, attention_power=attention_power, min_weight=min_weight,
            required_stage=int(query["stage"]) if stage_constrained_matching else None)
        raw_score = float(score)
        if patch_profile is not None:
            from .patch_normality import patch_evidence
            score, _ = patch_evidence(patches, weights, float(match["progress"]), patch_profile)
        rows.append(dict(held_out_video_id=run_id, query_frame_id=int(query["frame_id"]),
            query_bag=str(getattr(query.get('source'), 'bag_path', query.get('video_path', ''))),
            query_progress=float(query["progress"]), matched_progress=float(match["progress"]),
            matched_video_id=int(match["video_id"]), matched_frame_id=int(match["frame_id"]),
            matched_timestamp_sec=float(match["timestamp_sec"]), score=float(score), raw_global_score=raw_score,
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

    # Group quantiles prevent a long bag from dominating the distribution.
    grouped = "held_out_video_id" in calibration_df
    global_scores = (calibration_df.groupby("held_out_video_id").score.quantile(quantile).to_numpy()
                     if grouped else scores)
    global_threshold = float(np.quantile(global_scores, quantile) * margin)

    threshold_floor = (
        (float(calibration_df.groupby("held_out_video_id").score.quantile(threshold_floor_quantile).median())
         if grouped else float(np.quantile(scores, threshold_floor_quantile)))
        if threshold_floor_quantile is not None
        else 0.0
    )

    threshold_values: list[float] = []
    counts: list[int] = []
    run_counts: list[int] = []

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

        bin_rows = calibration_df.loc[mask]
        bin_scores = (bin_rows.groupby("held_out_video_id").score.quantile(quantile).to_numpy()
                      if grouped else bin_rows.score.to_numpy())
        counts.append(len(bin_rows))
        run_counts.append(len(bin_scores))

        if len(bin_rows) >= min_points_per_bin and len(bin_scores) >= (2 if grouped else 1):
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
            "run_count": run_counts,
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
        smoothed = (
            threshold_df["adaptive_threshold"]
            .rolling(
                window=smooth_window,
                center=True,
                min_periods=1,
            )
            .median()
        )
        # A median filter previously lowered narrow, valid nominal peaks.
        threshold_df["adaptive_threshold"] = np.maximum(threshold_df.adaptive_threshold, smoothed)

    return threshold_df
