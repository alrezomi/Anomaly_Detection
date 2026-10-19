"""Fixed and adaptive threshold calibration for the DINOv2 detector."""

from __future__ import annotations

import numpy as np
import pandas as pd


SEQUENCE_CALIBRATION_METHOD = "split_nominal_sequence_max_v1"
NOTEBOOK_CALIBRATION_METHOD = "notebook_patch_rms_frame_quantile_v1"


def build_notebook_threshold(calibration_df):
    """Notebook cell 47: pooled frame 95th percentile times 1.10.

    These are empirical frame statistics, not a whole-run false-alarm guarantee.
    Queries must be held out from memory by the caller, at recording level.
    """
    rows = calibration_df.copy()
    if rows.empty or not {'score', 'held_out_video_id'}.issubset(rows):
        raise ValueError('Notebook calibration requires held-out frame scores and run IDs.')
    scores = rows.score.to_numpy(float)
    if not np.isfinite(scores).all() or (scores < 0).any():
        raise ValueError('Calibration scores must be finite and nonnegative.')
    bound = float(np.quantile(scores, .95)) * 1.10
    rows['adaptive_threshold'] = bound
    rows['above_threshold'] = scores > bound
    rows['calibration_role'] = 'frame_quantile'
    table = pd.DataFrame(dict(stage_center=[0., 1.], adaptive_threshold=bound,
                              count=len(scores), quantile=.95, margin=1.10,
                              method=NOTEBOOK_CALIBRATION_METHOD))
    table.attrs['score_mode'] = 'notebook'
    print(f'Notebook DINO calibration: {rows.held_out_video_id.nunique()} held-out runs, '
          f'{len(rows)} frames; 95th percentile x 1.10 = {bound:.6g}.')
    print('Calibration crossings (not independent evaluation): '
          f'{rows.above_threshold.mean():.2%} of frames; '
          f'{rows.groupby("held_out_video_id").above_threshold.any().mean():.2%} of runs.')
    return table, rows


def calibrate_notebook_leave_one_video_out(nominal_memory, stage_constrained_matching=False):
    ids = sorted({item['video_id'] for item in nominal_memory})
    if len(ids) < 2:
        raise ValueError('Notebook leave-one-recording-out calibration needs at least two nominal bags.')
    rows = []
    for run_id in ids:
        print(f'Notebook calibration: holding out complete nominal run {run_id}', flush=True)
        queries = [item for item in nominal_memory if item['video_id'] == run_id]
        references = [item for item in nominal_memory if item['video_id'] != run_id]
        rows.extend(score_calibration_memory(queries, references, run_id,
                    stage_constrained_matching=stage_constrained_matching, notebook_scoring=True))
    return build_notebook_threshold(pd.DataFrame(rows))


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


def build_sequence_threshold(calibration_df, n_bins=30, quantile=.90,
                             margin=1.10, smooth_window=3):
    """FIDeL-inspired scalar score band, using two disjoint nominal subsets.

    One subset estimates mean/scale versus *matched memory* progress. The other
    supplies one maximum standardized deviation per complete recording. Neither
    subset is part of retrieval memory. This is not FIDeL's patch/OT scorer.
    """
    required = {"held_out_video_id", "score", "matched_progress"}
    if not required.issubset(calibration_df) or calibration_df.empty:
        raise ValueError("Sequence calibration needs run IDs, scores and matched progress.")
    values = calibration_df[["score", "matched_progress"]].to_numpy(float)
    if (not np.isfinite(values).all() or (values[:, 0] < 0).any()
            or ((values[:, 1] < 0) | (values[:, 1] > 1)).any()):
        raise ValueError("Calibration scores/progress must be finite and in range.")
    if n_bins < 1 or not 0 < quantile < 1 or not np.isfinite(margin) or margin < 1:
        raise ValueError("Invalid sequence calibration bins, quantile or margin.")
    ids = np.asarray(sorted(calibration_df.held_out_video_id.unique()))
    # Fixed, data-independent partition. Keep complete recordings together.
    ids = np.random.default_rng(42).permutation(ids)
    split = len(ids) // 2
    shape_ids, limit_ids = ids[:split], ids[split:]
    rank = int(np.ceil((len(limit_ids) + 1) * quantile))
    if split < 2 or rank > len(limit_ids):
        raise ValueError("Too few independent calibration bags for the sequence quantile. "
                         "At 0.90, provide at least 18 bags (9 for the limit subset).")
    rows = calibration_df.copy()
    rows["calibration_role"] = np.where(rows.held_out_video_id.isin(shape_ids), "shape", "limit")
    shape = rows.loc[rows.calibration_role == "shape"].copy()
    centers = (np.arange(n_bins) + .5) / n_bins
    shape["bin"] = np.minimum((shape.matched_progress * n_bins).astype(int), n_bins - 1)
    shape["score_squared"] = shape.score ** 2
    # Equal run weights within each bin; a long bag must not dominate the shape.
    moments = shape.groupby(["held_out_video_id", "bin"])[["score", "score_squared"]].mean()
    local = moments.groupby("bin").mean().reindex(range(n_bins)).interpolate().bfill().ffill()
    mean = local.score
    scale = np.sqrt(np.maximum(local.score_squared - mean ** 2, 0))
    global_moments = shape.groupby("held_out_video_id")[["score", "score_squared"]].mean().mean()
    global_std = np.sqrt(max(0., float(global_moments.score_squared - global_moments.score ** 2)))
    floor = max(float(global_std) * .1, 1e-8)
    # Smooth the band shape BEFORE calibrating maxima; never lower a fitted bound afterwards.
    window = max(1, int(smooth_window or 1))
    mean = mean.rolling(window, center=True, min_periods=1).mean().to_numpy()
    scale = scale.rolling(window, center=True, min_periods=1).mean().clip(lower=floor).to_numpy()
    rows["baseline_mean"] = np.interp(rows.matched_progress, centers, mean)
    rows["baseline_scale"] = np.interp(rows.matched_progress, centers, scale)
    rows["standardized_deviation"] = (rows.score - rows.baseline_mean) / rows.baseline_scale
    maxima = rows.loc[rows.calibration_role == "limit"].groupby("held_out_video_id").standardized_deviation.max()
    bandwidth = max(0., float(np.sort(maxima.to_numpy())[rank - 1]))
    table = pd.DataFrame(dict(stage_center=centers,
        adaptive_threshold=np.maximum(mean + margin * bandwidth * scale, 1e-8),
        baseline_mean=mean, baseline_scale=scale,
        count=shape.groupby("bin").size().reindex(range(n_bins), fill_value=0).to_numpy(),
        bandwidth=bandwidth, method=SEQUENCE_CALIBRATION_METHOD,
        shape_runs=len(shape_ids), limit_runs=len(limit_ids), quantile=quantile, margin=margin))
    rows["adaptive_threshold"] = np.interp(rows.matched_progress, centers, table.adaptive_threshold)
    rows["above_threshold"] = rows.score > rows.adaptive_threshold
    print(f"DINO sequence calibration: {len(shape_ids)} shape bags, {len(limit_ids)} limit bags; "
          f"maximum-deviation rank {rank}/{len(limit_ids)}, bandwidth {bandwidth:.6g}")
    print("Calibration crossings (not independent evaluation):",
          rows.groupby("calibration_role").above_threshold.mean().to_dict())
    return table, rows


def score_calibration_memory(query_memory, reference_memory, run_id, *, top_k_cls=5,
                             attention_power=1.5, min_weight=.10, stage_constrained_matching=False,
                             patch_maps=None, notebook_scoring=False):
    """Score a separate successful recording using exactly the runtime matcher."""
    rows = []
    for query in query_memory:
        score, patches, _, _, match, distance = compare_frame_to_nominal_memory(
            query["features"], query["attention"], query["cls"], reference_memory,
            top_k_cls=top_k_cls, attention_power=attention_power, min_weight=min_weight,
            required_stage=int(query["stage"]) if stage_constrained_matching else None,
            local_scoring=patch_maps is not None, notebook_scoring=notebook_scoring)
        if patch_maps is not None:
            patch_maps.append(np.asarray(patches, dtype=np.float32))
        rows.append(dict(held_out_video_id=run_id, query_frame_id=int(query["frame_id"]),
            query_bag=str(getattr(query.get('source'), 'bag_path', query.get('video_path', ''))),
            query_progress=float(query["progress"]), matched_progress=float(match["progress"]),
            matched_video_id=int(match["video_id"]), matched_frame_id=int(match["frame_id"]),
            matched_timestamp_sec=float(match["timestamp_sec"]), score=float(score),
            cls_distance=float(distance), query_stage=int(query.get("stage", -1)),
            matched_stage=int(match.get("stage", -1))))
    return rows

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
    stage_constrained_matching: bool = False,
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
                required_stage=(
                    int(query_item.get("stage", -1))
                    if stage_constrained_matching else None
                ),
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
                    "query_stage": int(query_item.get("stage", -1)),
                    "matched_stage": int(matched_item.get("stage", -1)),
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
