"""Location-aware DINO patch scoring, calibrated on separate nominal runs.

The existing matcher chooses the reference frame. This module then tests all
3x3 patch neighborhoods without treating CLS attention as an object mask.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LOCAL_METHOD = "split_nominal_local_regions_v3_attention_spatial"


def score_matched_frame(global_score, patch_scores, progress, threshold_table, threshold_scale=1.0,
                        normalized_feature_dimension=None):
    """Shared by replay, batch CSV scoring and exported heatmap video."""
    from .anomaly_detection import threshold_for_progress
    profile = threshold_table.attrs.get('local_profile')
    if profile is None:
        base = threshold_for_progress(progress, threshold_table)
        return dict(score=float(global_score), threshold=base * threshold_scale,
                    base_threshold=base, heatmap=None, score_mode='mean')
    decision = local_decision(patch_scores, progress, profile,
        threshold_scale * threshold_table.attrs.get('threshold_scale', 1.0),
        normalized_feature_dimension=normalized_feature_dimension)
    decision.update(global_score=float(global_score), score_mode='local')
    return decision


def region_distances(patch_scores):
    patches = np.asarray(patch_scores, dtype=np.float64)
    grid = int(np.sqrt(patches.size))
    if patches.ndim != 1 or grid * grid != patches.size or grid < 1:
        raise ValueError("Local scoring requires a square patch grid.")
    if not np.isfinite(patches).all() or (patches < 0).any():
        raise ValueError("Patch distances must be finite and nonnegative.")
    window = min(3, grid)
    regions = np.lib.stride_tricks.sliding_window_view(patches.reshape(grid, grid), (window, window))
    return regions.mean(axis=(-1, -2)).reshape(-1), grid, window


def _interpolate(profile, progress):
    centers = profile['centers']
    progress = float(np.clip(progress, centers[0], centers[-1]))
    right = int(np.searchsorted(centers, progress, side='right'))
    right = min(right, len(centers) - 1)
    left = max(right - 1, 0)
    weight = (progress - centers[left]) / max(float(centers[right] - centers[left]), 1e-12)
    return tuple((1 - weight) * profile[key][left] + weight * profile[key][right]
                 for key in ('mean', 'scale'))


def local_decision(patch_scores, progress, profile, threshold_scale=1.0,
                   normalized_feature_dimension=None):
    """Return raw distance/bound of the region with the greatest bound ratio.

    All region tests share the sequence-max calibrated bandwidth. Choosing the
    largest ratio makes the displayed decision exactly equivalent to any region
    crossing its bound. A positive global multiplier cannot change this choice.
    """
    from .threshold import validate_threshold_scale
    threshold_scale = validate_threshold_scale(threshold_scale)
    regions, grid, window = region_distances(patch_scores)
    if grid != int(profile['grid']) or window != int(profile['window']):
        raise ValueError("Local calibration grid does not match the DINO features. Rebuild vision-memory.")
    mean, scale = _interpolate(profile, progress)
    bounds = np.maximum(mean + float(profile['bandwidth']) * float(profile['margin']) * scale, 1e-8)
    ratios = regions / (bounds * threshold_scale)
    selected = int(np.argmax(ratios))
    side = grid - window + 1
    y, x = divmod(selected, side)
    # Paint every patch by the strongest calibrated region covering it.
    heat = np.zeros((grid, grid), dtype=float)
    ratio_grid = ratios.reshape(side, side)
    for dy in range(window):
        for dx in range(window):
            np.maximum(heat[dy:dy + side, dx:dx + side], ratio_grid,
                       out=heat[dy:dy + side, dx:dx + side])
    raw_selected = int(np.argmax(regions))
    raw_y, raw_x = divmod(raw_selected, side)
    result = dict(score=float(regions[selected]), threshold=float(bounds[selected] * threshold_scale),
                base_threshold=float(bounds[selected]), epsilon=float(ratios[selected]),
                region_x=x, region_y=y, region_size=window, patch_grid=grid,
                max_region_distance=float(regions[raw_selected]),
                max_region_threshold=float(bounds[raw_selected] * threshold_scale),
                max_region_x=raw_x, max_region_y=raw_y,
                max_patch_distance=float(np.max(patch_scores)),
                heatmap=heat.reshape(-1))
    if normalized_feature_dimension is not None:
        # For unit vectors u,v in R^d, RMS(u-v) <= 2/sqrt(d).
        # A mean of these distances has the same ceiling. Do not clip bounds:
        # clipping would conceal a calibration problem, not restore sensitivity.
        ceiling = normalized_distance_ceiling(normalized_feature_dimension)
        result.update(distance_ceiling=ceiling,
                      unreachable_region_fraction=float(np.mean(bounds * threshold_scale >= ceiling)))
    return result


def normalized_distance_ceiling(feature_dimension):
    if int(feature_dimension) != feature_dimension or feature_dimension < 1:
        raise ValueError('Feature dimension must be a positive integer.')
    return float(2 / np.sqrt(feature_dimension))


def local_profile_diagnostics(profile, feature_dimension, threshold_scale=1.0):
    """Read-only health check; it never modifies or silently lowers thresholds."""
    from .threshold import validate_threshold_scale
    multiplier = validate_threshold_scale(threshold_scale)
    bounds = (profile['mean'] + float(profile['margin']) * float(profile['bandwidth']) * profile['scale']) * multiplier
    ceiling = normalized_distance_ceiling(feature_dimension)
    return dict(bandwidth=float(profile['bandwidth']), distance_ceiling=ceiling,
                threshold_min=float(np.min(bounds)), threshold_median=float(np.median(bounds)),
                threshold_max=float(np.max(bounds)),
                unreachable_region_fraction=float(np.mean(bounds >= ceiling)))


def fit_local_threshold(calibration_df, patch_maps, n_bins=30, quantile=.90, margin=1.10):
    """Fit region-specific normal variation, then max over regions AND time.

    Shape and limit recordings remain disjoint from one another and memory.
    Equal run weighting is used within each progress bin. Averaging 3x3 patch
    neighborhoods reduces isolated artifacts without averaging the whole scene.
    """
    rows = calibration_df.copy()
    if len(rows) != len(patch_maps) or rows.empty:
        raise ValueError("Local calibration requires one patch map per scored frame.")
    progress = rows.matched_progress.to_numpy(float)
    if not np.isfinite(progress).all() or ((progress < 0) | (progress > 1)).any():
        raise ValueError("Calibration progress must be finite and in [0, 1].")
    if n_bins < 1 or not 0 < quantile < 1 or not np.isfinite(margin) or margin < 1:
        raise ValueError("Invalid local threshold settings.")
    ids = np.random.default_rng(42).permutation(sorted(rows.held_out_video_id.unique()))
    shape_ids, limit_ids = ids[:len(ids) // 2], ids[len(ids) // 2:]
    rank = int(np.ceil((len(limit_ids) + 1) * quantile))
    if len(shape_ids) < 2 or rank > len(limit_ids):
        raise ValueError("Local calibration needs at least 18 separate successful bags at quantile 0.90.")
    distances = [region_distances(patches) for patches in patch_maps]
    grid, window = distances[0][1:]
    if any(item[1:] != (grid, window) for item in distances):
        raise ValueError("Calibration patch grids differ.")
    values = np.stack([item[0] for item in distances])
    bins = np.minimum((progress * n_bins).astype(int), n_bins - 1)
    run_ids = rows.held_out_video_id.to_numpy()
    rows['calibration_role'] = np.where(np.isin(run_ids, shape_ids), 'shape', 'limit')
    shape_moments = []
    means, squares, counts = [], [], []
    for run_id in shape_ids:
        run = values[run_ids == run_id]
        shape_moments.append((run.mean(axis=0), (run ** 2).mean(axis=0)))
    global_mean = np.mean([x[0] for x in shape_moments], axis=0)
    global_square = np.mean([x[1] for x in shape_moments], axis=0)
    # A nearly constant region in the shape split can otherwise produce a huge
    # standardized residual in a limit run. Because bandwidth is shared, that
    # single denominator can disable detection everywhere. Regularize with the
    # median location-wise nominal variability, learned ONLY from shape runs.
    # The median avoids letting a few noisy robot/background regions set this
    # floor. Larger region-specific variations are still retained below.
    global_std = np.sqrt(np.maximum(global_square - global_mean ** 2, 0))
    pooled_floor = max(float(np.median(global_std)), 1e-6)
    floor = np.maximum(.1 * global_std, pooled_floor)
    for bin_id in range(n_bins):
        per_run = [values[(run_ids == run_id) & (bins == bin_id)] for run_id in shape_ids]
        per_run = [samples for samples in per_run if len(samples)]
        counts.append(sum(map(len, per_run)))
        means.append(np.mean([x.mean(axis=0) for x in per_run], axis=0) if per_run
                     else np.full(values.shape[1], np.nan))
        squares.append(np.mean([(x ** 2).mean(axis=0) for x in per_run], axis=0) if per_run
                       else np.full(values.shape[1], np.nan))
    mean = pd.DataFrame(means).interpolate().bfill().ffill().to_numpy()
    square = pd.DataFrame(squares).interpolate().bfill().ffill().to_numpy()
    scale = np.maximum(np.sqrt(np.maximum(square - mean ** 2, 0)), floor)
    profile = dict(centers=(np.arange(n_bins) + .5) / n_bins, mean=mean, scale=scale,
                   grid=np.asarray(grid), window=np.asarray(window), margin=np.asarray(margin),
                   pooled_scale_floor=np.asarray(pooled_floor),
                   bandwidth=np.asarray(0.))
    deviations, locations, peak_values, peak_means, peak_scales = [], [], [], [], []
    for scores, p in zip(values, progress):
        mu, sigma = _interpolate(profile, p)
        standardized = (scores - mu) / sigma
        selected = int(np.argmax(standardized))
        deviations.append(float(standardized[selected]))
        locations.append(divmod(selected, grid - window + 1))
        peak_values.append(float(scores[selected]))
        peak_means.append(float(mu[selected]))
        peak_scales.append(float(sigma[selected]))
    rows['maximum_region_deviation'] = deviations
    rows['deviation_region_y'] = [y for y, x in locations]
    rows['deviation_region_x'] = [x for y, x in locations]
    rows['deviation_region_distance'] = peak_values
    rows['deviation_region_mean'] = peak_means
    rows['deviation_region_scale'] = peak_scales
    rows['run_maximum_region_deviation'] = rows.groupby('held_out_video_id').maximum_region_deviation.transform('max')
    maxima = rows.loc[rows.calibration_role == 'limit'].groupby('held_out_video_id').maximum_region_deviation.max()
    bandwidth = max(0., float(np.sort(maxima.to_numpy())[rank - 1]))
    profile['bandwidth'] = np.asarray(bandwidth)
    # Preserve the original whole-image distance for comparison, not decisions.
    rows['global_score'] = rows.score
    for index, (patches, p) in enumerate(zip(patch_maps, progress)):
        decision = local_decision(patches, p, profile)
        for column, value in [('score', decision['score']), ('adaptive_threshold', decision['threshold']),
                              ('region_x', decision['region_x']), ('region_y', decision['region_y'])]:
            rows.loc[rows.index[index], column] = value
    rows['above_threshold'] = rows.score > rows.adaptive_threshold
    # CSV curve is a summary only: runtime always uses the selected region's bound.
    table = pd.DataFrame(dict(stage_center=profile['centers'], count=counts,
        adaptive_threshold=np.mean(mean + margin * bandwidth * scale, axis=1),
        method=LOCAL_METHOD, shape_runs=len(shape_ids), limit_runs=len(limit_ids),
        bandwidth=bandwidth, margin=margin, quantile=quantile, pooled_scale_floor=pooled_floor))
    table.attrs['local_profile'] = profile
    print(f"DINO local calibration: {len(shape_ids)} shape / {len(limit_ids)} limit bags; "
          f"{grid}x{grid} patches, {window}x{window} regions, maximum-deviation rank {rank}/{len(limit_ids)}")
    print("Largest nominal region deviations by limit bag:", maxima.to_dict())
    print(f"Nominal pooled scale floor: {pooled_floor:.6g}; calibrated bandwidth: {bandwidth:.6g}")
    return table, rows
