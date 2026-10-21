"""Normal matched-patch residuals, learned only from retrieval recordings.

This is a scalar residual model, not a full-covariance PaDiM implementation.
Match first using the existing CLS/attention matcher, then measure how unusual
each patch difference is at that matched progress. No test-time adaptation.
"""
from __future__ import annotations

import numpy as np


def fit_patch_profile(memory, n_bins=30, *, top_k_cls=5, attention_power=1.5,
                      min_weight=.1, stage_constrained_matching=False):
    """Cross-recording matches avoid the zero residual from matching oneself.

    Accumulate moments per recording/bin before combining them. Repeating frames
    in a long recording therefore cannot give it extra weight. Keep only one
    recording's small residual accumulator in RAM, not feature-vector copies.
    """
    from .anomaly_detection import compare_frame_to_nominal_memory
    ids = sorted({item["video_id"] for item in memory})
    if len(ids) < 2:
        raise ValueError("Patch normality requires at least two nominal recordings.")
    if n_bins < 1:
        raise ValueError("Patch normality needs at least one progress bin.")
    patch_count = memory[0]["features"].shape[0]
    totals = np.zeros((n_bins, patch_count), dtype=np.float64)
    squares = np.zeros_like(totals)
    runs = np.zeros(n_bins, dtype=int)
    global_means, global_squares = [], []
    for run_id in ids:
        print(f"Learning normal patch variation: holding out nominal run {run_id}", flush=True)
        references = [item for item in memory if item["video_id"] != run_id]
        sums = np.zeros_like(totals)
        sums2 = np.zeros_like(totals)
        counts = np.zeros(n_bins, dtype=int)
        for query in memory:
            if query["video_id"] != run_id:
                continue
            _, residual, _, _, match, _ = compare_frame_to_nominal_memory(
                query["features"], query["attention"], query["cls"], references,
                top_k_cls=top_k_cls, attention_power=attention_power, min_weight=min_weight,
                required_stage=int(query["stage"]) if stage_constrained_matching else None)
            progress = float(match["progress"])
            if not np.isfinite(progress) or not 0 <= progress <= 1:
                raise ValueError("Nominal matched progress must be finite and in [0, 1].")
            residual = np.asarray(residual, dtype=np.float64)
            if residual.shape != (patch_count,) or not np.isfinite(residual).all():
                raise ValueError("Nominal patch residuals have inconsistent shape or nonfinite values.")
            index = min(int(progress * n_bins), n_bins - 1)
            sums[index] += residual
            sums2[index] += residual ** 2
            counts[index] += 1
        valid = counts > 0
        totals[valid] += sums[valid] / counts[valid, None]
        squares[valid] += sums2[valid] / counts[valid, None]
        runs[valid] += 1
        global_means.append(sums.sum(axis=0) / counts.sum())
        global_squares.append(sums2.sum(axis=0) / counts.sum())
    global_mean = np.mean(global_means, axis=0)
    global_square = np.mean(global_squares, axis=0)
    # Three recording-equivalents of shrinkage prevent sparse bins estimating
    # near-zero variance. This also gives empty bins a learned finite fallback.
    trust = runs[:, None] / (runs[:, None] + 3.)
    mean = trust * totals / np.maximum(runs[:, None], 1) + (1 - trust) * global_mean
    second = trust * squares / np.maximum(runs[:, None], 1) + (1 - trust) * global_square
    variance = np.maximum(second - mean ** 2, 0.)
    global_std = np.sqrt(np.maximum(global_square - global_mean ** 2, 0.))
    # Use the typical nonzero nominal residual as the noise scale too. A nearly
    # constant patch must not acquire enormous evidence from tiny numerical or
    # lighting differences merely because its estimated variance is near zero.
    positive_mean = global_mean[global_mean > 0]
    floor = max(float(np.median(global_std)),
                float(np.median(positive_mean)) if positive_mean.size else 0., 1e-6)
    profile = dict(centers=(np.arange(n_bins) + .5) / n_bins,
                   mean=mean.astype(np.float32),
                   scale=np.sqrt(variance + floor ** 2).astype(np.float32),
                   scale_floor=np.asarray(floor), run_count=runs,
                   training_runs=np.asarray(len(ids)))
    validate_patch_profile(profile, patch_count)
    print(f"Patch normality: {len(ids)} nominal runs; scale floor={floor:.6g}; "
          f"{int(np.sum(runs < 2))}/{n_bins} bins have fewer than two contributing runs.", flush=True)
    return profile


def validate_patch_profile(profile, patch_count=None):
    required = {"centers", "mean", "scale", "scale_floor", "run_count", "training_runs"}
    if not required.issubset(profile):
        raise ValueError("Incomplete DINO patch-normality profile; rebuild vision-memory.")
    centers = np.asarray(profile["centers"])
    mean, scale = np.asarray(profile["mean"]), np.asarray(profile["scale"])
    if (centers.ndim != 1 or not len(centers) or not np.isfinite(centers).all()
            or (np.diff(centers) <= 0).any() or (centers < 0).any() or (centers > 1).any()
            or mean.ndim != 2 or scale.shape != mean.shape or len(mean) != len(centers)
            or (patch_count is not None and mean.shape[1] != patch_count)
            or not np.isfinite(mean).all() or not np.isfinite(scale).all()
            or (mean < 0).any() or (scale <= 0).any()
            or np.asarray(profile["run_count"]).shape != centers.shape
            or not np.isfinite(profile["run_count"]).all() or (profile["run_count"] < 0).any()
            or not np.isfinite(profile["scale_floor"]) or float(profile["scale_floor"]) <= 0
            or int(profile["training_runs"]) < 2):
        raise ValueError("Invalid DINO patch-normality profile; rebuild vision-memory.")


def patch_evidence(patches, attention_weights, progress, profile):
    """Return one dimensionless score and its spatially pooled evidence map.

    Subtract expected residuals and divide by regularized nominal variability.
    Restore soft attention (at least half weight), without redistributing weight
    from matched regions. Average 3x3 neighborhoods, then the largest
    1% of those responses, to retain small objects without relying on one pixel.
    """
    patches = np.asarray(patches, dtype=np.float64)
    weights = np.asarray(attention_weights, dtype=np.float64)
    grid = int(np.sqrt(patches.size))
    if (patches.ndim != 1 or grid * grid != patches.size or weights.shape != patches.shape
            or not np.isfinite(patches).all() or (patches < 0).any()
            or not np.isfinite(weights).all() or (weights < 0).any() or (weights > 1).any()
            or not np.isfinite(progress) or not 0 <= progress <= 1):
        raise ValueError("Invalid patch distances, weights or progress.")
    if profile["mean"].shape[1] != patches.size:
        raise ValueError("Patch-normality grid does not match extracted features.")
    centers = profile["centers"]
    right = min(int(np.searchsorted(centers, progress, side="right")), len(centers) - 1)
    left = max(right - 1, 0)
    fraction = np.clip((progress - centers[left]) /
                       max(float(centers[right] - centers[left]), 1e-12), 0., 1.)
    mean = (1 - fraction) * profile["mean"][left] + fraction * profile["mean"][right]
    scale = (1 - fraction) * profile["scale"][left] + fraction * profile["scale"][right]
    evidence = np.maximum(patches - mean, 0.) / scale * (.5 + .5 * weights)
    padded = np.pad(evidence.reshape(grid, grid), 1, mode="edge")
    heatmap = np.lib.stride_tricks.sliding_window_view(padded, (3, 3)).mean(axis=(-1, -2)).ravel()
    count = max(1, int(np.ceil(.01 * heatmap.size)))
    score = float(np.partition(heatmap, heatmap.size - count)[-count:].mean())
    return score, heatmap


def plot_patch_reliability(profile, path):
    """Show learned down-weighting per progress bin on one fixed color scale."""
    import matplotlib.pyplot as plt
    grid = int(np.sqrt(profile["mean"].shape[1]))
    bins = len(profile["centers"])
    figure, axes = plt.subplots(int(np.ceil(bins / 5)), 5, squeeze=False, figsize=(14, 2.5 * np.ceil(bins / 5)))
    for index, axis in enumerate(axes.flat):
        axis.axis("off")
        if index < bins:
            image = axis.imshow((float(profile["scale_floor"]) / profile["scale"][index]).reshape(grid, grid),
                                vmin=0, vmax=1, cmap="viridis")
            axis.set_title(f'{profile["centers"][index]:.0%} progress; {profile["run_count"][index]} runs')
    figure.colorbar(image, ax=list(axes.flat), label="Relative reliability (low = more normal variation)", shrink=.6)
    figure.suptitle("Nominal patch reliability — learned from retrieval recordings only")
    figure.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(figure)
