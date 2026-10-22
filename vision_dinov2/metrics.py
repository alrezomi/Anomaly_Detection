"""Locally matched residuals for a regularized diagonal Mahalanobis model.

Keep a fixed subset of at most 32 feature channels, avoiding a large, poorly
estimated full covariance per patch/bin. Frame retrieval is unchanged. Local
correspondences minimize Euclidean distance in this fixed subspace; the fitted
metric scores both directions afterwards. No query data enters the fit.
"""
import numpy as np


def validate_metric(value):
    if value not in {"euclidean", "mahalanobis"}:
        raise ValueError('dino_metric must be "euclidean" or "mahalanobis".')
    return value


def local_residual_vectors(query, reference, channels):
    query, reference = np.asarray(query, dtype=np.float32), np.asarray(reference, dtype=np.float32)
    if (query.ndim != 2 or query.shape != reference.shape or not np.isfinite(query).all()
            or not np.isfinite(reference).all()):
        raise ValueError('Mahalanobis matching needs equal finite feature arrays.')
    query, reference = query[:, channels], reference[:, channels]
    grid = int(np.sqrt(len(query)))
    if query.shape != reference.shape or grid * grid != len(query):
        raise ValueError("Mahalanobis matching requires equal square patch grids.")
    query, reference = query.reshape(grid, grid, -1), reference.reshape(grid, grid, -1)
    forward = np.full((grid, grid), np.inf)
    reverse = np.full_like(forward, np.inf)
    fv, rv = np.zeros_like(query), np.zeros_like(query)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            y0, y1 = max(0, -dy), min(grid, grid - dy)
            x0, x1 = max(0, -dx), min(grid, grid - dx)
            q = (slice(y0, y1), slice(x0, x1))
            r = (slice(y0 + dy, y1 + dy), slice(x0 + dx, x1 + dx))
            delta = query[q] - reference[r]
            distance = np.mean(delta * delta, axis=-1)
            better = distance < forward[q]
            fv[q][better] = delta[better]
            forward[q] = np.minimum(forward[q], distance)
            better = distance < reverse[r]
            rv[r][better] = delta[better]
            reverse[r] = np.minimum(reverse[r], distance)
    return np.stack([fv.reshape(-1, len(channels)), rv.reshape(-1, len(channels))])


def interpolate_profile(profile, progress, key):
    centers = profile["centers"]
    right = min(int(np.searchsorted(centers, progress, side="right")), len(centers) - 1)
    left = max(right - 1, 0)
    fraction = np.clip((progress - centers[left]) /
                       max(float(centers[right] - centers[left]), 1e-12), 0., 1.)
    return (1 - fraction) * profile[key][left] + fraction * profile[key][right]


def mahalanobis_patches(query, reference, progress, profile):
    if 'feature_dimension' in profile and np.asarray(query).shape[-1] != int(profile['feature_dimension']):
        raise ValueError('Mahalanobis feature dimension differs from calibration.')
    residuals = local_residual_vectors(query, reference, profile["channels"])
    mean = interpolate_profile(profile, progress, "vector_mean")
    variance = interpolate_profile(profile, progress, "vector_variance")
    distances = np.sqrt(np.sum((residuals - mean) ** 2 / variance, axis=-1))
    return np.max(distances, axis=0)
