"""Attention-weighted, bidirectional local patch matching (no extra network)."""
from __future__ import annotations

import numpy as np


def spatial_attention_distances(query, reference, query_attention, reference_attention,
                                attention_power=1.5, min_weight=.10):
    """Allow one patch of displacement, penalizing normalized image distance.

    Each patch searches a 3x3 neighbourhood, including its original location.
    Cost squared = feature RMS squared + squared normalized displacement / d.
    The maximum of forward and reverse nearest costs retains missing/extra
    content. This is feature correspondence, not object tracking or optical flow.
    """
    from .anomaly_detection import make_attention_weights
    query, reference = np.asarray(query), np.asarray(reference)
    if query.ndim != 2 or query.shape != reference.shape or query.shape[1] < 1:
        raise ValueError('Spatial comparison requires matching [patch, feature] arrays.')
    grid = int(np.sqrt(len(query)))
    if grid * grid != len(query) or grid < 1:
        raise ValueError('Spatial comparison requires a square patch grid.')
    if not np.isfinite(query).all() or not np.isfinite(reference).all():
        raise ValueError('Spatial comparison features must be finite.')
    dimension = query.shape[1]
    q = query.astype(np.float32, copy=False).reshape(grid, grid, dimension)
    r = reference.astype(np.float32, copy=False).reshape(grid, grid, dimension)
    forward = np.mean((q - r) ** 2, axis=-1)
    reverse = forward.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if not (dy or dx):
                continue
            ys, xs = slice(max(0, -dy), min(grid, grid - dy)), slice(max(0, -dx), min(grid, grid - dx))
            yr, xr = slice(max(0, dy), min(grid, grid + dy)), slice(max(0, dx), min(grid, grid + dx))
            displacement = (dx * dx + dy * dy) / (grid * grid * dimension)
            cost = np.mean((q[ys, xs] - r[yr, xr]) ** 2, axis=-1) + displacement
            np.minimum(forward[ys, xs], cost, out=forward[ys, xs])
            np.minimum(reverse[yr, xr], cost, out=reverse[yr, xr])
    weights = make_attention_weights(np.asarray(query_attention), np.asarray(reference_attention),
                                     attention_power=attention_power, min_weight=min_weight)
    if weights.shape != (grid * grid,) or not np.isfinite(weights).all():
        raise ValueError('Attention must contain one finite value per patch.')
    return np.sqrt(np.maximum(forward, reverse)).reshape(-1) * weights
