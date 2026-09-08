"""Homography helpers with explicit source/destination semantics."""

from __future__ import annotations

import numpy as np


def transform_points_source_to_destination(
    points_xy: np.ndarray | list[tuple[float, float]],
    homography_source_to_destination: np.ndarray,
) -> np.ndarray:
    """Apply a 3x3 homography returned by ``findHomography(src, dst)``."""
    points = np.asarray(points_xy, dtype=np.float64)
    matrix = np.asarray(homography_source_to_destination, dtype=np.float64)
    if points.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"Expected points with shape (N, 2), got {points.shape}")
    if matrix.shape != (3, 3):
        raise ValueError(f"Expected homography with shape (3, 3), got {matrix.shape}")
    homogeneous = np.column_stack((points, np.ones(len(points))))
    projected = (matrix @ homogeneous.T).T
    if np.any(np.isclose(projected[:, 2], 0.0)):
        raise ValueError("Homography maps at least one point to infinity")
    return projected[:, :2] / projected[:, 2, None]
