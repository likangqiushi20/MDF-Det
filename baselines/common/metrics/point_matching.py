"""Strict one-to-one matching for point detections."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class PointMatchResult:
    matches: tuple[tuple[int, int, float], ...]
    unmatched_predictions: tuple[int, ...]
    unmatched_truths: tuple[int, ...]

    @property
    def true_positives(self) -> int:
        return len(self.matches)

    @property
    def false_positives(self) -> int:
        return len(self.unmatched_predictions)

    @property
    def false_negatives(self) -> int:
        return len(self.unmatched_truths)


def _as_points(values: np.ndarray | list[tuple[float, float]]) -> np.ndarray:
    points = np.asarray(values, dtype=np.float64)
    if points.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"Expected point array with shape (N, 2), got {points.shape}")
    if not np.isfinite(points).all():
        raise ValueError("Points must contain only finite values")
    return points


def match_points(
    predictions: np.ndarray | list[tuple[float, float]],
    truths: np.ndarray | list[tuple[float, float]],
    radius: float,
) -> PointMatchResult:
    """Maximize valid one-to-one matches, then minimize total pixel distance."""
    if not np.isfinite(radius) or radius < 0:
        raise ValueError("radius must be a finite non-negative number")
    predicted = _as_points(predictions)
    target = _as_points(truths)
    if len(predicted) == 0 or len(target) == 0:
        return PointMatchResult(
            matches=(),
            unmatched_predictions=tuple(range(len(predicted))),
            unmatched_truths=tuple(range(len(target))),
        )

    distances = np.linalg.norm(predicted[:, None, :] - target[None, :, :], axis=2)
    invalid_cost = (max(len(predicted), len(target)) + 1) * (radius + 1.0)
    cost = np.where(distances <= radius, distances, invalid_cost)
    prediction_indexes, truth_indexes = linear_sum_assignment(cost)

    matches = tuple(
        sorted(
            (
                (int(prediction_index), int(truth_index), float(distances[prediction_index, truth_index]))
                for prediction_index, truth_index in zip(prediction_indexes, truth_indexes)
                if distances[prediction_index, truth_index] <= radius
            ),
            key=lambda match: match[0],
        )
    )
    matched_predictions = {match[0] for match in matches}
    matched_truths = {match[1] for match in matches}
    return PointMatchResult(
        matches=matches,
        unmatched_predictions=tuple(
            index for index in range(len(predicted)) if index not in matched_predictions
        ),
        unmatched_truths=tuple(
            index for index in range(len(target)) if index not in matched_truths
        ),
    )
