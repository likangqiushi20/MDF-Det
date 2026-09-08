"""Foreground-region extraction and paper-specified blob filtering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

from catloss.data import temporal_change_score


@dataclass(frozen=True)
class ForegroundRegion:
    x: int
    y: int
    width: int
    height: int
    area: int
    center_x: float
    center_y: float

    @property
    def extent(self) -> float:
        return self.area / (self.width * self.height)

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height


def extract_foreground_regions(
    images: Sequence[np.ndarray],
    *,
    foreground_quantile: float = 0.98,
    minimum_area: int = 4,
    maximum_area: int = 1000,
    minimum_extent: float = 0.3,
    minimum_aspect: float = 0.2,
    maximum_aspect: float = 5.0,
) -> tuple[np.ndarray, list[ForegroundRegion]]:
    """Extract connected regions using a causal temporal-median foreground map."""
    if not 0.0 < foreground_quantile < 1.0:
        raise ValueError("foreground_quantile must be between zero and one")
    score = temporal_change_score(images)
    positive_scores = score[score > 0]
    if not len(positive_scores):
        return np.zeros(score.shape, dtype=np.uint8), []
    threshold = float(np.quantile(positive_scores, foreground_quantile))
    mask = (score >= threshold).astype(np.uint8)
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    filtered_mask = np.zeros_like(mask)
    regions = []
    for label in range(1, count):
        x, y, width, height, area = stats[label].tolist()
        region = ForegroundRegion(
            x=x,
            y=y,
            width=width,
            height=height,
            area=area,
            center_x=float(centroids[label, 0]),
            center_y=float(centroids[label, 1]),
        )
        if not minimum_area < region.area < maximum_area:
            continue
        if region.extent <= minimum_extent:
            continue
        if not minimum_aspect < region.aspect_ratio < maximum_aspect:
            continue
        regions.append(region)
        filtered_mask[labels == label] = 1
    return filtered_mask, regions
