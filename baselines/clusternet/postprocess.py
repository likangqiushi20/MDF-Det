"""ROOBI extraction and FoveaNet heatmap-to-point postprocessing."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from clusternet.geometry import FOVEANET_GEOMETRY, roobi_input_bounds


@dataclass(frozen=True)
class ROOBI:
    output_bounds: tuple[int, int, int, int]
    input_bounds: tuple[int, int, int, int]
    peak_score: float


def extract_roobis(scoremap: np.ndarray, threshold: float) -> list[ROOBI]:
    mask = (scoremap >= threshold).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    output = []
    for label in range(1, count):
        x, y, width, height, _area = stats[label].tolist()
        bounds = (x, y, x + width - 1, y + height - 1)
        output.append(
            ROOBI(
                output_bounds=bounds,
                input_bounds=roobi_input_bounds(*bounds),
                peak_score=float(scoremap[labels == label].max()),
            )
        )
    return output


def heatmap_centroids(
    heatmap: np.ndarray,
    threshold: float,
    *,
    minimum_area: int = 100,
    maximum_area: int = 900,
    upsample: int = 2,
) -> list[tuple[float, float]]:
    """Convert a stitched FoveaNet output into input-coordinate centroids."""
    resized = cv2.resize(
        heatmap.astype(np.float32),
        None,
        fx=upsample,
        fy=upsample,
        interpolation=cv2.INTER_LINEAR,
    )
    mask = (resized >= threshold).astype(np.uint8)
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    detections = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < minimum_area:
            continue
        # The paper splits >900-pixel merged components with circular centers.
        # L1 retains the centroid and flags the split as a later L2 component.
        x = FOVEANET_GEOMETRY.first_center + float(centroids[label, 0])
        y = FOVEANET_GEOMETRY.first_center + float(centroids[label, 1])
        detections.append((x, y))
    return detections


def _split_large_component(component: np.ndarray) -> list[tuple[float, float]]:
    """Split a merged response using separated distance-transform centers."""
    distance = cv2.distanceTransform(component.astype(np.uint8), cv2.DIST_L2, 5)
    if not np.any(distance):
        return []
    dilated = cv2.dilate(distance, np.ones((11, 11), dtype=np.uint8))
    peaks = (distance == dilated) & (distance >= 0.3 * distance.max())
    count, _, _, centroids = cv2.connectedComponentsWithStats(
        peaks.astype(np.uint8), 8
    )
    return [
        (float(centroids[index, 0]), float(centroids[index, 1]))
        for index in range(1, count)
    ]


def global_heatmap_centroids(
    heatmap: np.ndarray,
    threshold: float,
    *,
    minimum_area: int = 100,
    maximum_area: int = 900,
) -> list[tuple[float, float]]:
    """Apply the paper's component rules to an input-resolution heatmap."""
    mask = (heatmap >= threshold).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    detections = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < minimum_area:
            continue
        if area <= maximum_area:
            detections.append(
                (float(centroids[label, 0]), float(centroids[label, 1]))
            )
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        component = (labels[y:y + height, x:x + width] == label).astype(np.uint8)
        centers = _split_large_component(component)
        detections.extend((x + cx, y + cy) for cx, cy in centers)
    return detections
