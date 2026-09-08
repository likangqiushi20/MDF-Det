from __future__ import annotations

from pathlib import Path
import sys

import cv2
import numpy as np


AOIS = ("01", "02", "03", "34", "40", "41")
AOI_TO_CODE = {name: index for index, name in enumerate(AOIS)}
CODE_TO_AOI = {index: name for name, index in AOI_TO_CODE.items()}


def add_external_repo(path: Path) -> None:
    resolved = str(path.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


def axis_origins(left: int, right: int, limit: int, size: int = 128, stride: int = 96) -> list[int]:
    left = max(0, min(limit, int(left)))
    right = max(left + 1, min(limit, int(right)))
    maximum = max(0, limit - size)
    if right - left <= size:
        return [int(np.clip(round((left + right - size) / 2), 0, maximum))]
    values = list(range(max(0, left), max(0, right - size) + 1, stride))
    values.append(int(np.clip(right - size, 0, maximum)))
    return sorted(set(values))


def origins_cover_bounds(bounds: tuple[int, int, int, int], width: int, height: int) -> list[tuple[int, int]]:
    left, top, right, bottom = bounds
    xs = axis_origins(left, right, width)
    ys = axis_origins(top, bottom, height)
    return [(x, y) for y in ys for x in xs]


def gaussian_target(
    points_xy: list[tuple[float, float]],
    origin_x: int,
    origin_y: int,
    *,
    output_size: int = 64,
    stride: float = 2.0,
    sigma: float = 1.75,
) -> np.ndarray:
    target = np.zeros((output_size, output_size), dtype=np.float32)
    if not points_xy:
        return target[None]
    yy, xx = np.mgrid[:output_size, :output_size].astype(np.float32)
    for x, y in points_xy:
        px = (x - origin_x) / stride
        py = (y - origin_y) / stride
        if 0 <= px < output_size and 0 <= py < output_size:
            value = np.exp(-((xx - px) ** 2 + (yy - py) ** 2) / (2.0 * sigma * sigma))
            target = np.maximum(target, value)
    return target[None]


def decode_local_maxima(probability: np.ndarray, threshold: float, radius: int = 4) -> list[tuple[float, float, float]]:
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
    maxima = cv2.dilate(probability.astype(np.float32), kernel)
    mask = ((probability >= threshold) & (probability >= maxima - 1e-7)).astype(np.uint8)
    count, labels, _, centroids = cv2.connectedComponentsWithStats(mask, 8)
    points = []
    for label in range(1, count):
        values = probability[labels == label]
        points.append((float(centroids[label, 0]), float(centroids[label, 1]), float(values.max())))
    return points
