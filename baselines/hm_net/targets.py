"""Two-class center, motion and subpixel precision targets for HM-Net."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor

from hmrn.targets import gaussian_heatmap


@dataclass(frozen=True)
class ClassifiedPoint:
    track_id: int
    x: float
    y: float
    class_id: int


def build_targets(
    current: list[ClassifiedPoint],
    previous: list[ClassifiedPoint],
    height: int,
    width: int,
    *,
    classes: int = 2,
    sigma: float = 3.0,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    center = torch.zeros(classes, height, width)
    for class_id in range(classes):
        points = [
            (math.floor(point.x), math.floor(point.y))
            for point in current
            if point.class_id == class_id
        ]
        center[class_id] = gaussian_heatmap(height, width, points, sigma)
    motion = torch.zeros(2, height, width)
    precision = torch.zeros(2, height, width)
    motion_mask = torch.zeros(1, height, width, dtype=torch.bool)
    precision_mask = torch.zeros(1, height, width, dtype=torch.bool)
    previous_by_id = {point.track_id: point for point in previous}
    for point in current:
        x, y = math.floor(point.x), math.floor(point.y)
        if not (0 <= x < width and 0 <= y < height):
            continue
        precision[:, y, x] = torch.tensor([point.x - x, point.y - y])
        prior = previous_by_id.get(point.track_id)
        if prior is not None:
            motion[:, y, x] = torch.tensor(
                [prior.x - point.x, prior.y - point.y]
            )
            motion_mask[0, y, x] = True
        precision_mask[0, y, x] = True
    return center, motion, precision, motion_mask, precision_mask
