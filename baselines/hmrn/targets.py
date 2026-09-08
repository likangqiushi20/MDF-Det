"""Center and displacement targets for HMRN."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class TrackedPoint:
    track_id: int
    x: float
    y: float


def gaussian_heatmap(
    height: int,
    width: int,
    points_xy: list[tuple[float, float]],
    sigma: float,
) -> Tensor:
    rows = torch.arange(height, dtype=torch.float32)[:, None]
    columns = torch.arange(width, dtype=torch.float32)[None, :]
    output = torch.zeros(height, width)
    for x, y in points_xy:
        output = torch.maximum(
            output,
            torch.exp(
                -((columns - x).square() + (rows - y).square())
                / (2 * sigma * sigma)
            ),
        )
    return output


def build_targets(
    current: list[TrackedPoint],
    previous: list[TrackedPoint],
    input_height: int,
    input_width: int,
    *,
    stride: int = 4,
    sigma: float = 1.5,
) -> tuple[Tensor, Tensor, Tensor]:
    height, width = input_height // stride, input_width // stride
    # CenterNet defines positives on the discrete output lattice.  Keeping a
    # fractional Gaussian center would make every target value smaller than
    # one and therefore leave the focal loss with no positive pixels.
    centers = [
        (round(point.x / stride), round(point.y / stride))
        for point in current
    ]
    heatmap = gaussian_heatmap(height, width, centers, sigma)[None]
    displacement = torch.zeros(2, height, width)
    mask = torch.zeros(1, height, width, dtype=torch.bool)
    previous_by_id = {point.track_id: point for point in previous}
    for point in current:
        prior = previous_by_id.get(point.track_id)
        if prior is None:
            continue
        x = round(point.x / stride)
        y = round(point.y / stride)
        if 0 <= x < width and 0 <= y < height:
            displacement[0, y, x] = (prior.x - point.x) / stride
            displacement[1, y, x] = (prior.y - point.y) / stride
            mask[0, y, x] = True
    return heatmap, displacement, mask
