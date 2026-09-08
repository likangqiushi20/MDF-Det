"""Binary center targets and local-maximum extraction."""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as functional


def binary_center_map(
    height: int,
    width: int,
    points_xy: list[tuple[float, float]],
    *,
    device: torch.device | str | None = None,
) -> Tensor:
    target = torch.zeros((height, width), dtype=torch.float32, device=device)
    for x, y in points_xy:
        column = math.floor(x + 0.5)
        row = math.floor(y + 0.5)
        if 0 <= column < width and 0 <= row < height:
            target[row, column] = 1.0
    return target


def extract_peaks(
    heatmap: Tensor,
    threshold: float,
    minimum_distance: int = 1,
) -> list[list[tuple[int, int, float]]]:
    if heatmap.ndim == 3:
        heatmap = heatmap[:, None]
    if heatmap.ndim != 4 or heatmap.shape[1] != 1:
        raise ValueError("Expected heatmap shape (N,H,W) or (N,1,H,W)")
    kernel = 2 * minimum_distance + 1
    pooled = functional.max_pool2d(
        heatmap,
        kernel_size=kernel,
        stride=1,
        padding=minimum_distance,
    )
    maxima = (heatmap >= threshold) & (heatmap == pooled)
    output = []
    for batch_index in range(len(heatmap)):
        rows, columns = torch.where(maxima[batch_index, 0])
        values = heatmap[batch_index, 0, rows, columns]
        order = torch.argsort(values, descending=True)
        output.append(
            [
                (
                    int(columns[index]),
                    int(rows[index]),
                    float(values[index]),
                )
                for index in order
            ]
        )
    return output
