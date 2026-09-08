"""HMRN heatmap decoding and confidence-ordered greedy association."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor
from torch.nn import functional as functional


@dataclass(frozen=True)
class Detection:
    x: float
    y: float
    score: float
    dx: float
    dy: float


@dataclass(frozen=True)
class TrackPoint:
    track_id: int
    x: float
    y: float
    score: float


def decode(
    center: Tensor,
    motion: Tensor,
    threshold: float,
    *,
    stride: int = 4,
) -> list[list[Detection]]:
    pooled = functional.max_pool2d(center, 3, stride=1, padding=1)
    peaks = (center == pooled) & (center >= threshold)
    batches = []
    for batch in range(len(center)):
        rows, columns = torch.where(peaks[batch, 0])
        detections = [
            Detection(
                x=float(column * stride),
                y=float(row * stride),
                score=float(center[batch, 0, row, column]),
                dx=float(motion[batch, 0, row, column] * stride),
                dy=float(motion[batch, 1, row, column] * stride),
            )
            for row, column in zip(rows, columns)
        ]
        batches.append(sorted(detections, key=lambda item: item.score, reverse=True))
    return batches


def associate(
    detections: list[Detection],
    previous: list[TrackPoint],
    next_track_id: int,
    *,
    radius: float = 20.0,
) -> tuple[list[TrackPoint], int]:
    unmatched = set(range(len(previous)))
    output = []
    for detection in sorted(detections, key=lambda item: item.score, reverse=True):
        predicted_x = detection.x + detection.dx
        predicted_y = detection.y + detection.dy
        candidates = [
            (
                math.hypot(
                    predicted_x - previous[index].x,
                    predicted_y - previous[index].y,
                ),
                index,
            )
            for index in unmatched
        ]
        candidates = [item for item in candidates if item[0] <= radius]
        if candidates:
            _, index = min(candidates)
            unmatched.remove(index)
            track_id = previous[index].track_id
        else:
            track_id = next_track_id
            next_track_id += 1
        output.append(
            TrackPoint(track_id, detection.x, detection.y, detection.score)
        )
    return output, next_track_id
