"""HM-Net decoding and gated Hungarian track association."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor

from hm_net.sgr import local_maxima


@dataclass(frozen=True)
class Detection:
    x: float
    y: float
    class_id: int
    score: float
    dx: float
    dy: float


@dataclass(frozen=True)
class Track:
    track_id: int
    x: float
    y: float
    class_id: int
    score: float
    active: bool = True


def decode(
    center: Tensor,
    motion: Tensor,
    precision: Tensor,
    *,
    threshold: float = 0.32,
    nms_window: int = 15,
    max_detections_per_class: int | None = 5000,
) -> list[list[Detection]]:
    peaks = local_maxima(center, nms_window)
    output = []
    for batch in range(len(center)):
        detections = []
        indices = torch.nonzero(
            peaks[batch] >= threshold, as_tuple=False
        )
        if max_detections_per_class is not None:
            retained = []
            for class_id in range(center.shape[1]):
                subset = indices[indices[:, 0] == class_id]
                if len(subset) > max_detections_per_class:
                    scores = center[
                        batch, subset[:, 0], subset[:, 1], subset[:, 2]
                    ]
                    subset = subset[
                        torch.topk(scores, max_detections_per_class).indices
                    ]
                retained.append(subset)
            indices = torch.cat(retained, dim=0) if retained else indices
        for class_id, row, column in indices:
            detections.append(
                Detection(
                    float(column + precision[batch, 0, row, column]),
                    float(row + precision[batch, 1, row, column]),
                    int(class_id),
                    float(center[batch, class_id, row, column]),
                    float(motion[batch, 0, row, column]),
                    float(motion[batch, 1, row, column]),
                )
            )
        output.append(
            sorted(detections, key=lambda item: item.score, reverse=True)
        )
    return output


def associate_hungarian(
    detections: list[Detection],
    previous: list[Track],
    next_track_id: int,
    *,
    gate: float = 20,
) -> tuple[list[Track], int]:
    if not detections:
        return [], next_track_id
    assigned: dict[int, int] = {}
    if previous:
        costs = np.full((len(detections), len(previous)), gate + 1.0)
        for detection_index, detection in enumerate(detections):
            prior_x = detection.x + detection.dx
            prior_y = detection.y + detection.dy
            for previous_index, track in enumerate(previous):
                distance = np.hypot(prior_x - track.x, prior_y - track.y)
                if distance <= gate:
                    # Distance is primary; confidence breaks near ties.
                    costs[detection_index, previous_index] = (
                        distance - 1e-3 * detection.score
                    )
        rows, columns = linear_sum_assignment(costs)
        assigned = {
            int(row): int(column)
            for row, column in zip(rows, columns)
            if costs[row, column] <= gate
        }
    tracks = []
    for index, detection in enumerate(detections):
        if index in assigned:
            track_id = previous[assigned[index]].track_id
        else:
            track_id = next_track_id
            next_track_id += 1
        tracks.append(
            Track(
                track_id,
                detection.x,
                detection.y,
                detection.class_id,
                detection.score,
            )
        )
    return tracks, next_track_id
