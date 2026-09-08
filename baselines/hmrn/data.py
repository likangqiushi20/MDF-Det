"""Paper-sized crop and three-channel temporal input construction for HMRN."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from hmrn.targets import TrackedPoint, build_targets, gaussian_heatmap


@dataclass(frozen=True)
class TrainingSample:
    inputs: Tensor
    center: Tensor
    displacement: Tensor
    displacement_mask: Tensor
    origin_xy: tuple[int, int]


def _normalize(image: np.ndarray) -> np.ndarray:
    values = image.astype(np.float32)
    if np.issubdtype(image.dtype, np.integer):
        return values / float(np.iinfo(image.dtype).max)
    maximum = float(np.nanmax(values)) if values.size else 0
    return values / maximum if maximum > 1 else values


def _local_points(
    points: list[TrackedPoint], left: int, top: int, width: int, height: int
) -> list[TrackedPoint]:
    return [
        TrackedPoint(point.track_id, point.x - left, point.y - top)
        for point in points
        if left <= point.x < left + width and top <= point.y < top + height
    ]


def select_crop_origin(
    image_shape: tuple[int, int],
    current: list[TrackedPoint],
    rng: np.random.Generator,
    *,
    width: int = 960,
    height: int = 544,
) -> tuple[int, int]:
    image_height, image_width = image_shape
    if width > image_width or height > image_height:
        raise ValueError("Crop is larger than the AOI image")
    if current:
        anchor = current[int(rng.integers(len(current)))]
        left = round(anchor.x - width / 2)
        top = round(anchor.y - height / 2)
    else:
        left = int(rng.integers(image_width - width + 1))
        top = int(rng.integers(image_height - height + 1))
    return (
        min(max(left, 0), image_width - width),
        min(max(top, 0), image_height - height),
    )


def build_training_sample(
    previous_image: np.ndarray,
    current_image: np.ndarray,
    previous_points: list[TrackedPoint],
    current_points: list[TrackedPoint],
    rng: np.random.Generator,
    *,
    crop_width: int = 960,
    crop_height: int = 544,
    stride: int = 4,
    previous_sigma: float = 6.0,
    target_sigma: float = 1.5,
) -> TrainingSample:
    if previous_image.shape != current_image.shape:
        raise ValueError("Consecutive images must share a shape")
    left, top = select_crop_origin(
        current_image.shape,
        current_points,
        rng,
        width=crop_width,
        height=crop_height,
    )
    rows = slice(top, top + crop_height)
    columns = slice(left, left + crop_width)
    local_previous = _local_points(
        previous_points, left, top, crop_width, crop_height
    )
    # Keep out-of-crop prior coordinates for motion supervision when a track
    # crosses the crop boundary; only the rendered feedback map is clipped.
    shifted_previous = [
        TrackedPoint(point.track_id, point.x - left, point.y - top)
        for point in previous_points
    ]
    local_current = _local_points(
        current_points, left, top, crop_width, crop_height
    )
    previous_heatmap = gaussian_heatmap(
        crop_height,
        crop_width,
        [(point.x, point.y) for point in local_previous],
        previous_sigma,
    ).numpy()
    inputs = np.stack(
        [
            _normalize(current_image[rows, columns]),
            _normalize(previous_image[rows, columns]),
            previous_heatmap,
        ]
    )
    center, displacement, mask = build_targets(
        local_current,
        shifted_previous,
        crop_height,
        crop_width,
        stride=stride,
        sigma=target_sigma,
    )
    return TrainingSample(
        torch.from_numpy(inputs),
        center,
        displacement,
        mask,
        (left, top),
    )
