"""Patch construction and negative-pool sampling for CATLoss reproduction."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy.spatial import cKDTree

from catloss.heatmaps import binary_center_map


@dataclass(frozen=True)
class PatchPools:
    positive: tuple[tuple[int, int], ...]
    hard_negative: tuple[tuple[int, int], ...]
    ordinary_negative: tuple[tuple[int, int], ...]


def valid_center(x: float, y: float, width: int, height: int, size: int) -> bool:
    half = size // 2
    return half <= x < width - half and half <= y < height - half


def crop_stack(
    images: Sequence[np.ndarray],
    center_x: int,
    center_y: int,
    size: int,
) -> np.ndarray:
    if size % 2 != 1:
        raise ValueError("Patch size must be odd")
    if not images:
        raise ValueError("At least one temporal image is required")
    shape = images[0].shape
    if any(image.shape != shape for image in images):
        raise ValueError("All temporal images must have the same shape")
    if not valid_center(center_x, center_y, shape[1], shape[0], size):
        raise ValueError("Patch center falls outside the valid image interior")
    half = size // 2
    patch = np.stack(
        [
            image[
                center_y - half : center_y + half + 1,
                center_x - half : center_x + half + 1,
            ]
            for image in images
        ]
    )
    if np.issubdtype(patch.dtype, np.integer):
        return patch.astype(np.float32) / float(np.iinfo(patch.dtype).max)
    patch = patch.astype(np.float32)
    maximum = float(np.nanmax(patch)) if patch.size else 0.0
    return patch / maximum if maximum > 1.0 else patch


def _far_from_truth(
    x: int,
    y: int,
    truth_xy: Sequence[tuple[float, float]],
    exclusion_radius: float,
) -> bool:
    return all(
        math.hypot(x - truth_x, y - truth_y) > exclusion_radius
        for truth_x, truth_y in truth_xy
    )


def temporal_change_score(images: Sequence[np.ndarray]) -> np.ndarray:
    """Return current-vs-past temporal-median absolute difference."""
    if len(images) < 2:
        raise ValueError("BGS hard-negative scoring requires at least two frames")
    stack = np.stack(images).astype(np.float32)
    background = np.median(stack[:-1], axis=0)
    return np.abs(stack[-1] - background)


def build_patch_pools(
    images: Sequence[np.ndarray],
    truth_xy: Sequence[tuple[float, float]],
    *,
    patch_size: int = 21,
    ordinary_stride: int = 21,
    hard_stride: int = 5,
    hard_quantile: float = 0.995,
    exclusion_radius: float = 15.0,
) -> PatchPools:
    """Build positive, BGS-like hard-negative and ordinary-negative pools."""
    if not 0.0 < hard_quantile < 1.0:
        raise ValueError("hard_quantile must be between zero and one")
    height, width = images[-1].shape
    half = patch_size // 2
    rounded_truth = ((round(x), round(y)) for x, y in truth_xy)
    positives = tuple(
        dict.fromkeys(
            (x, y)
            for x, y in rounded_truth
            if valid_center(x, y, width, height, patch_size)
        )
    )

    score = temporal_change_score(images)
    threshold = float(np.quantile(score, hard_quantile))
    tree = cKDTree(np.asarray(truth_xy, dtype=np.float64)) if truth_xy else None

    def candidate_grid(stride: int) -> np.ndarray:
        xs = np.arange(half, width - half, stride, dtype=np.int32)
        ys = np.arange(half, height - half, stride, dtype=np.int32)
        xx, yy = np.meshgrid(xs, ys)
        return np.column_stack((xx.ravel(), yy.ravel()))

    def far_mask(candidates: np.ndarray) -> np.ndarray:
        if tree is None:
            return np.ones(len(candidates), dtype=bool)
        distances, _ = tree.query(candidates, k=1, workers=1)
        return distances > exclusion_radius

    hard_candidates = candidate_grid(hard_stride)
    hard_keep = far_mask(hard_candidates)
    hard_keep &= (
        score[hard_candidates[:, 1], hard_candidates[:, 0]]
        > max(threshold, 0.0)
    )
    ordinary_candidates = candidate_grid(ordinary_stride)
    ordinary_keep = far_mask(ordinary_candidates)

    hard = tuple(map(tuple, hard_candidates[hard_keep].tolist()))
    ordinary = tuple(map(tuple, ordinary_candidates[ordinary_keep].tolist()))
    return PatchPools(positives, hard, ordinary)


def sample_objectness_epoch(
    images: Sequence[np.ndarray],
    pools: PatchPools,
    rng: np.random.Generator,
    *,
    negatives_per_positive: int = 1,
    hard_fraction: float = 0.5,
    patch_size: int = 21,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample objectness negatives for one epoch, as required by the paper."""
    positives = list(pools.positive)
    if not positives:
        raise ValueError("Objectness sampling requires positive centers")
    positive_count = len(positives)
    negative_count = positive_count * negatives_per_positive
    hard_count = min(round(negative_count * hard_fraction), len(pools.hard_negative))
    ordinary_count = negative_count - hard_count
    if ordinary_count > len(pools.ordinary_negative):
        raise ValueError("Insufficient ordinary negatives")
    hard_indices = rng.choice(len(pools.hard_negative), hard_count, replace=False)
    ordinary_indices = rng.choice(
        len(pools.ordinary_negative), ordinary_count, replace=False
    )
    centers = list(positives)
    centers += [pools.hard_negative[index] for index in hard_indices]
    centers += [pools.ordinary_negative[index] for index in ordinary_indices]
    inputs = np.stack(
        [crop_stack(images, x, y, patch_size) for x, y in centers]
    )
    targets = np.asarray(
        [1] * positive_count + [0] * negative_count,
        dtype=np.int64,
    )
    permutation = rng.permutation(len(targets))
    return inputs[permutation], targets[permutation]


def build_localization_batch(
    images: Sequence[np.ndarray],
    truth_xy: Sequence[tuple[float, float]],
    centers: Iterable[tuple[int, int]],
    *,
    patch_size: int = 45,
    output_size: int = 15,
) -> tuple[np.ndarray, np.ndarray]:
    if patch_size % output_size != 0:
        raise ValueError("patch_size must be divisible by output_size")
    scale = patch_size / output_size
    inputs = []
    targets = []
    half = patch_size // 2
    for center_x, center_y in centers:
        origin_x = center_x - half
        origin_y = center_y - half
        local_points = [
            ((x - origin_x) / scale, (y - origin_y) / scale)
            for x, y in truth_xy
            if origin_x <= x <= origin_x + patch_size - 1
            and origin_y <= y <= origin_y + patch_size - 1
        ]
        if not local_points:
            continue
        inputs.append(crop_stack(images, center_x, center_y, patch_size))
        targets.append(binary_center_map(output_size, output_size, local_points).numpy())
    if not inputs:
        raise ValueError("Localization sampling requires at least one positive patch")
    return np.stack(inputs), np.stack(targets)[:, None]


def _zero_fill_shift(values: np.ndarray, shift_y: int, shift_x: int) -> np.ndarray:
    output = np.zeros_like(values)
    height, width = values.shape[-2:]
    source_y = slice(max(0, -shift_y), min(height, height - shift_y))
    source_x = slice(max(0, -shift_x), min(width, width - shift_x))
    target_y = slice(max(0, shift_y), min(height, height + shift_y))
    target_x = slice(max(0, shift_x), min(width, width + shift_x))
    output[..., target_y, target_x] = values[..., source_y, source_x]
    return output


def _edge_shift(values: np.ndarray, shift_y: int, shift_x: int) -> np.ndarray:
    pad_y, pad_x = abs(shift_y), abs(shift_x)
    padded = np.pad(
        values,
        [(0, 0)] * (values.ndim - 2) + [(pad_y, pad_y), (pad_x, pad_x)],
        mode="edge",
    )
    start_y = pad_y - shift_y
    start_x = pad_x - shift_x
    return padded[
        ..., start_y : start_y + values.shape[-2],
        start_x : start_x + values.shape[-1]
    ]


def augment_patch_batches(
    object_inputs: np.ndarray,
    local_inputs: np.ndarray,
    local_targets: np.ndarray,
    rng: np.random.Generator,
    *,
    intensity_range: tuple[float, float] = (0.9, 1.1),
    object_shift_px: int = 2,
    local_output_shift: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply per-patch intensity scaling and zero-filled translations."""
    object_inputs = object_inputs.copy()
    local_inputs = local_inputs.copy()
    local_targets = local_targets.copy()
    for index in range(len(object_inputs)):
        scale = rng.uniform(*intensity_range)
        object_inputs[index] = np.clip(object_inputs[index] * scale, 0, 1)
        sx, sy = rng.integers(-object_shift_px, object_shift_px + 1, size=2)
        object_inputs[index] = _edge_shift(object_inputs[index], sy, sx)
    for index in range(len(local_inputs)):
        scale = rng.uniform(*intensity_range)
        local_inputs[index] = np.clip(local_inputs[index] * scale, 0, 1)
        sx, sy = rng.integers(
            -local_output_shift, local_output_shift + 1, size=2
        )
        local_inputs[index] = _edge_shift(local_inputs[index], 3 * sy, 3 * sx)
        local_targets[index] = _zero_fill_shift(local_targets[index], sy, sx)
    return object_inputs, local_inputs, local_targets
