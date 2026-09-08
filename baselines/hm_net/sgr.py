"""Selective Gaussian Reconstruction plus paper training augmentations."""

from __future__ import annotations

import math
import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as functional


def local_maxima(heatmap: Tensor, window: int = 15) -> Tensor:
    if window % 2 != 1:
        raise ValueError("NMS window must be odd")
    pooled = functional.max_pool2d(
        heatmap, window, stride=1, padding=window // 2
    )
    return heatmap * heatmap.eq(pooled)


def selective_gaussian_reconstruction(
    heatmap: Tensor,
    *,
    filter_threshold: float = 0.28,
    detection_threshold: float = 0.32,
    amplification: float = 1.2,
    sigma: float = 3.0,
    nms_window: int = 15,
    max_peaks_per_class: int | None = 5000,
) -> Tensor:
    """Apply Eq. 3 and reconstruct confidence-weighted Gaussian peaks."""
    peaks = local_maxima(heatmap, nms_window)
    output = torch.zeros_like(heatmap)
    height, width = heatmap.shape[-2:]
    gaussian_radius = max(1, math.ceil(4 * sigma))
    offsets = torch.arange(
        -gaussian_radius,
        gaussian_radius + 1,
        device=heatmap.device,
        dtype=heatmap.dtype,
    )
    yy, xx = torch.meshgrid(offsets, offsets, indexing="ij")
    kernel = torch.exp(-(xx.square() + yy.square()) / (2 * sigma * sigma))
    # A test-time detection threshold may be calibrated below the paper
    # default. Never leave the feedback branch empty merely because the
    # separately configured SGR filter was not lowered with it.
    effective_filter = min(filter_threshold, detection_threshold)
    peak_indices = torch.nonzero(peaks >= effective_filter, as_tuple=False)
    if max_peaks_per_class is not None:
        retained = []
        for batch in range(heatmap.shape[0]):
            for class_id in range(heatmap.shape[1]):
                subset = peak_indices[
                    (peak_indices[:, 0] == batch)
                    & (peak_indices[:, 1] == class_id)
                ]
                if len(subset) > max_peaks_per_class:
                    scores = peaks[
                        subset[:, 0], subset[:, 1], subset[:, 2], subset[:, 3]
                    ]
                    subset = subset[
                        torch.topk(scores, max_peaks_per_class).indices
                    ]
                retained.append(subset)
        peak_indices = (
            torch.cat(retained, dim=0)
            if retained
            else peak_indices
        )
    for batch, class_id, row, column in peak_indices:
        confidence = peaks[batch, class_id, row, column]
        if confidence >= detection_threshold:
            confidence = confidence * amplification
        top = max(0, int(row) - gaussian_radius)
        bottom = min(height, int(row) + gaussian_radius + 1)
        left = max(0, int(column) - gaussian_radius)
        right = min(width, int(column) + gaussian_radius + 1)
        kernel_top = top - (int(row) - gaussian_radius)
        kernel_left = left - (int(column) - gaussian_radius)
        gaussian = confidence.clamp(max=1) * kernel[
            kernel_top : kernel_top + bottom - top,
            kernel_left : kernel_left + right - left,
        ]
        output[batch, class_id, top:bottom, left:right] = torch.maximum(
            output[batch, class_id, top:bottom, left:right],
            gaussian,
        )
    return output


def augment_feedback_centers(
    centers_xyc: list[tuple[float, float, int]],
    rng: np.random.Generator,
    *,
    rcr_probability: float = 0.7,
    rcr_max_reduction: float = 0.8,
    false_centers_per_object: int = 1,
    jitter_radius: float = 12,
) -> list[tuple[float, float, int, float]]:
    """Generate RCR confidences and RCP false centers for training feedback."""
    output = []
    for x, y, class_id in centers_xyc:
        confidence = 1.0
        if rng.random() < rcr_probability:
            confidence *= rng.uniform(1 - rcr_max_reduction, 1)
        output.append((x, y, class_id, confidence))
        for _ in range(false_centers_per_object):
            angle = rng.uniform(0, 2 * np.pi)
            radius = rng.uniform(1, jitter_radius)
            output.append(
                (
                    x + radius * np.cos(angle),
                    y + radius * np.sin(angle),
                    class_id,
                    rng.uniform(0.05, 0.4),
                )
            )
    return output


def render_weighted_centers(
    height: int,
    width: int,
    classes: int,
    centers: list[tuple[float, float, int, float]],
    *,
    sigma: float = 3,
) -> Tensor:
    output = torch.zeros(classes, height, width)
    rows = torch.arange(height, dtype=torch.float32)[:, None]
    columns = torch.arange(width, dtype=torch.float32)[None, :]
    for x, y, class_id, confidence in centers:
        gaussian = confidence * torch.exp(
            -((columns - x).square() + (rows - y).square())
            / (2 * sigma * sigma)
        )
        output[class_id] = torch.maximum(output[class_id], gaussian)
    return output.clamp(max=1)
