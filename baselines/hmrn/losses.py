"""HMRN CenterNet focal and masked displacement losses."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class CenterFocalLoss(nn.Module):
    def __init__(self, alpha: float = 2.0, beta: float = 4.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        predictions = predictions.clamp(1e-6, 1 - 1e-6)
        positive = targets.eq(1)
        negative = targets.lt(1)
        positive_loss = (
            (1 - predictions).pow(self.alpha)
            * predictions.log()
            * positive
        )
        negative_loss = (
            (1 - targets).pow(self.beta)
            * predictions.pow(self.alpha)
            * (1 - predictions).log()
            * negative
        )
        normalizer = positive.sum().clamp_min(1)
        return -(positive_loss.sum() + negative_loss.sum()) / normalizer


def masked_displacement_l1(
    predictions: Tensor,
    targets: Tensor,
    mask: Tensor,
) -> Tensor:
    expanded = mask.expand_as(predictions)
    if not expanded.any():
        return predictions.sum() * 0
    return torch.abs(predictions[expanded] - targets[expanded]).mean()
