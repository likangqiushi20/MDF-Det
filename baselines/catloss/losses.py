"""Loss functions used by the CATLoss reproduction."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as functional


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        targets = targets.long()
        ce = functional.cross_entropy(logits, targets, reduction="none")
        probabilities = torch.softmax(logits, dim=1)
        pt = probabilities.gather(1, targets[:, None]).squeeze(1)
        alpha_t = torch.where(
            targets == 1,
            torch.as_tensor(self.alpha, device=logits.device),
            torch.as_tensor(1.0 - self.alpha, device=logits.device),
        )
        return (alpha_t * (1.0 - pt).pow(self.gamma) * ce).mean()


def thresholded_loss_per_sample(
    predictions: Tensor,
    targets: Tensor,
    tau: float = 0.2,
) -> Tensor:
    if predictions.shape != targets.shape:
        raise ValueError(
            f"Predictions and targets must have equal shapes, got "
            f"{predictions.shape} and {targets.shape}"
        )
    if predictions.ndim < 2:
        raise ValueError("Expected a batch dimension plus heatmap dimensions")
    foreground = targets >= 0.5
    foreground_error = (predictions - targets).square()
    background_error = torch.relu(predictions - targets - tau).square()
    element_loss = torch.where(foreground, foreground_error, background_error)
    return element_loss.flatten(1).mean(dim=1)


class ThresholdedLoss(nn.Module):
    def __init__(self, tau: float = 0.2) -> None:
        super().__init__()
        self.tau = tau

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        return thresholded_loss_per_sample(predictions, targets, self.tau).mean()


class CrowdAwareThresholdedLoss(nn.Module):
    def __init__(self, tau: float = 0.2, q: float = 0.5) -> None:
        super().__init__()
        if q < 0:
            raise ValueError("q must be non-negative")
        self.tau = tau
        self.q = q

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        per_sample = thresholded_loss_per_sample(predictions, targets, self.tau)
        target_counts = (targets >= 0.5).flatten(1).sum(dim=1).to(per_sample.dtype)
        if torch.any(target_counts < 1):
            raise ValueError("CATLoss localization batches must not contain empty targets")
        return (target_counts.pow(self.q) * per_sample).mean()
