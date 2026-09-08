"""Weighted HM-Net center, motion and precision losses."""

from __future__ import annotations

from torch import Tensor

from hmrn.losses import CenterFocalLoss, masked_displacement_l1


def hmnet_loss(
    predictions: dict[str, Tensor],
    center: Tensor,
    motion: Tensor,
    precision: Tensor,
    motion_mask: Tensor,
    precision_mask: Tensor,
    *,
    center_weight: float = 1,
    motion_weight: float = 1,
    precision_weight: float = 1,
) -> tuple[Tensor, dict[str, Tensor]]:
    parts = {
        "center": CenterFocalLoss()(predictions["center"], center),
        "motion": masked_displacement_l1(
            predictions["motion"], motion, motion_mask
        ),
        "precision": masked_displacement_l1(
            predictions["precision"], precision, precision_mask
        ),
    }
    total = (
        center_weight * parts["center"]
        + motion_weight * parts["motion"]
        + precision_weight * parts["precision"]
    )
    return total, parts
