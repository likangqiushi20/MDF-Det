from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _cluster_block(inputs: int, outputs: int, kernel: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(inputs, outputs, kernel, stride=stride, padding=kernel // 2),
        nn.PReLU(outputs),
        nn.BatchNorm2d(outputs),
    )


class ClusterNet(nn.Module):
    """Frozen fully convolutional ROOBI proposal network."""

    def __init__(self, temporal_channels: int = 5) -> None:
        super().__init__()
        self.network = nn.Sequential(
            _cluster_block(temporal_channels, 32, 3, stride=2),
            nn.MaxPool2d(2, 2),
            _cluster_block(32, 64, 3, stride=2),
            _cluster_block(64, 128, 3),
            nn.MaxPool2d(2, 2),
            _cluster_block(128, 128, 1),
            nn.Conv2d(128, 1, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.network(inputs))


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.body(x))


class FoveaNet(nn.Module):
    """Small high-resolution detector with a 64x64 logit map for 128x128 input."""

    def __init__(self, temporal_channels: int = 5, base_channels: int = 16) -> None:
        super().__init__()
        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 3
        self.base_channels = base_channels
        self.stem = nn.Sequential(
            nn.Conv2d(temporal_channels, c1, 3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )
        self.encoder1 = nn.Sequential(
            nn.Conv2d(c1, c2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
            ResidualBlock(c2),
            ResidualBlock(c2),
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(c2, c3, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.SiLU(inplace=True),
            ResidualBlock(c3),
            ResidualBlock(c3),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(c3 + c2, c2, 3, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
            ResidualBlock(c2),
            nn.Conv2d(c2, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        skip = self.encoder1(x)
        low = self.encoder2(skip)
        low = F.interpolate(low, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.decoder(torch.cat([low, skip], dim=1))


def point_map_to_gaussian(point_map: torch.Tensor, sigma: float = 1.75) -> torch.Tensor:
    radius = int(round(3.0 * sigma))
    coordinates = torch.arange(-radius, radius + 1, device=point_map.device, dtype=point_map.dtype)
    kernel = torch.exp(-(coordinates[:, None].square() + coordinates[None, :].square()) / (2.0 * sigma * sigma))
    kernel = kernel[None, None]
    return F.conv2d(point_map, kernel, padding=radius).clamp_max_(1.0)


def focal_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    alpha: float = 0.25,
    gamma: float = 2.0,
    dice_weight: float = 0.5,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    probability = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    pt = probability * target + (1.0 - probability) * (1.0 - target)
    alpha_t = alpha * target + (1.0 - alpha) * (1.0 - target)
    focal = (alpha_t * (1.0 - pt).pow(gamma) * bce).mean()
    axes = tuple(range(1, target.ndim))
    intersection = (probability * target).sum(dim=axes)
    denominator = probability.sum(dim=axes) + target.sum(dim=axes)
    dice = (1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
    total = focal + dice_weight * dice
    return total, {"focal": focal.detach(), "dice": dice.detach()}
