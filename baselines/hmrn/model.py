"""Compact four-stage IDA reproduction of HMRN."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as functional


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return functional.relu(inputs + self.layers(inputs), inplace=True)


def _stage(inputs: int, outputs: int, stride: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(inputs, outputs, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(outputs),
        nn.ReLU(inplace=True),
        ResidualBlock(outputs),
    )


class HMRN(nn.Module):
    def __init__(self, input_channels: int = 3, head_channels: int = 64) -> None:
        super().__init__()
        self.stage1 = _stage(input_channels, 32, 2)
        self.stage2 = _stage(32, 64, 2)
        self.stage3 = _stage(64, 128, 2)
        self.stage4 = _stage(128, 256, 2)
        self.project2 = nn.Conv2d(64, head_channels, 1)
        self.project3 = nn.Conv2d(128, head_channels, 1)
        self.project4 = nn.Conv2d(256, head_channels, 1)
        self.aggregate = nn.Sequential(
            ResidualBlock(head_channels),
            nn.Conv2d(head_channels, head_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.center_head = nn.Sequential(
            nn.Conv2d(head_channels, head_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_channels, 1, 1),
        )
        self.motion_head = nn.Sequential(
            nn.Conv2d(head_channels, head_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_channels, 2, 1),
        )
        # Standard CenterNet prior: start with a low foreground probability.
        nn.init.constant_(self.center_head[-1].bias, -2.19)

    def forward(self, inputs: Tensor) -> dict[str, Tensor]:
        stage1 = self.stage1(inputs)
        stage2 = self.stage2(stage1)
        stage3 = self.stage3(stage2)
        stage4 = self.stage4(stage3)
        size = stage2.shape[-2:]
        aggregated = self.project2(stage2)
        aggregated = aggregated + functional.interpolate(
            self.project3(stage3), size=size, mode="bilinear", align_corners=False
        )
        aggregated = aggregated + functional.interpolate(
            self.project4(stage4), size=size, mode="bilinear", align_corners=False
        )
        features = self.aggregate(aggregated)
        return {
            "center": torch.sigmoid(self.center_head(features)),
            "motion": self.motion_head(features),
        }
