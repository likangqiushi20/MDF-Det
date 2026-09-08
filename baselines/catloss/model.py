"""PyTorch Objectness and Localization networks reconstructed from Fig. 6."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def _conv_block(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    dilation: int = 1,
) -> nn.Sequential:
    padding = dilation * (kernel_size // 2)
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class ObjectnessNetwork(nn.Module):
    def __init__(self, input_channels: int = 4, patch_size: int = 21) -> None:
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(input_channels, 32, 3),
            _conv_block(32, 32, 3),
            _conv_block(32, 64, 3),
            nn.MaxPool2d(2, 2),
            _conv_block(64, 128, 3),
        )
        with torch.no_grad():
            feature_count = self.features(
                torch.zeros(1, input_channels, patch_size, patch_size)
            ).numel()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_count, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.classifier(self.features(inputs))


class LocalizationNetwork(nn.Module):
    def __init__(
        self,
        input_channels: int = 4,
        patch_size: int = 45,
        output_size: int = 15,
        dilation_mode: str = "half",
    ) -> None:
        super().__init__()
        if dilation_mode not in {"half", "full"}:
            raise ValueError("dilation_mode must be 'half' or 'full'")
        early_dilation = 2 if dilation_mode == "full" else 1
        self.output_size = output_size
        self.features = nn.Sequential(
            _conv_block(input_channels, 32, 5, early_dilation),
            _conv_block(32, 32, 3, early_dilation),
            nn.MaxPool2d(2, 2),
            _conv_block(32, 64, 3, 2),
            _conv_block(64, 128, 3, 2),
        )
        with torch.no_grad():
            feature_count = self.features(
                torch.zeros(1, input_channels, patch_size, patch_size)
            ).numel()
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_count, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, output_size * output_size),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        values = self.regressor(self.features(inputs))
        return torch.sigmoid(
            values.reshape(-1, 1, self.output_size, self.output_size)
        )
