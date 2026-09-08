"""Two-stage, three-encoder HM-Net reconstructed from Figures 6 and 7."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class EncoderBlock(nn.Module):
    def __init__(self, inputs: int, outputs: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(inputs, outputs, 3, padding=1, bias=False),
            nn.BatchNorm2d(outputs),
            nn.ReLU(inplace=True),
            nn.Conv2d(outputs, outputs, 3, padding=1, bias=False),
            nn.BatchNorm2d(outputs),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.pool(self.features(inputs))

    def forward_with_skip(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        """Return the pooled tensor and the pre-pooling skip activation."""
        skip = self.features(inputs)
        return self.pool(skip), skip


class DecoderBlock(nn.Module):
    def __init__(self, inputs: int, outputs: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(inputs, outputs, 2, stride=2, bias=False),
            nn.Conv2d(outputs, outputs, 3, padding=1, bias=False),
            nn.BatchNorm2d(outputs),
            nn.ReLU(inplace=True),
            nn.Conv2d(outputs, outputs, 3, padding=1, bias=False),
            nn.BatchNorm2d(outputs),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.layers(inputs)


class Bottleneck(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.layers(inputs)


class Encoder(nn.Module):
    def __init__(self, input_channels: int) -> None:
        super().__init__()
        channels = [32, 64, 128, 256]
        blocks = []
        previous = input_channels
        for channel in channels:
            blocks.append(EncoderBlock(previous, channel))
            previous = channel
        self.blocks = nn.ModuleList(blocks)
        self.bottleneck = Bottleneck(256)

    def forward(self, inputs: Tensor) -> tuple[Tensor, list[Tensor]]:
        skips = []
        values = inputs
        for block in self.blocks:
            values, skip = block.forward_with_skip(values)
            skips.append(skip)
        return self.bottleneck(values), skips


class Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                DecoderBlock(256, 256),
                DecoderBlock(256, 128),
                DecoderBlock(128, 64),
                DecoderBlock(64, 32),
            ]
        )

    def forward(
        self, inputs: Tensor, summed_skips: list[Tensor]
    ) -> tuple[Tensor, list[Tensor]]:
        values = inputs
        outputs = []
        # Pre-pooling encoder activations are H, H/2, H/4, H/8. Reversing
        # them therefore aligns exactly with decoder outputs H/8..H.
        for block, skip in zip(self.blocks, reversed(summed_skips)):
            values = block(values)
            if values.shape != skip.shape:
                raise RuntimeError(
                    "HM-Net decoder/encoder skip mismatch: "
                    f"decoder={tuple(values.shape)}, skip={tuple(skip.shape)}"
                )
            values = values + skip
            outputs.append(values)
        return values, outputs


class PredictionHead(nn.Module):
    def __init__(self, inputs: int, outputs: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(inputs, inputs, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(inputs, outputs, 3, padding=1),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.layers(inputs)


class HMNet(nn.Module):
    def __init__(self, classes: int = 2) -> None:
        super().__init__()
        self.current_encoder = Encoder(1)
        self.previous_encoder = Encoder(1)
        self.feedback_encoder = Encoder(classes)
        self.fused_bottleneck = Bottleneck(256)
        self.decoder1 = Decoder()
        self.encoder2 = EncoderBlock(32, 32)
        self.encoder2_rest = nn.ModuleList(
            [
                EncoderBlock(32, 64),
                EncoderBlock(64, 128),
                EncoderBlock(128, 256),
            ]
        )
        self.bottleneck2 = Bottleneck(256)
        self.decoder2 = Decoder()
        self.center_head = PredictionHead(32, classes)
        self.motion_head = PredictionHead(32, 2)
        self.precision_head = PredictionHead(32, 2)
        nn.init.constant_(self.center_head.layers[-1].bias, -2.19)

    def forward(
        self, current: Tensor, previous: Tensor, feedback: Tensor
    ) -> dict[str, Tensor]:
        current_b, current_s = self.current_encoder(current)
        previous_b, previous_s = self.previous_encoder(previous)
        feedback_b, feedback_s = self.feedback_encoder(feedback)
        summed_skips = [
            first + second + third
            for first, second, third in zip(current_s, previous_s, feedback_s)
        ]
        fused = self.fused_bottleneck(current_b + previous_b + feedback_b)
        first, first_decoder_skips = self.decoder1(fused, summed_skips)
        second_skips = []
        values, skip = self.encoder2.forward_with_skip(first)
        second_skips.append(skip)
        for block in self.encoder2_rest:
            values, skip = block.forward_with_skip(values)
            second_skips.append(skip)
        values = self.bottleneck2(values)
        # Long connections combine first-decoder and second-encoder scales.
        combined = []
        for first_skip, second_skip in zip(
            first_decoder_skips, reversed(second_skips)
        ):
            if first_skip.shape != second_skip.shape:
                raise RuntimeError(
                    "HM-Net cascade skip mismatch: "
                    f"decoder1={tuple(first_skip.shape)}, "
                    f"encoder2={tuple(second_skip.shape)}"
                )
            combined.append(first_skip + second_skip)
        # Decoder expects skips in encoder order (H, H/2, H/4, H/8).
        final, _ = self.decoder2(values, list(reversed(combined)))
        return {
            "center": torch.sigmoid(self.center_head(final)),
            "motion": self.motion_head(final),
            "precision": torch.sigmoid(self.precision_head(final)),
        }
