"""
Lightweight two-level UNet for binary hyperspectral segmentation.

Architecture
------------
Encoder:    conv_block(C→6) → MaxPool → conv_block(6→12) → MaxPool
Bottleneck: conv_block(12→6)
Decoder:    Upsample + skip(6+12) → conv_block(18→12)
            Upsample + skip(12+6) → conv_block(18→6)
Output:     Conv2d(6→num_classes) + Sigmoid
"""

from __future__ import annotations

import torch
import torch.nn as nn


def conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """Conv2d → BatchNorm → ReLU (no bias needed with BN)."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same", bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class JustoUNetSimple(nn.Module):
    """
    Lightweight UNet for binary segmentation.

    Parameters
    ----------
    in_channels : int
        Number of input spectral bands.
    num_classes : int
        Number of output classes (1 for binary segmentation).
    """

    def __init__(self, in_channels: int = 25, num_classes: int = 1) -> None:
        super().__init__()

        self.encoder_conv1 = conv_block(in_channels, 6)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder_conv2 = conv_block(6, 12)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = conv_block(12, 6)

        self.upsample1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.decoder_conv1 = conv_block(6 + 12, 12)  # cat(bottleneck↑, skip2)

        self.upsample2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.decoder_conv2 = conv_block(12 + 6, 6)   # cat(decoder1↑, skip1)

        self.out_conv = nn.Conv2d(6, num_classes, kernel_size=3, padding="same")
        self.sigmoid = nn.Sigmoid()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        c1 = self.encoder_conv1(x)
        p1 = self.pool1(c1)

        c2 = self.encoder_conv2(p1)
        p2 = self.pool2(c2)

        # Bottleneck
        c3 = self.bottleneck(p2)

        # Decoder
        u1 = self.upsample1(c3)
        c4 = self.decoder_conv1(torch.cat([u1, c2], dim=1))

        u2 = self.upsample2(c4)
        c5 = self.decoder_conv2(torch.cat([u2, c1], dim=1))

        return self.sigmoid(self.out_conv(c5))


    def model_summary(self, input_size: tuple) -> None:
        """Print a torchinfo summary.  ``input_size`` = (B, C, H, W)."""
        try:
            import torchinfo  # optional dependency

            print("\n" + "=" * 70)
            print(f"Model Summary: {self.__class__.__name__}")
            print("=" * 70)
            torchinfo.summary(
                self,
                input_size=input_size,
                verbose=1,
                depth=3,
                col_names=["input_size", "output_size", "num_params", "trainable"],
            )
            print("=" * 70 + "\n")
        except ImportError:
            print("torchinfo not installed — run `pip install torchinfo` for a detailed summary.")
            print(self)
