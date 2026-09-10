"""A U-Net segmentation model for road label placement."""

from __future__ import annotations

import torch
from torch import nn


class UNetRoadLabeler(nn.Module):
    """Classic encoder-decoder U-Net with skip connections.

    The network downsamples the single-channel road network image through a
    series of convolutional blocks, then upsamples it back to the input
    resolution, concatenating the encoder features (skip connections) along the
    way. A final 1x1 convolution plus sigmoid produces a per-pixel label
    placement probability.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1):
        super().__init__()

        def conv_block(in_feat: int, out_feat: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_feat, out_feat, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_feat),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_feat, out_feat, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_feat),
                nn.ReLU(inplace=True),
            )

        # encoder (downsampling)
        self.f1 = conv_block(in_channels, 64)
        self.p1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.f2 = conv_block(64, 128)
        self.p2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.f3 = conv_block(128, 256)
        self.p3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # bottleneck
        self.bottleneck = conv_block(256, 512)

        # decoder (upsampling)
        # ConvTranspose2d upsamples, then the skip connection is concatenated
        # in forward().
        self.u3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = conv_block(512, 256)  # 512 because of concatenation (256+256)

        self.u2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = conv_block(256, 128)

        self.u1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = conv_block(128, 64)

        # classifier
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # encoder
        f1 = self.f1(x)
        p1 = self.p1(f1)

        f2 = self.f2(p1)
        p2 = self.p2(f2)

        f3 = self.f3(p2)
        p3 = self.p3(f3)

        # bottleneck
        bn = self.bottleneck(p3)

        # decoder
        u3 = self.u3(bn)
        u3 = torch.cat([u3, f3], dim=1)  # skip
        f4 = self.dec3(u3)

        u2 = self.u2(f4)
        u2 = torch.cat([u2, f2], dim=1)
        f5 = self.dec2(u2)

        u1 = self.u1(f5)
        u1 = torch.cat([u1, f1], dim=1)
        f6 = self.dec1(u1)

        outputs = self.final_conv(f6)
        return self.sigmoid(outputs)
