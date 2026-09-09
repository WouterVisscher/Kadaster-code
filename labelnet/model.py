"""U-Net architecture for predicting road label placement."""

import torch
from torch import nn


class UNetRoadLabeler(nn.Module):
    """Small U-Net that maps a road network image to a label placement heatmap.

    Input: (batch, in_channels, H, W) — the road network image.
    Output: (batch, out_channels, H, W) — sigmoid probabilities in [0, 1].
    The input size must be divisible by 8 (three pooling stages).
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

        # Encoder (downsampling)
        self.f1 = conv_block(in_channels, 64)
        self.p1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.f2 = conv_block(64, 128)
        self.p2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.f3 = conv_block(128, 256)
        self.p3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = conv_block(256, 512)

        # Decoder (upsampling), channel counts include the skip connection
        self.u3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = conv_block(512, 256)  # 256 upsampled + 256 skip

        self.u2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = conv_block(256, 128)  # 128 + 128

        self.u1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = conv_block(128, 64)  # 64 + 64

        # Final output layer
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        f1 = self.f1(x)
        p1 = self.p1(f1)
        f2 = self.f2(p1)
        p2 = self.p2(f2)
        f3 = self.f3(p2)
        p3 = self.p3(f3)

        # Bottleneck
        bn = self.bottleneck(p3)

        # Decoder with skip connections
        u3 = self.u3(bn)
        u3 = torch.cat([u3, f3], dim=1)
        f4 = self.dec3(u3)

        u2 = self.u2(f4)
        u2 = torch.cat([u2, f2], dim=1)
        f5 = self.dec2(u2)

        u1 = self.u1(f5)
        u1 = torch.cat([u1, f1], dim=1)
        f6 = self.dec1(u1)

        return self.sigmoid(self.final_conv(f6))
