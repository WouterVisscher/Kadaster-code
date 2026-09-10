"""A Stacked Hourglass segmentation model for road label placement.

The stacked hourglass architecture (newell et al.) repeatedly narrows the
feature map down to a low-resolution "bottleneck" and expands it back to the
full resolution, refining the prediction with each stack. Two stacks are used
here to keep inference fast.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class Residual(nn.Module):
    """The smallest building block: a few residual convolutions."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(out_channels // 2)
        self.conv2 = nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels // 2)
        self.conv3 = nn.Conv2d(out_channels // 2, out_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        if self.skip is not None:
            identity = self.skip(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out + identity)


class Hourglass(nn.Module):
    """One hourglass: narrow down, then expand back, refining features.

    The structure is recursive: as long as ``depth > 1`` it keeps making new
    hourglasses inside itself.
    """

    def __init__(self, depth: int, channels: int):
        super().__init__()
        self.depth = depth
        self.up1 = Residual(channels, channels)
        self.low1 = nn.MaxPool2d(2, stride=2)
        self.low2 = Residual(channels, channels)
        if self.depth > 1:
            self.low3 = Hourglass(depth - 1, channels)
        else:
            self.low3 = Residual(channels, channels)
        self.low4 = Residual(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        up1 = self.up1(x)
        low1 = self.low1(x)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        low4 = self.low4(low3)
        # to avoid size mismatches
        up2 = F.interpolate(low4, size=up1.shape[-2:], mode="nearest")
        return up1 + up2


class StackedHourglassRoadLabeler(nn.Module):
    """Stacked hourglass model producing a per-pixel label placement heatmap."""

    def __init__(self, input_channels: int = 1, num_stacks: int = 2, num_channels: int = 128):
        # In case of memory overflow, drop num_channels to e.g. 64.
        super().__init__()
        self.num_stacks = num_stacks
        self.pre = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            Residual(64, 128),
            nn.MaxPool2d(2, stride=2),
            Residual(128, 128),
            Residual(128, num_channels),
        )

        self.hgs = nn.ModuleList([Hourglass(4, num_channels) for _ in range(num_stacks)])
        self.features = nn.ModuleList(
            [
                nn.Sequential(
                    Residual(num_channels, num_channels),
                    nn.Conv2d(num_channels, num_channels, 1),
                    nn.BatchNorm2d(num_channels),
                    nn.ReLU(inplace=True),
                )
                for _ in range(num_stacks)
            ]
        )

        # heatmap prediction layers
        self.outs = nn.ModuleList([nn.Conv2d(num_channels, 1, 1) for _ in range(num_stacks)])
        self.merge_features = nn.ModuleList([nn.Conv2d(num_channels, num_channels, 1) for _ in range(num_stacks)])
        self.merge_preds = nn.ModuleList([nn.Conv2d(1, num_channels, 1) for _ in range(num_stacks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W)
        target_size = x.shape[-2:]
        x = self.pre(x)
        combined_outputs = []

        for i in range(self.num_stacks):
            hg = self.hgs[i](x)
            feature = self.features[i](hg)
            preds = self.outs[i](feature)
            combined_outputs.append(preds)

            if i < self.num_stacks - 1:
                x = x + self.merge_features[i](feature) + self.merge_preds[i](preds)

        # Resize the final prediction back to the input size.
        final_outputs = [F.interpolate(o, size=target_size, mode="bilinear") for o in combined_outputs]
        # Sigmoid for a continuous output.
        return torch.sigmoid(final_outputs[-1])
