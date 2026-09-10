"""A DeepLabV3 (ResNet-101) segmentation model for road label placement.

The pre-trained backbone is adapted to a single-channel input (the road
network image) by re-using the pre-trained RGB conv1 weights averaged over the
colour channels, and the classifier head is replaced with a single-channel
1x1 convolution.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models.segmentation import DeepLabV3_ResNet101_Weights
from torchvision.models.segmentation.fcn import FCNHead


class DeepLabRoadLabeler(nn.Module):
    """DeepLabV3 with a ResNet-101 backbone, adapted for road label placement."""

    def __init__(self, input_channels: int = 1, output_channels: int = 1, use_aux: bool = False):
        super().__init__()
        # The whole DeepLabV3 architecture with a ResNet backbone.
        weights = DeepLabV3_ResNet101_Weights.DEFAULT
        self.network = models.segmentation.deeplabv3_resnet101(weights=weights)

        # ResNet expects 3 channels, but we give it 1 (the binary road network).
        old_conv = self.network.backbone.conv1
        self.network.backbone.conv1 = nn.Conv2d(
            input_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )

        # Average the pre-trained RGB weights over the colour channels.
        with torch.no_grad():
            self.network.backbone.conv1.weight[:] = old_conv.weight.sum(dim=1, keepdim=True)

        # The DeepLab classifier is typically [Conv2d(2048, 256), Conv2d(256,
        # num_classes)]; replace just the final layer so the internal number
        # of channels stays consistent.
        if hasattr(self.network, "classifier") and len(self.network.classifier) >= 1:
            if isinstance(self.network.classifier, nn.Sequential):
                last = list(self.network.classifier.children())[-1]
                if isinstance(last, nn.Conv2d):
                    in_ch = last.in_channels
                    self.network.classifier[-1] = nn.Conv2d(in_ch, output_channels, kernel_size=(1, 1))
            else:
                self.network.classifier = nn.Conv2d(self.network.classifier.in_channels, output_channels, kernel_size=1)

        # Use the internal auxiliary arm during training to help convergence.
        if use_aux and self.network.aux_classifier is not None:
            self.network.aux_classifier = FCNHead(1024, output_channels)
        else:
            self.network.aux_classifier = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_shape = x.shape[-2:]

        # Output of the pre-coded DeepLab model.
        result = self.network(x)

        # Resize the output back to the original input size, in case the
        # stride caused a mismatch (360 is not divisible by 32).
        out = F.interpolate(result["out"], size=input_shape, mode="bilinear", align_corners=False)
        return torch.sigmoid(out)
