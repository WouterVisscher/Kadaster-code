"""Model registry: build any of the road labeler architectures by name.

Model names are matched by prefix so that checkpoint names such as
``unet_dice_ES_63.pth`` can be resolved back to their architecture.
"""

from __future__ import annotations

from ..config import MODEL_DEFAULTS
from .deeplab import DeepLabRoadLabeler
from .stacked_hourglass import StackedHourglassRoadLabeler
from .unet import UNetRoadLabeler

__all__ = [
    "UNetRoadLabeler",
    "DeepLabRoadLabeler",
    "StackedHourglassRoadLabeler",
    "build_model",
]


def build_model(name: str, in_channels: int = 1):
    """Instantiate a model by name (case-insensitive, prefix match)."""
    key = name.lower()
    if key.startswith("unet"):
        return UNetRoadLabeler(in_channels, 1)
    if key.startswith("deeplab"):
        # Trained with use_aux=False, so the checkpoint matches.
        return DeepLabRoadLabeler(in_channels, use_aux=False)
    if key.startswith("stackedhourglass"):
        return StackedHourglassRoadLabeler(in_channels)
    raise ValueError(
        f"Unknown model name '{name}'. Expected a prefix of one of: {sorted(MODEL_DEFAULTS)}"
    )
