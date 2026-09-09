"""Evaluation helpers: inference and visualization of predictions."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
import torch

from .model import UNetRoadLabeler
from .train import get_device


def predict(model: UNetRoadLabeler, x_test: np.ndarray, device: torch.device | None = None) -> np.ndarray:
    """Run inference on a (n, H, W) uint8 test set.

    Returns predictions of shape ``(n, H, W)`` with values in [0, 1].
    For a binary mask instead of a heatmap, threshold the result at 0.5.
    """
    device = device or get_device()
    model.eval()
    x = torch.from_numpy(x_test).unsqueeze(1).float().to(device)
    with torch.no_grad():
        predictions = model(x)
    return np.squeeze(predictions.cpu().numpy(), axis=1)


def visualize_results(x_test: np.ndarray, predictions: np.ndarray, y_test: np.ndarray, index: int = 0) -> Figure:
    """Plot input, prediction and target for one sample. Caller shows/saves the figure."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(x_test[index], cmap="gray")
    axes[0].set_title("Input (road network)")
    axes[1].imshow(predictions[index], cmap="gray")
    axes[1].set_title("Prediction")
    axes[2].imshow(y_test[index], cmap="gray")
    axes[2].set_title("Target (labels)")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    return fig
