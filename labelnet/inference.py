"""Run a trained model on road network images (single images or arrays)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .config import Config
from .models import build_model


def prepare_input(image: Image.Image, width: int, height: int, threshold: int = 128) -> torch.Tensor:
    """Resize a road network image to the model input size and binarise it.

    Returns a float tensor of shape (1, 1, height, width) with 0/1 values.
    """
    resized = image.convert("RGB").resize((width, height), Image.LANCZOS)
    array = np.asarray(resized)
    if array.ndim == 3 and array.shape[2] > 3:
        array = array[:, :, :3]
    gray = np.mean(array, axis=2, keepdims=True)
    binary = (gray >= threshold).astype(np.float32)
    return torch.from_numpy(binary).unsqueeze(0)


def predict_image(model: torch.nn.Module, image: Image.Image, config: Config, threshold: int = 128) -> np.ndarray:
    """Predict a label placement mask for one road network image.

    Returns a float array of shape (height, width) with values in [0, 1],
    resized back to the original image size.
    """
    width, height = image.size
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        tensor = prepare_input(image, config.input_image_width, config.input_image_height, threshold).to(device)
        output = model(tensor)
    mask = output.squeeze(0).squeeze(0).cpu().numpy()
    mask_image = Image.fromarray((mask * 255).astype(np.uint8))
    mask_image = mask_image.resize((width, height), Image.LANCZOS)
    return np.asarray(mask_image) / 255.0


def predict_arrays(model: torch.nn.Module, arrays: np.ndarray, batch_size: int = 16) -> np.ndarray:
    """Run the model on a stack of (N, H, W, 1) 0/1 arrays.

    Returns a (N, H, W) float array with values in [0, 1].
    """
    device = next(model.parameters()).device
    model.eval()
    tensor = torch.from_numpy(np.asarray(arrays)).permute(0, 3, 1, 2).float()
    predictions = []
    with torch.no_grad():
        for start in range(0, tensor.shape[0], batch_size):
            batch = tensor[start : start + batch_size].to(device)
            predictions.append(model(batch).cpu())
    out = torch.cat(predictions, dim=0).numpy().transpose(0, 2, 3, 1)
    return out[:, :, :, 0]


def infer_model_name(checkpoint: Path) -> str:
    """Infer the model family from a checkpoint name like ``unet_dice_ES_63.pth``."""
    from .config import MODEL_DEFAULTS

    stem = Path(checkpoint).stem.lower()
    for key in MODEL_DEFAULTS:
        if stem.startswith(key):
            return key
    raise ValueError(f"Cannot infer the model name from checkpoint '{checkpoint}'")


def load_model(model_name: str, checkpoint: Path, device: str = "auto") -> torch.nn.Module:
    """Build the model matching ``model_name`` and load the checkpoint weights."""
    from .train import resolve_device

    model = build_model(model_name)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.to(resolve_device(device))
    model.eval()
    return model
