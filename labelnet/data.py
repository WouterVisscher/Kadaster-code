"""Image loading, binarization and dataset preparation."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split as _sklearn_split

#: Pixel values at or above this threshold become 1 (white), below it 0 (black).
BINARY_THRESHOLD = 128


def _load_grayscale(path: Path, size: int) -> np.ndarray:
    """Load an image, crop it to ``size`` x ``size`` and reduce to one channel.

    The channel reduction uses the plain mean of RGB, matching the behaviour of
    the original data pipeline (near black-and-white map images).
    """
    array = np.asarray(Image.open(path))[:size, :size]
    if array.ndim == 3:
        array = array.mean(axis=2)
    return array


def _binarize(array: np.ndarray) -> np.ndarray:
    """Turn a grayscale array into a uint8 mask with values in {0, 1}."""
    return (array >= BINARY_THRESHOLD).astype(np.uint8)


def _image_index(path: Path) -> int:
    """Extract the numeric index from a filename like ``image_42.jpg``."""
    return int(path.stem[len("image_"):])


def load_image_pair(input_path: Path, target_path: Path, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Load one (road network, label placement) image pair, binarized.

    Returns two ``(size, size)`` uint8 arrays with values in {0, 1}.
    """
    return _binarize(_load_grayscale(input_path, size)), _binarize(_load_grayscale(target_path, size))


def load_dataset(input_dir: Path, target_dir: Path, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Load all image pairs from two folders matched by filename.

    Returns ``(inputs, targets)`` as uint8 arrays of shape ``(n, size, size)``
    with values in {0, 1}.
    """
    input_dir, target_dir = Path(input_dir), Path(target_dir)
    pairs = sorted(input_dir.glob("image_*.jpg"), key=_image_index)
    if not pairs:
        raise FileNotFoundError(f"No image_*.jpg files found in {input_dir}")

    inputs, targets = [], []
    for path in pairs:
        target_path = target_dir / path.name
        if not target_path.exists():
            raise FileNotFoundError(f"Missing target image: {target_path}")
        inputs.append(_binarize(_load_grayscale(path, size)))
        targets.append(_binarize(_load_grayscale(target_path, size)))
    return np.stack(inputs), np.stack(targets)


def train_test_split(
    inputs: np.ndarray, targets: np.ndarray, test_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split image arrays into train/test sets. Returns ``(x_train, y_train, x_test, y_test)``."""
    arrays = [np.asarray(arr) for arr in _sklearn_split(inputs, targets, test_size=test_fraction, random_state=seed)]
    x_train, x_test, y_train, y_test = arrays
    return x_train, y_train, x_test, y_test


def to_tensors(*arrays: np.ndarray) -> list[torch.Tensor]:
    """Convert (n, H, W) uint8 arrays to NCHW float32 tensors (values stay in [0, 1])."""
    return [torch.from_numpy(array).unsqueeze(1).float() for array in arrays]
