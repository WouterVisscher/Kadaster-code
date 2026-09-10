"""Shared fixtures: a synthetic dataset of road network / label image pairs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from labelnet.config import Config


def write_pair(input_dir: Path, label_dir: Path, index: int, width: int, height: int, seed: int) -> None:
    """Write one deterministic road network / label pair as RGB JPEGs."""
    rng = np.random.default_rng(seed)
    road = (rng.random((height, width, 3)) < 0.4).astype(np.uint8) * 255
    label = (rng.random((height, width, 3)) < 0.15).astype(np.uint8) * 255
    Image.fromarray(road).save(input_dir / f"image_{index}.jpg", format="JPEG")
    Image.fromarray(label).save(label_dir / f"image_{index}.jpg", format="JPEG")


@pytest.fixture
def dataset(tmp_path: Path) -> Config:
    """A synthetic combined dataset (8 pairs of 32x64 images) and a Config for it."""
    n_pairs = 8
    height, width = 32, 64
    input_dir = tmp_path / "data" / "roadnetwork" / "combined"
    label_dir = tmp_path / "data" / "labels" / "combined"
    input_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    for i in range(n_pairs):
        write_pair(input_dir, label_dir, i, width, height, seed=i)

    return Config(
        model_name="unet",
        data_root=tmp_path / "data",
        data_name="combined",
        input_image_height=height,
        input_image_width=width,
        data_split_proportion=0.2,
        checkpoints_dir=tmp_path / "checkpoints",
        predictions_dir=tmp_path / "predictions",
        outputs_dir=tmp_path / "outputs",
        device="cpu",
    )
