"""Compare model predictions at different training epochs.

Old flow from main.py with ``make_comparison = True``: for every tenth epoch
checkpoint, run the test set and save the prediction images, then build
comparison figures (input / ground truth / one column per epoch).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

from ..config import Config
from ..data import data_split
from ..inference import load_model, predict_arrays


def epoch_checkpoint_names(config: Config) -> list[str]:
    """Checkpoint names for the intermediate saves (every ``checkpoint_every`` epochs)."""
    return [f"{config.model_name}_{(i + 1) * config.checkpoint_every}" for i in range(10)]


def load_test_set(config: Config) -> tuple[np.ndarray, np.ndarray]:
    _, _, x_test, y_test = data_split(
        config.input_folder,
        config.output_folder,
        config.number_of_data_pairs,
        config.input_image_height,
        config.input_image_width,
        config.data_split_proportion,
        config.random_state,
    )
    return x_test, y_test


def predict_test_set(config: Config, name: str, x_test: np.ndarray, batch_size: int = 16) -> np.ndarray:
    """Run the checkpoint named ``name`` over the test set."""
    model = load_model(config.model_name, config.model_path / f"{name}.pth")
    return predict_arrays(model, x_test, batch_size=batch_size)


def save_predictions(config: Config, name: str, x_test: np.ndarray, prediction: np.ndarray) -> None:
    out_dir: Path = config.predictions_dir / config.model_name / name
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(x_test.shape[0]):
        im = Image.fromarray((prediction[i] * 255).astype(np.uint8))
        im.save(out_dir / f"prediction_{i}.png")


def compare_predictions(config: Config, x_test: np.ndarray, y_test: np.ndarray, n_comparisons: int = 11) -> None:
    comparisons_dir: Path = config.predictions_dir / config.model_name / "Comparisons"
    comparisons_dir.mkdir(parents=True, exist_ok=True)
    names = epoch_checkpoint_names(config)

    for i in range(n_comparisons):
        plt.subplot(3, 4, 1)
        plt.imshow(x_test[i])
        plt.title("Input image")
        plt.subplot(3, 4, 2)
        plt.imshow(y_test[i])
        plt.title("Ground truth")
        for epoch_index, name in enumerate(names):
            plt.subplot(3, 4, epoch_index + 3)
            plt.imshow(np.array(Image.open(config.predictions_dir / config.model_name / name / f"prediction_{i}.png")))
            plt.title("Epochs: " + str((epoch_index + 1) * config.checkpoint_every))
        plt.savefig(comparisons_dir / f"compare_{i}.png")
        plt.clf()


def run_comparison(config: Config, batch_size: int = 16) -> None:
    """Full comparison flow: predict with every checkpoint, then plot."""
    x_test, y_test = load_test_set(config)
    for name in epoch_checkpoint_names(config):
        prediction = predict_test_set(config, name, x_test, batch_size=batch_size)
        save_predictions(config, name, x_test, prediction)
    compare_predictions(config, x_test, y_test)
