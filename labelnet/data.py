"""Load road network / label image pairs and split them into train/test sets.

Images are near black-and-white JPEGs. To keep the tensors small and avoid any
shape mismatch in the model, each image is cropped to the configured size,
averaged over its colour channels and binarised to 0/1.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split


def _load_binary_image(path: Path, height: int, width: int) -> np.ndarray:
    """Load one image, crop it to (height, width) and binarise it to 0/1."""
    image = Image.open(path)
    array = np.array(image)

    # Crop to the expected size and drop any alpha channel.
    array = np.delete(
        np.delete(np.delete(array, np.s_[height::], 0), np.s_[width::], 1), np.s_[3::], 2
    )

    # Average the colour channels, then binarise.
    array = np.mean(array, axis=2, keepdims=True)
    array[array < 128] = 0
    array[array >= 128] = 1
    return np.array(array, dtype=int)


def load_images(
    input_folder: Path,
    output_folder: Path,
    number_of_data_pairs: int | None,
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Load ``number_of_data_pairs`` input/target image pairs as 0/1 arrays.

    Both arrays have shape (n, height, width, 1).
    """
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)

    if number_of_data_pairs is None:
        # Use however many input images exist on disk.
        number_of_data_pairs = len(list(input_folder.glob("image_*.jpg")))

    input_image_array = np.empty([number_of_data_pairs, height, width, 1], dtype=int)
    target_image_array = np.empty([number_of_data_pairs, height, width, 1], dtype=int)

    for i in range(number_of_data_pairs):
        input_image_array[i] = _load_binary_image(input_folder / f"image_{i}.jpg", height, width)
        target_image_array[i] = _load_binary_image(output_folder / f"image_{i}.jpg", height, width)

    return input_image_array, target_image_array


def data_split(
    input_folder: Path,
    output_folder: Path,
    number_of_data_pairs: int | None,
    height: int,
    width: int,
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the image pairs and split them into train and test sets."""
    inputs, targets = load_images(input_folder, output_folder, number_of_data_pairs, height, width)
    x_train, x_test, y_train, y_test = train_test_split(
        inputs, targets, test_size=test_size, random_state=random_state
    )
    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)
