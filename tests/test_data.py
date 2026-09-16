"""Loading image pairs and the train/test split."""

import numpy as np
import pytest

from labelnet.data import data_split, load_images


def test_load_images_shapes(dataset):
    inputs, targets = load_images(dataset.input_folder, dataset.output_folder, 8, 32, 64)
    assert inputs.shape == (8, 32, 64, 1)
    assert targets.shape == (8, 32, 64, 1)
    assert set(np.unique(inputs)) <= {0, 1}
    assert set(np.unique(targets)) <= {0, 1}


def test_load_images_none_uses_disk(dataset):
    inputs, _ = load_images(dataset.input_folder, dataset.output_folder, None, 32, 64)
    assert inputs.shape[0] == 8


def test_load_images_no_pairs_raises(tmp_path):
    input_folder = tmp_path / "input"
    output_folder = tmp_path / "output"
    input_folder.mkdir()
    output_folder.mkdir()
    with pytest.raises(ValueError, match="No image_\\*\\.jpg files found"):
        load_images(input_folder, output_folder, None, 32, 64)


def test_load_images_smaller_than_crop_raises(dataset):
    with pytest.raises(ValueError, match="smaller than the configured crop size"):
        load_images(dataset.input_folder, dataset.output_folder, 8, 32, 128)


def test_data_split(dataset):
    x_train, y_train, x_test, y_test = data_split(
        dataset.input_folder, dataset.output_folder, 8, 32, 64, 0.2, 42
    )
    assert x_train.shape == (6, 32, 64, 1)
    assert x_test.shape == (2, 32, 64, 1)
    assert y_train.shape == x_train.shape
    assert y_test.shape == x_test.shape


def test_data_split_reproducible(dataset):
    first = data_split(dataset.input_folder, dataset.output_folder, 8, 32, 64, 0.2, 42)
    second = data_split(dataset.input_folder, dataset.output_folder, 8, 32, 64, 0.2, 42)
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)
