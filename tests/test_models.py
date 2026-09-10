"""The model registry and the forward passes of the architectures."""

import pytest
import torch

from labelnet.config import Config
from labelnet.models import build_model


# Small input; the unet needs the dimensions divisible by 8.
INPUT_SHAPE = (1, 1, 32, 64)


@pytest.mark.parametrize(
    "name,shape",
    [
        ("unet", (1, 1, 32, 64)),  # small input; the unet needs the dimensions divisible by 8
        ("stackedhourglass", (1, 1, 64, 64)),  # the hourglass pools four times after pre-processing
    ],
)
def test_model_forward(name, shape):
    model = build_model(name)
    model.eval()
    with torch.no_grad():
        output = model(torch.rand(*shape))
    assert output.shape == shape
    assert torch.all((output >= 0) & (output <= 1))


def test_deeplab_forward():
    # Downloads the pre-trained ResNet101 backbone on first build.
    model = build_model("deeplab")
    model.eval()
    with torch.no_grad():
        output = model(torch.rand(*INPUT_SHAPE))
    assert output.shape == (1, 1, 32, 64)
    assert torch.all((output >= 0) & (output <= 1))


def test_build_model_prefix():
    # Checkpoint-style names resolve by prefix.
    build_model("unet_dice_ES_63.pth")


def test_build_model_rejects_unknown():
    with pytest.raises(ValueError):
        build_model("resnet50")


@pytest.mark.parametrize(
    "name,expected",
    [
        ("unet", (0.0006, 32)),
        ("deeplab", (0.00002, 16)),
        ("stackedhourglass", (0.00004, 16)),
    ],
)
def test_model_defaults(name, expected):
    hyper = Config(model_name=name).hyperparameters()
    assert (hyper.learning_rate, hyper.batch_size) == expected


def test_model_defaults_reject_unknown():
    with pytest.raises(ValueError):
        Config(model_name="resnet50").hyperparameters()
