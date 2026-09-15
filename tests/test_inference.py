"""Shape and behaviour checks for single-image inference."""

from PIL import Image

from labelnet.inference import predict_image, prepare_input
from labelnet.models import build_model


def test_prepare_input_shape():
    image = Image.new("RGB", (100, 80), "white")
    tensor = prepare_input(image, 64, 32)
    assert tensor.shape == (1, 1, 32, 64)
    assert set(tensor.flatten().unique().tolist()) <= {0.0, 1.0}


def test_predict_image_runs_on_unet(dataset, tmp_path):
    model = build_model("unet")
    image = Image.new("L", (300, 200), 30)
    mask = predict_image(model, image, dataset)
    assert mask.shape == (200, 300)
    assert mask.min() >= 0.0
    assert mask.max() <= 1.0
