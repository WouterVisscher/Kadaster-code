"""End-to-end smoke test on a small synthetic dataset (no real data needed).

Run with:
    python tests/smoke_test.py
or with pytest if available:
    pytest tests/
"""

import sys
import tempfile
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow running without installing

import matplotlib

matplotlib.use("Agg")  # headless: must be set before pyplot is imported

import numpy as np
import torch
from PIL import Image

from labelnet.config import Config
from labelnet.data import load_dataset, to_tensors, train_test_split
from labelnet.evaluate import predict, visualize_results
from labelnet.model import UNetRoadLabeler
from labelnet.train import load_model, train_model

N_PAIRS = 8
SIZE = 64  # small so the test stays fast on CPU


def _make_dataset(root: Path) -> None:
    rng = np.random.default_rng(0)
    for i in range(N_PAIRS):
        for name in ("roadnetwork", "labels"):
            (root / name).mkdir(parents=True, exist_ok=True)
            image = (rng.random((SIZE, SIZE, 3)) < 0.1).astype(np.uint8) * 255
            Image.fromarray(image).save(root / name / f"image_{i}.jpg")


def test_load_dataset() -> None:
    tmp = tempfile.TemporaryDirectory()
    root = Path(tmp.name)
    _make_dataset(root)
    inputs, targets = load_dataset(root / "roadnetwork", root / "labels", size=SIZE)
    assert inputs.shape == (N_PAIRS, SIZE, SIZE)
    assert targets.shape == (N_PAIRS, SIZE, SIZE)
    assert set(np.unique(inputs)) <= {0, 1}
    tmp.cleanup()


def test_train_test_split() -> None:
    tmp = tempfile.TemporaryDirectory()
    root = Path(tmp.name)
    _make_dataset(root)
    inputs, targets = load_dataset(root / "roadnetwork", root / "labels", size=SIZE)
    x_train, y_train, x_test, y_test = train_test_split(inputs, targets, test_fraction=0.25, seed=42)
    assert len(x_train) + len(x_test) == N_PAIRS
    assert len(x_test) == int(N_PAIRS * 0.25)
    assert len(x_train) == len(y_train) and len(x_test) == len(y_test)
    tmp.cleanup()


def test_to_tensors() -> None:
    array = np.zeros((2, 8, 8), dtype=np.uint8)  # (n, H, W)
    array[0, 0, 0] = 1
    (tensor,) = to_tensors(array)
    assert tensor.shape == (2, 1, 8, 8)
    assert tensor.dtype == torch.float32
    assert tensor[0, 0, 0, 0].item() == 1.0


def test_model_forward_shapes() -> None:
    model = UNetRoadLabeler(1, 1)
    for size in (64, 512):  # 512 is the real dataset size
        x = torch.rand(1, 1, size, size)
        y = model(x)
        assert y.shape == (1, 1, size, size)
        assert torch.all(y >= 0) and torch.all(y <= 1)


def test_full_pipeline() -> None:
    tmp = tempfile.TemporaryDirectory()
    root = Path(tmp.name)
    _make_dataset(root)
    config = replace(Config(), data_dir=root, image_size=SIZE, epochs=1, batch_size=4, checkpoint_dir=root / "checkpoints")

    inputs, targets = load_dataset(config.input_dir, config.target_dir, config.image_size)
    x_train, y_train, x_test, y_test = train_test_split(inputs, targets, config.test_fraction, config.seed)
    model = train_model(config, x_train, y_train, x_test, y_test, device=torch.device("cpu"))
    assert config.checkpoint_path.exists()

    reloaded = load_model(config.checkpoint_path, device=torch.device("cpu"))
    assert torch.equal(reloaded.state_dict()["final_conv.weight"], model.state_dict()["final_conv.weight"])

    predictions = predict(model, x_test, device=torch.device("cpu"))
    assert predictions.shape == x_test.shape[:3]

    fig = visualize_results(x_test, predictions, y_test)
    out = root / "evaluation.png"
    fig.savefig(out)
    assert out.exists()
    tmp.cleanup()


if __name__ == "__main__":
    for test in (test_load_dataset, test_train_test_split, test_to_tensors, test_model_forward_shapes, test_full_pipeline):
        test()
        print(f"PASS {test.__name__}")
    print("All smoke tests passed.")
