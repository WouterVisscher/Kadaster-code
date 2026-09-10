"""A one-epoch training run on the synthetic dataset."""

import torch

from labelnet.models import build_model
from labelnet.train import train_model


def test_train_model_one_epoch(dataset):
    dataset.epochs = 1
    result = train_model(dataset, intermediate_saves=False)

    assert len(result.train_losses) == 1
    assert len(result.val_losses) == 1
    assert result.best_epoch == 1
    assert len(result.checkpoints) == 1
    assert result.checkpoints[0].is_file()

    # The checkpoint loads back into a fresh model of the same architecture.
    model = build_model("unet")
    model.load_state_dict(torch.load(result.checkpoints[0], map_location="cpu", weights_only=True))
