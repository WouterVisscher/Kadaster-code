"""Tests for the loss functions and early stopping helper."""

from __future__ import annotations

import torch
from torch import nn

from labelnet.losses import EarlyStopping


def test_early_stopping_snapshot_survives_further_training():
    """best_model_state must not be mutated by training steps after it is captured.

    ``model.state_dict()`` returns tensors that alias the live parameter
    storage, so capturing it without a deep copy would let later in-place
    parameter updates corrupt the "best" snapshot.
    """
    model = nn.Linear(2, 2)
    early_stopping = EarlyStopping(patience=2, delta=0.0)

    early_stopping(val_loss=1.0, model=model)
    snapshot = {k: v.clone() for k, v in early_stopping.best_model_state.items()}

    # Simulate further optimizer steps mutating the model's parameters in place.
    with torch.no_grad():
        for param in model.parameters():
            param.add_(1.0)

    for key, value in snapshot.items():
        assert torch.equal(early_stopping.best_model_state[key], value), (
            f"best_model_state[{key!r}] changed after further training"
        )

    early_stopping.load_best_model(model)
    for key, value in snapshot.items():
        assert torch.equal(dict(model.state_dict())[key], value)
