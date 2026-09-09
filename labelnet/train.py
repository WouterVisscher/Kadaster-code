"""Training loop for the label placement U-Net."""

from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from .config import Config
from .data import to_tensors
from .model import UNetRoadLabeler


def get_device(preferred: str | None = None) -> torch.device:
    """Return the best available device, honouring an explicit preference."""
    if preferred == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(
    config: Config,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    device: torch.device | None = None,
) -> UNetRoadLabeler:
    """Train the U-Net and return the trained model.

    ``x_*``/``y_*`` are uint8 NHWC arrays with values in {0, 1}.
    The final weights are saved to ``config.checkpoint_path``.
    """
    device = device or get_device()

    model = UNetRoadLabeler(1, 1).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    x_train_t, y_train_t = to_tensors(x_train, y_train)
    x_test_t, y_test_t = to_tensors(x_test, y_test)

    train_loader = DataLoader(TensorDataset(x_train_t, y_train_t), batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_test_t, y_test_t), batch_size=config.batch_size, shuffle=False)

    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0
        for inputs, masks in train_loader:
            inputs, masks = inputs.to(device), masks.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, masks in val_loader:
                inputs, masks = inputs.to(device), masks.to(device)
                val_loss += criterion(model(inputs), masks).item()

        print(
            f"Epoch {epoch + 1}/{config.epochs} | "
            f"Train Loss: {train_loss / len(train_loader):.4f} | "
            f"Val Loss: {val_loss / len(val_loader):.4f}"
        )

    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), config.checkpoint_path)
    print(f"Model saved to {config.checkpoint_path}")
    return model


def load_model(checkpoint: Path, device: torch.device | None = None) -> UNetRoadLabeler:
    """Load a checkpoint saved by :func:`train_model` into a fresh model."""
    device = device or get_device()
    model = UNetRoadLabeler(1, 1).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    return model
