"""Loss functions and early stopping for the training loop."""

from __future__ import annotations

import torch
from torch import nn


class DiceBCELoss(nn.Module):
    """BCE + dice loss. Inputs are sigmoid probabilities, not logits."""

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.BCELoss()(inputs, targets)

        # Dice loss on the probabilities.
        intersection = (inputs * targets).sum()
        dice = 1 - (2.0 * intersection / (inputs.sum() + targets.sum() + 1e-7))
        return bce + dice


class AdaptiveWingLoss(nn.Module):
    """Adaptive wing loss (alternative, kept for experimentation)."""

    def __init__(self, omega: float = 14, theta: float = 0.5, epsilon: float = 1, alpha: float = 1):
        super().__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        delta_y = torch.abs(y_true - y_pred)
        loss = torch.where(
            delta_y < self.theta,
            self.omega * torch.log(1 + (delta_y / self.epsilon) ** self.alpha),
            self.omega * (delta_y - self.theta)
            + self.omega * torch.log(torch.tensor(1 + (self.theta / self.epsilon) ** self.alpha)),
        )
        return torch.mean(loss)


class EarlyStopping:
    """Stops training when the validation loss stops improving."""

    def __init__(self, patience: int = 5, delta: float = 0.001):
        self.patience = patience
        self.delta = delta
        self.best_score: float | None = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state: dict | None = None

    def __call__(self, val_loss: float, model: nn.Module) -> None:
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.counter = 0

    def load_best_model(self, model: nn.Module) -> None:
        """Restore the best-seen weights into ``model`` (in place)."""
        model.load_state_dict(self.best_model_state)
