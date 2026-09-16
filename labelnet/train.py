"""Training loop: data loading, epochs with early stopping and checkpoints."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from .config import Config
from .data import data_split
from .logger import log
from .losses import DiceBCELoss, EarlyStopping
from .models import build_model


@dataclass
class TrainingResult:
    train_losses: list[float]
    val_losses: list[float]
    best_epoch: int
    checkpoints: list[Path]


def configure_rocm_environment() -> None:
    """Work around known ROCm/MIOpen issues, before the HIP runtime starts up.

    This must run before the *first* ``torch.cuda`` call in the process: HSA
    only reads ``HSA_OVERRIDE_GFX_VERSION`` when it enumerates GPU agents, so
    setting it any later has no effect. ``resolve_device`` calls this before
    touching ``torch.cuda`` for that reason; nothing else in this codebase
    touches it earlier.

    - ``HSA_OVERRIDE_GFX_VERSION=11.0.0``: gfx1151 (the Radeon 8060S /
      "Strix Halo" APU) support in the ROCm wheels is still immature. On the
      rocm7.1 wheels, MIOpen JIT-compiles its kernels for this target and a
      few of them (e.g. the batch norm spatial kernel) fail with an
      inline-asm codegen error; on the rocm10.0 wheels, the gfx1151 code
      objects fail to load outright under WSL2 (``hipErrorInvalidImage``).
      Reporting the ISA-compatible gfx1100 (e.g. the RX 7900 XTX, which is
      fully supported) instead makes torch load the known-good gfx1100
      binaries and avoids both failure modes. This is a no-op on GPUs that
      already report gfx1100.
    - ``MIOPEN_FIND_MODE=FAST``: works around a GPU-timer resolution issue
      seen under WSL2, where MIOpen's convolution algorithm search (which
      benchmarks candidate kernels) raises "Invalid elapsed time detected in
      EvaluateInvokers". FAST mode picks a heuristic algorithm instead of
      benchmarking, which sidesteps the timing call entirely.

    Both are set with ``setdefault``, so an explicit value already present in
    the environment (set by the user, or by a previous call) always wins.
    """
    if torch.version.hip is None:
        return
    os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
    os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")


def resolve_device(name: str = "auto") -> torch.device:
    configure_rocm_environment()
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return torch.device("xpu")
        return torch.device("cpu")
    return torch.device(name)


def disable_unusable_miopen(device: torch.device) -> None:
    """Last-resort fallback: disable MIOpen entirely if it still can't run.

    ``resolve_device`` already works around the known gfx1151 MIOpen issues
    (see ``configure_rocm_environment``), so this should rarely trigger in
    practice. It stays as a safety net for any other MIOpen failure: falls
    back to the precompiled ATen kernels by disabling MIOpen (the
    ``torch.backends.cudnn`` flag) so convolutions and batch norm use the
    (slower, but always available) native HIP kernels instead.
    """
    if device.type != "cuda":
        return
    try:
        x = torch.randn(1, 1, 8, 8, device=device)
        torch.nn.functional.batch_norm(
            x, None, None, torch.ones(1, device=device), torch.zeros(1, device=device), True, 0.1, 1e-5
        )
        torch.cuda.synchronize()
    except RuntimeError:
        torch.backends.cudnn.enabled = False
        log("MIOpen unavailable on this GPU, falling back to native kernels")


def to_tensor(array: np.ndarray) -> torch.Tensor:
    """(N, H, W, 1) numpy array -> (N, 1, H, W) float tensor."""
    return torch.from_numpy(array).permute(0, 3, 1, 2).float()


def save_checkpoint(model: torch.nn.Module, config: Config, epoch: int) -> Path:
    config.model_path.mkdir(parents=True, exist_ok=True)
    path = config.model_path / f"{config.model_name}_{epoch}.pth"
    torch.save(model.state_dict(), path)
    return path


def train_model(config: Config, intermediate_saves: bool = True) -> TrainingResult:
    """Train the configured model and save checkpoints.

    Saves a checkpoint every ``config.checkpoint_every`` epochs (for the
    per-epoch comparison plots), and always saves the final model. If early
    stopping triggers, the best-seen weights are saved under the stopping epoch.
    """
    model = build_model(config.model_name)
    hyper = config.hyperparameters()
    learning_rate = config.learning_rate if config.learning_rate is not None else hyper.learning_rate
    batch_size = config.batch_size if config.batch_size is not None else hyper.batch_size

    device = resolve_device(config.device)
    disable_unusable_miopen(device)
    model.to(device)
    log(f"Using device: {device}")

    criterion = DiceBCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    early_stopping = EarlyStopping(config.early_stopping_patience, config.early_stopping_delta)

    x_train, y_train, x_test, y_test = data_split(
        config.input_folder,
        config.output_folder,
        config.number_of_data_pairs,
        config.input_image_height,
        config.input_image_width,
        config.data_split_proportion,
        config.random_state,
    )

    train_loader = DataLoader(
        TensorDataset(to_tensor(x_train), to_tensor(y_train)), batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(to_tensor(x_test), to_tensor(y_test)), batch_size=batch_size, shuffle=False
    )

    config.model_path.mkdir(parents=True, exist_ok=True)
    log("start training")

    train_losses: list[float] = []
    val_losses: list[float] = []
    checkpoints: list[Path] = []
    early_stopped = False

    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0
        for inputs, masks in train_loader:
            inputs, masks = inputs.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_losses.append(train_loss / len(train_loader))

        # Evaluate the validation loss in each epoch.
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, masks in test_loader:
                inputs, masks = inputs.to(device), masks.to(device)
                val_loss += criterion(model(inputs), masks).item()
        val_losses.append(val_loss / len(test_loader))

        log(f"Epoch {epoch + 1}/{config.epochs} | Train Loss: {train_losses[-1]:.5f} | Val Loss: {val_losses[-1]:.5f}")

        early_stopping(val_losses[-1], model)

        if early_stopping.early_stop:
            early_stopped = True
            log(f"Early stopping at epoch: {epoch + 1}")
            # Restore the best-seen weights before saving them.
            early_stopping.load_best_model(model)
            checkpoints.append(save_checkpoint(model, config, epoch + 1))
            break

        # Intermediate checkpoints, for the per-epoch comparison plots.
        if intermediate_saves and (epoch + 1) % config.checkpoint_every == 0 and epoch + 1 != config.epochs:
            checkpoints.append(save_checkpoint(model, config, epoch + 1))

    if not early_stopped:
        checkpoints.append(save_checkpoint(model, config, config.epochs))

    best_epoch = int(np.argmin(val_losses)) + 1
    return TrainingResult(train_losses, val_losses, best_epoch, checkpoints)


def tune_model(
    config: Config,
    learning_rates: list[float],
    batch_sizes: list[int],
) -> list[TrainingResult]:
    """Grid search over learning rate and batch size (old Hyperparameter_tuning).

    Saves one loss curve per combination under ``outputs/loss_curves/<model>/``.
    """
    curves_dir = config.outputs_dir / "loss_curves" / config.model_name
    results: list[TrainingResult] = []
    for lr in learning_rates:
        for bs in batch_sizes:
            log(f"Running {config.model_name} training with learning rate: {lr} and batch size: {bs}")
            trial = Config(
                model_name=config.model_name,
                data_root=config.data_root,
                data_name=config.data_name,
                number_of_data_pairs=config.number_of_data_pairs,
                input_image_height=config.input_image_height,
                input_image_width=config.input_image_width,
                data_split_proportion=config.data_split_proportion,
                random_state=config.random_state,
                epochs=config.epochs,
                early_stopping_patience=config.early_stopping_patience,
                early_stopping_delta=config.early_stopping_delta,
                checkpoint_every=config.checkpoint_every,
                device=config.device,
                # Keep tuning checkpoints out of the way of the real ones.
                checkpoints_dir=config.checkpoints_dir / "tuning",
                predictions_dir=config.predictions_dir,
                outputs_dir=config.outputs_dir,
                learning_rate=lr,
                batch_size=bs,
            )
            result = train_model(trial)

            curves_dir.mkdir(parents=True, exist_ok=True)
            plt.figure(figsize=(10, 5))
            plt.plot(result.train_losses, label="Train Loss")
            plt.plot(result.val_losses, label="Val Loss")
            plt.ylim(0, 0.5)
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"Learning Curves {config.model_name} - Learning rate={lr}, Batch size={bs}")
            plt.legend()
            plt.savefig(curves_dir / f"loss_bs_{bs}_lr_{lr}.png")
            plt.close()
            results.append(result)
    return results
