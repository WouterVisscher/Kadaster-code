"""Pipeline configuration.

All knobs live in a single frozen dataclass so runs are reproducible and
overridable from the command line (see ``main.py``).
"""

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Config:
    # Data
    data_dir: Path = PROJECT_ROOT / "data"
    image_size: int = 512
    test_fraction: float = 0.2
    seed: int = 42

    # Training
    learning_rate: float = 1e-3
    epochs: int = 50
    batch_size: int = 5

    # Artifacts
    checkpoint_dir: Path = PROJECT_ROOT / "checkpoints"

    @property
    def input_dir(self) -> Path:
        """Road network (input) images."""
        return self.data_dir / "roadnetwork"

    @property
    def target_dir(self) -> Path:
        """Label placement (target) images."""
        return self.data_dir / "labels"

    @property
    def checkpoint_path(self) -> Path:
        """Where the final model weights are saved after training."""
        return self.checkpoint_dir / "labelnet_last.pt"
