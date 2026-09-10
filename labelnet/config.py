"""Central configuration for labelnet.

All knobs live in a single ``Config`` dataclass so that training, inference and
serving can be driven from one place (or overridden on the command line). Paths
are resolved relative to ``data_root`` by default, so the package no longer
depends on the current working directory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelHyperparameters:
    learning_rate: float
    batch_size: int


# Per-model hyperparameters, the result of the tuning runs
# (previously hardcoded in Model.compile_model).
MODEL_DEFAULTS: dict[str, ModelHyperparameters] = {
    "unet": ModelHyperparameters(learning_rate=0.0006, batch_size=32),
    "deeplab": ModelHyperparameters(learning_rate=0.00002, batch_size=16),
    "stackedhourglass": ModelHyperparameters(learning_rate=0.00004, batch_size=16),
}


@dataclass
class Config:
    # --- model ---------------------------------------------------------------
    # Which architecture to use; matched by prefix against MODEL_DEFAULTS.
    model_name: str = "stackedhourglass"

    # --- data ----------------------------------------------------------------
    # Root directory that holds the fetched dataset (see scripts/fetch_data.sh).
    data_root: Path = Path("data")
    # Sub-folder of roadnetwork/ and labels/ to use as the training set.
    data_name: str = "combined"
    # Number of image pairs to use (None = use however many exist on disk).
    number_of_data_pairs: int | None = None
    # Geometry of the input/target images.
    input_image_height: int = 360
    input_image_width: int = 640
    # Fraction of the data held out as the test set.
    data_split_proportion: float = 0.2
    random_state: int = 42

    # --- training ------------------------------------------------------------
    # None: fall back to the per-model defaults in MODEL_DEFAULTS.
    learning_rate: float | None = None
    batch_size: int | None = None
    epochs: int = 100
    early_stopping_patience: int = 5
    early_stopping_delta: float = 0.001
    # Save a checkpoint every N epochs (for the per-epoch comparison plots).
    checkpoint_every: int = 10
    # Device: "auto" picks cuda if available, else cpu.
    device: str = "auto"

    # --- outputs -------------------------------------------------------------
    checkpoints_dir: Path = Path("checkpoints")
    predictions_dir: Path = Path("predictions")
    outputs_dir: Path = Path("outputs")

    # --- derived paths -------------------------------------------------------
    @property
    def input_folder(self) -> Path:
        return self.data_root / "roadnetwork" / self.data_name

    @property
    def output_folder(self) -> Path:
        return self.data_root / "labels" / self.data_name

    @property
    def json_folder(self) -> Path:
        return self.data_root / "json_files"

    @property
    def test_folder(self) -> Path:
        return self.data_root / "test"

    @property
    def ground_truth_folder(self) -> Path:
        return self.data_root / "ground_truth"

    @property
    def model_path(self) -> Path:
        return self.checkpoints_dir / self.model_name

    def hyperparameters(self) -> ModelHyperparameters:
        """The tuned hyperparameters for the configured model."""
        name = self.model_name.lower()
        for key, hyper in MODEL_DEFAULTS.items():
            if name.startswith(key):
                return hyper
        raise ValueError(
            f"Unknown model name '{self.model_name}'. "
            f"Expected a prefix of one of: {sorted(MODEL_DEFAULTS)}"
        )

    def resolve(self) -> None:
        """Create the output directories if they do not exist yet."""
        self.model_path.mkdir(parents=True, exist_ok=True)
        (self.predictions_dir / self.model_name).mkdir(parents=True, exist_ok=True)
