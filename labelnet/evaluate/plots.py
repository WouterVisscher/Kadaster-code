"""Shared plotting helpers for the evaluation (3D bar charts over the alpha/beta grid)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps as cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)


def bar3d(x, y, z, out_path: Path, title: str, zlabel: str) -> Path:
    """Save a 3D bar chart of ``z`` over the (alpha, beta) grid to ``out_path``."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    z = [float(v) for v in z]

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    vmin, vmax = min(z), max(z)
    normalize = Normalize(vmin=vmin, vmax=vmax) if vmin != vmax else Normalize(vmin=0, vmax=1)
    colormap = cm.get_cmap("coolwarm")
    colors = [colormap(normalize(value)) for value in z]

    ax.bar3d(x, y, np.zeros(len(z)), 0.045, 0.045, z, color=colors)
    ax.set_zlim(0, 1)
    ax.set_xlabel("Alpha")
    ax.set_ylabel("Beta")
    ax.set_zlabel(zlabel)
    ax.set_title(title)

    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path
