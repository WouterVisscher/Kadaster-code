# labelnet

Neural networks that place road labels (street names) on road network map
images. The model takes a single-channel road network image and predicts a
per-pixel placement probability, which can be thresholded into label polygons
(GeoJSON) for use on a map.

The repository is organised around the four concerns of the workflow:

| Concern          | Where                                                        |
| ---------------- | ------------------------------------------------------------ |
| Data collection  | `labelnet/collect/` — QGIS panning, WMS downloads, rvimage annotation extraction |
| Data / training set | `labelnet/data.py`, loaded from the [Kadaster-data](https://github.com/WouterVisscher/Kadaster-data) repository |
| Training         | `labelnet/train.py`, `labelnet/models/`, `labelnet/losses.py` |
| Serving / running | `labelnet/inference.py`, `labelnet/serve.py` (FastAPI), CLI `labelnet predict` / `labelnet serve` |

Research evaluation (per-epoch comparison, threshold sweeps, ground-truth
accuracy) lives in `labelnet/evaluate/`.

## Installation

Python >= 3.10. Create a venv, then pick the setup that matches your
hardware. Each `requirements*.txt` pins the exact research environment
(torch 2.10.0 / torchvision 0.25.0) for its backend:

| Backend                | Requirements file      | Backend wheel index                                   |
| ---------------------- | ---------------------- | ----------------------------------------------------- |
| NVIDIA GPU (CUDA 12.8) | `requirements.txt`     | `https://download.pytorch.org/whl/cu128`              |
| AMD GPU (ROCm 7.1)     | `requirements-amd.txt` | `https://download.pytorch.org/whl/rocm7.1`            |
| CPU only               | `requirements-cpu.txt` | `https://download.pytorch.org/whl/cpu`                |

### NVIDIA GPU

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### AMD GPU (ROCm)

Works on any ROCm-compatible GPU. Only a working AMD GPU driver is
required — verify with `rocm-smi`; the torch wheel bundles the HIP
runtime, so no separate ROCm installation is needed.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-amd.txt
pip install -e .
```

PyTorch exposes the AMD device through the regular `torch.cuda` API, so
`--device auto|cuda` and everything else works unchanged.

labelnet never touches ROCm/HIP environment variables itself — if your
card needs any (e.g. `HSA_OVERRIDE_GFX_VERSION` for GPUs with immature
ROCm support, such as the Radeon 8060S "Strix Halo" gfx1151 APU, which
needs `HSA_OVERRIDE_GFX_VERSION=11.0.0` to report as the better-supported
gfx1100 target), set them in your shell before running any `labelnet`
command:

```bash
export HSA_OVERRIDE_GFX_VERSION=11.0.0  # only if your GPU needs it
labelnet train --device auto ...
```

### CPU only

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-cpu.txt
pip install -e .
```

As an alternative to the requirements files, `pip install -e ".[dev]"`
installs with loose floors from `pyproject.toml`: it keeps a pre-installed
CPU or ROCm torch in place, and otherwise pulls the CUDA build from PyPI.
The requirements files work with a plain venv if you prefer not to install
the package itself (then run the CLI via `python -m labelnet`).

## Getting the data

The dataset (625 road network / label image pairs, 512x512) lives in the
separate [Kadaster-data](https://github.com/WouterVisscher/Kadaster-data)
repository:

```bash
git clone https://github.com/WouterVisscher/Kadaster-data.git ../Kadaster-data
scripts/fetch_data.sh   # copies roadnetwork/ and labels/ into ./data
```

Set `KADASTER_DATA_SOURCE=/path/to/Kadaster-data` to use a checkout that is not
a sibling directory. The data repository's README documents the folder layout.

## Training

The fetched dataset is a flat set of 625 pairs of 512x512 images in
`data/roadnetwork/` and `data/labels/`, so point the training at it with
`--data-name ""` and `--image-size 512x512`:

```bash
labelnet train --model unet --epochs 100 --data-name "" --image-size 512x512 --batch-size 16
```

- The defaults (`--data-name combined`, `--image-size 640x360`) match the
  multi-city `combined` set built with `labelnet collect combine`.
- Images are cropped, not resized, so the size must fit the actual images.
- The default U-Net batch size of 32 was tuned for 640x360; on 512x512 it
  needs more than 24 GB of GPU memory, so pass `--batch-size 16` (or lower).
- `--model` is matched by prefix, so checkpoint names like
  `unet_dice_ES_63.pth` work too.
- `--lr` / `--batch-size` override the per-model tuned defaults in
  `labelnet/config.py`; `labelnet tune --lr 0.0006 0.001 --batch-size 16 32`
  grid-searches them and writes loss curves to `outputs/loss_curves/`.
- Checkpoints are written to `checkpoints/<model>/`; by default the final
  model's predictions on the held-out test set are saved to
  `predictions/<model>/` (use `--compare` for the per-epoch comparison plots
  instead).
- U-Net needs image dimensions divisible by 8 (512 and 640x360 both work).

## Running predictions

On a single image (`--image-size` must match the size the checkpoint was
trained with — default 640x360, or 512x512 for the fetched dataset):

```bash
labelnet predict --checkpoint checkpoints/unet/unet_100.pth --image road.jpg --out mask.png
labelnet predict --checkpoint checkpoints/unet/unet_100.pth --image road.jpg \
    --image-size 512x512 --out mask.png
labelnet predict --checkpoint ... --image road.jpg --geojson labels.geojson \
    --bbox minx,miny,maxx,maxy --threshold 0.5
```

As a service:

```bash
labelnet serve --checkpoint checkpoints/unet/unet_100.pth --port 8000
labelnet serve --checkpoint checkpoints/unet/unet_100.pth --image-size 512x512 --port 8000

curl -F image=@road.jpg http://localhost:8000/predict -o prediction.png
curl -F image=@road.jpg -F bbox=minx,miny,maxx,maxy -F geojson=true \
    http://localhost:8000/predict
```

`GET /health` reports the loaded model; `POST /predict` returns a mask PNG or
a GeoJSON `FeatureCollection` of label polygons.

## Data collection

`labelnet/collect/` rebuilds the training set: download road network and
label images from the WMS request logs, render the rvimage annotation JSON
into label masks, and combine per-city folders into the `combined` training
set. See [`labelnet/collect/README.md`](labelnet/collect/README.md) for the
full workflow (including the QGIS panning script and the local WMS servers).

CLI shortcuts:

```bash
labelnet collect download --json data/json_files/enschede_LU.json --out labels/lu
labelnet collect extract --json data/json_files/LU.json --out labels/lu
labelnet collect combine --source labels --out data/labels/combined lu ...
```

## Evaluation (research)

These tools reproduce the research experiments and expect the two WMS servers
to be running (road network on port 80, labels on port 8080, see
`labelnet/collect/README.md`):

```bash
labelnet compare --model unet            # per-epoch comparison plots
labelnet sweep --checkpoint ... --image-index 0
labelnet accuracy --checkpoint ...       # against the ground truth masks
```

## Configuration

All knobs live in the `Config` dataclass in `labelnet/config.py`; the CLI
flags above map onto it. Paths resolve relative to `data_root` (default
`./data`), so the package does not depend on the working directory.

## Development

```bash
pytest
```

The test suite runs on the synthetic data generated under `tmp_path`; the
DeepLab test downloads pre-trained ResNet101 weights on first use.
