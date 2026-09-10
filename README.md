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

Python >= 3.10.

```bash
pip install -e ".[dev]"
```

The default `torch` install is the CUDA build. For a CPU-only environment,
install the CPU wheel first and then the rest:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
```

The CPU wheel matches the version pin, so the second step leaves it in place.
`requirements.txt` mirrors `pyproject.toml` and works with a plain venv if you
prefer not to install the package itself.

## Getting the data

The dataset (1200 training image pairs, test images, ground truth masks and
WMS request logs) lives in the separate
[Kadaster-data](https://github.com/WouterVisscher/Kadaster-data) repository:

```bash
git clone https://github.com/WouterVisscher/Kadaster-data.git ../Kadaster-data
scripts/fetch_data.sh   # copies roadnetwork/, labels/, test/, ground_truth/, json_files/, outputs/ into ./data
```

Set `KADASTER_DATA_SOURCE=/path/to/Kadaster-data` to use a checkout that is not
a sibling directory. The data repository's README documents the folder layout.

## Training

```bash
labelnet train --model unet --epochs 100
```

- `--model` is matched by prefix, so checkpoint names like
  `unet_dice_ES_63.pth` work too.
- `--lr` / `--batch-size` override the per-model tuned defaults in
  `labelnet/config.py`; `labelnet tune --lr 0.0006 0.001 --batch-size 16 32`
  grid-searches them and writes loss curves to `outputs/loss_curves/`.
- Checkpoints are written to `checkpoints/<model>/`; by default the final
  model's predictions on the held-out test set are saved to
  `predictions/<model>/` (use `--compare` for the per-epoch comparison plots
  instead).
- U-Net needs image dimensions divisible by 8 (the dataset is 640x360).

## Running predictions

On a single image:

```bash
labelnet predict --checkpoint checkpoints/unet/unet_100.pth --image road.jpg --out mask.png
labelnet predict --checkpoint ... --image road.jpg --geojson labels.geojson \
    --bbox minx,miny,maxx,maxy --threshold 0.5
```

As a service:

```bash
labelnet serve --checkpoint checkpoints/unet/unet_100.pth --port 8000

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
