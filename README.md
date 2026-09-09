# Kadaster-code (labelnet)

A U-Net pipeline that learns the "best" road label placement on a map: it maps a
road network image (input) to a label placement heatmap (target), trained on
image pairs harvested from the PDOK/BRT map (see [data harvesting](data_harvesting/README.md)).

## Project layout

```
├── main.py                  # entry point: train + evaluate + visualize
├── labelnet/                # the Python package
│   ├── config.py            # all knobs in one frozen dataclass
│   ├── data.py              # image loading, binarization, train/test split
│   ├── model.py             # UNetRoadLabeler architecture
│   ├── train.py             # training loop, checkpoint save/load
│   └── evaluate.py          # inference + visualization
├── data_harvesting/         # toolchain to collect new image pairs (WMS + QGIS)
├── tests/smoke_test.py      # end-to-end smoke test on synthetic data
└── data/                    # image pairs (NOT in git; see "Data" below)
```

## Setup

Python 3.10+ (developed on 3.12).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### PyTorch / ROCm

`requirements.txt` installs standard (CPU/CUDA) torch/torchvision. For AMD ROCm
GPUs install the platform wheels first instead, then the rest:

**Linux (ROCm 7.2):**

```bash
pip3 uninstall torch torchvision triton torchaudio
pip3 install -r requirements-rocm.txt
pip3 install -r requirements.txt   # torch already satisfied, installs the rest
```

**Windows (ROCm 7.2):**

```bash
pip install --no-cache-dir \
  https://repo.radeon.com/rocm/windows/rocm-rel-7.2/torch-2.9.1%2Brocmsdk20260116-cp312-cp312-win_amd64.whl \
  https://repo.radeon.com/rocm/windows/rocm-rel-7.2/torchaudio-2.9.1%2Brocmsdk20260116-cp312-cp312-win_amd64.whl \
  https://repo.radeon.com/rocm/windows/rocm-rel-7.2/torchvision-0.24.1%2Brocmsdk20260116-cp312-cp312-win_amd64.whl
pip install -r requirements.txt
```

Useful env vars for ROCm memory behaviour:

```bash
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8,max_split_size_mb:512
```

## Data

The image pairs (625 road-network/label images, ~31 MB) are **not** in this repo;
they live in the separate [`Kadaster-data`](https://github.com/IlanWS/Kadaster-data)
repository. Get them into `./data` like this:

```bash
git clone git@github.com:IlanWS/Kadaster-data.git
mkdir -p data
cp -r Kadaster-data/roadnetwork Kadaster-data/labels data/
cp Kadaster-data/zutphen-met-labels.json data/
```

`data/` must contain:

```
data/
├── roadnetwork/   # image_0.jpg ... image_624.jpg  (input)
├── labels/        # image_0.jpg ... image_624.jpg  (target)
└── zutphen-met-labels.json   # QGIS request log used to harvest the images
```

To collect new image pairs yourself, follow [data_harvesting/README.md](data_harvesting/README.md).

## Usage

```bash
# Train (50 epochs by default) and plot input / prediction / target
python main.py

# Quick run on CPU, fewer epochs
python main.py --epochs 2 --device cpu

# Evaluate an existing checkpoint without training
python main.py --checkpoint checkpoints/labelnet_last.pt --no-show
```

Training saves the final weights to `checkpoints/labelnet_last.pt` and prints
train/validation loss per epoch.

## Tests

```bash
python tests/smoke_test.py   # or: pytest tests/
```

The smoke test builds a small synthetic dataset and runs load → train →
checkpoint reload → predict → plot end to end, so it needs no real data.
