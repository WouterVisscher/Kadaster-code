"""Helpers for the WMS request logs (JSON) and the local map servers.

The harvesting workflow serves two mapserver WMS instances locally: the road
network on port 80 and the label placement on port 8080 (see
``labelnet/collect/run_wms.sh``). The JSON logs recorded by QGIS contain the
request URL and the georeference (BBOX, CRS, WIDTH, HEIGHT) of every request.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import requests
from PIL import Image

from ..config import Config


def load_log(json_path: Path) -> list[dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_port(url: str, which: str) -> str:
    # The road network is served on port 80 (standard), labels on 8080.
    if which == "roadnetwork":
        return url.replace("localhost:8080/", "localhost/")
    return url.replace("localhost/", "localhost:8080/")


def image_url(json_path: Path, index: int, which: str = "roadnetwork") -> str:
    data = load_log(json_path)
    if index >= len(data):
        raise IndexError(f"Index {index} out of range. JSON contains {len(data)} items.")
    url = data[index].get("URL")
    if not url:
        raise ValueError(f"No URL found at index {index} of {json_path}")
    return _normalize_port(url, which)


def query_for(json_path: Path, index: int) -> dict:
    data = load_log(json_path)
    if index >= len(data):
        raise IndexError(f"Index {index} out of range. JSON contains {len(data)} items.")
    return data[index].get("Query", {})


def parse_bbox(query: dict) -> tuple[float, float, float, float]:
    bbox = query["BBOX"].split("%2C")
    return tuple(float(v) for v in bbox)  # type: ignore[return-value]


def parse_crs(query: dict) -> str:
    return query["CRS"].replace("%3A", ":")


def download_image(url: str, timeout_seconds: int = 5) -> Image.Image | None:
    try:
        response = requests.get(url, timeout=timeout_seconds)
    except Exception as e:
        print(f"Error downloading from URL: {e}")
        print(f"URL: {url}")
        return None
    if response.status_code != 200:
        print(f"Error downloading from URL: Status {response.status_code}")
        print(f"URL: {url}")
        return None
    return Image.open(io.BytesIO(response.content))


def fetch_road_network(config: Config, json_name: str, index: int = 0) -> Image.Image | None:
    """Download the road network image for entry ``index`` of ``json_name``."""
    return download_image(image_url(config.json_folder / json_name, index, "roadnetwork"))


def fetch_label_mask(config: Config, json_name: str, index: int = 0, threshold: int = 10) -> np.ndarray | None:
    """Download a label image and binarise it into a [0, 1] mask.

    The label WMS renders sparse white polygons on black, so a low threshold
    keeps the anti-aliased edges intact after the downscale (matching the old
    ``get_labels`` behaviour).
    """
    url = image_url(config.json_folder / json_name, index, "labels")
    image = download_image(url)
    if image is None:
        return None

    original_width, original_height = image.size
    resized = image.resize((config.input_image_width, config.input_image_height), Image.NEAREST)
    array = np.asarray(resized)
    if array.ndim == 3 and array.shape[2] > 3:
        array = array[:, :, :3]
    gray = np.mean(array, axis=2)
    binary = (gray >= threshold).astype(np.uint8)
    mask_image = Image.fromarray(binary * 255).resize((original_width, original_height), Image.NEAREST)
    return np.asarray(mask_image) / 255.0
