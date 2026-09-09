"""Download map images from the URLs recorded in a QGIS request log.

The WMS must still be running (see ``run_wms.sh``). Run this script twice with
the same request log: once with the road network WMS active (save to
``data/roadnetwork``) and once with the labels WMS active (save to
``data/labels``).

Example:
    python data_harvesting/download_images.py \
        --json data/zutphen-met-labels.json --output data/roadnetwork
"""

import argparse
import json
from pathlib import Path

import requests

TIMEOUT_SECONDS = 5


def download_images(json_path: Path, output_dir: Path, timeout: int = TIMEOUT_SECONDS) -> int:
    """Download every image from the request log into ``output_dir``.

    Images are named ``image_<index>.jpg`` using their position in the log, so
    both WMS runs (road network and labels) produce matching filenames.
    Returns the number of images saved.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(json_path) as f:
        data = json.load(f)
    print(f"Downloading {len(data)} images...")

    saved = 0
    for index, item in enumerate(data):
        url = item.get("URL")
        if not url:
            print(f"Skipping index {index}: no URL found.")
            continue
        filename = f"image_{index}.jpg"
        try:
            response = requests.get(url, timeout=timeout)
        except requests.RequestException as exc:
            print(f"Error downloading {url}: {exc}")
            continue
        if response.status_code != 200:
            print(f"Error for {url}: status {response.status_code}")
            continue
        (output_dir / filename).write_bytes(response.content)
        print(f"Saved: {filename}")
        saved += 1
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, required=True, help="QGIS request log JSON file")
    parser.add_argument("--output", type=Path, required=True, help="directory to save the images in")
    args = parser.parse_args()

    saved = download_images(args.json, args.output)
    print(f"Done: {saved} images saved to {args.output}")


if __name__ == "__main__":
    main()
