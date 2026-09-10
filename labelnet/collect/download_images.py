"""Download the images of a WMS request log (JSON) to disk.

Run while the WMS instance is up (see ``run_wms.sh``). Run it once with
the road network WMS to fill the input images, and once with the label WMS
to fill the label images.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests


def download_images(json_path: Path, out_dir: Path, timeout_seconds: int = 5) -> int:
    """Download every image of the request log to ``out_dir``.

    The images are saved as ``image_0.jpg``, ``image_1.jpg``, ... in the
    order of the request log. Returns the number of saved images.
    """
    json_path = Path(json_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Downloading {len(data)} images from {json_path} ...")
    saved = 0
    for index, item in enumerate(data):
        url = item.get("URL")
        if not url:
            print(f"Skipping index {index}: no URL found.")
            continue
        try:
            response = requests.get(url, timeout=timeout_seconds)
        except Exception as e:
            print(f"Error downloading {url}: {e}")
            continue
        if response.status_code != 200:
            print(f"Error for {url}: status {response.status_code}")
            continue
        (out_dir / f"image_{index}.jpg").write_bytes(response.content)
        saved += 1
    print(f"Saved {saved} images to {out_dir}")
    return saved


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download the images of a WMS request log to disk")
    parser.add_argument("--json", type=Path, required=True, help="WMS request log (JSON)")
    parser.add_argument("--out", type=Path, required=True, help="directory to save the images in")
    args = parser.parse_args(argv)
    download_images(args.json, args.out)


if __name__ == "__main__":
    main()
