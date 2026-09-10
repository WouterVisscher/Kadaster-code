"""Turn rvimage annotation JSON files into label mask images.

The rvimage JSON format stores the annotation polygons of every image in
``tools_data_map -> {tool} -> specifics -> {tool} -> annotations_map``.
:func:`extract_rvimage_masks` renders the polygons of every image as a
white-on-black JPEG mask, and :func:`combine_label_folders` copies the
per-city label folders into one combined training folder, renaming the
images to ``image_0``, ``image_1``, ...

Combined data comes from different JSON files, so extraction has to be
run once per file, followed by one combine step.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from PIL import Image, ImageDraw


def _find_annotations_maps(data: dict):
    """Yield every annotations_map in an rvimage JSON file (Bbox, Rot90, Brush tools)."""
    for tool_name in ("Bbox", "Rot90", "Brush"):
        tool_data = data.get("tools_data_map", {}).get(tool_name, {})
        specifics = tool_data.get("specifics", {})
        tool_map = specifics.get(tool_name, {})
        annotations_map = tool_map.get("annotations_map")
        if isinstance(annotations_map, dict) and annotations_map:
            yield annotations_map


def extract_rvimage_masks(json_path: Path, output_dir: Path, width: int = 640, height: int = 360) -> int:
    """Render the annotation polygons of every image in the rvimage JSON.

    Images with labels are saved as white-on-black JPEG masks named after
    the source image, in ``output_dir``. Returns the number of saved masks.
    """
    json_path = Path(json_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    maps = list(_find_annotations_maps(data))
    if not maps:
        raise ValueError(f"No annotations_map found in {json_path}")
    annotations_map = maps[0]

    saved = 0
    for file_path, content in annotations_map.items():
        if len(content) < 2:
            print(f"Skipping {file_path}: no annotation data")
            continue
        anno_data, size_data = content[0], content[1]
        image_width = size_data.get("w", width)
        image_height = size_data.get("h", height)

        mask = Image.new("RGB", (image_width, image_height), (0, 0, 0))
        draw = ImageDraw.Draw(mask)
        has_labels = False
        for elt in anno_data.get("elts", []):
            poly_data = elt.get("Poly")
            if not poly_data:
                continue
            points = [(p["x"], p["y"]) for p in poly_data.get("points", []) if "x" in p and "y" in p]
            if len(points) >= 3:
                draw.polygon(points, fill=(255, 255, 255))
                has_labels = True

        if has_labels:
            mask.save(output_dir / Path(file_path).name, format="JPEG")
            saved += 1
    print(f"Saved {saved} masks to {output_dir}")
    return saved


def combine_label_folders(
    source_root: Path,
    output_dir: Path,
    folder_sequence: list[str],
    start_points: dict[str, int] | None = None,
) -> int:
    """Copy images from label subfolders into one destination folder.

    The images are renamed to ``image_0``, ``image_1``, ... in the combined
    folder. ``start_points`` can force the start index of any folder.
    Returns the number of combined images.
    """
    source_root = Path(source_root)
    output_dir = Path(output_dir)
    if start_points is None:
        start_points = {}
    output_dir.mkdir(parents=True, exist_ok=True)

    current_index = 0
    for folder_name in folder_sequence:
        if folder_name in start_points:
            current_index = start_points[folder_name]

        folder_path = source_root / folder_name
        if not folder_path.is_dir():
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        image_files = sorted(
            f for f in folder_path.iterdir() if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        for image_file in image_files:
            destination = output_dir / f"image_{current_index}{image_file.suffix.lower()}"
            shutil.copy2(image_file, destination)
            print(f"Copied {folder_name}/{image_file.name} -> {destination.name}")
            current_index += 1

    print(f"Combined images written to: {output_dir}")
    return current_index


def _parse_start_points(specs: list[str] | None) -> dict[str, int]:
    start_points: dict[str, int] = {}
    for spec in specs or []:
        folder, index = spec.split("=", 1)
        start_points[folder] = int(index)
    return start_points


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Extract label masks from rvimage annotation JSON files")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("extract", help="render the annotation polygons as label masks")
    p.add_argument("--json", type=Path, required=True, help="rvimage annotation JSON file")
    p.add_argument("--out", type=Path, required=True, help="directory to save the masks in")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=360)

    p = sub.add_parser("combine", help="copy label folders into one combined folder")
    p.add_argument("--source", type=Path, required=True, help="root directory of the label folders")
    p.add_argument("--out", type=Path, required=True, help="combined output directory")
    p.add_argument("folders", nargs="+", help="label folders, in the order to combine them")
    p.add_argument("--start", action="append", default=None, metavar="FOLDER=INDEX", help="start index for a folder (repeatable)")

    args = parser.parse_args(argv)
    if args.command == "extract":
        extract_rvimage_masks(args.json, args.out, args.width, args.height)
    else:
        combine_label_folders(args.source, args.out, args.folders, _parse_start_points(args.start))


if __name__ == "__main__":
    main()
