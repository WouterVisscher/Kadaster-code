"""The command line interface."""

import json

import numpy as np
import pytest
from PIL import Image

from labelnet.cli import main


def test_help():
    with pytest.raises(SystemExit):
        main(["--help"])


@pytest.mark.parametrize(
    "command",
    ["train", "predict", "tune", "compare", "sweep", "accuracy", "collect", "serve"],
)
def test_subcommand_help(command):
    with pytest.raises(SystemExit):
        main([command, "--help"])


def test_collect_combine(tmp_path):
    source = tmp_path / "source"
    for folder, count in (("a", 2), ("b", 3)):
        folder_dir = source / folder
        folder_dir.mkdir(parents=True)
        for i in range(count):
            Image.new("RGB", (8, 8), "white").save(folder_dir / f"img_{i}.jpg")

    out = tmp_path / "combined"
    main(["collect", "combine", "--source", str(source), "--out", str(out), "a", "b", "--start", "b=10"])
    assert sorted(p.name for p in out.glob("image_*.jpg")) == [
        "image_0.jpg",
        "image_1.jpg",
        "image_10.jpg",
        "image_11.jpg",
        "image_12.jpg",
    ]


def test_collect_extract(tmp_path):
    payload = {
        "tools_data_map": {
            "Bbox": {
                "specifics": {
                    "Bbox": {
                        "annotations_map": {
                            "city/street.jpg": [
                                {
                                    "elts": [
                                        {
                                            "Poly": {
                                                "points": [
                                                    {"x": 10, "y": 10},
                                                    {"x": 50, "y": 10},
                                                    {"x": 50, "y": 30},
                                                ]
                                            }
                                        }
                                    ]
                                },
                                {"w": 64, "h": 40},
                            ]
                        }
                    }
                }
            }
        }
    }
    json_path = tmp_path / "annotations.json"
    json_path.write_text(json.dumps(payload), encoding="utf-8")
    out = tmp_path / "labels"
    main(["collect", "extract", "--json", str(json_path), "--out", str(out)])

    mask = np.asarray(Image.open(out / "street.jpg").convert("L"))
    assert mask[15, 40] == 255  # inside the annotated polygon
    assert mask[0, 0] == 0  # outside
