"""Accuracy of the predicted labels against the ground truth.

This is the old ``Evaluate_part2.py`` workflow, over the five curated
samples per scale context (``LU1`` .. ``SU5``). Each sample is identified
by its index in the ``enschede_{ctx}.json`` request log, which provides
the georeference. The test images on disk in ``data/test`` were harvested
in request log order and do not line up with the curated samples, so the
road network and label images are re-requested from the local WMS
instances (road network on port 80, labels on port 8080).

1. ``create_samples``          - run the model and vectorise the
                                 predictions for every (alpha, beta) pair.
2. ``get_original_labels``     - vectorise the original engine labels.
3. ``get_ground_truth_labels`` - vectorise the ground truth masks from
                                 ``data/ground_truth``.
4. ``calculate_accuracy``      - TP/TN/FP/FN per original label: a label is
                                 positive when it is part of the ground
                                 truth (IoU above the overlap threshold),
                                 and predicted when it is in the model output.
5. ``visualize_accuracy``      - accuracy/recall/precision per (alpha, beta).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
from PIL import Image

from .. import inference, vectorize, wms
from ..config import Config
from . import metrics, plots

CONTEXTS = ("LU", "LR", "SU", "SR")
IMAGES_PER_CONTEXT = 5

# The index in enschede_{ctx}.json of samples 1..5 of each context.
JSON_INDICES = {
    "LU": [0, 4, 9, 11, 18],
    "LR": [0, 7, 10, 13, 14],
    "SU": [1, 8, 11, 15, 19],
    "SR": [0, 5, 11, 13, 17],
}

# 0.0 .. 0.5 in steps of 0.05.
ALPHAS = [round(0.05 * i, 2) for i in range(11)]
BETAS = list(ALPHAS)

# Two polygons are considered the same label when their IoU exceeds this.
OVERLAP_THRESHOLD = 0.1


@dataclass
class AccuracyResult:
    alpha: float
    beta: float
    tp: int
    tn: int
    fp: int
    fn: int


def _georeference(config: Config, context: str, image_index: int):
    query = wms.query_for(config.json_folder / f"enschede_{context}.json", image_index)
    return wms.parse_bbox(query), wms.parse_crs(query)


def _load_binary_mask(path: Path) -> np.ndarray:
    array = np.asarray(Image.open(path).convert("RGB"))
    return metrics.binarize(array)


def get_ground_truth_labels(config: Config) -> dict[str, Path]:
    """Vectorise the ground truth masks from ``data/ground_truth``.

    Returns a mapping like ``{"LU1": <geojson path>}``.
    """
    out_dir = config.outputs_dir / "accuracy" / "ground_truth"
    paths: dict[str, Path] = {}
    for context in CONTEXTS:
        for i in range(1, IMAGES_PER_CONTEXT + 1):
            name = f"{context}{i}"
            mask_path = config.ground_truth_folder / context / f"{name}.jpg"
            if not mask_path.is_file():
                print(f"Skipping {mask_path}: not found")
                continue
            bbox, crs = _georeference(config, context, JSON_INDICES[context][i - 1])
            mask = _load_binary_mask(mask_path)
            transform = vectorize.bbox_transform(*bbox, mask.shape[1], mask.shape[0])
            polygons = vectorize.mask_to_polygons(mask, transform)
            path = out_dir / f"{name}.geojson"
            vectorize.write_vectors(path, polygons, crs, value=[1] * len(polygons))
            paths[name] = path
            print(f"Saved ground truth for {name}: {len(polygons)} polygons")
    return paths


def get_original_labels(config: Config) -> dict[str, Path]:
    """Vectorise the original engine labels (label WMS on port 8080)."""
    out_dir = config.outputs_dir / "accuracy" / "original_labels"
    paths: dict[str, Path] = {}
    for context in CONTEXTS:
        for i in range(1, IMAGES_PER_CONTEXT + 1):
            name = f"{context}{i}"
            image_index = JSON_INDICES[context][i - 1]
            label_mask = wms.fetch_label_mask(config, f"enschede_{context}.json", image_index)
            if label_mask is None:
                print(f"Skipping {name}: could not download the label image")
                continue
            bbox, crs = _georeference(config, context, image_index)
            transform = vectorize.bbox_transform(*bbox, label_mask.shape[1], label_mask.shape[0])
            polygons = vectorize.mask_to_polygons((label_mask > 0.5).astype(np.uint8), transform)
            if not polygons:
                print(f"No label polygons found for {name}")
                continue
            path = out_dir / f"{name}.geojson"
            vectorize.write_vectors(path, polygons, crs, value=[1] * len(polygons))
            paths[name] = path
            print(f"Saved original labels for {name}: {len(polygons)} polygons")
    return paths


def create_samples(config: Config, checkpoint: Path, alphas: list[float] | None = None, betas: list[float] | None = None) -> int:
    """Run the model on the curated samples and write the predictions per (alpha, beta)."""
    alphas = alphas or ALPHAS
    betas = betas or BETAS
    model = inference.load_model(config.model_name, checkpoint, config.device)
    out_dir = config.outputs_dir / "accuracy" / "predictions"
    written = 0
    for context in CONTEXTS:
        for i in range(1, IMAGES_PER_CONTEXT + 1):
            name = f"{context}{i}"
            image_index = JSON_INDICES[context][i - 1]
            json_name = f"enschede_{context}.json"
            road_image = wms.fetch_road_network(config, json_name, image_index)
            if road_image is None:
                print(f"Skipping {name}: could not download the road network image")
                continue
            label_mask = wms.fetch_label_mask(config, json_name, image_index)
            if label_mask is None:
                print(f"Skipping {name}: could not download the label image")
                continue
            prediction = inference.predict_image(model, road_image, config)
            bbox, crs = _georeference(config, context, image_index)
            transform_prediction = vectorize.bbox_transform(*bbox, prediction.shape[1], prediction.shape[0])
            transform_label = vectorize.bbox_transform(*bbox, label_mask.shape[1], label_mask.shape[0])
            for alpha in alphas:
                for beta in betas:
                    label_path = out_dir / f"{name}_labels_{alpha}_{beta}.geojson"
                    prediction_path = out_dir / f"{name}_prediction_{alpha}_{beta}.geojson"
                    vectorize.label_polygons_with_overlap_scores(
                        prediction,
                        label_mask,
                        transform_prediction,
                        transform_label,
                        alpha,
                        beta,
                        label_path,
                        crs,
                        prediction_path,
                    )
                    written += 1
    print(f"Wrote {written} prediction sets to {out_dir}")
    return written


def _matches(geom, candidates: list, threshold: float) -> bool:
    """Whether ``geom`` overlaps one of ``candidates`` with IoU above ``threshold``."""
    for candidate in candidates:
        if not candidate.is_valid:
            candidate = candidate.buffer(0)
        if candidate.is_empty:
            continue
        union_area = geom.union(candidate).area
        if union_area > 0 and geom.intersection(candidate).area / union_area > threshold:
            return True
    return False


def _read_geometries(path: Path, reference_crs) -> list:
    if not path.is_file():
        return []
    gdf = gpd.read_file(path)
    if reference_crs is not None and gdf.crs is not None and gdf.crs != reference_crs:
        try:
            gdf = gdf.to_crs(reference_crs)
        except Exception as exc:
            print(f"Could not reproject {path}: {exc}")
            return []
    return list(gdf.geometry)


def calculate_accuracy(
    config: Config,
    alphas: list[float] | None = None,
    betas: list[float] | None = None,
    overlap_threshold: float = OVERLAP_THRESHOLD,
) -> list[AccuracyResult]:
    """Classify every original label for each (alpha, beta) combination."""
    alphas = alphas or ALPHAS
    betas = betas or BETAS
    original_dir = config.outputs_dir / "accuracy" / "original_labels"
    ground_truth_dir = config.outputs_dir / "accuracy" / "ground_truth"
    prediction_dir = config.outputs_dir / "accuracy" / "predictions"

    results: list[AccuracyResult] = []
    for alpha in alphas:
        for beta in betas:
            print(f"Evaluating for alpha={alpha}, beta={beta}")
            tp = tn = fp = fn = 0
            for context in CONTEXTS:
                for i in range(1, IMAGES_PER_CONTEXT + 1):
                    name = f"{context}{i}"
                    original_path = original_dir / f"{name}.geojson"
                    if not original_path.is_file():
                        continue
                    original_gdf = gpd.read_file(original_path)
                    gt_polys = _read_geometries(ground_truth_dir / f"{name}.geojson", original_gdf.crs)
                    model_polys = _read_geometries(prediction_dir / f"{name}_labels_{alpha}_{beta}.geojson", original_gdf.crs)
                    for orig_poly in original_gdf.geometry:
                        if not orig_poly.is_valid:
                            orig_poly = orig_poly.buffer(0)
                        if orig_poly.is_empty:
                            continue
                        in_gt = _matches(orig_poly, gt_polys, overlap_threshold)
                        in_model = _matches(orig_poly, model_polys, overlap_threshold)
                        if in_gt and in_model:
                            tp += 1
                        elif not in_gt and in_model:
                            fp += 1
                        elif in_gt and not in_model:
                            fn += 1
                        else:
                            tn += 1
            results.append(AccuracyResult(alpha, beta, tp, tn, fp, fn))
    return results


def visualize_accuracy(results: list[AccuracyResult], out_dir: Path | None = None) -> dict[str, list[float]]:
    """Accuracy/recall/precision per (alpha, beta), drawn as 3D bar charts."""
    out_dir = out_dir or Path("outputs") / "accuracy" / "plots"
    grid = [r for r in results if r.alpha > 0 and r.beta > 0]
    if not grid:
        return {}
    alphas = [r.alpha for r in grid]
    betas = [r.beta for r in grid]
    accuracy = [(r.tp + r.tn) / (r.tp + r.tn + r.fp + r.fn) if (r.tp + r.tn + r.fp + r.fn) else 0.0 for r in grid]
    recall = [r.tp / (r.tp + r.fn) if (r.tp + r.fn) else 0.0 for r in grid]
    precision = [r.tp / (r.tp + r.fp) if (r.tp + r.fp) else 0.0 for r in grid]
    f_score = [2 * p * r / (p + r) if (p + r) else 0.0 for p, r in zip(precision, recall)]

    baseline = next((r for r in results if r.alpha == 0 and r.beta == 0), None)
    if baseline is not None:
        total = baseline.tp + baseline.tn + baseline.fp + baseline.fn
        print(f"original accuracy: {(baseline.tp + baseline.tn) / total if total else 0.0:.4f}")
        print(f"original recall: {baseline.tp / (baseline.tp + baseline.fn) if (baseline.tp + baseline.fn) else 0.0:.4f}")
        print(f"original precision: {baseline.tp / (baseline.tp + baseline.fp) if (baseline.tp + baseline.fp) else 0.0:.4f}")
    print(f"highest accuracy: {max(accuracy):.4f}")
    print(f"highest recall: {max(recall):.4f}")
    print(f"highest precision: {max(precision):.4f}")

    for name, values in (("accuracy", accuracy), ("recall", recall), ("precision", precision)):
        plots.bar3d(
            alphas,
            betas,
            values,
            out_dir / f"{name}.png",
            f"{name.title()} for different alpha and beta values",
            name.title(),
        )
    return {"accuracy": accuracy, "recall": recall, "precision": precision, "f_score": f_score}


def run_accuracy(config: Config, checkpoint: Path, alphas: list[float] | None = None, betas: list[float] | None = None) -> list[AccuracyResult]:
    """The full part-2 workflow: samples, labels, ground truth, accuracy, plots."""
    alphas = alphas or ALPHAS
    betas = betas or BETAS
    create_samples(config, checkpoint, alphas, betas)
    get_original_labels(config)
    get_ground_truth_labels(config)
    results = calculate_accuracy(config, alphas, betas)
    print("True Positives:", [r.tp for r in results])
    print("False Negatives:", [r.fn for r in results])
    visualize_accuracy(results)
    return results
