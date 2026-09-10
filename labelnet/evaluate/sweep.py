"""Alpha/beta threshold sweep over the four scale contexts.

This is the old ``Evaluate_part1.py`` workflow. For one request of each
scale context (the WMS request log in ``data/json_files``), the model
predicts a label placement, which is then evaluated for every combination
of:

- ``alpha``: threshold function - only predicted pixels above alpha are kept;
- ``beta``:  omission function - label polygons whose overlap with the
  prediction is below beta are dropped.

Each combination is scored on unambiguity (labels on the road), legibility
(buffered labels that do not overlap each other) and label ratio (fraction
of the original labels that is kept), combined into a label score. The
results are drawn as 3D bar charts under ``outputs/sweep/plots``.

The local WMS instances need to be running (road network on port 80, labels
on port 8080; see ``labelnet/collect/run_wms.sh``).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np

from .. import inference, vectorize, wms
from ..config import Config
from . import metrics, plots

# The four scale contexts, named after their WMS request logs.
SCALE_CONTEXTS = (
    "small_scale_urban",
    "small_scale_rural",
    "large_scale_urban",
    "large_scale_rural",
)

# 0.0 .. 0.95 in steps of 0.05.
ALPHAS = [round(0.05 * i, 2) for i in range(20)]
BETAS = list(ALPHAS)


@dataclass
class SweepResult:
    context: str
    alpha: float
    beta: float
    unambiguity: float
    legibility: float
    label_ratio: float
    label_score: float
    n_labels: int


def _grid(results: list[SweepResult], context: str) -> list[SweepResult]:
    """The sweep points of ``context`` with both functions active (alpha, beta > 0)."""
    return [r for r in results if r.context == context and r.alpha > 0 and r.beta > 0]


def _sweep_context(config: Config, model, context: str, alphas: list[float], betas: list[float], image_index: int = 0):
    """Run the model once for ``context`` and score every (alpha, beta) pair."""
    json_name = f"{context}.json"
    road_image = wms.fetch_road_network(config, json_name, image_index)
    if road_image is None:
        print(f"Skipping {context}: could not download the road network image")
        return None, None

    start_time = time.time()
    prediction = inference.predict_image(model, road_image, config)
    inference_time = time.time() - start_time

    label_mask = wms.fetch_label_mask(config, json_name, image_index)
    if label_mask is None:
        print(f"Skipping {context}: could not download the label image")
        return None, None

    query = wms.query_for(config.json_folder / json_name, image_index)
    bbox = wms.parse_bbox(query)
    crs = wms.parse_crs(query)
    transform_prediction = vectorize.bbox_transform(*bbox, prediction.shape[1], prediction.shape[0])
    transform_label = vectorize.bbox_transform(*bbox, label_mask.shape[1], label_mask.shape[0])

    # The road network and the number of original labels are fixed for the context.
    road_binary = metrics.binarize(np.asarray(road_image))
    merged_road = metrics.road_network_geometry(road_binary, transform_prediction)
    total_labels = metrics.count_labels((label_mask > 0.5).astype(np.uint8), transform_label)
    pixel_size = (bbox[2] - bbox[0]) / prediction.shape[1]

    out_dir = config.outputs_dir / "sweep" / context
    results: list[SweepResult] = []
    for alpha in alphas:
        for beta in betas:
            label_path = out_dir / f"labels_{alpha}_{beta}.geojson"
            prediction_path = out_dir / f"prediction_{alpha}_{beta}.geojson"
            kept, scores = vectorize.label_polygons_with_overlap_scores(
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
            labels_gdf = gpd.GeoDataFrame({"score": scores}, geometry=kept, crs=crs)
            unambiguity_value, n_labels = metrics.unambiguity(merged_road, labels_gdf)
            legibility_value = metrics.legibility(metrics.buffer(labels_gdf, pixel_size))
            label_ratio = n_labels / total_labels if total_labels else 0.0
            results.append(
                SweepResult(
                    context=context,
                    alpha=alpha,
                    beta=beta,
                    unambiguity=unambiguity_value,
                    legibility=legibility_value,
                    label_ratio=label_ratio,
                    label_score=metrics.label_score(unambiguity_value, legibility_value, label_ratio),
                    n_labels=n_labels,
                )
            )

    total_time = time.time() - start_time
    print(f"{context}: {len(results)} combinations, inference {inference_time:.3f}s, total {total_time:.3f}s")
    return results, (context, inference_time, total_time)


def run_sweep(
    config: Config,
    checkpoint: Path,
    alphas: list[float] | None = None,
    betas: list[float] | None = None,
    image_index: int = 0,
    plot: bool = True,
) -> tuple[list[SweepResult], list[tuple[str, float, float]]]:
    """Sweep the (alpha, beta) grid for all scale contexts.

    Returns the results and the per-context timings
    ``(context, inference_time, total_time)``.
    """
    alphas = alphas or ALPHAS
    betas = betas or BETAS
    model = inference.load_model(config.model_name, checkpoint, config.device)

    results: list[SweepResult] = []
    timings: list[tuple[str, float, float]] = []
    for context in SCALE_CONTEXTS:
        context_results, context_timing = _sweep_context(config, model, context, alphas, betas, image_index)
        if context_results is not None:
            results.extend(context_results)
        if context_timing is not None:
            timings.append(context_timing)

    if plot and results:
        plot_results(results)
        plot_final_results(results)
    time_analysis(timings)
    return results, timings


def plot_results(results: list[SweepResult], out_dir: Path | None = None) -> None:
    """Per-context 3D bar charts, and the original (alpha=beta=0) baseline."""
    out_dir = out_dir or Path("outputs") / "sweep" / "plots"
    for context in SCALE_CONTEXTS:
        grid = _grid(results, context)
        if not grid:
            continue
        alphas = [r.alpha for r in grid]
        betas = [r.beta for r in grid]
        for name, values in (
            ("unambiguity", [r.unambiguity for r in grid]),
            ("legibility", [r.legibility for r in grid]),
            ("label_ratio", [r.label_ratio for r in grid]),
            ("label_score", [r.label_score for r in grid]),
        ):
            plots.bar3d(
                alphas,
                betas,
                values,
                out_dir / f"{context}_{name}.png",
                f"Average {name.title()} for different alpha and beta values for {context}",
                f"Average {name.title().replace('_', ' ')}",
            )
        baseline = next((r for r in results if r.context == context and r.alpha == 0 and r.beta == 0), None)
        if baseline is not None:
            print(
                f"Original labels for {context}: unambiguity={baseline.unambiguity:.4f}, "
                f"legibility={baseline.legibility:.4f}, number of labels={baseline.n_labels}"
            )


def plot_final_results(results: list[SweepResult], out_dir: Path | None = None) -> None:
    """The label score averaged over the four scale contexts."""
    out_dir = out_dir or Path("outputs") / "sweep" / "plots"
    per_context = {context: [r.label_score for r in _grid(results, context)] for context in SCALE_CONTEXTS}
    per_context = {context: values for context, values in per_context.items() if values}
    if len(per_context) < 2:
        return
    if len({len(values) for values in per_context.values()}) != 1:
        print("Cannot average the results: the (alpha, beta) grids differ between contexts")
        return

    reference = next(iter(per_context.values()))
    alphas = [r.alpha for r in _grid(results, next(iter(per_context)))]
    betas = [r.beta for r in _grid(results, next(iter(per_context)))]
    average = [sum(values[i] for values in per_context.values()) / len(per_context) for i in range(len(reference))]
    plots.bar3d(
        alphas,
        betas,
        average,
        out_dir / "all_contexts_label_score.png",
        "Average Label Score for different alpha and beta values for all cases",
        "Average Label Score",
    )


def time_analysis(timings: list[tuple[str, float, float]]) -> None:
    for context, inference_time, total_time in timings:
        print(f"Average inference time for {context}: {inference_time:.4f} seconds")
        print(f"Average total time for {context}: {total_time:.4f} seconds")
