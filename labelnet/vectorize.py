"""Convert binary masks to vector polygons and score label overlaps.

This is the vector part of the inference pipeline: a predicted (or label)
mask is polygonised with a georeferencing transform, and the overlap between
the predicted road pieces and the original labels is scored per label.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio.features
from affine import Affine
from shapely.geometry import shape
from shapely.ops import unary_union
from shapely.validation import make_valid


def bbox_transform(minx: float, miny: float, maxx: float, maxy: float, width: int, height: int) -> Affine:
    """Build the affine transform from pixel coordinates to the map CRS."""
    pixel_width = (maxx - minx) / width
    pixel_height = (maxy - miny) / height
    return Affine.translation(minx, maxy) * Affine.scale(pixel_width, -pixel_height)


def mask_to_polygons(mask: np.ndarray, transform: Affine) -> list:
    """Extract the connected components of a binary mask as polygons."""
    polygons = []
    for geom, value in rasterio.features.shapes(mask, transform=transform):
        if value == 1:
            geom_shape = shape(geom)
            if not geom_shape.is_valid:
                geom_shape = make_valid(geom_shape)
            if not geom_shape.is_empty:
                polygons.append(geom_shape)
    return polygons


def merge_polygons(polygons: list):
    """Merge a list of polygons into a single geometry (or None)."""
    return unary_union(polygons) if polygons else None


def write_vectors(path: Path, geometries: list, crs: str | None, **properties) -> None:
    """Write polygons to a shapefile (or GeoJSON, by extension)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf = gpd.GeoDataFrame(properties, geometry=list(geometries), crs=crs)
    if path.suffix.lower() in {".geojson", ".json"}:
        gdf.to_file(path, driver="GeoJSON")
    else:
        gdf.to_file(path, driver="ESRI Shapefile")


def label_polygons_with_overlap_scores(
    predictions: np.ndarray,
    labels: np.ndarray,
    transform_predictions: Affine,
    transform_labels: Affine,
    alpha: float,
    beta: float,
    label_shapefile_path: Path,
    crs: str | None,
    prediction_shapefile_path: Path | None = None,
) -> tuple[list, list[float]]:
    """Vectorise the prediction and label masks and score their overlap.

    ``predictions`` and ``labels`` are float masks with values in [0, 1].
    Pixels above ``alpha`` are kept in the prediction (threshold function);
    label polygons whose overlap with the prediction is below ``beta`` are
    omitted from the output (omission function).

    Returns the kept label geometries and their overlap scores.
    """
    binary_predictions = (np.asarray(predictions) > alpha).astype(np.uint8)
    binary_labels = (np.asarray(labels) > 0.5).astype(np.uint8)

    prediction_polygons = mask_to_polygons(binary_predictions, transform_predictions)
    all_predictions = merge_polygons(prediction_polygons)

    label_polygons = mask_to_polygons(binary_labels, transform_labels)
    kept_geometries = []
    kept_scores = []
    for geom_shape in label_polygons:
        if all_predictions is not None:
            intersection_area = geom_shape.intersection(all_predictions).area
            score = intersection_area / geom_shape.area if geom_shape.area > 0 else 0.0
        else:
            score = 0.0
        if score >= beta:
            kept_geometries.append(geom_shape)
            kept_scores.append(float(score))

    write_vectors(label_shapefile_path, kept_geometries, crs, score=kept_scores)

    if prediction_shapefile_path is not None:
        write_vectors(
            prediction_shapefile_path,
            prediction_polygons,
            crs,
            value=[1] * len(prediction_polygons),
        )

    return kept_geometries, kept_scores
