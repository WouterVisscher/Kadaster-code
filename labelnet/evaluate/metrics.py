"""Geometry metrics for label placement quality.

These are the building blocks used by :mod:`labelnet.evaluate.sweep` and
:mod:`labelnet.evaluate.accuracy` (the helpers from the old
``Evaluate_part1.py``):

- ``unambiguity``: the fraction of the label area that lies on the road network.
- ``legibility``: 1 - the fraction of buffered label area that overlaps other labels.
- ``label_score``: the combined score, weighted by how many of the original
  labels are kept.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
from shapely.ops import unary_union

from ..vectorize import mask_to_polygons, merge_polygons


def binarize(image_array: np.ndarray, threshold: int = 128) -> np.ndarray:
    """Convert an image array to a 0/1 uint8 mask (grayscale, thresholded)."""
    array = np.asarray(image_array)
    if array.ndim == 3:
        array = array[:, :, :3]
    gray = np.mean(array, axis=2)
    return (gray >= threshold).astype(np.uint8)


def road_network_geometry(binary_mask: np.ndarray, transform) -> object:
    """Merge the road network polygons of a binary mask into one geometry."""
    return merge_polygons(mask_to_polygons(binary_mask, transform))


def count_labels(binary_mask: np.ndarray, transform) -> int:
    """Count the label polygons (connected components) of a binary mask."""
    return len(mask_to_polygons(binary_mask, transform))


def unambiguity(merged_road_network, labels: gpd.GeoDataFrame) -> tuple[float, int]:
    """Average overlap of the labels with the road network, and the label count."""
    if merged_road_network is None:
        return 0.0, 0

    total_intersection_area = 0.0
    total_labels_area = 0.0
    number_of_labels = 0
    for idx, row in labels.iterrows():
        label_geom = row.geometry
        total_labels_area += label_geom.area
        number_of_labels += 1
        total_intersection_area += label_geom.intersection(merged_road_network).area

    if total_labels_area > 0:
        return total_intersection_area / total_labels_area, number_of_labels
    return 0.0, number_of_labels


def buffer(labels: gpd.GeoDataFrame, pixel_size: float, buffer_pixels: int = 50) -> gpd.GeoDataFrame:
    """Buffer the label geometries (buffer size in pixels, converted to map units)."""
    buffer_distance = buffer_pixels * pixel_size
    buffered = labels.copy()
    buffered["geometry"] = buffered.geometry.buffer(buffer_distance)
    return buffered


def legibility(buffered_labels: gpd.GeoDataFrame) -> float:
    """1 - the fraction of buffered label area that overlaps other buffered labels."""
    total_buffer_area = buffered_labels.geometry.area.sum()
    if total_buffer_area == 0:
        return 1.0
    union_area = unary_union(buffered_labels.geometry).area
    self_overlap_area = max(total_buffer_area - union_area, 0.0)
    return 1 - self_overlap_area / total_buffer_area


def label_score(unambiguity_value: float, legibility_value: float, label_ratio: float) -> float:
    """The combined score: mean of unambiguity and legibility, weighted by
    the square root of the fraction of original labels that are kept."""
    return ((unambiguity_value + legibility_value) / 2) * label_ratio**0.5
