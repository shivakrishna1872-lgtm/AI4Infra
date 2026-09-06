"""Instance extraction: grid connected components and PCA-based component metrics.

This is the layer that turns *points* into *instances*. The pipeline feeds each
detector a list of point indices per connected component, and detectors decide
whether a component matches an asset class from its measured metrics.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Dict, Iterable, List, Tuple

import numpy as np


def grid_components(
    x: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    resolution: float,
    min_cells: int = 1,
) -> List[np.ndarray]:
    """Connected components of ``mask`` points on an XY grid (8-connectivity).

    Returns lists of point indices, one per component with at least
    ``min_cells`` occupied cells.
    """
    selected = np.flatnonzero(mask)
    if not len(selected):
        return []
    coordinates = np.floor(np.column_stack((x[selected], y[selected])) / resolution).astype(np.int64)
    index_by_cell: Dict[Tuple[int, int], List[int]] = {}
    for index, cell in zip(selected.tolist(), coordinates.tolist()):
        index_by_cell.setdefault((cell[0], cell[1]), []).append(index)
    return [
        np.asarray([index for cell in group for index in index_by_cell[cell]], dtype=np.int64)
        for group in _connected_cells(index_by_cell)
        if len(group) >= min_cells
    ]


def _connected_cells(cells: Dict[Tuple[int, int], List[int]]) -> List[List[Tuple[int, int]]]:
    remaining = set(cells)
    groups: List[List[Tuple[int, int]]] = []
    while remaining:
        start = remaining.pop()
        group = [start]
        queue = deque([start])
        while queue:
            cell = queue.popleft()
            for dx, dy in (
                (1, 0), (-1, 0), (0, 1), (0, -1),
                (1, 1), (1, -1), (-1, 1), (-1, -1),
            ):
                candidate = (cell[0] + dx, cell[1] + dy)
                if candidate in remaining:
                    remaining.remove(candidate)
                    group.append(candidate)
                    queue.append(candidate)
        groups.append(group)
    return groups


def component_metrics(x: np.ndarray, y: np.ndarray, z: np.ndarray, indices: np.ndarray) -> Dict[str, float]:
    """Measured metrics for a component: bbox, dimensions, PCA geometry.

    Returns a plain dict of floats so detectors can stay pure functions:
      bounds        (min_x, min_y, min_z, max_x, max_y, max_z)
      length_m, width_m, height_m
      centroid_x/y/z
      orientation_deg   (azimuth of the dominant XY axis, 0-180)
      planarity, linearity, verticality  (eigenvalue ratios, 0-1)
    """
    points = np.column_stack((x[indices], y[indices], z[indices]))
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    bounds = (float(lo[0]), float(lo[1]), float(lo[2]), float(hi[0]), float(hi[1]), float(hi[2]))
    centroid = points.mean(axis=0)
    # Degenerate components (1-2 points, or zero variance along an axis) make
    # np.cov/np.linalg.eigh raise on real-world tiles (USGS 3DEP airborne data
    # is full of tiny fragments). Fall back to axis-aligned metrics instead of
    # crashing the whole run.
    planarity = linearity = verticality = columnarity = 0.0
    orientation = 0.0
    if len(points) >= 3 and np.isfinite(points).all():
        try:
            cov = np.cov(points, rowvar=False)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            eigenvalues = np.clip(np.nan_to_num(eigenvalues, nan=0.0), 0.0, None)
            if np.isfinite(eigenvalues).all():
                order = np.argsort(eigenvalues)
                e0, e1, e2 = eigenvalues[order[0]], eigenvalues[order[1]], eigenvalues[order[2]]
                # Dominant axis for elongated structures is the eigenvector of e2
                dominant = eigenvectors[:, order[2]]
                orientation = (math.degrees(math.atan2(dominant[1], dominant[0])) + 360) % 180
                planarity = float((e1 - e0) / e2) if e2 > 1e-12 else 0.0
                linearity = float((e2 - e1) / e2) if e2 > 1e-12 else 0.0
                # Verticality: how vertical the dominant eigenvector is (0 horizontal .. 1 vertical)
                verticality = float(abs(dominant[2])) if eigenvalues[order[2]] > 1e-12 else 0.0
                # Columnarity: cross-section symmetry (e0 ~ e1). A vertical *column* (pole)
                # has e0 ~ e1; a vertical *plane* (sign panel, wall) has e0 << e1.
                columnarity = float(e0 / e1) if e1 > 1e-12 else 0.0
        except (np.linalg.LinAlgError, ValueError):  # pragma: no cover - defensive
            pass
    return {
        "bounds": bounds,
        "length_m": float(hi[0] - lo[0]),
        "width_m": float(hi[1] - lo[1]),
        "height_m": float(hi[2] - lo[2]),
        "centroid_x": float(centroid[0]),
        "centroid_y": float(centroid[1]),
        "centroid_z": float(centroid[2]),
        "orientation_deg": float(orientation),
        "planarity": planarity,
        "linearity": linearity,
        "verticality": verticality,
        "columnarity": columnarity,
        "point_count": int(len(indices)),
    }


def longitudinal_lateral(metrics: Dict[str, float], orientation_deg: float) -> Tuple[float, float]:
    """Split bbox extents into longitudinal (along orientation) and lateral (across)."""
    length, width = metrics["length_m"], metrics["width_m"]
    if orientation_deg is not None and 30 <= orientation_deg <= 150:
        return max(length, width), min(length, width)
    return length, width


def compactness(metrics: Dict[str, float]) -> float:
    """Occupied-volume fraction of the component's bounding box (0..1).

    Degenerate (zero-extent) bounding-box dimensions are clamped to a minimum so
    perfectly flat objects do not divide by zero.
    """
    volume = max(metrics["length_m"], 0.05) * max(metrics["width_m"], 0.05) * max(metrics["height_m"], 0.05)
    if volume <= 0:
        return 0.0
    points_estimate = metrics["point_count"] * 0.001  # rough per-point volume share
    return min(1.0, points_estimate / volume)


def group_line_segments(segments: List[Tuple[float, float, float, float, float, float]], gap_m: float = 1.5) -> List[List[Tuple[float, float, float, float, float, float]]]:
    """Group 3D line segments (x1,y1,z1,x2,y2,z2) into clusters by endpoint proximity.

    Used by the RoadMarkingExtraction DXF adapter to re-assemble vectorized
    segments into individual marking assets.
    """
    groups: List[List[Tuple[float, float, float, float, float, float]]] = []
    for segment in segments:
        placed = False
        for group in groups:
            for other in group:
                if _segment_near(segment, other, gap_m):
                    group.append(segment)
                    placed = True
                    break
            if placed:
                break
        if not placed:
            groups.append([segment])
    return groups


def _segment_near(a: Tuple[float, float, float, float, float, float], b: Tuple[float, float, float, float, float, float], gap_m: float) -> bool:
    a1 = np.asarray(a[:3], dtype=np.float64)
    a2 = np.asarray(a[3:], dtype=np.float64)
    b1 = np.asarray(b[:3], dtype=np.float64)
    b2 = np.asarray(b[3:], dtype=np.float64)
    return min(np.linalg.norm(a1 - b1), np.linalg.norm(a1 - b2), np.linalg.norm(a2 - b1), np.linalg.norm(a2 - b2)) <= gap_m