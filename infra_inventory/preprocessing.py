"""Spatial tiling, ground estimation, and per-point feature computation.

The pipeline is designed for large mobile-LiDAR clouds: tiling is done in
streaming passes, tiles are persisted to disk, and each tile is processed one
at a time. Features here (height above ground, local density, cell z-range) are
the measurable inputs every geometry detector is built on.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from .models import ProcessingSettings


def tile_keys(x: np.ndarray, y: np.ndarray, tile_size_m: float) -> Tuple[np.ndarray, np.ndarray]:
    """Integer tile coordinates for each point (floor division)."""
    tx = np.floor(x / tile_size_m).astype(np.int64)
    ty = np.floor(y / tile_size_m).astype(np.int64)
    return tx, ty


def unique_tiles(tx: np.ndarray, ty: np.ndarray) -> np.ndarray:
    return np.unique(np.column_stack((tx, ty)), axis=0)


def estimate_ground_z(z: np.ndarray, x: np.ndarray, y: np.ndarray, bin_m: float, quantile: float = 0.15) -> float:
    """Robust scalar ground elevation for one tile.

    Bin the XY plane, take the 1st-percentile z per populated cell (removes
    vegetation/object outliers), then return the ``quantile`` of the cell
    minima. This is far more robust than a global z quantile on roads with
    curbs, drainage, and parked objects.
    """
    if len(z) == 0:
        return 0.0
    cell_x = np.floor(x / bin_m).astype(np.int64)
    cell_y = np.floor(y / bin_m).astype(np.int64)
    minima: Dict[Tuple[int, int], float] = {}
    for key, zz in zip(zip(cell_x.tolist(), cell_y.tolist()), z.tolist()):
        if not np.isfinite(zz):
            continue
        current = minima.get(key)
        if current is None or zz < current:
            minima[key] = zz
    if not minima:
        return float(np.quantile(z, quantile))
    values = np.asarray(list(minima.values()), dtype=np.float64)
    low = float(np.quantile(values, 0.01))
    filtered = values[values >= low]
    # Fall back to a global quantile if the cells are degenerate (e.g. all points vertical)
    if len(filtered) < 3:
        return float(np.quantile(z, quantile))
    return float(np.quantile(filtered, quantile))


def height_above_ground(z: np.ndarray, ground: float) -> np.ndarray:
    return z - ground


def cell_occupancy_counts(x: np.ndarray, y: np.ndarray, resolution: float) -> Dict[Tuple[int, int], int]:
    """Count of points per cell at the given resolution."""
    counts: Dict[Tuple[int, int], int] = {}
    cell_x = np.floor(x / resolution).astype(np.int64)
    cell_y = np.floor(y / resolution).astype(np.int64)
    for key in zip(cell_x.tolist(), cell_y.tolist()):
        counts[key] = counts.get(key, 0) + 1
    return counts


def local_density(x: np.ndarray, y: np.ndarray, resolution: float, radius_cells: int = 1) -> np.ndarray:
    """Per-point local density: points within a (2*radius+1)^2 cell neighbourhood."""
    counts = cell_occupancy_counts(x, y, resolution)
    result = np.zeros(len(x), dtype=np.float64)
    cell_x = np.floor(x / resolution).astype(np.int64)
    cell_y = np.floor(y / resolution).astype(np.int64)
    for index, (cx, cy) in enumerate(zip(cell_x.tolist(), cell_y.tolist())):
        total = 0
        for dx in range(-radius_cells, radius_cells + 1):
            for dy in range(-radius_cells, radius_cells + 1):
                total += counts.get((cx + dx, cy + dy), 0)
        result[index] = total
    return result


def cell_z_ranges(x: np.ndarray, y: np.ndarray, z: np.ndarray, resolution: float) -> np.ndarray:
    """Per-point z-range of its cell (vertical extent signal, useful for poles)."""
    ranges: Dict[Tuple[int, int], Tuple[float, float]] = {}
    cell_x = np.floor(x / resolution).astype(np.int64)
    cell_y = np.floor(y / resolution).astype(np.int64)
    for key, zz in zip(zip(cell_x.tolist(), cell_y.tolist()), z.tolist()):
        lo, hi = ranges.get(key, (zz, zz))
        ranges[key] = (min(lo, zz), max(hi, zz))
    result = np.zeros(len(x), dtype=np.float64)
    for index, key in enumerate(zip(cell_x.tolist(), cell_y.tolist())):
        lo, hi = ranges[key]
        result[index] = hi - lo
    return result


def occupied_cells(x: np.ndarray, y: np.ndarray, resolution: float) -> set:
    return set(zip(np.floor(x / resolution).astype(np.int64).tolist(), np.floor(y / resolution).astype(np.int64).tolist()))


def nearest_occupied_distance(
    x: np.ndarray, y: np.ndarray, occupied: set, resolution: float, max_search_cells: int = 12
) -> np.ndarray:
    """Per-point Chebyshev distance in cells to the nearest occupied cell.

    Ring-by-ring search: the first matching ring is the nearest distance.
    """
    result = np.full(len(x), float(max_search_cells), dtype=np.float64)
    cx = np.floor(x / resolution).astype(np.int64)
    cy = np.floor(y / resolution).astype(np.int64)
    for index, (key_x, key_y) in enumerate(zip(cx.tolist(), cy.tolist())):
        for ring in range(max_search_cells + 1):
            if ring == 0:
                if (int(key_x), int(key_y)) in occupied:
                    result[index] = 0.0
                    break
                continue
            found = False
            for offset in range(-ring, ring + 1):
                if (
                    (int(key_x) + ring, int(key_y) + offset) in occupied
                    or (int(key_x) - ring, int(key_y) + offset) in occupied
                    or (int(key_x) + offset, int(key_y) + ring) in occupied
                    or (int(key_x) + offset, int(key_y) - ring) in occupied
                ):
                    found = True
                    break
            if found:
                result[index] = float(ring)
                break
    return result


def normalize_coordinates(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Zero-centred, unit-scaled XYZ (feature normalization for learned backends)."""
    centroid = np.asarray([x.mean(), y.mean(), z.mean()])
    scale = float(np.sqrt(np.mean((np.column_stack((x, y, z)) - centroid) ** 2))) or 1.0
    return (np.column_stack((x, y, z)) - centroid) / scale