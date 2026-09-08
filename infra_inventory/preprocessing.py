"""Spatial tiling, ground estimation, and per-point feature computation.

The pipeline is designed for large mobile-LiDAR clouds: tiling is done in
streaming passes, tiles are persisted to disk, and each tile is processed one
at a time. Features here (height above ground, local density, cell z-range) are
the measurable inputs every geometry detector is built on.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np

from .models import ProcessingSettings


def voxel_downsample(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    target: int,
    min_cell_m: float = 0.05,
    max_cell_m: float = 8.0,
) -> np.ndarray:
    """Spatial LOD: keep the first point of every occupied voxel, ~``target`` cells.

    This is the LAStools ``las2las -thin``-style spatial decimation the viewer
    needs: uniform *strided* sampling keeps the densest areas over-represented
    and can still ship tens of megabytes of a multi-gigabyte cloud, while a
    voxel grid guarantees one point per cell of space - the background cloud
    covers the whole scene at bounded size. Deterministic (first point per
    voxel, input order preserved), vectorized, and memory-bounded.

    The cell edge is chosen so a surface-dominant scene yields ~``target``
    occupied cells (``cell = sqrt(horizontal_extent / target)``); 3-D clutter
    (poles, wires) simply adds fewer cells than the target.

    Returns the kept point indices (a subset of ``np.arange(len(x))``).
    """
    if len(x) == 0:
        return np.zeros(0, dtype=np.int64)
    extent_x = float(np.ptp(x))
    extent_y = float(np.ptp(y))
    area = max(extent_x * extent_y, 1e-6)
    cell = min(max(math.sqrt(area / max(target, 1)), min_cell_m), max_cell_m)
    ix = np.floor((x - float(np.min(x))) / cell).astype(np.int64)
    iy = np.floor((y - float(np.min(y))) / cell).astype(np.int64)
    iz = np.floor((z - float(np.min(z))) / cell).astype(np.int64)
    cells = np.column_stack((ix, iy, iz))
    _, first = np.unique(cells, axis=0, return_index=True)
    return np.sort(first)


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

    Vectorized: per-cell minima come from a sort-by-cell + ``np.minimum.reduceat``
    (identical values to the old dict loop; quantiles are order-insensitive).
    """
    if len(z) == 0:
        return 0.0
    finite = np.isfinite(z)
    if not finite.any():
        return float(np.quantile(z, quantile))
    xf, yf, zf = x[finite], y[finite], z[finite]
    cell_x = np.floor(xf / bin_m).astype(np.int64)
    cell_y = np.floor(yf / bin_m).astype(np.int64)
    order = np.lexsort((cell_y, cell_x))
    sorted_z = zf[order]
    key = np.column_stack((cell_x, cell_y))[order]
    starts = np.flatnonzero(np.concatenate(([True], (key[1:] != key[:-1]).any(axis=1))))
    values = np.minimum.reduceat(sorted_z, starts)
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
    """Per-point z-range of its cell (vertical extent signal, useful for poles).

    Vectorized: points are sorted by cell, per-cell min/max come from
    ``np.minimum/maximum.reduceat``, and each point receives its cell's range.
    Values are identical to the old per-point dict loop.
    """
    cell_x = np.floor(x / resolution).astype(np.int64)
    cell_y = np.floor(y / resolution).astype(np.int64)
    order = np.lexsort((cell_y, cell_x))
    sorted_z = z[order]
    key = np.column_stack((cell_x, cell_y))[order]
    starts = np.flatnonzero(np.concatenate(([True], (key[1:] != key[:-1]).any(axis=1))))
    ends = np.concatenate((starts[1:], [len(order)]))
    lo = np.minimum.reduceat(sorted_z, starts)
    hi = np.maximum.reduceat(sorted_z, starts)
    result = np.empty(len(x), dtype=np.float64)
    result[order] = np.repeat(hi - lo, ends - starts)
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