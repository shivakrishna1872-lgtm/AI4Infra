from __future__ import annotations

import numpy as np

from infra_inventory.preprocessing import (
    cell_occupancy_counts,
    cell_z_ranges,
    estimate_ground_z,
    height_above_ground,
    local_density,
    normalize_coordinates,
    occupied_cells,
    tile_keys,
    unique_tiles,
)


def test_tile_keys_and_unique() -> None:
    x = np.array([0.0, 39.9, 40.0, 79.9, -0.1])
    y = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    tx, ty = tile_keys(x, y, 40.0)
    assert tx.tolist() == [0, 0, 1, 1, -1]
    tiles = unique_tiles(tx, ty)
    assert len(tiles) == 3


def test_ground_estimation_flat() -> None:
    rng = np.random.default_rng(0)
    x = np.arange(0, 40, 0.5)
    y = np.arange(0, 40, 0.5)
    gx, gy = np.meshgrid(x, y)
    z = 12.5 + rng.normal(0, 0.01, gx.shape).ravel()
    assert abs(estimate_ground_z(z, gx.ravel(), gy.ravel(), 2.0) - 12.5) < 0.05


def test_ground_estimation_ignores_objects() -> None:
    x = np.arange(0, 40, 0.5)
    y = np.arange(0, 40, 0.5)
    gx, gy = np.meshgrid(x, y)
    z = np.zeros(gx.size)
    # a tall pole column at one cell
    z[gx.size // 2:] = 5.0
    ground = estimate_ground_z(z, gx.ravel(), gy.ravel(), 2.0)
    assert abs(ground) < 0.1  # object cells do not drag the ground estimate


def test_height_above_ground() -> None:
    z = np.array([10.0, 10.5, 12.0])
    heights = height_above_ground(z, 10.0)
    np.testing.assert_allclose(heights, [0.0, 0.5, 2.0])


def test_density_and_zranges() -> None:
    x = np.array([0.0, 0.1, 0.2, 5.0, 5.1])
    y = np.array([0.0, 0.1, 0.2, 5.0, 5.1])
    z = np.array([0.0, 0.0, 0.0, 5.0, 0.0])
    density = local_density(x, y, 0.5)
    assert density[0] >= 3  # cluster is denser
    assert density[3] == 2  # two points share the far cell
    zrange = cell_z_ranges(x, y, z, 0.5)
    assert zrange[0] == 0.0
    assert zrange[3] == 5.0  # z spread within the far cell


def test_occupied_cells_and_normalize() -> None:
    x = np.array([0.0, 0.0, 4.0])
    y = np.array([0.0, 0.0, 4.0])
    assert len(occupied_cells(x, y, 1.0)) == 2
    normalized = normalize_coordinates(x, y, np.array([0.0, 0.0, 0.0]))
    assert normalized.shape == (3, 3)
    assert abs(normalized.mean()) < 1e-9


def test_cell_occupancy_counts() -> None:
    x = np.array([0.1, 0.2, 1.5])
    y = np.array([0.1, 0.2, 1.5])
    counts = cell_occupancy_counts(x, y, 1.0)
    assert counts[(0, 0)] == 2
    assert counts[(1, 1)] == 1


def test_voxel_downsample_bounds_and_spreads() -> None:
    """LOD voxel grid: bounded output, one point per cell, full spatial spread."""
    from infra_inventory.preprocessing import voxel_downsample

    rng = np.random.default_rng(1)
    n = 100_000
    x = rng.uniform(0, 400, n)
    y = rng.uniform(-12, 12, n)
    z = rng.uniform(0, 0.8, n)  # surface-dominant corridor, like real mobile LiDAR
    keep = voxel_downsample(x, y, z, target=5_000)
    assert 0 < len(keep) <= 5_500  # bounded near the target
    assert len(keep) < n
    assert np.unique(keep).size == len(keep)  # no duplicate indices
    # Coverage is spatial: the kept cloud spans the whole extent.
    assert np.ptp(x[keep]) > 350
    assert np.ptp(y[keep]) > 20
    # Densest area no longer dominates: points per metre are roughly uniform.
    cell = 10.0
    bins = np.floor(x[keep] / cell).astype(int)
    counts = np.bincount(bins)
    assert counts.max() / max(counts.mean(), 1.0) < 5.0


def test_voxel_downsample_empty_and_dense() -> None:
    from infra_inventory.preprocessing import voxel_downsample

    assert len(voxel_downsample(np.zeros(0), np.zeros(0), np.zeros(0), 100)) == 0
    x = np.array([0.0, 0.001, 0.002, 0.003])  # four points in one 5 cm cell
    keep = voxel_downsample(x, np.zeros(4), np.zeros(4), target=10)
    assert len(keep) == 1  # first point of the cell wins