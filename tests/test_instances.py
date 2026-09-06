from __future__ import annotations

import numpy as np

from infra_inventory.instances import (
    component_metrics,
    grid_components,
    group_line_segments,
)


def test_grid_components_two_clusters() -> None:
    x = np.concatenate((np.full(50, 0.0), np.full(30, 5.0), np.full(10, 20.0)))
    y = np.concatenate((np.linspace(0, 1, 50), np.linspace(0, 1, 30), np.linspace(0, 1, 10)))
    mask = np.ones(len(x), dtype=bool)
    components = grid_components(x, y, mask, resolution=0.5, min_cells=1)
    sizes = sorted(len(component) for component in components)
    assert sizes == [10, 30, 50]


def test_grid_components_min_cells_filter() -> None:
    x = np.array([0.0, 0.1, 0.2, 5.0])
    y = np.array([0.0, 0.1, 0.2, 5.0])
    # three points share one cell, one point is isolated
    components = grid_components(x, y, np.ones(4, dtype=bool), resolution=0.5, min_cells=1)
    assert len(components) == 2
    # with min_cells=2 both single-cell clusters are filtered out
    assert grid_components(x, y, np.ones(4, dtype=bool), resolution=0.5, min_cells=2) == []


def test_component_metrics_vertical_column() -> None:
    rng = np.random.default_rng(0)
    angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    zs = np.arange(0, 6, 0.1)
    x = (0.1 * np.cos(angles))[:, None] + rng.normal(0, 0.001, (16, len(zs)))
    y = (0.1 * np.sin(angles))[:, None] + rng.normal(0, 0.001, (16, len(zs)))
    z = np.broadcast_to(zs[None, :], x.shape)
    metrics = component_metrics(x.ravel(), y.ravel(), z.ravel(), np.arange(x.size))
    assert metrics["height_m"] > 5.5
    assert metrics["length_m"] < 0.3
    assert metrics["verticality"] > 0.9
    assert metrics["columnarity"] > 0.5
    assert metrics["planarity"] < 0.5


def test_component_metrics_planar_panel() -> None:
    ys = np.arange(-0.5, 0.5, 0.05)
    zs = np.arange(2.0, 3.0, 0.05)
    gy, gz = np.meshgrid(ys, zs)
    x = np.zeros(gy.size)
    metrics = component_metrics(x, gy.ravel(), gz.ravel(), np.arange(gy.size))
    assert metrics["planarity"] > 0.9
    assert metrics["columnarity"] < 0.2


def test_group_line_segments() -> None:
    segments = [
        (0, 0, 0, 1, 0, 0),
        (1, 0, 0, 2, 0, 0),  # near the first
        (10, 0, 0, 11, 0, 0),  # far away
    ]
    groups = group_line_segments(segments, gap_m=1.5)
    assert len(groups) == 2
    assert len(groups[0]) == 2