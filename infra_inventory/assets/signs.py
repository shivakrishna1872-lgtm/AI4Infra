"""Traffic sign detection.

A sign is a *planar* cluster at road-side elevation whose measured planarity,
area, and thickness match a sign panel. When a vertical support cluster exists
beneath the panel it is grouped into the same logical asset (so the inventory
contains the sign as a unit, not a pile of points).
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..instances import component_metrics, grid_components
from ..models import Asset
from .common import TileContext, build_asset


def detect_signs(ctx: TileContext) -> List[Asset]:
    settings = ctx.settings
    sign_mask = (ctx.height >= settings.sign_min_height_m) & (ctx.height <= settings.sign_max_height_m)
    assets: List[Asset] = []
    for component in grid_components(ctx.x, ctx.y, sign_mask, settings.sign_resolution_m, min_cells=2):
        if len(component) < settings.sign_min_points:
            continue
        metrics = component_metrics(ctx.x, ctx.y, ctx.z, component)
        # Panel dimensions from covariance eigenvalues: for a rectangle a x b,
        # e = a^2/12 so a = sqrt(12*e). This stays correct for perfectly flat
        # panels whose bbox width is zero (x = const).
        points = np.column_stack((ctx.x[component], ctx.y[component], ctx.z[component]))
        cov = np.cov(points, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        eigenvalues = np.clip(eigenvalues, 0.0, None)
        e1, e2 = eigenvalues[1], eigenvalues[2]
        side_a, side_b = float(np.sqrt(12 * e2)), float(np.sqrt(12 * e1))
        area = float(12 * np.sqrt(e1 * e2))
        # Thickness = extent along the panel normal (smallest-eigenvalue axis), so
        # a vertical panel is thin even though its z-extent is large.
        normal = eigenvectors[:, 0]
        projection = points @ normal
        thickness = float(projection.max() - projection.min())
        panel_height = metrics["centroid_z"] - ctx.ground
        is_panel = (
            metrics["planarity"] >= settings.sign_min_planarity
            and settings.sign_min_panel_area_m2 <= area <= settings.sign_max_panel_area_m2
            and side_a <= settings.sign_max_panel_side_m
            and thickness <= settings.sign_max_thickness_m
        )
        if not is_panel:
            continue
        support_indices = _find_support(ctx, component, metrics)
        subclass = "panel_with_support" if support_indices is not None else "panel_only"
        merged = component if support_indices is None else np.concatenate((component, support_indices))
        panel_geometry = 0.65 + 0.2 * metrics["planarity"] + 0.15 * min(1.0, metrics["point_count"] / 300.0)
        support_note = f" grouped with a vertical support ({len(support_indices)} points)" if support_indices is not None else " no support found within search radius"
        explanation = (
            f"Planar cluster at {panel_height:.1f} m above ground, area {area:.2f} m^2, "
            f"thickness {thickness:.2f} m;{support_note}."
        )
        asset = build_asset(
            ctx,
            asset_class="traffic_sign",
            subclass=subclass,
            indices=merged,
            geometry_score=min(0.95, panel_geometry),
            explanation=explanation,
            method="geometry-v1",
            model_target_classes=("traffic_sign",),
        )
        if asset is not None:
            assets.append(asset)
    return assets


def _find_support(ctx: TileContext, panel: np.ndarray, metrics: dict) -> Optional[np.ndarray]:
    """A vertical, narrow cluster below the panel centroid within the search radius."""
    settings = ctx.settings
    panel_centroid_x, panel_centroid_y = metrics["centroid_x"], metrics["centroid_y"]
    panel_bottom = metrics["bounds"][2]
    dx = ctx.x - panel_centroid_x
    dy = ctx.y - panel_centroid_y
    # Strictly below the panel (not panel bottom rows), above the ground scatter
    near = (np.hypot(dx, dy) <= settings.sign_support_search_m) & (ctx.z < panel_bottom - 0.1)
    near &= ctx.z > ctx.ground + 0.12
    if int(near.sum()) < 8:
        return None
    supports: List[np.ndarray] = []
    for component in grid_components(ctx.x, ctx.y, near, 0.2, min_cells=1):
        if len(component) < 8:
            continue
        metrics_c = component_metrics(ctx.x, ctx.y, ctx.z, component)
        footprint = max(metrics_c["length_m"], metrics_c["width_m"])
        support_height = metrics_c["bounds"][5] - ctx.ground
        if footprint <= 0.6 and support_height >= 0.5:
            supports.append(component)
    if not supports:
        return None
    chosen = max(supports, key=len)
    return chosen