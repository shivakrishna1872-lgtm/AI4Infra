"""Utility asset detection: poles, overhead conductors, and cabinets.

Rules are deliberately conservative and every rule is a *measured* geometric
property (footprint, vertical extent, linearity, density):

* Pole        - narrow horizontal footprint, multi-metre vertical extent
* Conductor   - elevated, thin, elongated linear chain (often near poles)
* Cabinet     - compact box near the ground with dense point support
"""
from __future__ import annotations

from typing import List

import numpy as np

from ..instances import component_metrics, grid_components
from ..models import Asset
from .common import TileContext, build_asset


def detect_utilities(ctx: TileContext) -> List[Asset]:
    assets: List[Asset] = []
    settings = ctx.settings

    # ---- Poles ----------------------------------------------------------------
    # Exclude the ground surface itself (height < 0.3) so the pole stands as its
    # own component; pole points above 0.3 m still dominate the component.
    pole_mask = ctx.height >= max(0.3, settings.pole_min_height_m * 0.12)
    for component in grid_components(ctx.x, ctx.y, pole_mask, settings.pole_resolution_m, min_cells=1):
        if len(component) < settings.pole_min_points:
            continue
        metrics = component_metrics(ctx.x, ctx.y, ctx.z, component)
        footprint = max(metrics["length_m"], metrics["width_m"])
        if (
            metrics["height_m"] >= settings.pole_min_height_m
            and footprint <= settings.pole_max_footprint_m
            and metrics["columnarity"] >= settings.pole_min_columnarity
        ):
            # Geometry: how pole-like is it? Dominant vertical extent + narrow base.
            vertical_ratio = metrics["height_m"] / max(footprint, 1e-3)
            geometry_score = min(0.95, 0.45 + min(vertical_ratio / 8.0, 0.35) + metrics["verticality"] * 0.15)
            explanation = (
                f"Narrow footprint ({footprint:.2f} m) with {metrics['height_m']:.1f} m vertical "
                "extent satisfies utility-pole geometry rules."
            )
            asset = build_asset(
                ctx,
                asset_class="utility_pole",
                subclass="vertical_support",
                indices=component,
                geometry_score=geometry_score,
                explanation=explanation,
                method="geometry-v1",
                model_target_classes=("utility_pole",),
            )
            if asset is not None:
                assets.append(asset)

    # ---- Overhead conductors ---------------------------------------------------
    conductor_mask = ctx.height >= settings.conductor_min_height_m
    for component in grid_components(ctx.x, ctx.y, conductor_mask, settings.conductor_resolution_m, min_cells=3):
        if len(component) < settings.conductor_min_points:
            continue
        metrics = component_metrics(ctx.x, ctx.y, ctx.z, component)
        length = max(metrics["length_m"], metrics["width_m"])
        lateral = min(metrics["length_m"], metrics["width_m"])
        if (
            length >= settings.conductor_min_length_m
            and lateral <= settings.conductor_max_width_m
            and metrics["height_m"] <= settings.conductor_max_height_extent_m
            and metrics["linearity"] >= 0.55
        ):
            geometry_score = min(0.7, 0.35 + metrics["linearity"] * 0.25 + min(length / 50.0, 0.2))
            explanation = (
                f"Elevated thin linear chain ({length:.1f} m long, {lateral:.2f} m across) is "
                "consistent with an overhead conductor; heuristic detection."
            )
            asset = build_asset(
                ctx,
                asset_class="overhead_conductor",
                subclass="conductor",
                indices=component,
                geometry_score=geometry_score,
                explanation=explanation,
                method="geometry-v2-heuristic",
                model_target_classes=(),
            )
            if asset is not None:
                assets.append(asset)

    # ---- Cabinets ---------------------------------------------------------------
    cabinet_mask = (ctx.height >= settings.cabinet_min_height_m) & (ctx.height <= settings.cabinet_max_height_m)
    for component in grid_components(ctx.x, ctx.y, cabinet_mask, settings.cabinet_resolution_m, min_cells=2):
        if len(component) < settings.cabinet_min_points:
            continue
        metrics = component_metrics(ctx.x, ctx.y, ctx.z, component)
        side_x, side_y, height = metrics["length_m"], metrics["width_m"], metrics["height_m"]
        if (
            settings.cabinet_min_side_m <= side_x <= settings.cabinet_max_side_m
            and settings.cabinet_min_side_m <= side_y <= settings.cabinet_max_side_m
            and settings.cabinet_min_height_m <= height <= settings.cabinet_max_height_m
        ):
            boxiness = 1.0 - min(1.0, abs(side_x - side_y) / max(max(side_x, side_y), 1e-3))
            geometry_score = min(0.85, 0.45 + boxiness * 0.25 + min(metrics["point_count"] / 2000.0, 0.2))
            explanation = (
                f"Compact near-ground box ({side_x:.2f} x {side_y:.2f} x {height:.2f} m) with dense "
                "point support is consistent with a utility cabinet."
            )
            asset = build_asset(
                ctx,
                asset_class="utility_cabinet",
                subclass="cabinet",
                indices=component,
                geometry_score=geometry_score,
                explanation=explanation,
                method="geometry-v1",
                model_target_classes=("utility_cabinet",),
            )
            if asset is not None:
                assets.append(asset)
    return assets