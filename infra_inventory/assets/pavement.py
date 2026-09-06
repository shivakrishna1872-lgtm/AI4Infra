"""Pavement asset detection: travelled surface + painted pavement markings.

Marking subclasses are decided from measured shape (aspect ratio, area):

* lane_line / edge_line  - long, thin (length >= lane_line_min_length_m)
* stop_line              - short band spanning the travel direction (width >= stop_line_min_width_m)
* crosswalk              - very wide band (width >= crosswalk_min_width_m)
* symbol                 - compact (both sides <= symbol_max_side_m)
* other                  - anything in between
"""
from __future__ import annotations

import math
from typing import List, Optional

import numpy as np

from ..instances import component_metrics, grid_components
from ..models import Asset
from .common import TileContext, bright_near_ground_mask, build_asset


def detect_pavement(ctx: TileContext) -> List[Asset]:
    settings = ctx.settings
    assets: List[Asset] = []
    ground = ctx.ground

    # ---- 1. Travelled surface -------------------------------------------------
    pavement_mask = ctx.height <= settings.pavement_height_m
    pavement_count = int(pavement_mask.sum())
    if pavement_count >= settings.min_pavement_points:
        geometry_score = _pavement_geometry_score(ctx, pavement_mask)
        asset = build_asset(
            ctx,
            asset_class="pavement",
            subclass="travelled_surface",
            indices=np.flatnonzero(pavement_mask),
            geometry_score=geometry_score,
            explanation=(
                "Low-elevation surface with continuous point support; geometry from "
                "elevation distribution and local density within this spatial tile."
            ),
            method="geometry-v1",
            model_target_classes=("pavement",),
        )
        if asset is not None:
            assets.append(asset)
        # Record pavement occupancy for road-proximity context of roadside classes
        ctx.pavement_mask = pavement_mask

    # ---- 2. Painted markings --------------------------------------------------
    marking_mask = bright_near_ground_mask(ctx, settings.marking_height_m)
    for component in grid_components(ctx.x, ctx.y, marking_mask, settings.marking_resolution_m, min_cells=3):
        if len(component) < settings.marking_min_points:
            continue
        asset = _marking_asset(ctx, component)
        if asset is not None:
            assets.append(asset)
    return assets


def _pavement_geometry_score(ctx: TileContext, mask: np.ndarray) -> float:
    """Pavement geometry quality: elevation flatness + density coverage."""
    settings = ctx.settings
    heights = ctx.height[mask]
    flatness = 1.0 - min(1.0, float(np.std(heights)) / max(settings.pavement_height_m, 1e-3))
    cell_count = float(len(set(zip(np.floor(ctx.x[mask] / 0.5).astype(np.int64).tolist(), np.floor(ctx.y[mask] / 0.5).astype(np.int64).tolist()))))
    expected_cells = (settings.tile_size_m / 0.5) ** 2
    coverage = min(1.0, cell_count / max(expected_cells, 1.0))
    return round(0.55 * flatness + 0.45 * coverage, 3)


def _marking_asset(ctx: TileContext, component: np.ndarray) -> Optional[Asset]:
    settings = ctx.settings
    metrics = component_metrics(ctx.x, ctx.y, ctx.z, component)
    length = max(metrics["length_m"], metrics["width_m"])
    width = min(metrics["length_m"], metrics["width_m"])
    area = metrics["length_m"] * metrics["width_m"]

    subclass: str
    if length >= settings.lane_line_min_length_m and width <= settings.lane_line_max_width_m:
        subclass = "lane_line"
    elif metrics["width_m"] >= settings.crosswalk_min_width_m:
        subclass = "crosswalk"
    elif metrics["width_m"] >= settings.stop_line_min_width_m:
        subclass = "stop_line"
    elif max(metrics["length_m"], metrics["width_m"]) <= settings.symbol_max_side_m:
        subclass = "symbol"
    else:
        subclass = "other"
    # Sanity caps: a real marking is bounded; the whole ground plane is not a marking.
    if subclass in ("crosswalk", "stop_line") and metrics["length_m"] > 15.0:
        return None
    if subclass == "other" and (metrics["length_m"] > 30.0 or metrics["width_m"] > 10.0):
        return None

    # Geometry quality: shape regularity (aspect-consistent) + linearity for lines
    aspect = (length / max(width, 1e-3)) if width > 0 else 0.0
    shape_score = 1.0 - min(1.0, abs(math.log10(aspect + 1.0) - _expected_aspect_log(subclass)) / 2.0) if subclass != "other" else 0.6
    geometry_score = round(0.6 * max(0.0, min(1.0, shape_score)) + 0.4 * metrics["linearity"], 3)
    explanation = (
        f"Near-ground connected component with high reflectance/brightness; "
        f"shape classified as {subclass.replace('_', ' ')} (measured {length:.1f} x {width:.2f} m)."
    )
    return build_asset(
        ctx,
        asset_class="pavement_marking",
        subclass=subclass,
        indices=component,
        geometry_score=geometry_score,
        explanation=explanation,
        method="native-roadmarking-v1",
        model_target_classes=("pavement",),
    )


def _expected_aspect_log(subclass: str) -> float:
    # log10(aspect + 1) expectations per subclass
    return {
        "lane_line": math.log10(6 + 1),
        "crosswalk": math.log10(2 + 1),
        "stop_line": math.log10(2 + 1),
        "symbol": math.log10(1 + 1),
        "other": 0.0,
    }.get(subclass, 0.0)