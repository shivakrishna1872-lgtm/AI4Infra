"""Safety asset detection: guardrails, concrete barriers, and rumble strips.

All three are roadside structures; the difference is measured geometry:

* Guardrail      - long, low, narrow; small vertical extent (W-beam)
* Safety barrier - long, wider cross-section (concrete barrier / Jersey barrier)
* Rumble strip   - short bright transverse bands repeated at regular spacing
"""
from __future__ import annotations

import math
from typing import List, Optional

import numpy as np

from ..instances import component_metrics, grid_components
from ..models import Asset
from .common import TileContext, bright_near_ground_mask, build_asset


def detect_safety(ctx: TileContext) -> List[Asset]:
    assets: List[Asset] = []
    settings = ctx.settings

    # ---- Guardrails + barriers (shared candidate mask) -------------------------
    rail_mask = (ctx.height >= settings.guardrail_min_height_m) & (ctx.height <= settings.guardrail_max_height_m)
    for component in grid_components(ctx.x, ctx.y, rail_mask, settings.guardrail_resolution_m, min_cells=5):
        if len(component) < settings.guardrail_min_points:
            continue
        metrics = component_metrics(ctx.x, ctx.y, ctx.z, component)
        length = max(metrics["length_m"], metrics["width_m"])
        lateral = min(metrics["length_m"], metrics["width_m"])

        if (
            length >= settings.guardrail_min_length_m
            and lateral <= settings.guardrail_max_width_m
            and metrics["height_m"] <= settings.guardrail_max_height_extent_m
        ):
            # Guardrail vs barrier: barrier has a wider cross-section
            if lateral >= settings.barrier_min_width_m:
                assets.append(_barrier_asset(ctx, component, metrics, length, lateral))
            else:
                assets.append(_guardrail_asset(ctx, component, metrics, length, lateral))

    # ---- Rumble strips -----------------------------------------------------------
    assets.extend(_rumble_strips(ctx))
    return assets


def _guardrail_asset(ctx: TileContext, component: np.ndarray, metrics: dict, length: float, lateral: float) -> Optional[Asset]:
    settings = ctx.settings
    straightness = metrics["linearity"]
    elevation_band = metrics["centroid_z"] - ctx.ground
    geometry_score = min(0.92, 0.5 + straightness * 0.3 + min(elevation_band / 1.0, 0.15))
    explanation = (
        f"Long ({length:.1f} m), low, narrow ({lateral:.2f} m) roadside structure with "
        f"straightness {straightness:.2f} matches W-beam guardrail geometry."
    )
    return build_asset(
        ctx,
        asset_class="guardrail",
        subclass="roadside_barrier",
        indices=component,
        geometry_score=geometry_score,
        explanation=explanation,
        method="geometry-v1",
        model_target_classes=("safety_barrier",),
    )


def _barrier_asset(ctx: TileContext, component: np.ndarray, metrics: dict, length: float, lateral: float) -> Optional[Asset]:
    settings = ctx.settings
    if (
        length < settings.barrier_min_length_m
        or lateral > settings.barrier_max_width_m
        or metrics["height_m"] < settings.barrier_min_height_m
        or metrics["height_m"] > settings.barrier_max_height_m
    ):
        return None
    geometry_score = min(0.9, 0.5 + metrics["linearity"] * 0.25 + min(lateral / 0.8, 0.2))
    explanation = (
        f"Continuous ({length:.1f} m) roadside structure with wide cross-section "
        f"({lateral:.2f} m) matches concrete/Jersey barrier geometry."
    )
    return build_asset(
        ctx,
        asset_class="safety_barrier",
        subclass="concrete_barrier",
        indices=component,
        geometry_score=geometry_score,
        explanation=explanation,
        method="geometry-v1",
        model_target_classes=("safety_barrier",),
    )


def _rumble_strips(ctx: TileContext) -> List[Asset]:
    """Repeated short bright transverse bands -> one rumble-strip asset."""
    settings = ctx.settings
    if settings.marking_height_m <= 0:
        return []
    mask = bright_near_ground_mask(ctx, settings.marking_height_m)
    bands: List[dict] = []
    for component in grid_components(ctx.x, ctx.y, mask, settings.marking_resolution_m, min_cells=2):
        if len(component) < settings.rumble_min_band_points:
            continue
        metrics = component_metrics(ctx.x, ctx.y, ctx.z, component)
        length = max(metrics["length_m"], metrics["width_m"])
        width = min(metrics["length_m"], metrics["width_m"])
        if (
            settings.rumble_min_length_m <= length <= settings.rumble_max_length_m
            and settings.rumble_min_width_m <= width <= settings.rumble_max_width_m
        ):
            bands.append({
                "indices": component,
                "orientation": metrics["orientation_deg"],
                "cx": metrics["centroid_x"],
                "cy": metrics["centroid_y"],
                "length": length,
            })
    assets: List[Asset] = []
    for group in _group_bands(bands, settings.rumble_max_gap_m):
        if len(group) < settings.rumble_min_bands:
            continue
        merged = np.concatenate([band["indices"] for band in group])
        geometry_score = min(0.65, 0.35 + 0.05 * len(group))
        gap = settings.rumble_max_gap_m
        explanation = (
            f"{len(group)} short bright transverse bands in a repeating pattern "
            f"(spacing <= {gap:.1f} m) are consistent with rumble strips; heuristic detection."
        )
        asset = build_asset(
            ctx,
            asset_class="rumble_strip",
            subclass="rumble_strip",
            indices=merged,
            geometry_score=geometry_score,
            explanation=explanation,
            method="geometry-v2-heuristic",
            model_target_classes=(),
        )
        if asset is not None:
            assets.append(asset)
    return assets


def _group_bands(bands: List[dict], max_gap_m: float) -> List[List[dict]]:
    """Group bands whose centroids lie within ``max_gap_m`` of a consecutive band."""
    remaining = list(bands)
    groups: List[List[dict]] = []
    while remaining:
        seed = remaining.pop(0)
        group = [seed]
        changed = True
        while changed:
            changed = False
            for band in list(remaining):
                if any(math.hypot(band["cx"] - member["cx"], band["cy"] - member["cy"]) <= max_gap_m for member in group):
                    group.append(band)
                    _remove_by_identity(remaining, band)
                    changed = True
        groups.append(group)
    return groups


def _remove_by_identity(items: List[dict], target: dict) -> None:
    """Remove ``target`` by object identity.

    Band dicts carry numpy arrays (``indices``); ``list.remove`` compares with
    ``==``, which makes numpy raise a broadcast error when two bands have
    different point counts (seen on real airborne tiles, e.g. Burnet County
    USGS 3DEP).
    """
    for index, item in enumerate(items):
        if item is target:
            items.pop(index)
            return