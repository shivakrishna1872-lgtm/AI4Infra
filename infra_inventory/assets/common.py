"""Shared types and the asset builder for the detector modules.

Kept in its own module so detector packages can import it without circular
imports through ``assets/__init__``.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from ..attribution import (
    intensity_stats,
    orientation_deg,
    rgb_stats,
    run_label,
    sample_indices,
    scanner_label,
)
from ..confidence import (
    class_consistency,
    model_factor_from_prior,
    score_asset,
    spatial_context,
    support_score,
)
from ..instances import compactness, component_metrics
from ..models import Asset, ConfidenceFactors, ProcessingSettings, VALID_TAXONOMY


@dataclass
class TileContext:
    """Everything a detector needs about one spatial tile."""

    tile: str
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    height: np.ndarray  # above estimated ground
    ground: float
    intensity: np.ndarray
    rgb: Optional[np.ndarray]
    source_indices: np.ndarray  # tile-local point indices
    gps_labels: Optional[np.ndarray]  # per-point run label (1/2) or None
    point_source_id: Optional[np.ndarray]
    crs: Optional[str]
    model_class: Optional[np.ndarray]  # upstream class id per point, -1 = none
    class_names: List[str]
    mapping: Dict[str, Dict[str, object]]
    settings: ProcessingSettings
    id_counts: Counter
    sample_stride: int  # viewer sampling stride (global)
    viewer_limit: int
    pavement_mask: Optional[np.ndarray] = None


def bright_near_ground_mask(ctx: TileContext, height_max: float) -> np.ndarray:
    """Near-ground points that are locally *bright* (reflective marking evidence).

    Brightness uses a contrast floor in addition to the relative quantile, so a
    uniformly bright surface (e.g. fresh asphalt or a sunlit ground plane) never
    turns the whole tile into a marking. Points inside vertical structures (cells
    with a large z-range, e.g. pole bases) are excluded.
    """
    settings = ctx.settings
    base = ctx.height <= height_max
    if not len(ctx.intensity):
        return np.zeros(len(ctx.x), dtype=bool)
    intensity_floor = 1.5 * float(np.median(ctx.intensity[base])) if base.any() else 0.0
    intensity_quantile = float(np.quantile(ctx.intensity[base], settings.marking_intensity_quantile))
    bright = ctx.intensity >= max(intensity_floor, intensity_quantile)
    if ctx.rgb is not None and base.any():
        brightness = ctx.rgb.mean(axis=1)
        brightness_floor = 1.3 * float(np.median(brightness[base]))
        brightness_quantile = float(np.quantile(brightness[base], settings.marking_brightness_quantile))
        bright |= brightness >= max(brightness_floor, brightness_quantile)
    # Exclude points inside tall structures (pole bases, posts) via cell z-range
    from ..preprocessing import cell_z_ranges

    zrange = cell_z_ranges(ctx.x, ctx.y, ctx.z, max(settings.marking_resolution_m, 0.25))
    bright &= zrange <= 0.6
    return bright


def road_proximity_m(ctx: TileContext, indices: np.ndarray, resolution: float = 1.0) -> Optional[float]:
    """Mean distance from component points to the nearest pavement-occupied cell."""
    if ctx.pavement_mask is None or ctx.pavement_mask.sum() == 0:
        return None
    if len(indices) == 0:
        return None
    from ..preprocessing import occupied_cells

    pavement_cells = occupied_cells(ctx.x[ctx.pavement_mask], ctx.y[ctx.pavement_mask], resolution)
    px = np.floor(ctx.x[indices] / resolution).astype(np.int64)
    py = np.floor(ctx.y[indices] / resolution).astype(np.int64)
    max_search = 12
    distances = np.full(len(px), float(max_search), dtype=np.float64)
    for index, (cx, cy) in enumerate(zip(px.tolist(), py.tolist())):
        # Ring-by-ring Chebyshev search: the first matching ring is the nearest.
        for ring in range(max_search + 1):
            if _cell_in_ring(pavement_cells, int(cx), int(cy), ring):
                distances[index] = float(ring)
                break
    return float(distances.mean())


def _cell_in_ring(cells: set, cx: int, cy: int, ring: int) -> bool:
    """Whether any occupied cell lies at Chebyshev distance exactly ``ring``."""
    if ring == 0:
        return (cx, cy) in cells
    for offset in range(-ring, ring + 1):
        if (cx + ring, cy + offset) in cells or (cx - ring, cy + offset) in cells:
            return True
        if (cx + offset, cy + ring) in cells or (cx + offset, cy - ring) in cells:
            return True
    return False


def build_asset(
    ctx: TileContext,
    *,
    asset_class: str,
    subclass: Optional[str],
    indices: np.ndarray,
    geometry_score: float,
    explanation: str,
    method: str,
    model_target_classes: tuple[str, ...] = (),
) -> Optional[Asset]:
    """Turn a detected component into a fully attributed Asset."""
    if len(indices) < ctx.settings.min_asset_points:
        return None
    assert asset_class in VALID_TAXONOMY, f"Unknown asset class {asset_class}"
    if subclass is not None and subclass not in VALID_TAXONOMY[asset_class]:
        raise ValueError(f"Subclass {subclass!r} not valid for {asset_class}")

    mask = indices
    metrics = component_metrics(ctx.x, ctx.y, ctx.z, mask)
    box = metrics["bounds"]
    dimensions = {
        "length_m": round(metrics["length_m"], 3),
        "width_m": round(metrics["width_m"], 3),
        "height_m": round(metrics["height_m"], 3),
    }
    model_factor: Optional[float] = None
    if ctx.model_class is not None and model_target_classes:
        model_factor = model_factor_from_prior(
            ctx.model_class[mask], ctx.class_names, ctx.mapping, model_target_classes
        )
    support = support_score(len(mask), ctx.settings.min_asset_points, ctx.settings.max_support_points)
    consistency = class_consistency(asset_class, ctx.intensity[mask], ctx.rgb[mask] if ctx.rgb is not None else None)
    proximity = road_proximity_m(ctx, mask) if asset_class in ("guardrail", "safety_barrier", "pavement_marking", "rumble_strip", "pavement", "traffic_sign") else None
    compact = compactness(metrics)
    context = spatial_context(asset_class, proximity, compact)
    factors = ConfidenceFactors(
        model=model_factor,
        geometry=round(max(0.0, min(1.0, geometry_score)), 3),
        support=round(support, 3),
        spatial_context=context,
        class_consistency=consistency,
    )
    confidence, confidence_explanation = score_asset(asset_class, factors)

    source_id, scanner = scanner_label(ctx.point_source_id, mask)
    run = run_label(ctx.gps_labels, mask)
    prior_class_name: Optional[str] = None
    if ctx.model_class is not None and ctx.class_names and model_factor is not None:
        ids = ctx.model_class[mask]
        present = [ctx.class_names[i] for i in np.unique(ids) if 0 <= i < len(ctx.class_names)]
        if present:
            prior_class_name = present[0]

    return Asset(
        asset_id="",  # assigned by the pipeline from id_counts
        asset_class=asset_class,
        subclass=subclass,
        center={
            "x": round(float(metrics["centroid_x"]), 4),
            "y": round(float(metrics["centroid_y"]), 4),
            "z": round(float(metrics["centroid_z"]), 4),
        },
        bounding_box=tuple(round(float(value), 4) for value in box),
        dimensions=dimensions,
        point_count=int(len(mask)),
        source_tile=ctx.tile,
        source_point_indices_sample=sample_indices(ctx.source_indices[mask], ctx.settings.point_index_sample_limit),
        coordinate_reference_system=ctx.crs,
        confidence=confidence,
        confidence_factors=factors.to_dict(),
        confidence_explanation=confidence_explanation,
        detection_method=method,
        intensity_stats=intensity_stats(ctx.intensity[mask]),
        rgb_stats=rgb_stats(ctx.rgb[mask] if ctx.rgb is not None else None),
        orientation_deg=orientation_deg(ctx.x[mask], ctx.y[mask]),
        source_run=run,
        source_scanner=scanner,
        source_point_source_id=source_id,
        model_prior_class=prior_class_name,
        model_confidence=model_factor,
        processing_version="0.2.0",
        geometry={
            "ground_elevation_m": round(ctx.ground, 4),
            "eigen_planarity": round(metrics["planarity"], 3),
            "eigen_linearity": round(metrics["linearity"], 3),
            "eigen_verticality": round(metrics["verticality"], 3),
        },
    )


def assign_ids(asset: Asset, ctx: TileContext) -> Asset:
    from ..models import CLASS_PREFIXES

    prefix = CLASS_PREFIXES[asset.asset_class]
    ctx.id_counts[prefix] += 1
    asset.asset_id = f"{prefix}-{ctx.id_counts[prefix]:05d}"
    return asset


def geometry_score_from_ratio(measured: float, ideal: float, tolerance: float) -> float:
    """Similarity of a measured value to an ideal, 0..1 (Gaussian-ish falloff)."""
    if ideal <= 0:
        return 0.0
    return float(max(0.0, min(1.0, math.exp(-0.5 * ((measured - ideal) / max(tolerance, 1e-6)) ** 2))))