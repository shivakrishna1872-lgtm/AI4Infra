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
    # Learned component classifier (loaded once per run; None = geometry-only).
    # Declared last because it defaults while earlier fields do not.
    classifier: Optional[object] = None
    # Infrared channel (LAS 1.4 point format 8); fused into marking brightness.
    nir: Optional[np.ndarray] = None


#: Classes whose class label is resolved at span level (merged across tiles)
#: and whose per-tile class assignment is legitimately ambiguous (a rail vs a
#: barrier by cross-section, a deck vs shoulders by extent). The learned veto
#: never applies to them: a confident "wrong" class here is fixed by the merge
#: pass, not by deleting the detection (measured on QuickSim: a guardrail tile
#: measured 0.63 m wide, was labeled safety_barrier, and the veto deleted a
#: true positive).
_VETO_EXEMPT_CLASSES = frozenset({"pavement", "pavement_marking", "guardrail", "safety_barrier"})


def _signal_alive(values: Optional[np.ndarray], base: np.ndarray) -> bool:
    """True when a radiometric channel carries real signal on the near-ground set.

    LAS exporters sometimes keep a channel's dimension but never fill it (a
    "no-color" export leaves RGB at zeros; some scans write intensity as zeros).
    An all-zero channel must never vote "bright": with floor = quantile = 0,
    every near-ground point would pass and the whole road would merge into one
    giant marking blob. A channel is alive when it has a meaningful share of
    positive values with a positive maximum.
    """
    if values is None or not len(values) or not base.any():
        return False
    v = np.asarray(values, dtype=np.float64)[base]
    v = v[np.isfinite(v)]
    if not len(v) or float(v.max()) <= 0.0:
        return False
    return float((v > 0).mean()) >= 0.01


def bright_near_ground_mask(ctx: TileContext, height_max: float) -> np.ndarray:
    """Near-ground points that are locally *bright* (reflective marking evidence).

    Brightness uses a contrast floor in addition to the relative quantile, so a
    uniformly bright surface (e.g. fresh asphalt or a sunlit ground plane) never
    turns the whole tile into a marking. Points inside vertical structures (cells
    with a large z-range, e.g. pole bases) are excluded.
    """
    settings = ctx.settings
    base = ctx.height <= height_max
    # No intensity channel, or no near-ground points at all (a tile whose points
    # all sit above ``height_max`` — elevated decks, upper structure tiles): no
    # marking evidence exists, so return an all-False mask instead of letting
    # np.quantile crash on an empty selection.
    if not len(ctx.intensity) or not base.any():
        return np.zeros(len(ctx.x), dtype=bool)
    bright = np.zeros(len(ctx.x), dtype=bool)
    # Each channel votes independently, and only when it carries real signal.
    # A channel that exists but is all zeros (no-color exports, unfilled
    # intensity) is ignored instead of marking every near-ground point bright
    # (floor = quantile = 0 would pass everything).
    #
    # Relative-contrast floor: a marking is *brighter than its surroundings*, not
    # merely in the upper tail. Without this, the bright tail of road/ground
    # reflectivity merges into real markings and pollutes their components.
    # The floor multiplier is deliberately modest (2.0x/1.6x median, configurable)
    # so the effective threshold lands at the configured quantile (0.90 = the
    # 88th-90th percentile band) instead of drifting to ~95th on bright asphalt,
    # which starved real markings (measured: 50 records on the 112.8M-point
    # Mannford corridor at the old 2.5x floor; docs/ACCURACY_REPORT.md §4a).
    if _signal_alive(ctx.intensity, base):
        intensity_floor = settings.marking_intensity_floor_multiplier * float(np.median(ctx.intensity[base]))
        intensity_quantile = float(np.quantile(ctx.intensity[base], settings.marking_intensity_quantile))
        bright |= ctx.intensity >= max(intensity_floor, intensity_quantile)
    if _signal_alive(ctx.rgb.mean(axis=1) if ctx.rgb is not None else None, base):
        brightness = ctx.rgb.mean(axis=1)
        brightness_floor = settings.marking_brightness_floor_multiplier * float(np.median(brightness[base]))
        brightness_quantile = float(np.quantile(brightness[base], settings.marking_brightness_quantile))
        bright |= brightness >= max(brightness_floor, brightness_quantile)
    if _signal_alive(ctx.nir, base):
        # Infrared (LAS 1.4 point format 8) is the strongest road-paint
        # discriminator on mobile LiDAR; same quantile/floor policy as intensity.
        nir_floor = settings.marking_intensity_floor_multiplier * float(np.median(ctx.nir[base]))
        nir_quantile = float(np.quantile(ctx.nir[base], settings.marking_intensity_quantile))
        bright |= ctx.nir >= max(nir_floor, nir_quantile)
    # Exclude points inside tall structures (pole bases, posts) via cell z-range
    from ..preprocessing import cell_z_ranges

    zrange = cell_z_ranges(ctx.x, ctx.y, ctx.z, max(settings.marking_resolution_m, 0.25))
    bright &= zrange <= 0.6
    return bright


def road_proximity_m(ctx: TileContext, indices: np.ndarray, resolution: float = 1.0) -> Optional[float]:
    """Mean distance from component points to the nearest pavement-occupied cell.

    Vectorized: a Chebyshev distance map is grown from the pavement cells over
    the tile's cell grid (ring dilation is 8-connectivity, matching the old
    per-point ring search exactly). Falls back to the per-point ring search when
    the tile extent would make the grid impractically large.
    """
    if ctx.pavement_mask is None or ctx.pavement_mask.sum() == 0:
        return None
    if len(indices) == 0:
        return None
    from ..preprocessing import occupied_cells

    pavement_cells = occupied_cells(ctx.x[ctx.pavement_mask], ctx.y[ctx.pavement_mask], resolution)
    px = np.floor(ctx.x[indices] / resolution).astype(np.int64)
    py = np.floor(ctx.y[indices] / resolution).astype(np.int64)
    max_search = 12
    origin_x = int(np.floor(float(ctx.x.min()) / resolution))
    origin_y = int(np.floor(float(ctx.y.min()) / resolution))
    span_x = int(np.floor(float(ctx.x.max()) / resolution)) - origin_x + 1
    span_y = int(np.floor(float(ctx.y.max()) / resolution)) - origin_y + 1
    if span_x * span_y <= 4_000_000 and span_x > 0 and span_y > 0:
        grid = np.zeros((span_y, span_x), dtype=bool)
        for cx, cy in pavement_cells:
            gx, gy = int(cx) - origin_x, int(cy) - origin_y
            if 0 <= gx < span_x and 0 <= gy < span_y:
                grid[gy, gx] = True
        distances_map = np.full((span_y, span_x), float(max_search), dtype=np.float64)
        frontier = grid.copy()
        seen = grid.copy()
        for ring in range(max_search + 1):
            distances_map[frontier] = float(ring)
            if bool(seen.all()):
                break
            padded = np.pad(frontier, 1)
            dilated = (
                padded[1:-1, 1:-1] | padded[:-2, 1:-1] | padded[2:, 1:-1]
                | padded[1:-1, :-2] | padded[1:-1, 2:]
                | padded[:-2, :-2] | padded[:-2, 2:] | padded[2:, :-2] | padded[2:, 2:]
            )
            frontier = dilated & ~seen
            seen |= dilated
        gx = (px - origin_x).astype(np.int64)
        gy = (py - origin_y).astype(np.int64)
        inside = (gx >= 0) & (gx < span_x) & (gy >= 0) & (gy < span_y)
        distances = np.full(len(px), float(max_search), dtype=np.float64)
        distances[inside] = distances_map[gy[inside], gx[inside]]
        return float(distances.mean())
    # Fallback: per-point ring-by-ring Chebyshev search.
    distances = np.full(len(px), float(max_search), dtype=np.float64)
    for index, (cx, cy) in enumerate(zip(px.tolist(), py.tolist())):
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
    min_points: Optional[int] = None,
) -> Optional[Asset]:
    """Turn a detected component into a fully attributed Asset.

    ``min_points`` defaults to the global ``min_asset_points`` floor; detectors
    with a class-specific support gate (e.g. ``conductor_min_points=15`` vs the
    global 25) pass their own minimum so a fragment that passed every detector
    rule is not silently dropped here. Dropping 15-24-point wire fragments
    stripped the ends of spans and biased their reported centroids.
    """
    if len(indices) < (min_points if min_points is not None else ctx.settings.min_asset_points):
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
    prior_class_name: Optional[str] = None
    if ctx.model_class is not None and model_target_classes:
        model_factor = model_factor_from_prior(
            ctx.model_class[mask], ctx.class_names, ctx.mapping, model_target_classes
        )
    # Training-data collection: record every candidate component's features
    # with the detector's label (relabeled against ground truth by the training
    # script). Independent of the classifier so collection works pre-training.
    if getattr(ctx.settings, "collect_training", False):
        from ..learned import collect_component_example
        collect_component_example(ctx, asset_class, mask, ctx.ground)
    # Learned component classifier (CPU prior): scores this component's measured
    # features against its assigned class. Fills the `model` factor when no
    # Pointcept/OpenPCSeg prior is present, and can veto candidates the trained
    # model confidently rejects (the geometry gates still ran first).
    if ctx.classifier is not None:
        from ..learned import component_features, learned_prior_factor
        features, _extras = component_features(
            ctx.x, ctx.y, ctx.z, mask, ctx.intensity, ctx.rgb, ctx.ground,
        )
        scores = ctx.classifier.scores(features)
        factor, learned_name = learned_prior_factor(
            ctx.classifier, asset_class, features, scores=scores,
        )
        if model_factor is None:
            model_factor = factor
            if learned_name is not None:
                # Provenance: the classifier's own best class; `*` marks agreement.
                prior_class_name = f"{learned_name}*" if learned_name == asset_class else learned_name
        s = ctx.settings
        # Veto purely on the trained model's own scores: the classifier rejects
        # this component's assigned class with high confidence AND a concrete
        # alternative. Independent of which prior filled `model_factor`, so a
        # strong Pointcept prior can never mask a learned rejection.
        #
        # Area/linear classes are exempt: a guardrail fragment can legitimately
        # measure wider than its neighbours (merged with a sign or pole
        # footprint) and be labeled safety_barrier while the rest of the span
        # is guardrail - the span-level merge resolves that, so vetoing the
        # fragment deletes a true positive. Compact classes keep the veto.
        if (
            s.learned_veto
            and asset_class not in _VETO_EXEMPT_CLASSES
            and scores.get(asset_class, 0.0) < s.learned_veto_margin
            and max((p for c, p in scores.items() if c != asset_class), default=0.0)
            >= s.learned_veto_runner_margin
        ):
            return None  # vetoed: the trained model confidently rejects this class
        if getattr(s, "collect_training", False):
            collect_component_example(ctx, asset_class, mask, ctx.ground)
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
    if prior_class_name is None and ctx.model_class is not None and ctx.class_names and model_factor is not None:
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
        processing_version="0.3.0",
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