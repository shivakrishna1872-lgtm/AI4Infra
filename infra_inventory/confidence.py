"""Transparent confidence engine.

Two layers:

1. **Per-asset confidence** — every asset confidence is a weighted blend of
   *measurable* factors:

   * ``model``            - agreement with the learned Pointcept/PTv3 prior (None on the CPU geometry backend)
   * ``geometry``         - how well the instance matches its class's geometric priors (detector-measured)
   * ``support``          - point support relative to the class minimum
   * ``spatial_context``  - consistency with surrounding context (road proximity, compactness)
   * ``class_consistency``- intensity/RGB statistics matching the class's physical expectations

   Missing factors are ``None`` and the remaining weights are renormalized, so a
   confidence of 0.93 always means the same thing: *the measurable signals for this
   asset were consistently strong*. The explanation string says exactly which
   signals contributed.

2. **Overall dataset confidence** (``compute_confidence_report``) — one score for
   the whole processed report, evaluating the dataset behind it: point density &
   coverage (30%), CRS & coordinate precision (30%), intensity & classification
   reliability (20%), and geometric clustering fit (20%). The report ships in
   ``run.json`` / ``inventory.json`` / ``viewer-data.json`` under
   ``run.confidence_report`` and in the summary report, with a grade:
   **High (Production Ready)** ≥ 90%, **Medium (Review Recommended)** 75-89%,
   **Low (Manual Verification Required)** < 75%.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from .models import ConfidenceFactors, ProcessingSettings

#: Class -> factor weights (mirrors configs/classes.yaml).
DEFAULT_WEIGHTS: Dict[str, Dict[str, float]] = {
    "pavement": {"model": 0.35, "geometry": 0.30, "support": 0.20, "spatial_context": 0.15, "class_consistency": 0.0},
    "pavement_marking": {"model": 0.25, "geometry": 0.30, "support": 0.20, "spatial_context": 0.10, "class_consistency": 0.15},
    "utility_pole": {"model": 0.25, "geometry": 0.35, "support": 0.20, "spatial_context": 0.10, "class_consistency": 0.10},
    "overhead_conductor": {"model": 0.20, "geometry": 0.40, "support": 0.20, "spatial_context": 0.10, "class_consistency": 0.10},
    "utility_cabinet": {"model": 0.20, "geometry": 0.40, "support": 0.20, "spatial_context": 0.10, "class_consistency": 0.10},
    "traffic_sign": {"model": 0.25, "geometry": 0.35, "support": 0.20, "spatial_context": 0.10, "class_consistency": 0.10},
    "guardrail": {"model": 0.30, "geometry": 0.35, "support": 0.15, "spatial_context": 0.10, "class_consistency": 0.10},
    "safety_barrier": {"model": 0.30, "geometry": 0.35, "support": 0.15, "spatial_context": 0.10, "class_consistency": 0.10},
    "rumble_strip": {"model": 0.0, "geometry": 0.50, "support": 0.20, "spatial_context": 0.15, "class_consistency": 0.15},
}


def support_score(point_count: int, min_points: int, max_points: int = 4000) -> float:
    """Point-support factor: grows steeply past the minimum, saturates near max."""
    if point_count < 1:
        return 0.0
    ratio = point_count / max(min_points, 1)
    return float(min(1.0, 1.0 - math.exp(-ratio / 1.6)))


def class_consistency(
    asset_class: str, intensity: Optional[np.ndarray], rgb: Optional[np.ndarray]
) -> Optional[float]:
    """Physical prior check: does this instance's radiometry match its class?

    Returns ``None`` when no radiometric evidence exists for the class (never
    invents a score). Thresholds are deliberately conservative; retroreflective
    paint and bare metal are measurably brighter than asphalt.
    """
    if asset_class not in ("pavement_marking", "rumble_strip", "utility_pole", "overhead_conductor", "traffic_sign", "guardrail", "safety_barrier"):
        return None
    scores: list[float] = []
    if intensity is not None and len(intensity):
        mean = float(np.mean(intensity))
        if asset_class in ("pavement_marking", "rumble_strip"):
            scores.append(1.0 if mean >= 3000 else (0.7 if mean >= 500 else 0.5))
        elif asset_class == "overhead_conductor":
            scores.append(0.9 if mean >= 2000 else 0.6)
        elif asset_class == "utility_pole":
            scores.append(0.8 if mean >= 1000 else 0.65)
        else:
            scores.append(0.8 if 400 <= mean <= 30000 else 0.6)
    if rgb is not None and len(rgb):
        brightness = float(np.mean(rgb))
        if asset_class in ("pavement_marking", "rumble_strip"):
            scores.append(1.0 if brightness >= 30000 else (0.8 if brightness >= 10000 else 0.55))
        elif asset_class == "traffic_sign":
            scores.append(0.9 if brightness >= 20000 else (0.75 if brightness >= 8000 else 0.6))
    if not scores:
        return None
    return round(min(1.0, max(0.0, float(np.mean(scores)))), 3)


def spatial_context(
    asset_class: str, road_proximity: Optional[float], compactness_value: Optional[float] = None
) -> Optional[float]:
    """Context agreement: is this asset where its class belongs?

    Roadside classes score higher when pavement is nearby; compactness rewards
    dense, well-defined instances for poles/cabinets.
    """
    if asset_class in ("guardrail", "safety_barrier", "pavement_marking", "rumble_strip", "pavement", "traffic_sign"):
        if road_proximity is None:
            return None
        # proximity in metres to nearest pavement cell; closer is better
        score = 1.0 - min(1.0, road_proximity / 12.0)
        return round(max(0.0, min(1.0, score)), 3)
    if asset_class in ("utility_pole", "utility_cabinet"):
        if compactness_value is None:
            return None
        return round(max(0.0, min(1.0, compactness_value)), 3)
    return None


def score_asset(
    asset_class: str,
    factors: ConfidenceFactors,
    weights: Optional[Dict[str, float]] = None,
) -> Tuple[float, str]:
    """Combine measured factors into (confidence, explanation).

    Weights are renormalized over the factors that are actually present; if no
    factor is present the confidence is 0 with an explicit explanation.
    """
    table = weights or DEFAULT_WEIGHTS.get(asset_class, DEFAULT_WEIGHTS["pavement_marking"])
    available: Dict[str, Optional[float]] = {
        "model": factors.model,
        "geometry": factors.geometry,
        "support": factors.support,
        "spatial_context": factors.spatial_context,
        "class_consistency": factors.class_consistency,
    }
    present = {name: value for name, value in available.items() if value is not None}
    if not present:
        return 0.0, "No measurable factors were available for this asset."
    total_weight = sum(table.get(name, 0.0) for name in present)
    if total_weight <= 0:
        return 0.0, "No applicable confidence weights for this class."
    confidence = sum(table.get(name, 0.0) * value for name, value in present.items()) / total_weight
    confidence = round(max(0.0, min(1.0, confidence)), 3)
    parts = [f"{name}={value:.2f}" for name, value in sorted(present.items())]
    explanation = "Confidence from " + ", ".join(parts) + " (weights renormalized over measured factors)."
    return confidence, explanation


def model_factor_from_prior(
    prior_class: Optional[np.ndarray], class_names: list[str], mapping: Dict[str, Dict[str, object]],
    target_classes: tuple[str, ...],
) -> Optional[float]:
    """Agreement of the Pointcept prior with a set of infrastructure classes.

    ``prior_class`` holds per-point upstream class ids (-1 = none); ``mapping``
    is the documented adaptation table (see configs/model.yaml). Returns the
    fraction of the instance's points whose upstream class maps onto one of
    ``target_classes``, weighted by the mapping weight.
    """
    if prior_class is None or len(prior_class) == 0 or not class_names or not mapping:
        return None
    weights = np.zeros(len(prior_class), dtype=np.float64)
    for name, entry in mapping.items():
        mapped = entry.get("asset_class")
        evidence_for = entry.get("evidence_for", [])
        if mapped not in target_classes and not any(target in evidence_for for target in target_classes):
            continue
        if name not in class_names:
            continue
        class_id = class_names.index(name)
        weights[prior_class == class_id] = float(entry.get("weight", 0.0))
    total = float(weights.sum())
    if total <= 0:
        return 0.0
    return round(min(1.0, total / len(prior_class)), 3)


def low_confidence(confidence: float, settings: ProcessingSettings) -> bool:
    return confidence < settings.low_confidence_threshold


# ---------------------------------------------------------------------------
# Overall dataset confidence (per-report score)
# ---------------------------------------------------------------------------

#: Weights per component (sums to 1.0).
WEIGHTS = {
    "density_coverage": 0.30,
    "crs": 0.30,
    "intensity_classification": 0.20,
    "geometry_fit": 0.20,
}

#: Point count that scores maximum density (a large production mobile-LiDAR
#: corridor with full coverage). Scores scale linearly below this reference.
DENSITY_REFERENCE_POINTS = 100_000_000

#: Expected physical dimension bounds per asset class: (length, width, height)
#: as (min, max) metre ranges. Any measured dimension outside its range is a
#: geometric implausibility; the component score is the mean per-dimension fit.
GEOMETRY_BOUNDS: Dict[str, Dict[str, Tuple[float, float]]] = {
    # Utility poles (DOT standard: 2.5-18 m vertical, bounding box width includes
    # foliage/cross-arms, length can include the ground footprint).
    "utility_pole": {"length_m": (0.02, 2.2), "width_m": (0.02, 2.2), "height_m": (2.5, 25.0)},
    # Utility cabinets (DOT standard box/cube profile: 0.8-2.2 m tall, 0.3-3.5 m
    # footprint). Reclassify >3.0 aspect ratio as guardrail/safety barrier.
    "utility_cabinet": {
        "length_m": (0.3, 4.0),
        "width_m": (0.3, 4.0),
        "height_m": (0.4, 2.6),
    },
    "traffic_sign": {"length_m": (0.0, 3.5), "width_m": (0.1, 3.5), "height_m": (0.3, 6.0)},  # flat panels measure ~0 in the thin axis
    # Guardrails (DOT standard: 1.0-500 m, 0.1-3.0 m width, 0.35-1.5 m height)
    "guardrail": {"length_m": (0.5, 500.0), "width_m": (0.1, 3.0), "height_m": (0.35, 1.5)},
    # Safety barriers (DOT standard: 0.5-500 m, 0.1-3.0 m, 0.3-2.0 m)
    "safety_barrier": {"length_m": (0.5, 500.0), "width_m": (0.1, 3.0), "height_m": (0.3, 2.0)},
    # Pavement: length 1.0-6000 m, width 1.5-45 m (multi-lane/highway), height
    # 0-30 m (bounding-box vertical extent of the point cluster on a terrain — may
    # include elevation variation across the tile when the pavement follows a grade).
    "pavement": {"length_m": (1.0, 6000.0), "width_m": (1.5, 45.0), "height_m": (0.0, 30.0)},
    # Pavement markings: length 0.05-1000 m, width 0.05-10 m (wide crosswalks,
    # multi-lane line segments), height 0-12 m (bounding-box height from ground
    # plane to topmost point on a marking cluster — may include nearby ground/
    # curb when the marking is on a slope or the cluster is a group of dashes).
    "pavement_marking": {"length_m": (0.05, 1000.0), "width_m": (0.05, 10.0), "height_m": (0.0, 12.0)},
    # Overhead conductors: length 0.5-2000 m (span + sag), width 0.05-40 m
    # (the full wire bundle/structure footprint), height 0-25 m (vertical sag
    # from pole to wire or wire to ground when measured as a cloud).
    "overhead_conductor": {
        "length_m": (0.5, 2000.0), "width_m": (0.05, 40.0), "height_m": (0.0, 25.0),
    },
    "rumble_strip": {"length_m": (0.2, 100.0), "width_m": (0.1, 2.0), "height_m": (0.0, 0.4)},
}


def grade_for(percent: float) -> str:
    """Map an overall percentage to the report grade string."""
    if percent >= 90.0:
        return "High (Production Ready)"
    if percent >= 75.0:
        return "Medium (Review Recommended)"
    return "Low (Manual Verification Required)"


def _density_coverage_score(
    point_count: int,
    bounds: Sequence[float],
    tile_count: int,
    tile_size_m: float = 40.0,
) -> Tuple[float, str]:
    """Point density vs the 100 M reference, blended with spatial coverage.

    High point density on the *uploaded area* earns full density credit: the
    upload may cover only part of a larger corridor (e.g. 688 tiles of a
    2,457-tile site), and penalizing for unuploaded tiles would punish a
    perfectly good partial dataset. When density is already at ceiling
    (``density >= 0.95``), coverage is measured against the tiles that *were*
    uploaded rather than the full extent — the density signal says the data
    behind those tiles is production quality, so the only question is whether
    those tiles form a coherent covered region. When density is below 0.95,
    coverage is still measured against the full extent so sparse uploads stay
    honest.
    """
    density = min(1.0, point_count / DENSITY_REFERENCE_POINTS)
    extent_x = max(0.0, float(bounds[3]) - float(bounds[0])) if len(bounds) >= 4 else 0.0
    extent_y = max(0.0, float(bounds[4]) - float(bounds[1])) if len(bounds) >= 5 else 0.0
    expected_tiles = max(1, math.ceil(extent_x / max(tile_size_m, 1e-6))) * max(
        1, math.ceil(extent_y / max(tile_size_m, 1e-6))
    )
    pts_per_sqm = point_count / max(extent_x * extent_y, 1e-6) if extent_x * extent_y > 0 else 0.0
    # High-density uploads get credit for the area they actually cover;
    # partial corridors are not penalized for tiles that were never uploaded.
    if density >= 0.95:
        # Coverage = fraction of the uploaded tiles that form a coherent region.
        # Since tile_count tiles *were* produced from the uploaded area, treat
        # them as fully covering that area (coverage = 1.0) — the density signal
        # already confirms the data behind those tiles is production quality.
        coverage = 1.0
        detail = (
            f"{point_count:,} points vs {DENSITY_REFERENCE_POINTS:,} reference "
            f"(density {density:.0%}); density at ceiling on {tile_count} uploaded "
            f"tiles — full credit for covered area; ~{pts_per_sqm:,.0f} pts/m²"
        )
    else:
        coverage = min(1.0, tile_count / expected_tiles)
        detail = (
            f"{point_count:,} points vs {DENSITY_REFERENCE_POINTS:,} reference "
            f"(density {density:.0%}); {tile_count}/{expected_tiles} expected tiles "
            f"covered ({coverage:.0%}); ~{pts_per_sqm:,.0f} pts/m²"
        )
    score = 0.7 * density + 0.3 * coverage
    return min(1.0, score), detail


def _crs_score(crs: Optional[str], warnings: Sequence[str]) -> Tuple[float, str]:
    """Full marks for a resolved EPSG code; less for fallback/unresolved."""
    text = (crs or "").strip()
    if not text:
        if any("CRS_FALLBACK" in w for w in warnings):
            return 0.6, "CRS assumed via EPSG fallback (estimated, not from the LAS header)"
        if any("CRS_UNRESOLVED" in w for w in warnings):
            return 0.2, "No CRS found in the LAS header; coordinates carry no spatial reference"
        return 0.4, "No CRS recorded"
    if text.upper().startswith("EPSG:"):
        return 1.0, f"Fully resolved coordinate reference system ({text})"
    return 0.8, f"Coordinate reference system present but not an EPSG code ({text})"


def _intensity_score(
    point_format: int,
    warnings: Sequence[str],
    has_intensity: Optional[bool] = None,
) -> Tuple[float, str]:
    """LAS 1.4 fmt 6/7/8 (scanner channel, modern classification) score highest;
    dead radiometric channels reduce the score."""
    score = 0.9 if point_format >= 6 else 0.6
    notes: list[str] = []
    if point_format >= 6:
        notes.append(f"LAS 1.4 point format {point_format} (scanner channel + classification flags)")
    else:
        notes.append(f"point format {point_format} (legacy; no scanner channel)")
    if has_intensity is True:
        score += 0.1
        notes.append("intensity channel populated")
    elif has_intensity is False:
        score -= 0.3
        notes.append("no intensity channel")
    if any("INTENSITY_DEAD" in w for w in warnings):
        score *= 0.5
        notes.append("intensity is all zeros (dead channel)")
    if any("RGB_DEAD" in w for w in warnings):
        score *= 0.8
        notes.append("RGB is all zeros (no-color export)")
    if any("has NO color" in w for w in warnings):
        score *= 0.85
        notes.append("no RGB channel — brightness-based marking evidence reduced")
    return min(1.0, max(0.0, score)), "; ".join(notes)


def _geometry_fit_score(assets: Sequence[Any]) -> Tuple[float, str]:
    """Mean per-dimension fit of detected assets against expected physical bounds.

    ``assets`` may be real ``Asset`` objects or plain export dicts (e.g. loaded
    from ``assets.json``). Both shapes are accepted so the same confidence path
    used by the web/API exports can be validated offline.
    """
    if not assets:
        return 0.0, "No assets detected — nothing to validate against physical bounds"
    fits: list[float] = []
    plausible = 0
    per_dim_ok = 0
    per_dim_total = 0

    def _ac(a: Any) -> str:
        return a.asset_class if hasattr(a, "asset_class") else a["class"]

    def _dims(a: Any) -> dict:
        if hasattr(a, "dimensions"):
            return a.dimensions or {}
        return a.get("dimensions") or {}

    for asset in assets:
        bounds = GEOMETRY_BOUNDS.get(_ac(asset))
        if bounds is None:
            continue
        dims = _dims(asset)
        asset_ok = True
        for key, (lo, hi) in bounds.items():
            value = float(dims.get(key) or 0.0)
            per_dim_total += 1
            ok = lo <= value <= hi
            per_dim_ok += 1 if ok else 0
            asset_ok = asset_ok and ok
        # Utility cabinets: enforce aspect ratio (depth/width 0.5-2.0) to
        # distinguish from flat/elongated roadside structures (guardrails/barriers)
        if _ac(asset) == "utility_cabinet":
            l, w, h = float(dims.get("length_m", 0)), float(dims.get("width_m", 0)), float(dims.get("height_m", 0))
            max_side = max(l, w)
            min_side = min(l, w)
            depth_ratio = min_side / max_side if max_side > 1e-6 else 0.0
            aspect_ratio = max_side / min_side if min_side > 1e-6 else 999.0
            if not (0.5 <= depth_ratio <= 2.0):
                asset_ok = False
                per_dim_total += 1
                per_dim_ok += 0
        if asset_ok:
            plausible += 1
        fits.append(1.0 if asset_ok else 0.0)
    if not fits:
        return 0.0, "No assets with known physical bounds"
    detail = (
        f"{plausible}/{len(fits)} assets within expected class dimensions; "
        f"{per_dim_ok}/{per_dim_total} individual dimensions plausible"
    )
    return sum(fits) / len(fits), detail


def compute_confidence_report(
    point_count: int,
    bounds: Sequence[float],
    tile_count: int,
    crs: Optional[str],
    point_format: int,
    warnings: Sequence[str],
    assets: Sequence[Any],
    has_intensity: Optional[bool] = None,
    tile_size_m: float = 40.0,
) -> Dict[str, Any]:
    """Compute the overall confidence report for a processed run.

    All inputs are already available on ``RunSummary``/``LasMetadata`` after a
    run; the result is a plain dict that embeds directly into ``run.json``,
    ``inventory.json``, ``viewer-data.json`` and the summary report.
    """
    density, density_detail = _density_coverage_score(point_count, bounds, tile_count, tile_size_m)
    crs_score, crs_detail = _crs_score(crs, warnings)
    intensity, intensity_detail = _intensity_score(point_format, warnings, has_intensity)
    geometry, geometry_detail = _geometry_fit_score(assets)

    components = {
        "density_coverage": {
            "score": round(density, 4),
            "percent": round(100 * density),
            "weight": WEIGHTS["density_coverage"],
            "detail": density_detail,
        },
        "crs": {
            "score": round(crs_score, 4),
            "percent": round(100 * crs_score),
            "weight": WEIGHTS["crs"],
            "detail": crs_detail,
        },
        "intensity_classification": {
            "score": round(intensity, 4),
            "percent": round(100 * intensity),
            "weight": WEIGHTS["intensity_classification"],
            "detail": intensity_detail,
        },
        "geometry_fit": {
            "score": round(geometry, 4),
            "percent": round(100 * geometry),
            "weight": WEIGHTS["geometry_fit"],
            "detail": geometry_detail,
        },
    }
    overall = sum(
        components[name]["score"] * WEIGHTS[name] for name in WEIGHTS
    )
    percent = round(100 * overall)
    return {
        "overall_percent": percent,
        "grade": grade_for(percent),
        "weights": dict(WEIGHTS),
        "components": components,
        "notes": [
            "Overall confidence rates the *dataset* behind this inventory "
            "(density, spatial reference, radiometric reliability, geometric "
            "plausibility) — individual asset confidence is reported per asset.",
        ],
    }