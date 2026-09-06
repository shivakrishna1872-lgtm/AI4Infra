"""Quality control for the asset inventory.

Every asset is validated against measurable expectations. Suspicious assets are
flagged rather than deleted, so downstream consumers can decide whether to
trust them:

* ``LOW_CONFIDENCE``      - confidence below the configured threshold
* ``LIKELY_DUPLICATE``    - same class, overlapping/nearby detection (tile-boundary splits)
* ``UNUSUAL_DIMENSIONS``  - dimensions outside the class's typical envelope
* ``INVALID_GEOMETRY``    - non-finite or zero-size geometry (should not happen)
* ``BELOW_MIN_POINTS``    - point support below the configured minimum
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np

from .models import Asset, ProcessingSettings, VALID_TAXONOMY

#: (min_length, max_length, min_width, max_width, min_height, max_height) in metres.
#: Values outside this envelope are flagged UNUSUAL_DIMENSIONS. Deliberately wide.
DIMENSION_LIMITS: Dict[str, Tuple[float, float, float, float, float, float]] = {
    "pavement": (0.0, 1e9, 0.0, 1e9, 0.0, 1.0),
    "pavement_marking": (0.2, 500.0, 0.05, 20.0, 0.0, 1.0),
    "utility_pole": (0.0, 3.0, 0.0, 3.0, 1.5, 30.0),
    "overhead_conductor": (5.0, 800.0, 0.0, 2.0, 0.0, 6.0),
    "utility_cabinet": (0.2, 6.0, 0.2, 6.0, 0.2, 5.0),
    "traffic_sign": (0.1, 6.0, 0.05, 6.0, 0.0, 10.0),
    "guardrail": (4.0, 2000.0, 0.0, 3.0, 0.1, 3.0),
    "safety_barrier": (4.0, 2000.0, 0.1, 4.0, 0.2, 4.0),
    "rumble_strip": (0.3, 30.0, 0.1, 2.0, 0.0, 1.0),
}


def run_quality_control(assets: List[Asset], settings: ProcessingSettings) -> Tuple[List[Asset], List[dict]]:
    """Flag assets and produce a machine-readable QC report."""
    for asset in assets:
        asset.qc_flags = []
        asset.flagged = False
        _flag_geometry(asset)
        _flag_dimensions(asset)
        _flag_points(asset, settings)
        _flag_confidence(asset, settings)
        asset.flagged = bool(asset.qc_flags)
    _flag_duplicates(assets, settings)
    report: List[dict] = []
    for asset in assets:
        entry = asset.to_dict()
        entry["qc_flags"] = asset.qc_flags
        entry["flagged"] = asset.flagged
        report.append(entry)
    return assets, report


def _flag_geometry(asset: Asset) -> None:
    values = list(asset.bounding_box) + [asset.center["x"], asset.center["y"], asset.center["z"]]
    if not all(math.isfinite(value) for value in values):
        asset.qc_flags.append("INVALID_GEOMETRY")
        return
    if asset.dimensions["length_m"] == 0 and asset.dimensions["width_m"] == 0 and asset.dimensions["height_m"] == 0:
        asset.qc_flags.append("INVALID_GEOMETRY")


def _flag_dimensions(asset: Asset) -> None:
    limits = DIMENSION_LIMITS.get(asset.asset_class)
    if limits is None:
        return
    min_l, max_l, min_w, max_w, min_h, max_h = limits
    length, width, height = asset.dimensions["length_m"], asset.dimensions["width_m"], asset.dimensions["height_m"]
    if not (
        min_l <= length <= max_l and min_w <= width <= max_w and min_h <= height <= max_h
    ):
        asset.qc_flags.append("UNUSUAL_DIMENSIONS")


def _flag_points(asset: Asset, settings: ProcessingSettings) -> None:
    if asset.point_count < settings.min_asset_points:
        asset.qc_flags.append("BELOW_MIN_POINTS")


def _flag_confidence(asset: Asset, settings: ProcessingSettings) -> None:
    if asset.confidence < settings.low_confidence_threshold:
        asset.qc_flags.append("LOW_CONFIDENCE")


def _flag_duplicates(assets: List[Asset], settings: ProcessingSettings) -> None:
    """Flag same-class detections whose centroids are implausibly close."""
    by_class: Dict[str, List[Asset]] = {}
    for asset in assets:
        by_class.setdefault(asset.asset_class, []).append(asset)
    for group in by_class.values():
        for i, a in enumerate(group):
            for b in group[i + 1:]:
                distance = _centroid_distance(a, b)
                tolerance = max(settings.duplicate_distance_m, 0.12 * max(a.dimensions["length_m"], b.dimensions["length_m"]))
                if distance <= tolerance:
                    keeper = a if a.confidence >= b.confidence else b
                    other = b if keeper is a else a
                    other.qc_flags.append("LIKELY_DUPLICATE")
                    other.flagged = True


def _centroid_distance(a: Asset, b: Asset) -> float:
    return float(np.hypot(a.center["x"] - b.center["x"], a.center["y"] - b.center["y"]))


def summarize(assets: List[Asset]) -> Dict[str, object]:
    """Counts by class for the run report."""
    counts: Dict[str, int] = {}
    for asset in assets:
        counts[asset.asset_class] = counts.get(asset.asset_class, 0) + 1
    flagged = sum(1 for asset in assets if asset.flagged)
    total_confidence = sum(asset.confidence for asset in assets)
    return {
        "total_assets": len(assets),
        "by_class": dict(sorted(counts.items())),
        "flagged_assets": flagged,
        "mean_confidence": round(total_confidence / len(assets), 3) if assets else None,
    }