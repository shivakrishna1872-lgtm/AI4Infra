"""Transparent confidence engine.

Every asset confidence is a weighted blend of *measurable* factors:

* ``model``            - agreement with the learned Pointcept/PTv3 prior (None on the CPU geometry backend)
* ``geometry``         - how well the instance matches its class's geometric priors (detector-measured)
* ``support``          - point support relative to the class minimum
* ``spatial_context``  - consistency with surrounding context (road proximity, compactness)
* ``class_consistency``- intensity/RGB statistics matching the class's physical expectations

Missing factors are ``None`` and the remaining weights are renormalized, so a
confidence of 0.93 always means the same thing: *the measurable signals for this
asset were consistently strong*. The explanation string says exactly which
signals contributed.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

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