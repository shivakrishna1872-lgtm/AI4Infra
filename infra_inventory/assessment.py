"""ALP assessment layer: Observation -> Interpretation -> Recommended Action.

For every detected asset this module derives a **condition**, a **recommended
action**, and a **review flag** from the signals the pipeline actually measured
(geometry, intensity/RGB statistics, confidence factors, QC flags). It follows
the rules the ALP section of the competition brief demands:

* Every interpretation is grounded in a cited measurement (e.g. mean intensity,
  eigen-verticality). An unmeasured cause is never invented: if we cannot tell
  why a marking faded, we say "low measured retroreflectivity", not "sun damage".
* Confidence measures *how likely the detection is a true positive* (calibrated
  in scripts/calibrate_confidence.py against ground truth). Assets below the
  calibrated review threshold are sent to human review rather than trusted.
* Condition bands (GOOD / FAIR / POOR / REVIEW) map to calibrated precision
  intervals documented in docs/ALP.md, not arbitrary gut feelings.

The output fields are stored on each :class:`Asset` (condition,
recommended_action, review_required, assessment_reasoning) and exported with
every inventory artifact.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

from .models import Asset

#: Confidence band edges used for the base condition rating. The bands are
#: defined from the precision/recall calibration curve (scripts/calibrate_confidence.py)
#: and are documented with the measured numbers in docs/ALP.md: on the 400 m
#: simulated ground truth, compact-object precision reaches 1.00 at confidence
#: >= 0.80 (recall 0.89), so 0.80 is the GOOD/reliable edge; 0.60 is the
#: FAIR/POOR edge below which geometry is unlikely to be trustworthy.
GOOD_CONFIDENCE = 0.80
FAIR_CONFIDENCE = 0.60

#: Retroreflectivity proxy floor (LAS intensity mean, 0-65535 scale): markings
#: and sign panels measuring below this are interpreted as faded/low-retro.
RETROREFLECTIVE_FLOOR = 12_000
#: Lean angle (degrees from vertical) above which a pole/sign support is
#: flagged for a site lean check. Derived from eigen-verticality.
LEAN_ALERT_DEG = 8.0
#: Vertical extent of an overhead conductor span relative to its length above
#: which the span is interpreted as sagging (clearance review needed).
CONDUCTOR_SAG_RATIO = 0.12
#: Vertical spread (m) above which a detected travelled-surface patch is
#: interpreted as containing clutter/curb/deviation rather than clean pavement.
#: NOTE: sub-decimeter surface defects (cracks, potholes) are below the detector
#: resolution and are deliberately NOT claimed from this signal (docs/LIMITATIONS.md).
PAVEMENT_SPREAD_ALERT_M = 0.4

#: QC flags that force human review regardless of confidence.
REVIEW_FLAGS = {"LIKELY_DUPLICATE", "INVALID_GEOMETRY", "UNUSUAL_DIMENSIONS", "BELOW_MIN_POINTS"}

_ACTION_REVIEW = "Send to human review before including this asset in the report."
_ACTION_NONE = "No action — routine inventory record."

_CLASS_ACTIONS = {
    "pavement": {
        "fair": "Monitor — verify travelled-surface condition at the next inspection.",
        "poor": "Field-inspect the travelled surface; measured geometry deviates from a smooth plane.",
    },
    "pavement_marking": {
        "fair": "Monitor retroreflectivity — schedule a field striping check.",
        "poor": "Schedule restripe/refresh verification; measured retroreflectivity is low.",
    },
    "utility_pole": {
        "fair": "Monitor — verify pole condition at the next patrol.",
        "poor": "Inspect pole stability and lean at the site before relying on the span.",
    },
    "overhead_conductor": {
        "fair": "Monitor the span — verify clearance during the next survey.",
        "poor": "Perform a sag/clearance survey; measured span geometry suggests excessive droop.",
    },
    "utility_cabinet": {
        "fair": "Monitor — verify enclosure at the next visit.",
        "poor": "Inspect enclosure integrity and accessibility at the site.",
    },
    "traffic_sign": {
        "fair": "Monitor — verify panel retroreflectivity at the next inspection.",
        "poor": "Verify retroreflectivity per the governing standard (MUTCD) and replace if below.",
    },
    "guardrail": {
        "fair": "Monitor — verify rail continuity at the next patrol.",
        "poor": "Field-verify rail continuity and end treatments; geometry is fragmented or undersupported.",
    },
    "safety_barrier": {
        "fair": "Monitor — verify barrier condition at the next patrol.",
        "poor": "Inspect the barrier for cracks or displacement at the site.",
    },
    "rumble_strip": {
        "fair": "Monitor — verify strip condition at the next patrol.",
        "poor": "Field-verify strip geometry; measured bands are inconsistent.",
    },
}


def _lean_deg(verticality: Optional[float]) -> Optional[float]:
    """Lean angle from vertical in degrees, or None when unmeasurable."""
    if verticality is None or not math.isfinite(verticality):
        return None
    v = max(0.0, min(1.0, float(verticality)))
    if v >= 0.99999:
        return 0.0
    return math.degrees(math.acos(v))


def _intensity_mean(asset: Asset) -> Optional[float]:
    stats = asset.intensity_stats
    if not stats:
        return None
    for key in ("mean", "median"):
        value = stats.get(key)
        if value is not None and math.isfinite(float(value)):
            return float(value)
    return None


def _class_specific(asset: Asset) -> Tuple[Optional[str], Optional[str]]:
    """Return (override_condition, extra_reasoning) from class-specific signals.

    ``override_condition`` downgrades the base condition one step unless already
    POOR/REVIEW. Reasoning always cites the measurement that triggered it.
    """
    cls = asset.asset_class
    geometry = asset.geometry or {}
    intensity = _intensity_mean(asset)
    override: Optional[str] = None
    reason: Optional[str] = None

    if cls == "pavement_marking":
        if intensity is not None and intensity < RETROREFLECTIVE_FLOOR:
            override = "poor"
            reason = (
                f"measured mean intensity {intensity:,.0f} is below the "
                f"{RETROREFLECTIVE_FLOOR:,.0f} retroreflectivity floor (interpretation: faded "
                "marking; cause unmeasured, so action is re-verification, not a cause claim)."
            )
    elif cls == "traffic_sign":
        if intensity is not None and intensity < RETROREFLECTIVE_FLOOR:
            override = "poor"
            reason = (
                f"measured mean intensity {intensity:,.0f} suggests low panel retroreflectivity; "
                "verify per the governing standard before replacement."
            )
    elif cls == "utility_pole":
        lean = _lean_deg(geometry.get("eigen_verticality"))
        if lean is not None and lean > LEAN_ALERT_DEG:
            override = "poor"
            reason = (
                f"eigen-verticality implies a {lean:.1f} deg lean from vertical "
                f"(alert above {LEAN_ALERT_DEG} deg); a site lean/stability check is required."
            )
    elif cls == "overhead_conductor":
        dims = asset.dimensions
        length = max(dims.get("length_m", 0.0), dims.get("width_m", 0.0))
        sag = dims.get("height_m", 0.0)
        if length > 5.0 and sag / length > CONDUCTOR_SAG_RATIO:
            override = "poor"
            reason = (
                f"vertical extent {sag:.2f} m over a {length:.1f} m span exceeds the "
                f"{CONDUCTOR_SAG_RATIO:.0%} sag ratio; clearance survey recommended."
            )
    elif cls == "pavement":
        spread = asset.dimensions.get("height_m")
        if spread is not None and spread > PAVEMENT_SPREAD_ALERT_M:
            override = "poor"
            reason = (
                f"detected surface patch spans {spread:.2f} m vertically (alert above "
                f"{PAVEMENT_SPREAD_ALERT_M} m); interpretation: the patch includes curb, "
                "clutter or a local deviation — field inspection recommended. Sub-decimeter "
                "surface defects are not claimed at this resolution."
            )
    elif cls in ("guardrail", "safety_barrier"):
        dims = asset.dimensions
        length = max(dims.get("length_m", 0.0), dims.get("width_m", 0.0))
        min_length = 20.0 if cls == "safety_barrier" else 6.0
        if 0.0 < length < min_length:
            override = "poor"
            reason = (
                f"detected length {length:.1f} m is short for a {cls.replace('_', ' ')} "
                "(possible gap or fragmented detection); verify continuity at the site."
            )
    elif cls == "utility_cabinet":
        density = asset.point_count / max(asset.dimensions.get("length_m", 1.0) * asset.dimensions.get("width_m", 1.0), 1e-3)
        if density < 5.0:
            override = "fair"
            reason = (
                f"point density {density:.1f} pts/m^2 is sparse for an enclosure; "
                "verify completeness at the site."
            )
    elif cls == "rumble_strip":
        dims = asset.dimensions
        length = max(dims.get("length_m", 0.0), dims.get("width_m", 0.0))
        if length < 2.0:
            override = "fair"
            reason = "short strip fragment; verify extent at the site."
    return override, reason


def assess_asset(asset: Asset, review_threshold: float = 0.6) -> Asset:
    """Populate condition / recommended_action / review_required on the asset.

    Pure function over the asset's measured fields (idempotent, safe to re-run).
    """
    # Base condition from the calibrated confidence bands. The GOOD edge never
    # drops below the calibrated 0.80 precision bar; a stricter configured
    # review threshold raises it, and a looser one only changes review routing.
    confidence = asset.confidence
    good_edge = max(GOOD_CONFIDENCE, review_threshold)
    if confidence >= good_edge:
        base = "GOOD"
    elif confidence >= FAIR_CONFIDENCE:
        base = "FAIR"
    else:
        base = "POOR"

    override, extra_reason = _class_specific(asset)
    if override == "poor" and base in ("GOOD", "FAIR"):
        condition = "POOR"
    elif override == "fair" and base == "GOOD":
        condition = "FAIR"
    else:
        condition = base

    # Human review rules: low confidence OR structural QC flags.
    needs_review = confidence < review_threshold or bool(set(asset.qc_flags) & REVIEW_FLAGS)
    if needs_review and condition != "REVIEW":
        # Keep the underlying condition for reporting but flag review; if the
        # asset is also structurally flagged, the condition itself is uncertain.
        condition = "REVIEW" if set(asset.qc_flags) & REVIEW_FLAGS else condition

    # Recommended action.
    if condition == "REVIEW":
        action = _ACTION_REVIEW
    elif condition == "GOOD":
        action = _ACTION_NONE
    else:
        action = _CLASS_ACTIONS.get(asset.asset_class, {}).get(
            "poor" if condition == "POOR" else "fair",
            "Monitor — verify during the next inspection.",
        )

    parts: list[str] = [f"confidence {confidence:.2f} -> base {base}"]
    if extra_reason:
        parts.append(extra_reason)
    if asset.qc_flags:
        parts.append("QC flags: " + ", ".join(asset.qc_flags))
    if needs_review and condition != "REVIEW":
        parts.append(f"below review threshold {review_threshold:.2f}; human review recommended")
    if condition == "REVIEW":
        parts.append("structurally uncertain; human review required before downstream use")

    asset.condition = condition
    asset.recommended_action = action
    asset.review_required = needs_review
    asset.assessment_reasoning = " | ".join(parts)
    return asset


def assess_all(assets, review_threshold: float = 0.6) -> None:
    """Assess every asset in place (mutates each Asset)."""
    for asset in assets:
        assess_asset(asset, review_threshold)


#: Mapping used by docs/ALP.md and the web UI legend.
CONDITION_LABELS: Dict[str, str] = {
    "GOOD": "Good — no action",
    "FAIR": "Fair — monitor",
    "POOR": "Poor — schedule action",
    "REVIEW": "Review — human check required",
}
