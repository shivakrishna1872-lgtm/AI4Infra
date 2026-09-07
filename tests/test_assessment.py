"""Tests for the ALP assessment layer (condition / recommended action / review).

Every assertion checks that interpretations are grounded in measured signals
and that review routing follows the calibrated threshold, never a guess.
"""
from __future__ import annotations

from infra_inventory.assessment import assess_all, assess_asset
from infra_inventory.models import Asset

DEFAULT_REVIEW = 0.8  # calibrated value (configs/processing.yaml)


def _asset(
    asset_class: str,
    *,
    confidence: float = 0.9,
    intensity_mean: float | None = None,
    verticality: float | None = None,
    planarity: float | None = None,
    dimensions: dict | None = None,
    qc_flags: list[str] | None = None,
    subclass: str = "vertical_support",
) -> Asset:
    return Asset(
        asset_id="POL-00001",
        asset_class=asset_class,
        subclass=subclass,
        center={"x": 0.0, "y": 0.0, "z": 5.0},
        bounding_box=(-0.3, -0.3, 0.0, 0.3, 0.3, 10.0),
        dimensions=dimensions or {"length_m": 0.6, "width_m": 0.6, "height_m": 10.0},
        point_count=800,
        source_tile="tile_0_0",
        source_point_indices_sample=[],
        coordinate_reference_system="EPSG:26914",
        confidence=confidence,
        confidence_factors={"geometry": confidence},
        confidence_explanation="test",
        detection_method="test",
        intensity_stats={"mean": intensity_mean} if intensity_mean is not None else None,
        geometry={"eigen_verticality": verticality, "eigen_planarity": planarity}
        if (verticality is not None or planarity is not None) else None,
        qc_flags=list(qc_flags or []),
    )


def test_high_confidence_is_good_and_not_reviewed() -> None:
    asset = _asset("utility_pole", confidence=0.95)
    assess_asset(asset, DEFAULT_REVIEW)
    assert asset.condition == "GOOD"
    assert asset.review_required is False
    assert asset.recommended_action == "No action — routine inventory record."
    assert asset.assessment_reasoning


def test_below_calibrated_threshold_goes_to_review() -> None:
    asset = _asset("utility_pole", confidence=0.70)
    assess_asset(asset, DEFAULT_REVIEW)
    # 0.70 is above the FAIR floor but below the calibrated 0.80 review bar.
    assert asset.condition == "FAIR"
    assert asset.review_required is True
    assert "review" in asset.assessment_reasoning.lower()


def test_structural_qc_flag_forces_review_even_at_high_confidence() -> None:
    asset = _asset("utility_pole", confidence=0.95, qc_flags=["LIKELY_DUPLICATE"])
    assess_asset(asset, DEFAULT_REVIEW)
    assert asset.condition == "REVIEW"
    assert asset.review_required is True
    assert asset.recommended_action.startswith("Send to human review")


def test_low_intensity_marking_flagged_faded_and_restripe_action() -> None:
    asset = _asset("pavement_marking", confidence=0.9, intensity_mean=6_000)
    assess_asset(asset, DEFAULT_REVIEW)
    assert asset.condition == "POOR"
    assert "intensity" in asset.assessment_reasoning
    assert "restripe" in asset.recommended_action.lower()


def test_reflective_marking_stays_good() -> None:
    asset = _asset("pavement_marking", confidence=0.9, intensity_mean=30_000)
    assess_asset(asset, DEFAULT_REVIEW)
    assert asset.condition == "GOOD"


def test_leaning_pole_triggers_site_check() -> None:
    # eigen-verticality 0.90 -> ~26 deg lean proxy, far past the 8 deg alert.
    asset = _asset("utility_pole", confidence=0.9, verticality=0.90)
    assess_asset(asset, DEFAULT_REVIEW)
    assert asset.condition == "POOR"
    assert "lean" in asset.assessment_reasoning.lower()
    assert "inspect pole stability" in asset.recommended_action.lower()


def test_upright_pole_not_lean_flagged() -> None:
    asset = _asset("utility_pole", confidence=0.9, verticality=0.999)
    assess_asset(asset, DEFAULT_REVIEW)
    assert asset.condition == "GOOD"
    assert "lean" not in asset.assessment_reasoning.lower()


def test_sagging_conductor_triggers_clearance_survey() -> None:
    dims = {"length_m": 60.0, "width_m": 0.4, "height_m": 9.0}  # 15% sag ratio
    asset = _asset("overhead_conductor", confidence=0.9, dimensions=dims)
    assess_asset(asset, DEFAULT_REVIEW)
    assert asset.condition == "POOR"
    assert "clearance survey" in asset.recommended_action.lower()


def test_tall_spread_pavement_patch_recommends_field_inspection() -> None:
    dims = {"length_m": 39.0, "width_m": 8.0, "height_m": 0.6}  # includes curb/clutter
    asset = _asset("pavement", confidence=0.9, dimensions=dims)
    assess_asset(asset, DEFAULT_REVIEW)
    assert asset.condition == "POOR"
    assert "field" in asset.recommended_action.lower()


def test_clean_flat_pavement_is_not_penalized() -> None:
    # Flat travelled-surface patches read as elongated ribbons (low eigen
    # planarity by construction) but have a tight vertical spread.
    dims = {"length_m": 39.0, "width_m": 8.0, "height_m": 0.09}
    asset = _asset("pavement", confidence=0.9, dimensions=dims, planarity=0.025)
    assess_asset(asset, DEFAULT_REVIEW)
    assert asset.condition == "GOOD"


def test_short_guardrail_fragment_recommends_continuity_check() -> None:
    dims = {"length_m": 3.0, "width_m": 0.3, "height_m": 0.7}
    asset = _asset("guardrail", confidence=0.9, dimensions=dims, subclass="roadside_barrier")
    assess_asset(asset, DEFAULT_REVIEW)
    assert asset.condition == "POOR"
    assert "continuity" in asset.recommended_action.lower()


def test_assess_all_mutates_and_is_idempotent() -> None:
    assets = [_asset("traffic_sign", confidence=0.5), _asset("utility_cabinet", confidence=0.9)]
    assess_all(assets, DEFAULT_REVIEW)
    first = assets[0]
    snapshot = (first.condition, first.recommended_action, first.review_required, first.assessment_reasoning)
    assess_asset(first, DEFAULT_REVIEW)
    assert (first.condition, first.recommended_action, first.review_required, first.assessment_reasoning) == snapshot
    assert assets[0].review_required is True
    assert assets[1].review_required is False


def test_low_confidence_never_claimed_good() -> None:
    for confidence in (0.30, 0.45, 0.59):
        asset = _asset("utility_cabinet", confidence=confidence)
        assess_asset(asset, DEFAULT_REVIEW)
        assert asset.condition != "GOOD"
        assert asset.review_required is True