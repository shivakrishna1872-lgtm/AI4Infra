"""Tests for the overall confidence scoring engine (infra_inventory/confidence.py)."""

from __future__ import annotations

from types import SimpleNamespace

from infra_inventory.confidence import (
    WEIGHTS,
    compute_confidence_report,
    grade_for,
)


def _asset(asset_class: str, dims: dict) -> SimpleNamespace:
    return SimpleNamespace(asset_class=asset_class, dimensions=dims)


def _report(**kwargs) -> dict:
    defaults = dict(
        point_count=112_800_000,  # >100M: maximum density
        bounds=(0.0, -12.0, -1.0, 10_000.0, 12.0, 20.0),
        tile_count=500,  # 10km/40m x 1 = 250 expected; 500 covers it
        crs="EPSG:6553",
        point_format=7,
        warnings=[],
        assets=[
            _asset("utility_pole", {"length_m": 0.26, "width_m": 0.26, "height_m": 6.66}),
            _asset("guardrail", {"length_m": 59.86, "width_m": 0.2, "height_m": 0.66}),
            _asset("pavement", {"length_m": 79.8, "width_m": 18.13, "height_m": 0.285}),
            _asset("traffic_sign", {"length_m": 0.0, "width_m": 0.8, "height_m": 2.7}),
        ],
    )
    defaults.update(kwargs)
    return compute_confidence_report(**defaults)


def test_weights_sum_to_one() -> None:
    assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9


def test_full_quality_input_scores_high() -> None:
    report = _report()
    assert report["overall_percent"] >= 90
    assert report["grade"] == grade_for(report["overall_percent"])
    assert report["grade"].startswith("High")
    assert report["components"]["crs"]["percent"] == 100
    assert report["components"]["intensity_classification"]["percent"] >= 90


def test_grade_bands() -> None:
    assert grade_for(95).startswith("High")
    assert grade_for(90).startswith("High")
    assert grade_for(89).startswith("Medium")
    assert grade_for(75).startswith("Medium")
    assert grade_for(74).startswith("Low")
    assert grade_for(10).startswith("Low")


def test_tiny_point_count_lowers_density() -> None:
    report = _report(point_count=10_000)
    density = report["components"]["density_coverage"]
    # 10k pts vs the 100M reference: the density term is ~0, so the component
    # collapses to the coverage term (30% of the component at most).
    assert density["percent"] < 35
    assert report["overall_percent"] < 90
    assert report["overall_percent"] < _report()["overall_percent"]


def test_unresolved_crs_scores_lowest() -> None:
    report = _report(crs=None, warnings=["CRS_UNRESOLVED — header carries no CRS"])
    assert report["components"]["crs"]["percent"] == 20


def test_fallback_crs_scores_partial() -> None:
    report = _report(crs=None, warnings=["[CRS_FALLBACK] assumed EPSG:6553"])
    assert report["components"]["crs"]["percent"] == 60


def test_resolved_non_epsg_crs_scores_almost_full() -> None:
    report = _report(crs="NAD83(2011) / Oklahoma North (ftUS) WKT")
    assert report["components"]["crs"]["percent"] == 80


def test_dead_intensity_halves_intensity_component() -> None:
    clean = _report()
    dead = _report(warnings=["[INTENSITY_DEAD] intensity all zeros"])
    assert (
        dead["components"]["intensity_classification"]["score"]
        <= clean["components"]["intensity_classification"]["score"] * 0.55
    )


def test_legacy_format_scores_lower_intensity() -> None:
    modern = _report(point_format=7)
    legacy = _report(point_format=1)
    assert legacy["components"]["intensity_classification"]["percent"] < modern["components"]["intensity_classification"]["percent"]


def test_geometry_fit_penalizes_implausible_assets() -> None:
    good = _report()
    assert good["components"]["geometry_fit"]["percent"] == 100
    bad = _report(
        assets=[
            _asset("utility_pole", {"length_m": 0.26, "width_m": 0.26, "height_m": 80.0}),  # 80m pole
            _asset("guardrail", {"length_m": 59.86, "width_m": 0.2, "height_m": 0.66}),
        ]
    )
    assert bad["components"]["geometry_fit"]["percent"] == 50
    assert bad["overall_percent"] < good["overall_percent"]


def test_no_assets_geometry_scores_zero() -> None:
    report = _report(assets=[])
    assert report["components"]["geometry_fit"]["percent"] == 0


def test_overall_is_weighted_sum_of_components() -> None:
    report = _report()
    expected = sum(
        report["components"][name]["score"] * report["weights"][name] for name in WEIGHTS
    )
    assert abs(report["overall_percent"] / 100.0 - expected) <= 0.01


def test_report_structure() -> None:
    report = _report()
    assert set(report["components"]) == set(WEIGHTS)
    for name in WEIGHTS:
        component = report["components"][name]
        assert set(component) == {"score", "percent", "weight", "detail"}
        assert 0 <= component["score"] <= 1
        assert 0 <= component["percent"] <= 100
        assert component["weight"] == WEIGHTS[name]
    assert isinstance(report["notes"], list) and report["notes"]


def test_coverage_penalty_for_missing_tiles() -> None:
    # 10 km corridor implies 250 tiles; only 125 present -> coverage 50%.
    sparse = _report(tile_count=125)
    dense = _report(tile_count=250)
    assert (
        sparse["components"]["density_coverage"]["percent"]
        < dense["components"]["density_coverage"]["percent"]
    )