from __future__ import annotations

from infra_inventory.models import Asset, ProcessingSettings
from infra_inventory.qc import run_quality_control


def _asset(asset_id: str, asset_class: str, center, confidence: float, dimensions=None, point_count: int = 500) -> Asset:
    x, y, z = center
    return Asset(
        asset_id=asset_id,
        asset_class=asset_class,
        subclass=None,
        center={"x": x, "y": y, "z": z},
        bounding_box=(x - 1, y - 1, z, x + 1, y + 1, z + 5),
        dimensions=dimensions or {"length_m": 2.0, "width_m": 2.0, "height_m": 5.0},
        point_count=point_count,
        source_tile="tile_0_0",
        source_point_indices_sample=[],
        coordinate_reference_system=None,
        confidence=confidence,
        confidence_factors={"model": None, "geometry": confidence, "support": confidence, "spatial_context": None, "class_consistency": None},
        confidence_explanation="test",
        detection_method="test",
    )


def test_duplicate_flag(settings: ProcessingSettings = ProcessingSettings()) -> None:
    a = _asset("POL-00001", "utility_pole", (10.0, 10.0, 0.0), 0.9)
    b = _asset("POL-00002", "utility_pole", (10.4, 10.2, 0.0), 0.5)
    assets, report = run_quality_control([a, b], settings)
    assert "LIKELY_DUPLICATE" in b.qc_flags
    assert b.flagged
    assert "LIKELY_DUPLICATE" not in a.qc_flags
    assert len(report) == 2


def test_no_duplicate_flag_far_apart() -> None:
    a = _asset("POL-00001", "utility_pole", (10.0, 10.0, 0.0), 0.9)
    b = _asset("POL-00002", "utility_pole", (50.0, 10.0, 0.0), 0.9)
    assets, _ = run_quality_control([a, b], ProcessingSettings())
    assert "LIKELY_DUPLICATE" not in b.qc_flags


def test_low_confidence_flag() -> None:
    asset = _asset("MRK-00001", "pavement_marking", (0.0, 0.0, 0.0), 0.3)
    assets, _ = run_quality_control([asset], ProcessingSettings())
    assert "LOW_CONFIDENCE" in assets[0].qc_flags


def test_unusual_dimensions_flag() -> None:
    asset = _asset(
        "POL-00001", "utility_pole", (0.0, 0.0, 0.0), 0.9,
        dimensions={"length_m": 40.0, "width_m": 0.1, "height_m": 0.2},
    )
    assets, _ = run_quality_control([asset], ProcessingSettings())
    assert "UNUSUAL_DIMENSIONS" in assets[0].qc_flags


def test_report_shape() -> None:
    asset = _asset(
        "GRD-00001", "guardrail", (0.0, 0.0, 0.0), 0.8,
        dimensions={"length_m": 20.0, "width_m": 0.5, "height_m": 0.8},
    )
    _, report = run_quality_control([asset], ProcessingSettings())
    assert report[0]["asset_id"] == "GRD-00001"
    assert report[0]["qc_flags"] == []
    assert report[0]["flagged"] is False