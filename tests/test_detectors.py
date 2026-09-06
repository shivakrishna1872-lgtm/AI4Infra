from __future__ import annotations

from pathlib import Path

from infra_inventory.models import ProcessingSettings
from infra_inventory.pipeline import process_las


def _run(synthetic_las: Path, output_dir: Path):
    settings = ProcessingSettings()
    return process_las(synthetic_las, output_dir, settings, progress=False)


def test_all_expected_classes_detected(synthetic_las: Path, output_dir: Path) -> None:
    result = _run(synthetic_las, output_dir)
    classes = {asset.asset_class for asset in result.assets}
    assert "pavement" in classes
    assert "pavement_marking" in classes
    assert "utility_pole" in classes
    assert "traffic_sign" in classes
    assert "guardrail" in classes


def test_no_false_conductors_or_rumble_strips(synthetic_las: Path, output_dir: Path) -> None:
    result = _run(synthetic_las, output_dir)
    classes = {asset.asset_class for asset in result.assets}
    assert "overhead_conductor" not in classes
    assert "rumble_strip" not in classes
    assert "safety_barrier" not in classes
    assert "utility_cabinet" not in classes


def test_pole_is_real_pole(synthetic_las: Path, output_dir: Path) -> None:
    result = _run(synthetic_las, output_dir)
    poles = [asset for asset in result.assets if asset.asset_class == "utility_pole"]
    assert len(poles) == 1
    pole = poles[0]
    assert pole.dimensions["height_m"] > 3.0
    assert max(pole.dimensions["length_m"], pole.dimensions["width_m"]) < 1.0
    assert pole.point_count >= 40


def test_sign_is_panel_with_support(synthetic_las: Path, output_dir: Path) -> None:
    result = _run(synthetic_las, output_dir)
    signs = [asset for asset in result.assets if asset.asset_class == "traffic_sign"]
    assert len(signs) == 1
    assert signs[0].subclass == "panel_with_support"
    assert signs[0].point_count >= 200


def test_markings_are_lane_lines(synthetic_las: Path, output_dir: Path) -> None:
    result = _run(synthetic_las, output_dir)
    markings = [asset for asset in result.assets if asset.asset_class == "pavement_marking"]
    assert len(markings) >= 1
    for marking in markings:
        assert marking.subclass == "lane_line"
        assert marking.dimensions["length_m"] >= 2.0


def test_guardrails_are_elongated(synthetic_las: Path, output_dir: Path) -> None:
    result = _run(synthetic_las, output_dir)
    guardrails = [asset for asset in result.assets if asset.asset_class == "guardrail"]
    assert len(guardrails) >= 1
    for guardrail in guardrails:
        assert guardrail.dimensions["length_m"] >= 6.0
        assert guardrail.dimensions["height_m"] < 2.0


def test_confidence_in_range_and_explained(synthetic_las: Path, output_dir: Path) -> None:
    result = _run(synthetic_las, output_dir)
    for asset in result.assets:
        assert 0.0 <= asset.confidence <= 1.0
        assert asset.confidence_explanation
        assert asset.confidence_factors.get("geometry") is not None
        assert asset.source_tile.startswith("tile_")
        assert asset.coordinate_reference_system is None  # synthetic LAS has no CRS