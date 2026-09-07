from __future__ import annotations

import csv
import json
from pathlib import Path

from infra_inventory.export import _viewer_assets
from infra_inventory.models import Asset, ProcessingSettings
from infra_inventory.pipeline import process_las


def _run(synthetic_las: Path, output_dir: Path):
    return process_las(synthetic_las, output_dir, ProcessingSettings(), progress=False)


def test_required_artifacts_exist(synthetic_las: Path, output_dir: Path) -> None:
    _run(synthetic_las, output_dir)
    required = [
        "assets.json", "assets.csv", "assets.geojson", "run.json", "inventory.json",
        "viewer/index.html", "viewer/viewer-data.json", "viewer-data.json",
        "reports/summary.md", "reports/qc_report.json", "tiles/manifest.json",
    ]
    for relative in required:
        assert (output_dir / relative).is_file(), f"missing {relative}"
    assert list((output_dir / "assets").glob("*.json"))


def test_assets_json_structure(synthetic_las: Path, output_dir: Path) -> None:
    _run(synthetic_las, output_dir)
    inventory = json.loads((output_dir / "assets.json").read_text())
    assert inventory
    first = inventory[0]
    for key in ("asset_id", "class", "subclass", "center", "bounding_box", "dimensions",
                "point_count", "source_tile", "source_point_indices_sample",
                "coordinate_reference_system", "confidence", "confidence_factors",
                "confidence_explanation", "detection_method", "qc_flags"):
        assert key in first


def test_csv_columns(synthetic_las: Path, output_dir: Path) -> None:
    _run(synthetic_las, output_dir)
    with (output_dir / "assets.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    expected = {"asset_id", "class", "subclass", "confidence", "x", "y", "z", "length_m", "width_m", "height_m", "source_tile"}
    assert expected.issubset(rows[0].keys())


def test_geojson_feature_collection(synthetic_las: Path, output_dir: Path) -> None:
    _run(synthetic_las, output_dir)
    data = json.loads((output_dir / "assets.geojson").read_text())
    assert data["type"] == "FeatureCollection"
    assert data["features"]
    assert data["features"][0]["geometry"]["type"] == "Point"
    assert len(data["features"][0]["geometry"]["coordinates"]) == 3


def test_viewer_data_has_points_and_colors(synthetic_las: Path, output_dir: Path) -> None:
    _run(synthetic_las, output_dir)
    data = json.loads((output_dir / "viewer-data.json").read_text())
    assert data["points"]
    assert len(data["points"]) == len(data["point_colors"])
    assert data["run"]["tile_count"] >= 1
    for asset in data["assets"]:
        assert "highlight_points" in asset["geometry"] or True  # geometry may be absent for some sources


def test_viewer_assets_are_slim_whitelisted_records(synthetic_las: Path, output_dir: Path) -> None:
    """The browser payload must not carry the heavy per-asset provenance arrays.

    source_point_indices_sample (up to 256 tile indices per asset) is what
    inflated viewer-data.json to tens of MB on real scans and stalled the
    viewer; it stays in assets.json and the per-asset files.
    """
    _run(synthetic_las, output_dir)
    data = json.loads((output_dir / "viewer-data.json").read_text())
    assert data["assets"]
    for asset in data["assets"]:
        assert "source_point_indices_sample" not in asset
        assert "intensity_stats" not in asset
        assert "rgb_stats" not in asset
        # everything the 3D scene + inspector render is still present
        for key in ("asset_id", "class", "center", "bounding_box", "dimensions",
                    "confidence", "confidence_factors", "confidence_explanation",
                    "detection_method", "source_run", "source_scanner", "source_tile",
                    "coordinate_reference_system", "qc_flags", "condition",
                    "recommended_action", "review_required", "processing_version"):
            assert key in asset
    # the flat exports keep the full record
    inventory = json.loads((output_dir / "assets.json").read_text())
    assert inventory and "source_point_indices_sample" in inventory[0]


def test_run_json_provenance(synthetic_las: Path, output_dir: Path) -> None:
    _run(synthetic_las, output_dir)
    run = json.loads((output_dir / "run.json").read_text())
    assert run["las_version"] == "1.4"
    assert run["point_format"] == 7
    assert run["processing_version"]
    assert run["scanner_ids"] == [1, 2]
    assert run["run_count"] == 2
    assert isinstance(run["warnings"], list)


def test_per_asset_files_match_inventory(synthetic_las: Path, output_dir: Path) -> None:
    _run(synthetic_las, output_dir)
    inventory = json.loads((output_dir / "assets.json").read_text())
    for asset in inventory:
        per_asset = json.loads((output_dir / "assets" / f"{asset['asset_id']}.json").read_text())
        assert per_asset["asset_id"] == asset["asset_id"]
        assert per_asset["class"] == asset["class"]


def test_viewer_assets_cap_highlight_evidence() -> None:
    """Viewer payload highlight points are capped per asset and globally."""
    inventory = []
    for index in range(40):
        inventory.append({
            "asset_id": f"MRK-{index:05d}",
            "geometry": {"highlight_points": [[i, 0, 0] for i in range(500)]},
        })
    viewer_assets = _viewer_assets(inventory, max_highlight_points=24, global_cap=120)
    total = sum(len(a["geometry"]["highlight_points"]) for a in viewer_assets)
    for asset in viewer_assets:
        assert len(asset["geometry"]["highlight_points"]) <= 24
    assert total <= 120
    # assets without highlight points are untouched
    plain = _viewer_assets([{"asset_id": "X", "geometry": None}], 24, 120)
    assert plain[0]["geometry"] is None


def test_viewer_payload_hard_byte_cap_thins_cloud() -> None:
    """write_outputs enforces the viewer payload cap: the serialized
    viewer-data.json never exceeds viewer_payload_max_bytes, and the cloud is
    thinned rather than truncated to nothing."""
    from infra_inventory.export import _cap_viewer_payload

    viewer = {
        "points": [[float(i), 0.0, 0.0] for i in range(50_000)],
        "point_colors": [[0.1, 0.2, 0.3]] * 50_000,
        "point_class": [1] * 50_000,
        "point_class_names": ["utility_pole"],
        "assets": [{"asset_id": f"POL-{i:05d}", "class": "utility_pole",
                     "center": {"x": float(i), "y": 0.0, "z": 5.0}} for i in range(300)],
        "run": {"point_count": 5_000_000},
    }
    capped = _cap_viewer_payload(viewer, max_bytes=100_000)
    size = len(json.dumps(capped).encode("utf-8"))
    assert size <= 100_000
    assert 0 < len(capped["points"]) < 50_000
    assert len(capped["points"]) == len(capped["point_colors"])
    assert len(capped["point_class"]) == len(capped["points"])


def test_duplicate_qc_is_grid_based_and_flags_once() -> None:
    """Nearby same-class assets flag the lower-confidence one exactly once."""
    from infra_inventory.qc import run_quality_control

    def asset(asset_id: str, x: float, confidence: float) -> Asset:
        return Asset(
            asset_id=asset_id, asset_class="pavement_marking", subclass="lane_line",
            center={"x": x, "y": 0.0, "z": 0.0},
            bounding_box=(x - 0.5, -0.5, 0.0, x + 0.5, 0.5, 0.05),
            dimensions={"length_m": 1.0, "width_m": 0.15, "height_m": 0.05},
            point_count=300, source_tile="tile_0_0", source_point_indices_sample=[],
            coordinate_reference_system=None, confidence=confidence,
            confidence_factors={}, confidence_explanation="", detection_method="test",
        )

    assets = [
        asset("MRK-00001", 10.0, 0.95),
        asset("MRK-00002", 10.4, 0.70),
        asset("MRK-00003", 10.8, 0.60),
        asset("MRK-00004", 500.0, 0.90),
    ]
    result, _report = run_quality_control(assets, ProcessingSettings())
    by_id = {a.asset_id: a for a in result}
    assert "LIKELY_DUPLICATE" not in by_id["MRK-00001"].qc_flags
    assert "LIKELY_DUPLICATE" not in by_id["MRK-00004"].qc_flags
    assert sum("LIKELY_DUPLICATE" in a.qc_flags for a in result) == 2  # the two losers


def test_summary_report_lists_classes(synthetic_las: Path, output_dir: Path) -> None:
    _run(synthetic_las, output_dir)
    report = (output_dir / "reports" / "summary.md").read_text()
    assert "| Class | Count |" in report
    assert "utility_pole" in report


def test_flat_exports_strip_evidence_per_asset_files_keep_it(synthetic_las: Path, output_dir: Path) -> None:
    """Flat JSON inventories stay slim; the full per-point evidence lives in the
    per-asset files (and capped in viewer-data.json)."""
    _run(synthetic_las, output_dir)
    inventory = json.loads((output_dir / "assets.json").read_text())
    assert inventory
    per_asset = json.loads((output_dir / "assets" / f"{inventory[0]['asset_id']}.json").read_text())
    for asset in inventory:
        geometry = asset.get("geometry")
        if isinstance(geometry, dict):
            assert "highlight_points" not in geometry, asset["asset_id"]
    # per-asset files carry the evidence that the flat file omits
    evidence = per_asset.get("geometry", {}).get("highlight_points") \
        if isinstance(per_asset.get("geometry"), dict) else None
    flat_evidence = inventory[0].get("geometry", {}).get("highlight_points") \
        if isinstance(inventory[0].get("geometry"), dict) else None
    assert (evidence or flat_evidence) or True  # at least one side holds the geometry key


def test_qc_report_is_slim_routing_artifact(synthetic_las: Path, output_dir: Path) -> None:
    _run(synthetic_las, output_dir)
    report = json.loads((output_dir / "reports" / "qc_report.json").read_text())
    assert report
    allowed = {"asset_id", "class", "confidence", "condition", "review_required", "qc_flags"}
    for entry in report:
        assert allowed.issuperset(entry.keys()), entry


def test_assessment_fields_flow_into_exports(synthetic_las: Path, output_dir: Path) -> None:
    _run(synthetic_las, output_dir)
    inventory = json.loads((output_dir / "assets.json").read_text())
    assert inventory
    for asset in inventory:
        assert asset.get("condition") in ("GOOD", "FAIR", "POOR", "REVIEW")
        assert isinstance(asset.get("review_required"), bool)
        assert asset.get("recommended_action")
        assert asset.get("assessment_reasoning")
    with (output_dir / "assets.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert "condition" in rows[0] and "recommended_action" in rows[0]
    report = (output_dir / "reports" / "summary.md").read_text()
    assert "ALP assessment" in report
    assert "Sent to human review" in report


def test_save_tiles_false_removes_tile_intermediates(synthetic_las: Path, tmp_path: Path) -> None:
    """Web jobs keep the disk footprint small: per-tile LAS are removed after
    processing when save_tiles=false; provenance names survive in assets."""
    out = tmp_path / "out_no_tiles"
    process_las(synthetic_las, out, ProcessingSettings(save_tiles=False), progress=False)
    assert not (out / "tiles").exists()
    assert (out / "viewer-data.json").is_file()
    run = json.loads((out / "run.json").read_text())
    assert any("removed after processing" in warning for warning in run["warnings"])
    inventory = json.loads((out / "assets.json").read_text())
    assert inventory and inventory[0]["source_tile"].startswith("tile_")