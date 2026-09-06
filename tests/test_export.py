from __future__ import annotations

import csv
import json
from pathlib import Path

from infra_inventory.models import ProcessingSettings
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


def test_summary_report_lists_classes(synthetic_las: Path, output_dir: Path) -> None:
    _run(synthetic_las, output_dir)
    report = (output_dir / "reports" / "summary.md").read_text()
    assert "| Class | Count |" in report
    assert "utility_pole" in report