from __future__ import annotations

import json
from pathlib import Path

import pytest

from infra_inventory.errors import InfraError
from infra_inventory.las_reader import read_metadata
from infra_inventory.models import ProcessingSettings
from infra_inventory.pipeline import process_las


def test_processes_las_and_writes_inventory(synthetic_las: Path, tmp_path: Path) -> None:
    output = tmp_path / "out"
    result = process_las(synthetic_las, output, ProcessingSettings(), progress=False)
    assert result.point_count == read_metadata(synthetic_las).point_count
    assert result.tile_count >= 1
    assert result.las_version == "1.4"
    assert result.point_format == 7
    assert (output / "assets.json").is_file()
    assert (output / "assets.geojson").is_file()
    assert (output / "viewer" / "index.html").is_file()
    assert any(asset.asset_class == "pavement" for asset in result.assets)


def _asset_signature(path: Path) -> list:
    """Deterministic asset fingerprint for cross-run comparison."""
    return sorted(
        (
            asset["asset_id"], asset["class"],
            tuple(asset["center"][k] for k in ("x", "y", "z")),
            round(float(asset["confidence"]), 4),
            asset["source_tile"],
        )
        for asset in json.loads(path.read_text(encoding="utf-8"))
    )


def test_resumes_from_existing_tiles(synthetic_las: Path, tmp_path: Path) -> None:
    output = tmp_path / "out"
    first = process_las(synthetic_las, output, ProcessingSettings(save_tiles=True), progress=False)
    first_assets = _asset_signature(output / "assets.json")
    manifest = output / "tiles" / "manifest.json"
    assert manifest.is_file()
    assert json.loads(manifest.read_text()).get("input_sha256")
    # Simulate a crash after streaming/detection but before export: the viewer
    # payload is gone, tiles + manifest survive.
    (output / "viewer-data.json").unlink()
    resumed = process_las(
        synthetic_las, output, ProcessingSettings(save_tiles=True, resume_from_tiles=True),
        progress=False,
    )
    assert resumed.point_count == first.point_count
    assert any("Resumed from" in warning for warning in resumed.warnings)
    # Byte-identical inventory: same ids, classes, locations, confidence.
    assert _asset_signature(output / "assets.json") == first_assets
    assert (output / "viewer-data.json").is_file()


def test_resume_refuses_stale_manifest(synthetic_las: Path, tmp_path: Path) -> None:
    output = tmp_path / "out"
    process_las(synthetic_las, output, ProcessingSettings(save_tiles=True), progress=False)
    manifest = output / "tiles" / "manifest.json"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["input_sha256"] = "0" * 64  # tiles belong to a different file now
    manifest.write_text(json.dumps(data), encoding="utf-8")
    rerun = process_las(
        synthetic_las, output, ProcessingSettings(save_tiles=True, resume_from_tiles=True),
        progress=False,
    )
    assert not any("Resumed from" in warning for warning in rerun.warnings)
    assert rerun.point_count > 0
    assert rerun.tile_count >= 1


def test_streams_with_tiny_chunks(synthetic_las: Path, tmp_path: Path) -> None:
    settings = ProcessingSettings(chunk_size=3_000)
    result = process_las(synthetic_las, tmp_path / "out", settings, progress=False)
    assert result.point_count == read_metadata(synthetic_las).point_count
    assert result.tile_count >= 1
    # every tile written during streaming is readable
    from infra_inventory.las_reader import read_metadata as read_tile

    tiles = list((tmp_path / "out" / "tiles").glob("tile_*.las"))
    assert tiles
    total_tile_points = sum(read_tile(tile).point_count for tile in tiles)
    assert total_tile_points == result.point_count


def test_pointcept_backend_requires_full_configuration(synthetic_las: Path, tmp_path: Path) -> None:
    settings = ProcessingSettings(backend="pointcept")  # missing root/config/weight
    with pytest.raises(ValueError):
        process_las(synthetic_las, tmp_path / "out", settings, progress=False)


def test_roadmarking_backend_requires_command(synthetic_las: Path, tmp_path: Path) -> None:
    settings = ProcessingSettings(roadmarking_command=None)
    result = process_las(synthetic_las, tmp_path / "out", settings, progress=False)
    assert result.assets  # runs normally without the external subsystem


def test_roadmarking_missing_command_raises(synthetic_las: Path, tmp_path: Path) -> None:
    from infra_inventory.roadmarking import run_roadmarking

    with pytest.raises(InfraError):
        run_roadmarking("", None, tmp_path, "tile_0_0", tmp_path / "rm")


def test_unsupported_input_raises_clean_error(tmp_path: Path) -> None:
    bogus = tmp_path / "bogus.las"
    bogus.write_text("this is not a las file", encoding="utf-8")
    with pytest.raises(InfraError):
        process_las(bogus, tmp_path / "out", ProcessingSettings(), progress=False)


def test_missing_input_raises_clean_error(tmp_path: Path) -> None:
    with pytest.raises(InfraError):
        process_las(tmp_path / "missing.las", tmp_path / "out", ProcessingSettings(), progress=False)


def test_assets_have_provenance(synthetic_las: Path, tmp_path: Path) -> None:
    result = process_las(synthetic_las, tmp_path / "out", ProcessingSettings(), progress=False)
    for asset in result.assets:
        assert asset.source_point_indices_sample
        assert asset.processing_version
        assert asset.detection_method in ("geometry-v1", "geometry-v2-heuristic", "native-roadmarking-v1", "roadmarkingextraction-v1")