from __future__ import annotations

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