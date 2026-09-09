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
        assert asset.detection_method in (
            "geometry-v1", "geometry-v2-heuristic", "native-roadmarking-v1",
            "roadmarkingextraction-v1",
        ) or asset.detection_method.endswith("-merged")


def test_no_color_input_warns_and_color_input_does_not(synthetic_las: Path, tmp_path: Path) -> None:
    """A no-RGB export must warn that paint detection is degraded (honesty gate)."""
    import laspy

    las = laspy.read(synthetic_las)
    header = laspy.LasHeader(point_format=1, version="1.2")  # drops RGB
    header.scales = list(las.header.scales)
    header.offsets = list(las.header.offsets)
    cloud = laspy.LasData(header)
    dims = set(las.point_format.dimension_names)
    for name in ("x", "y", "z", "intensity", "return_number", "number_of_returns",
                 "classification", "gps_time", "point_source_id"):
        if name.upper() in dims:
            setattr(cloud, name, getattr(las, name).copy())
    no_color = tmp_path / "no-color.las"
    cloud.write(no_color)

    settings = ProcessingSettings(save_tiles=False)
    result = process_las(no_color, tmp_path / "out", settings, progress=False)
    assert any("no color" in warning.lower() for warning in result.warnings)
    assert result.assets  # the pipeline still runs end to end

    color_result = process_las(synthetic_las, tmp_path / "out_color", settings, progress=False)
    assert not any("no color" in warning.lower() for warning in color_result.warnings)


def test_linear_safety_merge_rebuilds_tile_fragments(tmp_path: Path) -> None:
    """Guardrail/barrier pieces that a tile boundary cut are re-joined into spans."""
    import numpy as np
    from infra_inventory.models import Asset
    from infra_inventory.instances import (
        component_metrics,
        merge_linear_safety_pieces,
    )

    settings = ProcessingSettings()
    # Two collinear one-tile pieces of the same roadside run.
    a = Asset(
        asset_id="", asset_class="guardrail", subclass="roadside_barrier",
        center={"x": 10.0, "y": 0.0, "z": 0.6},
        bounding_box=(6.0, -0.4, 0.2, 14.0, 0.4, 1.1),
        dimensions={"length_m": 8.0, "width_m": 0.8, "height_m": 0.9},
        point_count=80, source_tile="tile_A",
        source_point_indices_sample=[], coordinate_reference_system=None,
        confidence=0.85, confidence_factors={"geometry": 0.85},
        confidence_explanation="", detection_method="geometry-v1",
        processing_version="0.3.0",
    )
    b = Asset(
        asset_id="", asset_class="guardrail", subclass="roadside_barrier",
        center={"x": 18.0, "y": 0.0, "z": 0.6},
        bounding_box=(14.0, -0.4, 0.2, 22.0, 0.4, 1.1),
        dimensions={"length_m": 8.0, "width_m": 0.8, "height_m": 0.9},
        point_count=80, source_tile="tile_B",
        source_point_indices_sample=[], coordinate_reference_system=None,
        confidence=0.85, confidence_factors={"geometry": 0.85},
        confidence_explanation="", detection_method="geometry-v1",
        processing_version="0.3.0",
    )
    _raw_metrics_a = component_metrics(
        np.array([6.0, 10.0, 6.0]),
        np.array([-0.4, 0.0, 0.0]),
        np.array([0.2, 1.1, 0.2]),
        np.array([0, 1, 2], dtype=np.int64),
    )
    _raw_metrics_a["orientation_deg"] = 0.0
    _raw_metrics_a["centroid_z"] = 0.6
    a._metrics = _raw_metrics_a
    _raw_metrics_b = component_metrics(
        np.array([14.0, 18.0, 6.0]),
        np.array([-0.4, 0.0, 0.0]),
        np.array([0.2, 1.1, 0.2]),
        np.array([0, 1, 2], dtype=np.int64),
    )
    _raw_metrics_b["orientation_deg"] = 0.0
    _raw_metrics_b["centroid_z"] = 0.6
    b._metrics = _raw_metrics_b
    c = Asset(
        asset_id="", asset_class="utility_pole", subclass="vertical_support",
        center={"x": 14.0, "y": 5.0, "z": 5.0},
        bounding_box=(13.8, 4.8, 0.0, 14.2, 5.2, 5.2),
        dimensions={"length_m": 0.4, "width_m": 0.4, "height_m": 5.2},
        point_count=60, source_tile="tile_B",
        source_point_indices_sample=[], coordinate_reference_system=None,
        confidence=0.9, confidence_factors={"geometry": 0.9},
        confidence_explanation="", detection_method="geometry-v1",
        processing_version="0.3.0",
    )
    assets = merge_linear_safety_pieces([a, b, c], guardrail_max_gap_m=25.0)
    guardrails = [a for a in assets if a.asset_class == "guardrail"]
    assert len(guardrails) == 1
    assert guardrails[0].point_count == 160
    assert guardrails[0].source_tile == "tile_A+tile_B"
    assert len([a for a in assets if a.asset_class == "utility_pole"]) == 1


def test_elongated_compact_candidate_classified_as_guardrail(tmp_path: Path) -> None:
    """Long, low, narrow, above-ground candidate is routed to a linear safety class."""
    import numpy as np
    from collections import Counter

    from infra_inventory.assets.safety import detect_safety
    from infra_inventory.assets.common import TileContext
    from infra_inventory.models import ProcessingSettings

    settings = ProcessingSettings()
    x = np.linspace(0.0, 10.0, 200)
    y = np.zeros_like(x)
    z = np.full_like(x, 0.6)
    ctx = TileContext(
        tile="t", x=x, y=y, z=z, height=z, ground=0.0, intensity=np.zeros(200),
        rgb=None, source_indices=np.arange(200, dtype=np.int64), gps_labels=None,
        point_source_id=None, crs=None, model_class=None, class_names=[],
        mapping={}, settings=settings, id_counts=Counter(), sample_stride=1,
        viewer_limit=120_000,
    )
    all_assets = detect_safety(ctx)
    guardrails = [a for a in all_assets if a.asset_class == "guardrail"]
    cabinets = [a for a in all_assets if a.asset_class == "utility_cabinet"]
    print("guardrails:", guardrails)
    print("cabinets:", cabinets)
    print("all_assets:", [a.asset_class for a in all_assets])
    assert guardrails
    assert not cabinets


def test_linear_safety_bar_for_short_random_walk(tmp_path: Path) -> None:
    """Longer, low, narrow bar is still detected; compact tube is not."""
    import numpy as np
    from collections import Counter

    from infra_inventory.assets.safety import detect_safety
    from infra_inventory.assets.common import TileContext
    from infra_inventory.models import ProcessingSettings

    settings = ProcessingSettings()
    rng = np.random.default_rng(11)
    x_long = np.linspace(0.0, 10.0, 200)
    y_long = rng.normal(0.0, 0.05, len(x_long))
    z_long = np.full_like(x_long, 0.6)
    ctx_long = TileContext(
        tile="t_long", x=x_long, y=y_long, z=z_long, height=z_long, ground=0.0,
        intensity=np.zeros(200), rgb=None,
        source_indices=np.arange(200, dtype=np.int64), gps_labels=None,
        point_source_id=None, crs=None, model_class=None, class_names=[],
        mapping={}, settings=settings, id_counts=Counter(), sample_stride=1,
        viewer_limit=120_000,
    )
    long_assets = detect_safety(ctx_long)
    long_guardrails = [a for a in long_assets if a.asset_class == "guardrail"]
    long_cabinets = [a for a in long_assets if a.asset_class == "utility_cabinet"]
    assert long_guardrails
    assert not long_cabinets

    x_small = rng.uniform(0.0, 1.0, 40)
    y_small = rng.normal(0.0, 0.05, len(x_small))
    z_small = rng.normal(0.6, 0.05, len(x_small))
    ctx_small = TileContext(
        tile="t_small", x=x_small, y=y_small, z=z_small, height=z_small, ground=0.0,
        intensity=np.zeros(40), rgb=None,
        source_indices=np.arange(40, dtype=np.int64), gps_labels=None,
        point_source_id=None, crs=None, model_class=None, class_names=[],
        mapping={}, settings=settings, id_counts=Counter(), sample_stride=1,
        viewer_limit=120_000,
    )
    small_assets = detect_safety(ctx_small)
    small_guardrails = [a for a in small_assets if a.asset_class == "guardrail"]
    small_cabinets = [a for a in small_assets if a.asset_class == "utility_cabinet"]
    assert not small_guardrails
    assert not small_cabinets
