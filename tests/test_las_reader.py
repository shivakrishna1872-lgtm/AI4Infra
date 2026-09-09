from __future__ import annotations

from pathlib import Path

import numpy as np

from infra_inventory.las_reader import gps_run_labels, iter_chunks, read_metadata


def _concat_las(parts: list) -> "laspy.LasData":
    """Concatenate LasData objects of the same point format (no laspy.concatenate
    in this laspy version), preserving every dimension."""
    import laspy

    fmt = parts[0].point_format
    scales = list(parts[0].header.scales)
    offsets = list(parts[0].header.offsets)
    header = laspy.LasHeader(point_format=fmt.id, version="1.4")
    concat = laspy.LasData(header)
    concat.points = laspy.ScaleAwarePointRecord.zeros(
        sum(len(p.points) for p in parts), point_format=fmt, scales=scales, offsets=offsets,
    )
    for dim in fmt.dimension_names:
        concat.points[dim] = np.concatenate([np.asarray(p[dim]) for p in parts])
    return concat


def test_metadata_las14_format7(synthetic_las: Path) -> None:
    metadata = read_metadata(synthetic_las)
    assert metadata.version == "1.4"
    assert metadata.point_format == 7
    assert metadata.point_count > 0
    assert metadata.has_intensity
    assert metadata.has_rgb
    assert metadata.has_nir is False  # NIR is a point-format-8 dimension, not 7
    assert metadata.has_gps_time
    assert metadata.has_point_source_id
    assert metadata.bounds[3] > metadata.bounds[0]
    assert metadata.bounds[5] > metadata.bounds[2]
    # synthetic LAS carries no CRS VLR, so the documented fallback is applied
    # (laspy override_srs analogue) and flagged, not silently guessed
    assert metadata.crs == "EPSG:6553"
    assert metadata.crs_fallback_used is True


def test_read_metadata_no_fallback_reports_unknown(synthetic_las: Path) -> None:
    metadata = read_metadata(synthetic_las, crs_fallback=None)
    assert metadata.crs is None
    assert metadata.crs_fallback_used is False


def test_streaming_chunks_preserve_points(synthetic_las: Path) -> None:
    total = 0
    chunks = 0
    for _, _, chunk in iter_chunks(synthetic_las, chunk_size=4_000):
        total += len(chunk.x)
        chunks += 1
        assert np.isfinite(chunk.x).all()
        assert np.isfinite(chunk.z).all()
    assert total == read_metadata(synthetic_las).point_count
    assert chunks >= 3  # ~10k points / 4k chunk size


def test_streaming_chunks_stride_thins_and_counts_source_points(synthetic_las: Path) -> None:
    """LAStools las2las -thin analogue: stride keeps every Nth point per chunk,
    and global offsets still count source points so provenance stays truthful."""
    full = [(len(chunk.x), chunk.global_offset) for _, _, chunk in iter_chunks(synthetic_las, chunk_size=4_000)]
    thinned = [(len(chunk.x), chunk.global_offset) for _, _, chunk in iter_chunks(synthetic_las, chunk_size=4_000, stride=3)]
    assert all(length < full_length for (length, _), (full_length, _) in zip(thinned, full))
    # global offsets keep counting source points (monotonic, non-overlapping)
    offsets = [offset for _, offset in thinned]
    assert offsets == sorted(offsets)
    assert offsets[0] == 0
    # thinning keeps a strict minority of every chunk
    total_source = read_metadata(synthetic_las).point_count
    total_kept = sum(length for length, _ in thinned)
    assert total_kept < total_source
    assert total_kept >= total_source // 3 - 10  # stride 3 keeps ~1/3 of points


def test_streaming_chunks_global_offsets(synthetic_las: Path) -> None:
    offsets = []
    for _, _, chunk in iter_chunks(synthetic_las, chunk_size=10_000):
        offsets.append(chunk.global_offset)
    assert offsets == sorted(offsets)
    assert offsets[0] == 0


def test_gps_run_labels_split() -> None:
    gps = np.concatenate((np.linspace(1_000_000, 1_100_000, 2000), np.linspace(6_000_000, 6_100_000, 2000)))
    labels = gps_run_labels(gps)
    assert labels is not None
    assert set(np.unique(labels)) == {1, 2}
    assert labels[:2000].max() == 1 and labels[2000:].min() == 2


def test_gps_run_labels_single_population() -> None:
    gps = np.linspace(1_000_000, 1_100_000, 2000)
    assert gps_run_labels(gps) is None


def test_format8_nir_metadata_and_chunks(tmp_path: Path) -> None:
    """LAS 1.4 / point format 8 (RGB + NIR) is read natively: metadata flags the
    infrared channel and every streamed chunk carries it."""
    import laspy

    header = laspy.LasHeader(point_format=8, version="1.4")
    header.scales = [0.001, 0.001, 0.001]
    data = laspy.LasData(header)
    n = 2000
    data.x = np.linspace(0.0, 100.0, n)
    data.y = np.linspace(0.0, 5.0, n)
    data.z = np.zeros(n)
    data.intensity = np.full(n, 2000, dtype=np.uint16)
    data.red = np.full(n, 1000, dtype=np.uint16)
    data.green = np.full(n, 1200, dtype=np.uint16)
    data.blue = np.full(n, 1400, dtype=np.uint16)
    data.nir = np.full(n, 30000, dtype=np.uint16)
    data.gps_time = np.linspace(1_000_000.0, 1_100_000.0, n)
    data.point_source_id = np.full(n, 7, dtype=np.uint16)
    path = tmp_path / "fmt8.las"
    data.write(path)

    metadata = read_metadata(path)
    assert metadata.point_format == 8
    assert metadata.has_rgb
    assert metadata.has_nir
    assert metadata.to_dict()["has_nir"] is True
    _, _, chunk = next(iter_chunks(path, chunk_size=10_000))
    assert chunk.nir is not None
    assert float(chunk.nir.max()) == 30000.0
    assert chunk.rgb is not None and chunk.rgb.shape[1] == 3


def test_tiling_is_lossless_format7_all_dimensions(tmp_path: Path) -> None:
    """Streaming tiling is lossless for LAS 1.4 fmt 7 (the competition format):
    every point lands in exactly one tile, and every dimension — including the
    ones the tile writer used to drop (classification, user_data, scan_angle,
    scanner_channel, synthetic/keypoint/withheld/overlap) — round-trips
    bit-exactly."""
    import laspy

    from infra_inventory.pipeline import _stream_tiles
    from infra_inventory.models import ProcessingSettings

    header = laspy.LasHeader(point_format=7, version="1.4")
    header.scales = [0.001, 0.001, 0.001]
    data = laspy.LasData(header)
    n = 10_000
    arange = np.arange(n)
    # Exact multiples of the scale -> float roundtrip is exact. Spread across
    # 6 tiles (3 x-tiles, 2 y-tiles) including negative tile indices.
    data.x = (arange % 6000).astype(np.float64) * 0.001
    data.y = ((arange // 3000) % 2).astype(np.float64) * 4.0 - 1.0
    data.z = (arange % 13).astype(np.float64) * 0.001
    data.intensity = ((arange * 7) % 65536).astype(np.uint16)
    data.return_number = (arange % 5 + 1).astype(np.uint8)
    data.number_of_returns = (arange % 4 + 1).astype(np.uint8)
    data.classification = (arange % 100).astype(np.uint8)
    data.user_data = (arange % 255).astype(np.uint8)
    data.scan_angle = ((arange % 361) - 180).astype(np.int16)
    data.scanner_channel = (arange % 4).astype(np.uint8)
    data.synthetic = (arange % 2).astype(bool)
    data.key_point = ((arange // 3) % 2).astype(bool)
    data.withheld = ((arange // 5) % 2).astype(bool)
    data.overlap = ((arange // 7) % 2).astype(bool)
    data.gps_time = (1_000_000.0 + arange).astype(np.float64)  # unique per point
    data.point_source_id = ((arange % 8) + 1).astype(np.uint16)
    data.red = ((arange * 3) % 65536).astype(np.uint16)
    data.green = ((arange * 5) % 65536).astype(np.uint16)
    data.blue = ((arange * 7) % 65536).astype(np.uint16)
    path = tmp_path / "fmt7.las"
    data.write(path)

    out = tmp_path / "out"
    summary, tile_names, _ = _stream_tiles(
        path, out, ProcessingSettings(tile_size_m=2.0, chunk_size=3000), progress=False, progress_callback=None,
    )
    assert summary.point_count == n
    assert len(tile_names) >= 3, "test must actually exercise cross-tile splitting"

    # Rebuild the full cloud from tiles and check per-point equality keyed on
    # the unique gps_time.
    parts = []
    for tile_name in tile_names:
        with laspy.open(out / "tiles" / f"{tile_name}.las") as reader:
            parts.append(reader.read())
    concat = _concat_las(parts)
    assert len(concat.points) == n
    order_in = np.argsort(data.gps_time)
    order_out = np.argsort(np.asarray(concat.gps_time))
    dims = [
        "x", "y", "z", "intensity", "return_number", "number_of_returns",
        "classification", "user_data", "scan_angle", "scanner_channel",
        "synthetic", "key_point", "withheld", "overlap", "gps_time",
        "point_source_id", "red", "green", "blue",
    ]
    for dim in dims:
        src = np.asarray(getattr(data, dim))[order_in]
        dst = np.asarray(getattr(concat, dim))[order_out]
        assert np.array_equal(src, dst), f"dimension {dim} changed through tiling"


def test_tiling_is_lossless_legacy_format0(tmp_path: Path) -> None:
    """Legacy point formats (fmt 0-5: scan_angle_rank, scan_direction_flag,
    edge_of_flight_line, classification_flags) also round-trip bit-exactly
    through streaming tiling."""
    import laspy

    from infra_inventory.pipeline import _stream_tiles
    from infra_inventory.models import ProcessingSettings

    header = laspy.LasHeader(point_format=0, version="1.2")
    header.scales = [0.001, 0.001, 0.001]
    data = laspy.LasData(header)
    n = 5_000
    arange = np.arange(n)
    data.x = (arange % 4000).astype(np.float64) * 0.001
    data.y = ((arange // 4000) % 2).astype(np.float64) * 4.0 - 1.0
    data.z = np.zeros(n)
    data.intensity = ((arange * 3) % 65536).astype(np.uint16)
    data.return_number = (arange % 4 + 1).astype(np.uint8)
    data.number_of_returns = (arange % 3 + 1).astype(np.uint8)
    data.scan_direction_flag = (arange % 2).astype(bool)
    data.edge_of_flight_line = ((arange // 3) % 2).astype(bool)
    data.classification = (arange % 32).astype(np.uint8)
    data.synthetic = ((arange // 5) % 2).astype(bool)
    data.key_point = ((arange // 7) % 2).astype(bool)
    data.withheld = ((arange // 9) % 2).astype(bool)
    data.scan_angle_rank = ((arange % 181) - 90).astype(np.int8)
    data.user_data = (arange % 255).astype(np.uint8)
    data.point_source_id = ((arange % 6) + 1).astype(np.uint16)
    path = tmp_path / "fmt0.las"
    data.write(path)

    out = tmp_path / "out"
    summary, tile_names, _ = _stream_tiles(
        path, out, ProcessingSettings(tile_size_m=2.0, chunk_size=2000), progress=False, progress_callback=None,
    )
    assert summary.point_count == n
    assert len(tile_names) >= 2
    parts = []
    for tile_name in tile_names:
        with laspy.open(out / "tiles" / f"{tile_name}.las") as reader:
            parts.append(reader.read())
    concat = _concat_las(parts)
    assert len(concat.points) == n
    # fmt 0 has no gps_time; (x, y) is a unique per-point key here.
    order_in = np.lexsort((data.y, data.x))
    order_out = np.lexsort((np.asarray(concat.y), np.asarray(concat.x)))
    for dim in [
        "x", "y", "z", "intensity", "return_number", "number_of_returns",
        "scan_direction_flag", "edge_of_flight_line", "classification",
        "synthetic", "key_point", "withheld", "scan_angle_rank", "user_data",
        "point_source_id",
    ]:
        src = np.asarray(getattr(data, dim))[order_in]
        dst = np.asarray(getattr(concat, dim))[order_out]
        assert np.array_equal(src, dst), f"dimension {dim} changed through tiling"


def test_read_metadata_missing_file(tmp_path: Path) -> None:
    from infra_inventory.errors import InputNotFoundError

    try:
        read_metadata(tmp_path / "nope.las")
    except InputNotFoundError:
        return
    raise AssertionError("Expected InputNotFoundError")