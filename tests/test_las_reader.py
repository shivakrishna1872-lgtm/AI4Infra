from __future__ import annotations

from pathlib import Path

import numpy as np

from infra_inventory.las_reader import gps_run_labels, iter_chunks, read_metadata


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


def test_read_metadata_missing_file(tmp_path: Path) -> None:
    from infra_inventory.errors import InputNotFoundError

    try:
        read_metadata(tmp_path / "nope.las")
    except InputNotFoundError:
        return
    raise AssertionError("Expected InputNotFoundError")