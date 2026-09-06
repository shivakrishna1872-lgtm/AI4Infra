"""Regression tests for legacy LAS 1.0 / Point Data Record Format 1 inputs.

USGS 3DEP tiles from the mid-2000s are often LAS 1.0 / PDR 1. laspy cannot
*write* LAS 1.0 headers (``FileVersionNotSupported: 1.0``), so the first tile
write of the streaming pass used to crash the whole run at ~500k points with
``Processing failed: 1.0``. The pipeline now re-versions tile headers to a
writable LAS version while preserving scales/offsets/dimensions.

laspy cannot generate a LAS 1.0 file either, so this test builds one by hand
with ``struct`` (header layout per the LAS 1.0 spec).
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from infra_inventory.las_reader import read_metadata
from infra_inventory.pipeline import _stream_tiles, _tile_header
from infra_inventory.models import ProcessingSettings


def _write_las10(path: Path, n: int = 3_000) -> None:
    """Write a minimal LAS 1.0 / point format 1 file (227-byte header + 28-byte records)."""
    rng = np.random.default_rng(7)
    x = rng.uniform(0.0, 120.0, n)
    y = rng.uniform(-12.0, 12.0, n)
    z = rng.uniform(0.0, 6.0, n)
    intensity = rng.integers(0, 65535, n).astype(np.uint16)
    gps = np.linspace(1_000_000.0, 1_100_000.0, n)

    scale = 0.01
    offset = 0.0
    xs = np.round((x - offset) / scale).astype(np.int32)
    ys = np.round((y - offset) / scale).astype(np.int32)
    zs = np.round((z - offset) / scale).astype(np.int32)

    header = struct.pack(
        "<4sHHIHH8sBB32s32sHHHIIBHI5I3d3d6d",
        b"LASF",
        0,  # file source id
        0,  # global encoding
        0,  # project id 1
        0,  # project id 2
        0,  # project id 3
        b"\x00" * 8,  # project id 4
        1, 0,  # version 1.0
        b"AI4Infra test".ljust(32, b"\x00"),
        b"pytest".ljust(32, b"\x00"),
        0, 0,  # creation day/year
        227,  # header size
        227,  # offset to point data
        0,  # number of VLRs
        1,  # point data record format
        28,  # point data record length
        n,  # legacy number of point records
        0, 0, 0, 0, 0,  # legacy points by return
        scale, scale, scale,  # scales
        offset, offset, offset,  # offsets
        float(x.max()), float(x.min()), float(y.max()), float(y.min()),
        float(z.max()), float(z.min()),
    )
    assert len(header) == 227, len(header)

    records = bytearray()
    for i in range(n):
        records += struct.pack(
            "<iiiHBBbBHd",
            int(xs[i]), int(ys[i]), int(zs[i]),
            int(intensity[i]),
            1,  # return byte: return 1 of 1
            0,  # classification (0 = never classified, classic 3DEP)
            0,  # scan angle
            0,  # user data
            0,  # point source id
            float(gps[i]),
        )
    assert len(records) == n * 28
    path.write_bytes(header + bytes(records))


def test_metadata_reads_las10(tmp_path: Path) -> None:
    source = tmp_path / "legacy.las"
    _write_las10(source)
    metadata = read_metadata(source)
    assert metadata.version == "1.0"
    assert metadata.point_format == 1
    assert metadata.point_count == 3_000
    assert metadata.has_intensity and metadata.has_gps_time
    assert not metadata.has_rgb


def test_las10_tiles_are_rewritten_to_writable_version(tmp_path: Path) -> None:
    """The 500k crash regression: streaming a LAS 1.0 input must not hit
    FileVersionNotSupported when writing the first tile."""
    source = tmp_path / "legacy.las"
    _write_las10(source)

    settings = ProcessingSettings(chunk_size=1_000)  # 3 chunks, exercises all three
    output = tmp_path / "out"
    summary, tile_names, viewer = _stream_tiles(source, output, settings, progress=False, progress_callback=None)

    assert summary.point_count == 3_000
    assert len(tile_names) >= 3
    assert len(viewer["points"]) > 0
    # every tile is readable and written with a writable version + same format
    import laspy

    for name in tile_names:
        with laspy.open(output / "tiles" / f"{name}.las") as reader:
            version = f"{reader.header.version.major}.{reader.header.version.minor}"
            assert version >= "1.2"
            assert reader.header.point_format.id == 1
    # no warnings about a *failed* rewrite (a re-version notice is fine)
    assert not any("failed" in w.lower() for w in summary.warnings)


def test_las10_tile_header_preserves_scale_offset(tmp_path: Path) -> None:
    source = tmp_path / "legacy.las"
    _write_las10(source)
    import laspy

    with laspy.open(source) as reader:
        header = _tile_header(reader.header)
    assert tuple(header.scales) == (0.01, 0.01, 0.01)
    assert header.point_format.id == 1


def test_max_input_points_thins_las10(tmp_path: Path) -> None:
    """max_input_points (las2las -thin analogue) bounds huge inputs."""
    source = tmp_path / "legacy.las"
    _write_las10(source)
    settings = ProcessingSettings(chunk_size=1_000, max_input_points=600)
    output = tmp_path / "out"
    summary, tile_names, viewer = _stream_tiles(source, output, settings, progress=False, progress_callback=None)
    assert any("thinned" in w for w in summary.warnings)
    # ~600 target points -> stride 5 -> ~600 points kept, still covering the extent
    import laspy

    total = 0
    for name in tile_names:
        with laspy.open(output / "tiles" / f"{name}.las") as reader:
            total += int(reader.header.point_count)
    assert 400 <= total <= 900
    assert summary.point_count == 3_000  # provenance stays the true input count