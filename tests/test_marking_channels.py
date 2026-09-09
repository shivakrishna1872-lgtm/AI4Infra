"""Radiometric-channel robustness for pavement-marking detection.

LAS 1.4 format 7/8 files can carry dimensions that are never filled (a
"no-color" export keeps RGB at zeros; some scans record zero intensity). An
all-zero channel must not vote "bright" - with floor = quantile = 0 every
near-ground point would pass and the whole road would become one giant
marking blob. The format-8 infrared (NIR) channel is the strongest road-paint
discriminator on mobile LiDAR, so it is fused in when present.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import laspy
import numpy as np

from infra_inventory.assets.common import TileContext, bright_near_ground_mask
from infra_inventory.models import ProcessingSettings
from infra_inventory.pipeline import process_las


def _grid(n_points: int = 500) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A flat 10x50 grid with one point per 0.5 m cell (zrange cells stay tiny)."""
    x = np.tile(np.arange(10) * 0.5, 50).astype(np.float64)
    y = np.repeat(np.arange(50) * 0.5, 10).astype(np.float64)
    z = np.zeros(len(x))
    return x[:n_points], y[:n_points], z[:n_points]


def _ctx(x, y, z, intensity, rgb=None, nir=None) -> TileContext:
    return TileContext(
        tile="tile_0_0", x=x, y=y, z=z,
        height=np.zeros(len(x)), ground=0.0,
        intensity=intensity, rgb=rgb, source_indices=np.arange(len(x)),
        gps_labels=None, point_source_id=None, crs=None,
        model_class=None, class_names=[], mapping={},
        settings=ProcessingSettings(), id_counts=Counter(),
        sample_stride=1, viewer_limit=1000, nir=nir,
    )


def test_alive_intensity_with_dead_rgb_marks_only_bright_points() -> None:
    x, y, z = _grid()
    intensity = np.full(len(x), 100.0)
    intensity[:50] = 60000.0  # 10% reflective (markings)
    rgb = np.zeros((len(x), 3))  # no-color export: dimension present, all zeros
    mask = bright_near_ground_mask(_ctx(x, y, z, intensity, rgb=rgb), height_max=0.35)
    assert mask[:50].all() and not mask[50:].any()


def test_alive_rgb_with_dead_intensity_marks_only_bright_points() -> None:
    x, y, z = _grid()
    intensity = np.zeros(len(x))  # unfilled intensity channel
    rgb = np.full((len(x), 3), 100.0)
    rgb[:50] = 60000.0
    mask = bright_near_ground_mask(_ctx(x, y, z, intensity, rgb=rgb), height_max=0.35)
    assert mask[:50].all() and not mask[50:].any()


def test_nir_alone_detects_markings_when_other_channels_dead() -> None:
    """Format 8: NIR is the paint discriminator even with dead intensity+RGB."""
    x, y, z = _grid()
    intensity = np.zeros(len(x))
    rgb = np.zeros((len(x), 3))
    nir = np.full(len(x), 100.0)
    nir[:50] = 60000.0
    mask = bright_near_ground_mask(_ctx(x, y, z, intensity, rgb=rgb, nir=nir), height_max=0.35)
    assert mask[:50].all() and not mask[50:].any()


def test_all_channels_dead_never_floods() -> None:
    x, y, z = _grid()
    intensity = np.zeros(len(x))
    rgb = np.zeros((len(x), 3))
    nir = np.zeros(len(x))
    mask = bright_near_ground_mask(_ctx(x, y, z, intensity, rgb=rgb, nir=nir), height_max=0.35)
    assert not mask.any()


def _as_format(path: Path, out: Path, point_format: int, dead_rgb: bool = False) -> Path:
    """Rewrite a LAS file as format 7 (RGB) or 8 (RGB+NIR), optionally zeroing RGB."""
    with laspy.open(path) as reader:
        header = laspy.LasHeader(point_format=point_format, version="1.4")
        header.scales = list(reader.header.scales)
        header.offsets = list(reader.header.offsets)
        data = laspy.LasData(header)
        chunks = list(reader.chunk_iterator(1_000_000))
        data.x = np.concatenate([np.asarray(c.x) for c in chunks])
        data.y = np.concatenate([np.asarray(c.y) for c in chunks])
        data.z = np.concatenate([np.asarray(c.z) for c in chunks])
        data.intensity = np.concatenate([np.asarray(c.intensity) for c in chunks])
        data.red = np.concatenate([np.asarray(c.red) for c in chunks])
        data.green = np.concatenate([np.asarray(c.green) for c in chunks])
        data.blue = np.concatenate([np.asarray(c.blue) for c in chunks])
        data.gps_time = np.concatenate([np.asarray(c.gps_time) for c in chunks])
        data.point_source_id = np.concatenate([np.asarray(c.point_source_id) for c in chunks])
        data.return_number = np.concatenate([np.asarray(c.return_number) for c in chunks])
        data.number_of_returns = np.concatenate([np.asarray(c.number_of_returns) for c in chunks])
        if point_format == 8:
            data.nir = data.intensity.copy()  # paint is NIR-bright exactly where it is intensity-bright
        if dead_rgb:
            data.red[:] = 0
            data.green[:] = 0
            data.blue[:] = 0
        data.write(out)
    return out


def test_format8_nir_survives_streaming_and_detects_markings(
    synthetic_las: Path, tmp_path: Path,
) -> None:
    path = _as_format(synthetic_las, tmp_path / "fmt8.las", 8)
    summary = process_las(path, tmp_path / "out", ProcessingSettings(), progress=False)
    assert any(a.asset_class == "pavement_marking" for a in summary.assets)
    assert not any("DEAD" in w for w in summary.warnings)
    # The NIR dimension must survive the streaming -> per-tile LAS write.
    tiles = sorted((tmp_path / "out" / "tiles").glob("tile_*.las"))
    assert tiles
    with laspy.open(tiles[0]) as reader:
        assert "nir" in set(reader.header.point_format.dimension_names)


def test_dead_rgb_channel_warns_and_never_floods_markings(
    synthetic_las: Path, tmp_path: Path,
) -> None:
    """A no-color fmt-7 export (RGB zeros) is reported and still detects markings."""
    path = _as_format(synthetic_las, tmp_path / "nocolor.las", 7, dead_rgb=True)
    summary = process_las(path, tmp_path / "out", ProcessingSettings(), progress=False)
    assert any("[RGB_DEAD]" in w for w in summary.warnings)
    markings = [a for a in summary.assets if a.asset_class == "pavement_marking"]
    assert markings
    # No giant blob: markings stay thin (a flooded road would be ~7 m wide and
    # rejected as a whole-surface "other"). Long lengths are fine - the merge
    # pass joins lane-line segments into continuous features on purpose.
    for marking in markings:
        assert marking.dimensions["width_m"] <= 2.0
        assert marking.subclass != "other"