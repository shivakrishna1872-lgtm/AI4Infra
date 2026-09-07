"""MX9 dual-head / dual-run merge-clean: streaming merge, cell dedupe, provenance."""
from __future__ import annotations

from pathlib import Path

import laspy
import numpy as np

from infra_inventory.merge import merge_las_files


def _write_las(path: Path, x_off: float, n: int = 100, intensity: int = 500) -> None:
    header = laspy.LasHeader(point_format=7, version="1.4")
    header.scales = [0.001, 0.001, 0.001]
    header.offsets = [0.0, 0.0, 0.0]
    las = laspy.LasData(header)
    las.x = np.arange(n) * 0.1 + x_off
    las.y = np.zeros(n)
    las.z = np.zeros(n)
    las.intensity = np.full(n, intensity)
    las.red = las.green = las.blue = np.full(n, 4000, dtype=np.uint16)
    las.point_source_id = np.full(n, 1, dtype=np.uint16)
    las.write(path)


def _read(path: Path) -> laspy.LasData:
    with laspy.open(path) as reader:
        return next(reader.chunk_iterator(10_000_000))


def test_merge_removes_cross_file_duplicates(tmp_path: Path) -> None:
    a, b = tmp_path / "run1_left.las", tmp_path / "run1_right.las"
    _write_las(a, 0.0)      # x = 0.0..9.9
    _write_las(b, 5.0, 80)  # overlaps x = 5.0..9.9 (50 pts), adds x = 10.0..12.9
    out = tmp_path / "merged.las"
    summary = merge_las_files([a, b], out, dedupe_cell_m=0.05)
    assert summary["points_read"] == 180
    assert summary["points_kept"] == 130
    assert summary["duplicates_removed"] == 50
    assert summary["point_format"] == 7
    assert summary["las_version"] == "1.4"
    merged = _read(out)
    assert len(merged.x) == 130


def test_merge_stamps_scanner_ids_per_input(tmp_path: Path) -> None:
    a, b = tmp_path / "a.las", tmp_path / "b.las"
    _write_las(a, 0.0)
    _write_las(b, 5.0, 80)
    out = tmp_path / "merged.las"
    summary = merge_las_files([a, b], out, dedupe_cell_m=0.05)
    assert summary["scanner_ids"] == [1, 2]
    merged = _read(out)
    # Points in x >= 10.0 only exist in file b -> must carry id 2.
    ids = np.asarray(merged.point_source_id)
    xs = np.asarray(merged.x)
    assert set(ids.tolist()) == {1, 2}
    assert set(ids[xs >= 10.0].tolist()) == {2}
    assert set(ids[xs < 5.0].tolist()) == {1}


def test_merge_preserves_rgb_and_intensity(tmp_path: Path) -> None:
    a, b = tmp_path / "a.las", tmp_path / "b.las"
    _write_las(a, 0.0, intensity=900)
    _write_las(b, 5.0, 80, intensity=300)
    out = tmp_path / "merged.las"
    merge_las_files([a, b], out, dedupe_cell_m=0.05)
    merged = _read(out)
    assert set(np.asarray(merged.intensity).tolist()) == {300, 900}
    assert int(merged.red.max()) == 4000


def test_merge_writes_laz_with_lazrs(tmp_path: Path) -> None:
    a, b = tmp_path / "a.las", tmp_path / "b.las"
    _write_las(a, 0.0)
    _write_las(b, 5.0, 80)
    out = tmp_path / "merged.laz"
    summary = merge_las_files([a, b], out, dedupe_cell_m=0.05)
    assert summary["points_kept"] == 130
    merged = _read(out)
    assert len(merged.x) == 130


def test_merge_cell_zero_concatenates_without_dedupe(tmp_path: Path) -> None:
    a, b = tmp_path / "a.las", tmp_path / "b.las"
    _write_las(a, 0.0)
    _write_las(b, 0.0)  # identical points
    out = tmp_path / "merged.las"
    summary = merge_las_files([a, b], out, dedupe_cell_m=0.0)
    assert summary["duplicates_removed"] == 0
    assert summary["points_kept"] == 200


def test_merge_warns_on_crs_mismatch(monkeypatch, tmp_path: Path) -> None:
    """Inputs with different CRS values trigger a warning (pyproj is not
    required for the warning itself, so the headers are stubbed in-memory)."""
    from infra_inventory.merge import merge_las_files

    def make_reader(crs_value):
        header = laspy.LasHeader(point_format=7, version="1.4")
        header.scales = [0.001] * 3
        header.offsets = [0.0] * 3
        header.crs = crs_value  # instance attribute shadows the header getattr
        las = laspy.LasData(header)
        las.x = np.arange(10) * 0.1
        las.y = np.zeros(10)
        las.z = np.zeros(10)
        las.intensity = np.full(10, 500)
        las.point_source_id = np.full(10, 1, dtype=np.uint16)

        class FakeReader:
            def __init__(self, path):
                self.header = header

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def chunk_iterator(self, chunk_size):
                yield las

        return FakeReader

    from infra_inventory import merge as merge_module
    original_open = merge_module.laspy.open
    crs_by_path = {"a.las": "EPSG:2248", "b.las": "EPSG:4326"}

    def fake_open(path, *args, **kwargs):
        if "mode" in kwargs:  # output writes go to the real writer
            return original_open(path, *args, **kwargs)
        return make_reader(crs_by_path[Path(path).name])(path)

    monkeypatch.setattr(merge_module.laspy, "open", fake_open)

    out = tmp_path / "merged.las"
    summary = merge_las_files([tmp_path / "a.las", tmp_path / "b.las"], out, dedupe_cell_m=0.05)
    assert any("CRS" in warning for warning in summary["warnings"])