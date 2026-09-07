"""Tests for scripts/prepare_toronto3d.py against tiny synthetic PLY tiles."""
from __future__ import annotations

import importlib.util
import json
import struct
import sys
from pathlib import Path

import numpy as np

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_toronto3d.py"
_spec = importlib.util.spec_from_file_location("prepare_toronto3d", _SCRIPT)
assert _spec is not None and _spec.loader is not None
prep = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("prepare_toronto3d", prep)
_spec.loader.exec_module(prep)


def _write_ply(path: Path, count: int, seed: int, labeled: bool = True) -> None:
    """Write a minimal binary_little_endian PLY matching Toronto-3D's layout."""
    rng = np.random.default_rng(seed)
    x = (627285.0 + rng.uniform(0, 40, count)).astype(np.float64)
    y = (4841948.0 + rng.uniform(0, 40, count)).astype(np.float64)
    z = rng.uniform(0, 15, count).astype(np.float64)
    intensity = rng.integers(0, 256, count).astype(np.uint8)
    label = rng.integers(0, 9, count).astype(np.uint8) if labeled else np.zeros(count, np.uint8)
    red = rng.integers(0, 256, count).astype(np.uint8)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {count}\n"
        "property double x\n"
        "property double y\n"
        "property double z\n"
        "property uchar scalar_Intensity\n"
        "property uchar scalar_Label\n"
        "property uchar red\n"
        "end_header\n"
    )
    body = b"".join(struct.pack("<dddBBB", xi, yi, zi, int(ci), int(li), int(ri))
                    for xi, yi, zi, ci, li, ri in zip(x, y, z, intensity, label, red))
    path.write_bytes(header.encode("ascii") + body)


def test_header_parsing_and_chunks(tmp_path: Path) -> None:
    ply = tmp_path / "L001.ply"
    _write_ply(ply, count=2500, seed=1)
    fmt, properties, count = prep._parse_ply_header(ply)
    assert fmt == "binary_little_endian"
    assert count == 2500
    names = [name for name, _ in properties]
    assert names == ["x", "y", "z", "scalar_Intensity", "scalar_Label", "red"]

    chunks = list(prep.iter_ply_vertices(ply))
    total = sum(len(v["x"]) for v in chunks)
    assert total == 2500
    assert all(len(v["x"]) <= prep.PLY_CHUNK_ROWS for v in chunks)


def test_conversion_applies_utm_offset_and_writes_pairs(tmp_path: Path) -> None:
    ply = tmp_path / "original_ply" / "L001.ply"
    ply.parent.mkdir(parents=True)
    _write_ply(ply, count=1000, seed=2)

    out = tmp_path / "out"
    stats = prep.convert_tile(
        ply, out / "sequences" / "00" / "velodyne", out / "sequences" / "00" / "labels",
        prep.TORONTO3D_UTM_OFFSET,
    )
    assert stats["points"] == 1000 == stats["labels"]
    velodyne = out / "sequences" / "00" / "velodyne"
    labels = out / "sequences" / "00" / "labels"
    bins = sorted(velodyne.glob("*.bin"))
    lbls = sorted(labels.glob("*.label"))
    assert bins and len(bins) == len(lbls)

    rows = np.fromfile(bins[0], dtype=np.float32).reshape(-1, 4)
    assert rows[:, 0].min() >= -1.0  # UTM offset removed: local metres
    assert rows[:, 0].max() <= 41.0
    label_values = np.fromfile(lbls[0], dtype=np.uint32)
    assert label_values.max() <= 8  # Toronto-3D class ids


def test_end_to_end_main_split_and_manifest(tmp_path: Path, capsys: object) -> None:
    ply_dir = tmp_path / "original_ply"
    ply_dir.mkdir(parents=True)
    _write_ply(ply_dir / "L001.ply", count=600, seed=3)
    _write_ply(ply_dir / "L002.ply", count=400, seed=4)

    argv = ["prepare_toronto3d.py", "--input", str(tmp_path), "--output", str(tmp_path / "data")]
    original = sys.argv
    sys.argv = argv
    try:
        assert prep.main() == 0
    finally:
        sys.argv = original

    manifest = json.loads((tmp_path / "data" / "toronto3d_manifest.json").read_text())
    assert manifest["classes"]["6"] == "pole"  # JSON stringifies int keys
    assert manifest["train_tiles"] == ["L001", "L003", "L004"]
    assert manifest["test_tiles"] == ["L002"]
    assert manifest["utm_offset"] == [627285.0, 4841948.0, 0.0]
    # Train sequence 00 has L001; test sequence 08 has L002.
    train_points = sum(e["points"] for e in manifest["sequences"]["00"])
    test_points = sum(e["points"] for e in manifest["sequences"]["08"])
    assert train_points == 600 and test_points == 400
