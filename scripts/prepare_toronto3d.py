"""Prepare the Toronto-3D dataset for OpenPCSeg fine-tuning.

Toronto-3D (Tan et al., CVPRW 2020) is a mobile-LiDAR road survey — the closest
public analogue to the Mannford MX9 capture: vehicle-mounted MLS with
XYZ / intensity / RGB and an 8-class taxonomy that covers every competition
category (Road 1, Road marking 2, Natural 3, Building 4, Utility line 5, Pole
6, Car 7, Fence 8).

The dataset ships as binary PLY tiles under ``original_ply/`` with a
``scalar_Label`` class field (verified against the upstream preparation script
``data_prepare_toronto3d.py`` in WeikaiTan/RandLA-Net). This script converts
them into the SemanticKITTI-convention training pairs consumed by OpenPCSeg's
dataset loaders (``.bin`` float32 tensors + ``.label`` uint32 files), using the
upstream-documented train/test split (train = L001/L003/L004, test = L002) and
the upstream-documented UTM offset (``UTM_OFFSET = [627285, 4841948, 0]``) to
keep float32 precision (README "Data preparation tip").

Input layout (after unpacking the dataset download):

    <input>/original_ply/L001.ply L002.ply L003.ply L004.ply

Output layout:

    <output>/sequences/00/velodyne/*.bin + labels/*.label   (train split)
    <output>/sequences/08/velodyne/*.bin + labels/*.label   (test split, L002)
    <output>/toronto3d_manifest.json

Usage:

    python scripts/prepare_toronto3d.py --input /data/Toronto_3D --output data_root/toronto3d
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

# Toronto-3D class ids (upstream README, "Classes" section).
TORONTO3D_CLASSES = {
    0: "unclassified",
    1: "road",
    2: "road_marking",
    3: "natural",
    4: "building",
    5: "utility_line",
    6: "pole",
    7: "car",
    8: "fence",
}

# Upstream-documented UTM offset (README "Data preparation tip").
TORONTO3D_UTM_OFFSET = (627285.0, 4841948.0, 0.0)

# Upstream-documented split (WeikaiTan/RandLA-Net main_Toronto3D.py).
TRAIN_TILES = ("L001", "L003", "L004")
TEST_TILES = ("L002",)

PLY_CHUNK_ROWS = 1_000_000

# PLY scalar types -> (numpy dtype, byte size)
_PLY_TYPES = {
    "char": "i1", "uchar": "u1", "short": "i2", "ushort": "u2",
    "int": "i4", "uint": "u4", "float": "f4", "double": "f8",
    "int8": "i1", "uint8": "u1", "int16": "i2", "uint16": "u2",
    "int32": "i4", "uint32": "u4", "float32": "f4", "float64": "f8",
}


def _parse_ply_header(path: Path) -> Tuple[str, List[Tuple[str, str]], int]:
    """Return (format, [(property_name, numpy_base_type)], vertex_count)."""
    with path.open("rb") as handle:
        assert handle.readline().strip() == b"ply", f"{path} is not a PLY file"
        fmt = ""
        count = 0
        properties: List[Tuple[str, str]] = []
        in_vertex = False
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"PLY header ended without end_header: {path}")
            text = line.decode("ascii", errors="replace").strip()
            if not text:
                continue
            parts = text.split()
            keyword = parts[0]
            if keyword == "format":
                fmt = parts[1]
            elif keyword == "element":
                in_vertex = parts[1] == "vertex"
                if in_vertex:
                    count = int(parts[2])
            elif keyword == "property" and in_vertex:
                if parts[1] == "list":
                    raise ValueError(f"List properties are not supported in vertices: {path}")
                properties.append((parts[-1], _PLY_TYPES[parts[1]]))
            elif keyword == "end_header":
                break
        if fmt not in ("binary_little_endian", "ascii"):
            raise ValueError(f"Unsupported PLY format '{fmt}': {path}")
        return fmt, properties, count


def iter_ply_vertices(path: Path) -> Iterator[Dict[str, np.ndarray]]:
    """Yield dict-of-column chunks for a binary_little_endian PLY vertex element."""
    fmt, properties, count = _parse_ply_header(path)
    if fmt != "binary_little_endian":
        raise ValueError(
            f"ASCII PLY is not supported (re-download the binary release): {path}"
        )
    dtype = np.dtype([(name, base) for name, base in properties])
    itemsize = dtype.itemsize
    with path.open("rb") as handle:
        # Skip the header: find end_header's newline.
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"PLY header ended without end_header: {path}")
            if line.strip() == b"end_header":
                break
        remaining = count
        while remaining > 0:
            rows = min(PLY_CHUNK_ROWS, remaining)
            buffer = handle.read(rows * itemsize)
            if len(buffer) != rows * itemsize:
                raise ValueError(f"Truncated PLY vertex data: {path}")
            block = np.frombuffer(buffer, dtype=dtype, count=rows)
            yield {name: block[name] for name, _ in properties}
            remaining -= rows


def convert_tile(ply_path: Path, velodyne_dir: Path, labels_dir: Path, offset: Tuple[float, float, float]) -> dict:
    """Convert one Toronto-3D PLY tile to .bin/.label pairs; returns stats."""
    velodyne_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    stem = ply_path.stem
    points_written = 0
    labels_written = 0
    chunks = 0
    for index, vertices in enumerate(iter_ply_vertices(ply_path)):
        x = vertices["x"].astype(np.float64) - offset[0]
        y = vertices["y"].astype(np.float64) - offset[1]
        z = vertices["z"].astype(np.float64) - offset[2]
        intensity_name = next((n for n in vertices if n.lower() in ("scalar_intensity", "intensity")), None)
        if intensity_name is None:
            raise ValueError(f"No intensity property in {ply_path}")
        intensity = vertices[intensity_name].astype(np.float64)
        features = np.column_stack((x, y, z, intensity)).astype(np.float32)
        features.tofile(velodyne_dir / f"{stem}_{index:05d}.bin")
        points_written += len(features)
        label_name = next((n for n in vertices if n.lower() in ("scalar_label", "label", "class")), None)
        labels = (
            vertices[label_name].astype(np.uint32)
            if label_name is not None
            else np.zeros(len(x), dtype=np.uint32)
        )
        labels.tofile(labels_dir / f"{stem}_{index:05d}.label")
        labels_written += len(labels)
        chunks += 1
    return {"tile": stem, "points": points_written, "labels": labels_written, "chunks": chunks}


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert Toronto-3D PLY tiles to OpenPCSeg training pairs")
    parser.add_argument("--input", required=True, help="Directory holding original_ply/L001..L004.ply")
    parser.add_argument("--output", required=True, help="Output data root for OpenPCSeg")
    args = parser.parse_args()

    input_dir = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    ply_dir = input_dir / "original_ply"
    if ply_dir.is_dir():
        input_dir = ply_dir
    if not input_dir.is_dir():
        print(f"Input directory not found: {input_dir}", file=sys.stderr)
        return 1

    manifest: Dict[str, object] = {
        "dataset": "Toronto-3D",
        "source": "https://github.com/WeikaiTan/Toronto-3D",
        "license": "CC BY-NC 4.0",
        "classes": TORONTO3D_CLASSES,
        "utm_offset": list(TORONTO3D_UTM_OFFSET),
        "train_tiles": list(TRAIN_TILES),
        "test_tiles": list(TEST_TILES),
        "sequences": {},
    }
    splits = (("00", TRAIN_TILES), ("08", TEST_TILES))
    for sequence, tiles in splits:
        stats: List[dict] = []
        for tile in tiles:
            ply_path = input_dir / f"{tile}.ply"
            if not ply_path.is_file():
                print(f"  WARNING: {ply_path.name} not found in {input_dir}; skipping", file=sys.stderr)
                continue
            print(f"Converting {ply_path.name} -> sequence {sequence} ...", flush=True)
            stats.append(convert_tile(
                ply_path,
                output_dir / "sequences" / sequence / "velodyne",
                output_dir / "sequences" / sequence / "labels",
                TORONTO3D_UTM_OFFSET,
            ))
        manifest["sequences"][sequence] = stats
        total = sum(int(entry["points"]) for entry in stats)
        print(f"  sequence {sequence}: {total:,} points across {len(stats)} tiles")

    manifest_path = output_dir / "toronto3d_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Manifest: {manifest_path}")
    print(
        "\nNext step: point an OpenPCSeg training config's DATA_PATH at "
        f"{output_dir / 'sequences'} (see docs/LEARNED_MODELS.md)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
