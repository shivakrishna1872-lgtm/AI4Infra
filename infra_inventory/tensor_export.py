"""MX9 LAS -> learned-model tensor export (SemanticKITTI ``.bin`` convention).

OpenPCSeg's SemanticKITTI dataset loaders consume per-scan ``.bin`` files of
``float32[~, 4]`` rows (x, y, z, intensity) with 16-bit labels in ``.label``
files (low 16 bits = class id). This module exports the pipeline's 40 m tiles
into exactly that layout so OpenPCSeg's ``infer.py`` / ``train.py`` run on them
unchanged.

Precision: upstream datasets store local coordinates in float32 (Toronto-3D's
README documents the same failure mode for UTM-scale coordinates — a large
origin destroys decimal resolution in float32). The offset is written to
``tensor_manifest.json`` so predictions exported by the network are transformed
back into the original CRS for asset detection.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

TENSOR_DIMS = 4  # x, y, z, intensity — SemanticKITTI convention


def export_tiles_to_tensors(
    tiles_dir: Path, output_dir: Path, tile_names: Optional[List[str]] = None,
    labels_dir: Optional[Path] = None,
) -> Dict[str, object]:
    """Export tiles to ``.bin`` tensors (plus optional ``.label`` files).

    Args:
        tiles_dir: per-tile LAS files written by the streaming pass.
        output_dir: destination for ``<tile>.bin`` files and the manifest.
        tile_names: restrict the export; default is every ``*.las`` in tiles_dir.
        labels_dir: optional directory of per-tile predicted label arrays to
            export as 16-bit ``.label`` files (training-pair export).

    Returns:
        Manifest dict (input offset, per-tile point counts).
    """
    import laspy

    tiles_dir = Path(tiles_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    las_paths = sorted(tiles_dir.glob("*.las"))
    if tile_names is not None:
        wanted = {f"{name}.las" for name in tile_names}
        las_paths = [path for path in las_paths if path.name in wanted]

    offset = np.zeros(3, dtype=np.float64)
    counts: Dict[str, int] = {}
    first_header = True
    for las_path in las_paths:
        with laspy.open(las_path) as reader:
            header = reader.header
            if first_header:
                offset = np.asarray(header.offsets, dtype=np.float64)
                first_header = False
            for chunk in reader.chunk_iterator(2_000_000):
                x = np.asarray(chunk.x, dtype=np.float64) - offset[0]
                y = np.asarray(chunk.y, dtype=np.float64) - offset[1]
                z = np.asarray(chunk.z, dtype=np.float64) - offset[2]
                intensity = np.asarray(
                    chunk.intensity if "intensity" in chunk.point_format.dimension_names
                    else np.zeros(len(x)),
                    dtype=np.float64,
                )
                features = np.column_stack((x, y, z, intensity)).astype(np.float32)
                features.tofile(output_dir / f"{las_path.stem}.bin")
                counts[las_path.stem] = counts.get(las_path.stem, 0) + len(x)
        if labels_dir is not None and (labels_dir / f"{las_path.stem}.npy").is_file():
            labels = np.load(labels_dir / f"{las_path.stem}.npy").astype(np.uint32)
            labels.astype(np.uint32).tofile(output_dir / f"{las_path.stem}.label")
    manifest = {
        "input_offset": offset.tolist(),
        "tensor_dims": TENSOR_DIMS,
        "point_counts": counts,
        "convention": "SemanticKITTI float32 [x, y, z, intensity]; labels uint32 low-16-bit class id",
    }
    (output_dir / "tensor_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_tensor_points(tensor_path: Path, offset: Optional[List[float]] = None) -> np.ndarray:
    """Load one ``.bin`` tensor back to float64 XYZ (+intensity), re-applying the offset."""
    rows = np.fromfile(str(tensor_path), dtype=np.float32).reshape(-1, TENSOR_DIMS)
    points = rows.astype(np.float64)
    if offset:
        points[:, :3] += np.asarray(offset, dtype=np.float64)
    return points
