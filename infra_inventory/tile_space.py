"""Spatial tile + streaming point cloud export for the AI4Infra viewer.

This is the app's answer to "tile the point cloud and serve it for
range-request / overview streaming" without depending on PDAL or LAStools at
runtime. The pipeline already streams the input into one-per-tile LAS files
(``output/tiles/``); this module turns that tile tree into a *streaming point
cloud package*: a tiles manifest plus a voxel-downsampled overview that the
viewer can load in bounded chunks instead of a single multi-MB / multi-GB
point array.

Pipeline contract preserved:
- input is still a single LAS/LAZ (uploaded or streamed from cloud storage)
- processing still runs per-tile through the geometry detectors
- the viewer payload produced here is the same LOD background + asset overlay
  the rest of the app serves; this module only adds the *tile index* so a
  future viewer can lazy-load visible tiles instead of shipping one big blob.

File layout produced under ``<output>/tile-space/``:
    manifest.json            tile index (bounds, voxel, point count, file)
    overview.las             voxel-downsampled background cloud (LAS 1.4)
    overview.json            lightweight summary + color encoding for the viewer
    tile-<tx>-<ty>.las       per-tile LAS files from the pipeline tiling pass
    viewer/                  (copied) viewer payload that the frontend expects

The "COPC" wording in the user's notes is the goal (HTTP range-request
streaming, visible-region loading). Strictly speaking this package is not a
binary COPC file produced by PDAL — PDAL/LAStools are not installed in this
environment — but it satisfies the same workflow requirements for the app:
spatial tiling, an overview for fast loading, and a manifest the frontend can
use to load only what is visible.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .las_reader import gps_run_labels, iter_chunks
from .models import ProcessingSettings, RunSummary
from .preprocessing import estimate_ground_z, height_above_ground, voxel_downsample
from .viewer import write_viewer
from .viewer import elevation_color


def _tile_header(source_header):
    import laspy

    for version in (str(source_header.version), "1.2", "1.3", "1.4"):
        try:
            header = laspy.LasHeader(
                point_format=source_header.point_format, version=version
            )
            break
        except Exception:
            continue
    else:
        header = laspy.LasHeader(
            point_format=int(source_header.point_format.id), version="1.4"
        )
    header.scales = list(source_header.scales)
    header.offsets = list(source_header.offsets)
    try:
        header.vlrs = list(source_header.vlrs)
    except Exception:
        pass
    return header


def _read_tile(path: Path):
    import laspy

    with laspy.open(path) as reader:
        chunk = next(reader.chunk_iterator(20_000_000), None)
    if chunk is None:
        return None
    names = set(chunk.point_format.dimension_names)
    x = np.asarray(chunk.x, dtype=np.float64)
    y = np.asarray(chunk.y, dtype=np.float64)
    z = np.asarray(chunk.z, dtype=np.float64)
    intensity = np.asarray(chunk.intensity, dtype=np.float64) if "intensity" in names else None
    rgb = None
    if {"red", "green", "blue"}.issubset(names):
        rgb = np.column_stack((chunk.red.astype(np.float64), chunk.green.astype(np.float64), chunk.blue.astype(np.float64)))
    gps_time = np.asarray(chunk.gps_time, dtype=np.float64) if "gps_time" in names else None
    point_source_id = (
        np.asarray(chunk.point_source_id, dtype=np.int64)
        if "point_source_id" in names and chunk.point_source_id is not None
        else None
    )
    return {
        "path": path,
        "x": x,
        "y": y,
        "z": z,
        "intensity": intensity,
        "rgb": rgb,
        "gps_time": gps_time,
        "point_source_id": point_source_id,
        "point_count": int(len(x)),
    }


def _bounds_for_tile(tile: dict) -> Tuple[float, float, float, float, float, float]:
    x, y, z = tile["x"], tile["y"], tile["z"]
    return (float(x.min()), float(y.min()), float(z.min()), float(x.max()), float(y.max()), float(z.max()))


def _tile_resolution_m(tile: dict, target_points: int) -> float:
    x, y = tile["x"], tile["y"]
    area = max(float(np.ptp(x) * np.ptp(y)), 1e-6)
    return min(max(math.sqrt(area / max(target_points, 1)), 0.05), 8.0)


def build_tile_space(
    tiles_dir: Path,
    output_dir: Path,
    settings: Optional[ProcessingSettings] = None,
    tile_point_budget: int = 65536,
    overview_point_budget: int = 250000,
    progress: bool = True,
) -> Dict[str, Any]:
    """Build a streaming tile-space package from an existing pipeline tile tree.

    Returns a summary dict describing what was written.
    """
    settings = settings or ProcessingSettings()
    tiles_dir = tiles_dir.resolve()
    out = output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"

    tile_files = sorted(p for p in tiles_dir.glob("tile_*.las"))
    if not tile_files:
        raise ValueError(f"No tile_*.las files found under {tiles_dir}")

    tiles: List[Dict[str, Any]] = []
    all_x: List[np.ndarray] = []
    all_y: List[np.ndarray] = []
    all_z: List[np.ndarray] = []

    if progress:
        print(f"[tile-space] packaging {len(tile_files)} tiles", flush=True)

    for idx, tile_path in enumerate(tile_files):
        tile = _read_tile(tile_path)
        if tile is None:
            continue
        tile["tx"], tile["ty"] = _tile_resolution_m(tile, tile_point_budget), 0.0  # placeholder, replaced below
        tile["resolution_m"] = _tile_resolution_m(tile, tile_point_budget)
        tile["bounds"] = _bounds_for_tile(tile)
        tile["index"] = idx
        tiles.append(tile)
        all_x.append(tile["x"])
        all_y.append(tile["y"])
        all_z.append(tile["z"])
        if progress:
            print(
                f"  tile {tile_path.name}: {tile['point_count']:,} pts",
                flush=True,
            )

    if not tiles:
        raise ValueError("No readable tiles after packaging")

    xs = np.concatenate(all_x)
    ys = np.concatenate(all_y)
    zs = np.concatenate(all_z)
    xmin, ymin, zmin = float(xs.min()), float(ys.min()), float(zs.min())
    xmax, ymax, zmax = float(xs.max()), float(ys.max()), float(zs.max())

    for tile in tiles:
        tx = int(math.floor((tile["x"][0] if len(tile["x"]) else xmin) / settings.tile_size_m))
        ty = int(math.floor((tile["y"][0] if len(tile["y"]) else ymin) / settings.tile_size_m))
        tile["tx"], tile["ty"] = tx, ty

    # Overview background cloud: one voxel-downsampled LOD from the full tile set.
    overview_keep = voxel_downsample(xs, ys, zs, overview_point_budget)
    ov_x = xs[overview_keep]
    ov_y = ys[overview_keep]
    ov_z = zs[overview_keep]
    z01 = (ov_z - zmin) / max((zmax - zmin), 1e-6)
    ov_colors = elevation_color(z01)
    ov_colors_arr = np.asarray(ov_colors, dtype=np.float64)

    import laspy

    overview_path = out / "overview.las"
    header = laspy.LasHeader(point_format=3, version="1.4")
    header.scales = [0.01, 0.01, 0.01]
    header.offsets = [xmin, ymin, zmin]
    data = laspy.LasData(header)
    data.x, data.y, data.z = ov_x, ov_y, ov_z
    data.intensity = np.zeros(len(ov_x), dtype=np.uint16)
    data.red = (ov_colors_arr[:, 0] * 65535).round().astype(np.uint16)
    data.green = (ov_colors_arr[:, 1] * 65535).round().astype(np.uint16)
    data.blue = (ov_colors_arr[:, 2] * 65535).round().astype(np.uint16)

    with laspy.open(overview_path, mode="w", header=header) as writer:
        writer.write_points(data.points)

    overview_summary = {
        "point_count": int(len(ov_x)),
        "bounds": [xmin, ymin, zmin, xmax, ymax, zmax],
        "file": "overview.las",
        "color_mode": "elevation",
    }

    manifest = {
        "version": "1.0",
        "tile_size_m": float(settings.tile_size_m),
        "bounds": [xmin, ymin, zmin, xmax, ymax, zmax],
        "overview": overview_summary,
        "tiles": [
            {
                "file": f"tile-{t['tx']}-{t['ty']}.las",
                "tx": int(t["tx"]),
                "ty": int(t["ty"]),
                "point_count": int(t["point_count"]),
                "bounds": t["bounds"],
                "resolution_m": t["resolution_m"],
            }
            for t in tiles
        ],
        "tile_file_prefix": "tile-",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Copy tile files into the package with the manifest naming.
    for tile in tiles:
        src = tile["path"]
        dst = out / f"tile-{tile['tx']}-{tile['ty']}.las"
        if src != dst:
            shutil.copy2(src, dst)

    # Viewer payload: point class coloring is built from the per-tile assets,
    # but for a tile-space package without a full run summary we ship elevation
    # coloring and the manifest so the viewer can lazy-load tiles.
    viewer_dir = out / "viewer"
    writer = ViewerWriter(out)
    writer.write(
        points=np.column_stack((ov_x, ov_y, ov_z)).round(4).tolist(),
        colors=ov_colors,
        point_intensity=None,
        point_rgb=None,
        manifest=manifest,
        overview_point_count=int(len(ov_x)),
    )

    summary = {
        "tiles": len(tiles),
        "overview_points": int(len(ov_x)),
        "bounds": [xmin, ymin, zmin, xmax, ymax, zmax],
        "output": str(out),
        "manifest": str(manifest_path),
        "overview": str(overview_path),
    }
    if progress:
        print(f"[tile-space] wrote {summary}", flush=True)
    return summary


class ViewerWriter:
    """Write the viewer payload the frontend expects for a tile-space package."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir

    def write(
        self,
        points: List[List[float]],
        colors: List[List[float]],
        point_intensity: Optional[List[float]],
        point_rgb: Optional[List[List[float]]],
        manifest: Dict[str, Any],
        overview_point_count: int,
    ) -> None:
        viewer_dir = self.output_dir / "viewer"
        viewer_dir.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, Any] = {
            "points": points,
            "point_colors": colors,
            "point_rgb": point_rgb,
            "point_intensity": point_intensity,
            "manifest": manifest,
            "overview_point_count": overview_point_count,
            "tile_space": True,
            "source": "tile-space package",
        }
        (viewer_dir / "viewer-data.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        index = viewer_dir / "index.html"
        if not index.is_file():
            index.write_text(
                "<!doctype html><html><head><meta charset='utf-8'>"
                "<title>AI4Infra Tile Space</title></head><body>"
                "<pre>Tile-space package built. Open the app to view.</pre>"
                "</body></html>",
                encoding="utf-8",
            )


def run_tile_space_from_project(
    project_dir: Path,
    output_name: str = "tile-space",
    settings: Optional[ProcessingSettings] = None,
    tile_point_budget: int = 65536,
    overview_point_budget: int = 250000,
    progress: bool = True,
) -> Dict[str, Any]:
    """Convenience wrapper: build a tile-space package for a processed project."""
    tiles_dir = project_dir / "output" / "tiles"
    target = project_dir / output_name
    return build_tile_space(
        tiles_dir,
        target,
        settings=settings,
        tile_point_budget=tile_point_budget,
        overview_point_budget=overview_point_budget,
        progress=progress,
    )
