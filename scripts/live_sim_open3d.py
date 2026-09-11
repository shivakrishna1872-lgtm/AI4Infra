#!/usr/bin/env python3
"""Live 3D flythrough over a LAS/LAZ point cloud (desktop Open3D viewer).

Optional helper for a *local* machine with a display. The Terra Point web app has
its own browser 3D viewer + Quick Simulation, so this script is purely a
convenience for inspecting raw clouds:

* Streaming ingestion: laspy + the lazrs (LASzip) backend reads the file in
  chunks, then applies uniform thinning (LAStools las2las -thin style), so a
  multi-hundred-million-point tile never loads whole.
* Local-origin normalization: UTM/State-Plane coordinates are shifted by their
  minimums before rendering to avoid float32 GPU jitter (the same trick the web
  viewer uses).
* A frame-by-frame camera loop (Open3D's view-control rotate) simulates a
  drive/fly survey past the cloud.

Run on a machine with a display and Open3D installed:

    pip install "laspy[lazrs]" open3d numpy
    python scripts/live_sim_open3d.py TX_BurnetCo_2006_000014.laz --voxel 1.0

Flags: --stride N (keep every Nth point), --max-points N (auto-stride), --voxel M,
--no-autopilot (manual orbit instead of the flythrough).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_cloud(path: Path, *, stride: int = 1, max_points: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Stream + thin a LAS/LAZ into (xyz_local, colors). Never loads the file whole."""
    import laspy

    with laspy.open(path) as reader:
        metadata_count = int(reader.header.point_count)
        if max_points and metadata_count > max_points:
            stride = max(stride, metadata_count // max_points)
        keep = []
        for raw in reader.chunk_iterator(500_000):
            chunk = raw[::stride] if stride > 1 else raw
            xyz = np.column_stack((np.asarray(chunk.x), np.asarray(chunk.y), np.asarray(chunk.z)))
            if np.isfinite(xyz).all():
                keep.append(xyz)
        xyz = np.vstack(keep) if keep else np.empty((0, 3))

    # Local origin: shift by the minimums so float32 rendering has full precision.
    mins = xyz.min(axis=0)
    xyz_local = (xyz - mins).astype(np.float64)

    # Height-based color ramp (red = high, blue = low), as in the spec.
    z = xyz_local[:, 2]
    if len(z) and z.max() > z.min():
        ramp = (z - z.min()) / (z.max() - z.min())
        colors = np.column_stack((ramp, 0.25 + 0.35 * (1 - ramp), 1.0 - ramp))
    else:
        colors = np.tile([0.6, 0.6, 0.6], (len(xyz_local), 1))
    print(f"Loaded {len(xyz_local):,} points (source {metadata_count:,}, stride {stride}) "
          f"-> local origin {mins.round(2).tolist()}")
    return xyz_local.astype(np.float32), colors.astype(np.float64)


def fly(vis, pcd: object, frames: int = 360) -> None:
    """Rotate the camera smoothly around the scene ~30 fps (0.03 s per frame)."""
    import time

    ctr = vis.get_view_control()
    try:
        ctr.set_front([0.0, -1.0, 0.35])
        ctr.set_up([0.0, 0.0, 1.0])
    except Exception:
        pass
    for frame in range(frames):
        ctr.rotate(4.0, 0.0)
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.03)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="Input .las / .laz file")
    parser.add_argument("--stride", type=int, default=1, help="Keep every Nth point")
    parser.add_argument("--max-points", type=int, default=0,
                        help="Auto-stride so at most this many points are kept")
    parser.add_argument("--voxel", type=float, default=0.0,
                        help="Optional voxel downsample size (metres) after load")
    parser.add_argument("--frames", type=int, default=360, help="Flythrough frames")
    parser.add_argument("--no-autopilot", action="store_true", help="Manual orbit only")
    args = parser.parse_args()

    import open3d as o3d  # desktop-only import (not installed / headless-safe elsewhere)

    path = Path(args.input).expanduser().resolve()
    if not path.is_file():
        print(f"Input not found: {path}", file=sys.stderr)
        return 1
    xyz, colors = load_cloud(path, stride=args.stride, max_points=args.max_points)
    if len(xyz) == 0:
        print("No points survived thinning.", file=sys.stderr)
        return 2

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    if args.voxel > 0:
        pcd = pcd.voxel_down_sample(voxel_size=args.voxel)
        print(f"Voxel-downsampled to {len(pcd.points):,} points")

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Terra Point live LiDAR flythrough", width=1280, height=720)
    vis.add_geometry(pcd)
    if args.no_autopilot:
        vis.run()
    else:
        fly(vis, pcd, frames=args.frames)
    vis.destroy_window()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
