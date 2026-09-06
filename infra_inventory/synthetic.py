"""Synthetic mobile-LiDAR scene generator.

Builds a deterministic LAS 1.4 / Point Data Record Format 7 scene containing a
road, a painted lane line, a utility pole, a traffic sign with support, and a
guardrail - enough to exercise every geometry detector. Synthetic data exists
ONLY to test the pipeline and run demos; it must never be presented as
competition results.
"""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Optional, Tuple

import laspy
import numpy as np

GROUND_SPACING = 0.35


def _gps_time_for(x: np.ndarray, pass_limit: float = 40.0) -> np.ndarray:
    """Simulate a two-pass survey so gps_time separates into Run 1 / Run 2."""
    first = x <= pass_limit
    base = 1000000.0 + x * 100.0
    return np.where(first, base, base + 5_000_000.0)


def build_synthetic_las(path: str | Path, ground_spacing: float = GROUND_SPACING, seed: int = 7) -> dict:
    """Write a synthetic LAS and return a summary dict describing the scene."""
    rng = np.random.default_rng(seed)

    # ---- ground plane (road + shoulders) -------------------------------------
    xs = np.arange(0.0, 80.0, ground_spacing)
    ys = np.arange(-20.0, 20.0, ground_spacing)
    gx, gy = np.meshgrid(xs, ys)
    ground_x = gx.ravel()
    ground_y = gy.ravel()
    ground_z = np.full_like(ground_x, 0.0) + rng.normal(0.0, 0.01, ground_x.shape)
    ground_i = rng.integers(30, 50, ground_x.shape).astype(np.float64)
    ground_rgb = np.tile(np.array([8000, 8000, 8000]), (len(ground_x), 1)) + rng.integers(-300, 300, (len(ground_x), 3))

    # ---- painted lane line ------------------------------------------------------
    line_x = np.arange(10.0, 60.0, 0.15)
    line_y = np.full_like(line_x, -1.65) + rng.normal(0.0, 0.02, line_x.shape)
    line_z = np.full_like(line_x, 0.02)
    line_i = rng.integers(18000, 22000, line_x.shape).astype(np.float64)
    line_rgb = np.tile(np.array([62000, 62000, 62000]), (len(line_x), 1))

    # ---- utility pole -------------------------------------------------------------
    pole_angles = np.linspace(0, 2 * np.pi, 14, endpoint=False)
    pole_zs = np.arange(0.06, 7.2, 0.18)
    pole_x = 25.0 + 0.12 * np.cos(pole_angles)[:, None] + rng.normal(0.0, 0.005, (len(pole_angles), len(pole_zs)))
    pole_y = 8.0 + 0.12 * np.sin(pole_angles)[:, None] + rng.normal(0.0, 0.005, (len(pole_angles), len(pole_zs)))
    pole_z = np.broadcast_to(pole_zs[None, :], pole_x.shape)
    pole_x, pole_y, pole_z = pole_x.ravel(), pole_y.ravel(), pole_z.ravel()
    pole_i = rng.integers(900, 1400, pole_x.shape).astype(np.float64)
    pole_rgb = np.tile(np.array([25000, 25000, 26000]), (len(pole_x), 1))

    # ---- traffic sign (panel + support post) ---------------------------------------
    panel_ys = np.arange(-10.4, -9.6, 0.06)
    panel_zs = np.arange(2.0, 2.9, 0.06)
    py, pz = np.meshgrid(panel_ys, panel_zs)
    panel_x = np.full_like(py, 45.0).ravel()
    panel_y = py.ravel()
    panel_z = pz.ravel()
    panel_i = rng.integers(1500, 2000, panel_x.shape).astype(np.float64)
    panel_rgb = np.tile(np.array([9000, 45000, 9000]), (len(panel_x), 1))  # green sign

    post_zs = np.arange(0.06, 0.98, 0.08)
    post_x = np.full_like(post_zs, 45.0)
    post_y = np.full_like(post_zs, -10.0)
    post_z = post_zs
    post_i = rng.integers(800, 1100, post_x.shape).astype(np.float64)
    post_rgb = np.tile(np.array([20000, 20000, 20000]), (len(post_x), 1))

    # ---- guardrail --------------------------------------------------------------------
    rail_x = np.arange(10.0, 70.0, 0.2)
    rail_z_levels = (0.75, 0.95)
    rails_x, rails_z, rails_y = [], [], []
    for level in rail_z_levels:
        rails_x.append(rail_x + rng.normal(0.0, 0.01, rail_x.shape))
        rails_y.append(np.full_like(rail_x, 12.0) + rng.normal(0.0, 0.03, rail_x.shape))
        rails_z.append(np.full_like(rail_x, level) + rng.normal(0.0, 0.01, rail_x.shape))
    rail_x = np.concatenate(rails_x)
    rail_y = np.concatenate(rails_y)
    rail_z = np.concatenate(rails_z)
    rail_i = rng.integers(600, 900, rail_x.shape).astype(np.float64)
    rail_rgb = np.tile(np.array([32000, 32000, 33000]), (len(rail_x), 1))

    # ---- assemble ------------------------------------------------------------------------
    x = np.concatenate([ground_x, line_x, pole_x, panel_x, post_x, rail_x])
    y = np.concatenate([ground_y, line_y, pole_y, panel_y, post_y, rail_y])
    z = np.concatenate([ground_z, line_z, pole_z, panel_z, post_z, rail_z])
    intensity = np.concatenate([ground_i, line_i, pole_i, panel_i, post_i, rail_i])
    rgb = np.concatenate([ground_rgb, line_rgb, pole_rgb, panel_rgb, post_rgb, rail_rgb]).astype(np.uint16)
    # scanner: point_source_id 1 = Laser Left, 2 = Laser Right (split by y sign)
    point_source_id = np.where(y >= 0, 2, 1).astype(np.uint16)
    gps_time = _gps_time_for(x)

    header = laspy.LasHeader(point_format=7, version="1.4")
    header.scales = [0.005, 0.005, 0.005]
    header.offsets = [round(float(x.mean())), round(float(y.mean())), 0.0]
    cloud = laspy.LasData(header)
    cloud.x, cloud.y, cloud.z = x, y, z
    cloud.intensity = intensity.astype(np.uint16)
    cloud.red, cloud.green, cloud.blue = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    cloud.gps_time = gps_time
    cloud.point_source_id = point_source_id
    cloud.return_number = np.ones(len(x), dtype=np.uint8)
    cloud.number_of_returns = np.ones(len(x), dtype=np.uint8)

    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    cloud.write(path)

    return {
        "path": str(path),
        "points": len(x),
        "scene": {
            "ground": len(ground_x),
            "lane_line": len(line_x),
            "utility_pole": len(pole_x),
            "sign_panel": len(panel_x),
            "sign_support": len(post_x),
            "guardrail": len(rail_x),
        },
        "bounds": [float(header.mins[i]) for i in range(3)] + [float(header.maxs[i]) for i in range(3)],
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }