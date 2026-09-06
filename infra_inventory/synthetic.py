"""Synthetic mobile-LiDAR scene generators.

Two generators share the same LAS 1.4 / Point Data Record Format 7 output contract:

* ``build_synthetic_las`` - a small deterministic scene used by the test suite
  (road, lane line, pole, sign, guardrail).
* ``build_simulated_las`` - a larger, more realistic mobile-LiDAR environment
  used for demos: a two-lane road with markings, crosswalk, utility poles with
  overhead conductors, traffic signs, guardrails, a concrete barrier, and
  LiDAR-like noise, occlusion and density variation.

SIMULATED DATA IS NOT COMPETITION DATA. It exists only to exercise the pipeline
and demo the platform; results derived from it must be labeled as simulation.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Dict, Optional

import laspy
import numpy as np

GROUND_SPACING = 0.35

#: Uniqueness helpers for scene randomization.
_PART_COUNTER: Dict[str, int] = {}


def _next_id(prefix: str) -> str:
    _PART_COUNTER.setdefault(prefix, 0)
    _PART_COUNTER[prefix] += 1
    return f"{prefix}-{_PART_COUNTER[prefix]:03d}"


def _gps_time_for(x: np.ndarray, pass_limit: float = 40.0) -> np.ndarray:
    """Simulate a two-pass survey so gps_time separates into Run 1 / Run 2."""
    first = x <= pass_limit
    base = 1000000.0 + x * 100.0
    return np.where(first, base, base + 5_000_000.0)


def _scatter_dropout(mask: np.ndarray, rng: np.random.Generator, fraction: float) -> np.ndarray:
    """Simulate occlusion: drop a random fraction of an object's points."""
    drop = rng.random(mask.shape) < fraction
    return mask & ~drop


def _rng_for(seed: Optional[int]) -> np.random.Generator:
    return np.random.default_rng(seed if seed is not None else int(time.time() * 1000) % (2**31))


def _intensity_for(kind: str, rng: np.random.Generator) -> np.ndarray:
    tables = {
        "asphalt": (400, 1200),
        "road_edge": (800, 1600),
        "painted_marking": (15000, 21000),
        "utility_pole": (700, 1500),
        "crossarm": (700, 1500),
        "conductor": (300, 700),
        "sign_panel": (2600, 3600),
        "sign_post": (800, 1300),
        "guardrail": (900, 1500),
        "guardrail_post": (800, 1200),
        "concrete_barrier": (500, 900),
    }
    lo, hi = tables.get(kind, (500, 1000))
    return rng.integers(lo, hi, 1).astype(np.float64)[0]


def _rgb_for(kind: str, rng: np.random.Generator) -> np.ndarray:
    tables = {
        "asphalt": (9500, 9500, 10000),
        "road_edge": (9500, 9500, 10000),
        "painted_marking": (63000, 63000, 63000),
        "utility_pole": (28000, 27000, 26000),
        "crossarm": (28000, 27000, 26000),
        "conductor": (6000, 6000, 7500),
        "sign_panel_green": (5000, 58000, 5000),
        "sign_panel_white": (60000, 60000, 60000),
        "sign_panel_yellow": (58000, 56000, 8000),
        "sign_post": (22000, 22000, 22000),
        "guardrail": (33000, 33000, 34000),
        "guardrail_post": (30000, 30000, 31000),
        "concrete_barrier": (26000, 26000, 27000),
    }
    base = np.array(tables.get(kind, (8000, 8000, 8000)), dtype=np.int64)
    jitter = rng.integers(-400, 400, 3)
    return np.clip(base + jitter, 0, 65535).astype(np.uint16)


def _panel_mesh(px: float, py: float, height: float, width: float, z0: float, rng: np.random.Generator, color_kind: str):
    ys = np.arange(py - width / 2, py + width / 2, 0.05)
    zs = np.arange(z0, z0 + height, 0.05)
    gy, gz = np.meshgrid(ys, zs)
    x = np.full_like(gy, px).ravel()
    y = gy.ravel()
    z = gz.ravel()
    mask = _scatter_dropout(np.ones(x.shape, dtype=bool), rng, 0.08)
    return (x[mask], y[mask], z[mask],
            np.full(mask.sum(), _intensity_for("sign_panel", rng)),
            np.tile(_rgb_for(color_kind, rng), (mask.sum(), 1)))


def _post_mesh(px: float, py: float, top_z: float, rng: np.random.Generator):
    pz = np.arange(0.05, top_z - 0.05, 0.1)
    keep = _scatter_dropout(np.ones(pz.shape, dtype=bool), rng, 0.3)
    return (np.full(keep.sum(), px),
            np.full(keep.sum(), py),
            pz[keep],
            np.full(keep.sum(), _intensity_for("sign_post", rng)),
            np.tile(_rgb_for("sign_post", rng), (keep.sum(), 1)))


def _cylinder_points(px: float, py: float, radius: float, zs: np.ndarray, rng: np.random.Generator, intensity_kind: str, rgb_kind: str):
    angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    x = px + radius * np.cos(angles)[:, None] + rng.normal(0.0, 0.006, (len(angles), len(zs)))
    y = py + radius * np.sin(angles)[:, None] + rng.normal(0.0, 0.006, (len(angles), len(zs)))
    z = np.broadcast_to(zs[None, :], x.shape)
    occluded = _scatter_dropout(np.ones(x.shape, dtype=bool), rng, 0.25)
    flat_x, flat_y, flat_z = x[occluded], y[occluded], z[occluded]
    n = int(occluded.sum())
    intensity = np.full(n, _intensity_for(intensity_kind, rng))
    rgb = np.tile(_rgb_for(rgb_kind, rng), (n, 1))
    return flat_x, flat_y, flat_z, intensity, rgb


def _crossarm_points(px: float, py: float, rng: np.random.Generator):
    arm = np.arange(-1.0, 1.0, 0.08)
    ax = np.full_like(arm, px)
    ay = py + arm
    az = np.full_like(arm, 10.2) + rng.normal(0.0, 0.01, arm.shape)
    mask = _scatter_dropout(np.ones(arm.shape, dtype=bool), rng, 0.2)
    n = int(mask.sum())
    return (ax[mask], ay[mask], az[mask],
            np.full(n, _intensity_for("crossarm", rng)),
            np.tile(_rgb_for("crossarm", rng), (n, 1)))


def _conductor_points(p1: tuple[float, float], p2: tuple[float, float], rng: np.random.Generator):
    (px1, py1), (px2, py2) = p1, p2
    dist = math.hypot(px2 - px1, py2 - py1)
    steps = max(2, int(dist / 0.35))
    t = np.linspace(0.0, 1.0, steps)
    sag = 1.1
    cx = px1 + (px2 - px1) * t
    cy = py1 + (py2 - py1) * t
    cz = 10.2 - sag * 4 * t * (1 - t) + rng.normal(0.0, 0.03, t.shape)
    keep = _scatter_dropout(np.ones(t.shape, dtype=bool), rng, 0.35)
    n = int(keep.sum())
    return (cx[keep], cy[keep], cz[keep],
            np.full(n, _intensity_for("conductor", rng)),
            np.tile(_rgb_for("conductor", rng), (n, 1)))


def _guardrail_mesh(start_x: float, end_x: float, side: float, rng: np.random.Generator):
    rx = np.arange(start_x, end_x, 0.15)
    ry = np.full_like(rx, side * 5.05)
    parts = ([], [], [], [], [])
    for level in (0.72, 0.92):
        rxx = rx + rng.normal(0.0, 0.012, rx.shape)
        ryy = ry + rng.normal(0.0, 0.03, rx.shape)
        rzz = np.full_like(rx, level) + rng.normal(0.0, 0.01, rx.shape)
        parts[0].append(rxx); parts[1].append(ryy); parts[2].append(rzz)
        parts[3].append(np.full(len(rx), _intensity_for("guardrail", rng)))
        parts[4].append(np.tile(_rgb_for("guardrail", rng), (len(rx), 1)))
    for px_ in np.arange(max(start_x, 20.0), end_x, 2.0):
        pz = np.arange(0.05, 0.75, 0.12)
        keep = _scatter_dropout(np.ones(pz.shape, dtype=bool), rng, 0.15)
        nz = int(keep.sum())
        parts[0].append(np.full(nz, px_) + rng.normal(0.0, 0.01, (nz,)))
        parts[1].append(np.full(nz, ry[0]))
        parts[2].append(pz[keep])
        parts[3].append(np.full(nz, _intensity_for("guardrail_post", rng)))
        parts[4].append(np.tile(_rgb_for("guardrail_post", rng), (nz, 1)))
    return tuple(np.concatenate(p) for p in parts)


def _concrete_barrier_mesh(x0: float, x1: float, y_center: float, rng: np.random.Generator):
    bar_x = np.arange(x0, x1, 0.12)
    bar_ys = np.arange(y_center - 0.7, y_center + 0.7, 0.1)
    bx, by = np.meshgrid(bar_x, bar_ys)
    bz = np.full_like(bx, 0.8) * (by - (y_center - 0.7)) / 1.4
    bz = np.clip(bz, 0.0, 0.8)
    bxx = bx.ravel() + rng.normal(0.0, 0.01, bx.size)
    byy = by.ravel() + rng.normal(0.0, 0.01, by.size)
    bzz = bz.ravel() + rng.normal(0.0, 0.008, bz.size)
    n = bxx.size
    return (bxx, byy, bzz,
            np.full(n, _intensity_for("concrete_barrier", rng)),
            np.tile(_rgb_for("concrete_barrier", rng), (n, 1)))


def _road_surface_mesh(length_m: float, rng: np.random.Generator, road_half: float = 3.65, shoulder: float = 2.5, spacing: float = 0.25):
    xs = np.arange(0.0, length_m, spacing)
    ys = np.arange(-(road_half + shoulder), road_half + shoulder, spacing)
    gx, gy = np.meshgrid(xs, ys)
    ground_x = gx.ravel()
    ground_y = gy.ravel()
    crown = 0.02 * np.abs(ground_y) / road_half
    ground_z = crown + rng.normal(0.0, 0.012, ground_x.shape)
    on_road = np.abs(ground_y) <= road_half
    intensity = np.where(on_road, rng.integers(1200, 2600, ground_x.shape),
                         rng.integers(400, 1200, ground_x.shape)).astype(np.float64)
    rgb = np.tile(_rgb_for("asphalt", rng), (len(ground_x), 1)) + rng.integers(-400, 400, (len(ground_x), 3)).astype(np.int16)
    rgb = np.clip(rgb, 0, 65535).astype(np.uint16)
    return ground_x, ground_y, ground_z, intensity, rgb


def _painted_marking_points(x: np.ndarray, y: np.ndarray, z: float = 0.015, intensity_kind: str = "painted_marking", rng: np.random.Generator = None) -> tuple:
    if rng is None:
        raise ValueError("rng required")
    xx = x + rng.normal(0.0, 0.015, x.shape)
    yy = y + rng.normal(0.0, 0.02, y.shape)
    zz = np.full_like(x, z) + rng.normal(0.0, 0.006, x.shape)
    ii = rng.integers(16000, 21000, x.shape).astype(np.float64)
    rgb = np.tile(_rgb_for("painted_marking", rng), (len(x), 1))
    return xx, yy, zz, ii, rgb


def build_simulated_las(
    path: str | Path,
    length_m: float = 400.0,
    ground_spacing: float = 0.25,
    seed: Optional[int] = None,
) -> dict:
    """Write a realistic simulated mobile-LiDAR scene and return a summary.

    The scene models a two-lane rural road (MUTCD-style markings), utility
    poles with sagging overhead conductors, traffic signs, guardrails, a
    concrete barrier, and LiDAR measurement noise / occlusion. Everything is
    synthetic: ``scene`` records what was placed so results can be audited.
    """
    rng = _rng_for(seed)
    road_half = 3.65
    shoulder = 2.5

    parts: Dict[str, Dict[str, Any]] = {}

    # ---- road surface ----------------------------------------------------------
    ground_x, ground_y, ground_z, ground_i, ground_rgb = _road_surface_mesh(length_m, rng)
    parts["road"] = {"points": len(ground_x), "note": f"{length_m:.0f} m two-lane road, asphalt"}

    # ---- painted markings -------------------------------------------------------
    mark_x, mark_y, mark_z, mark_i, mark_rgb = [], [], [], [], []

    dash_x = np.concatenate([np.arange(start, min(start + 3.0, length_m - 1), 0.1)
                             for start in np.arange(6.0, length_m - 6.0, 12.0)])
    mx, my, mz, mi, mrgb = _painted_marking_points(dash_x, np.full_like(dash_x, 0.0), rng=rng)
    mark_x.append(mx); mark_y.append(my); mark_z.append(mz); mark_i.append(mi); mark_rgb.append(mrgb)

    for edge in (-3.0, 3.0):
        ex = np.arange(2.0, length_m - 2.0, 0.12)
        mx, my, mz, mi, mrgb = _painted_marking_points(ex, np.full_like(ex, edge), rng=rng)
        mark_x.append(mx); mark_y.append(my); mark_z.append(mz); mark_i.append(mi); mark_rgb.append(mrgb)

    for band_y in np.arange(-3.1, 3.2, 0.9):
        bx = np.arange(300.0, 306.0, 0.1)
        mx, my, mz, mi, mrgb = _painted_marking_points(np.full_like(bx, 300.4), np.full_like(bx, band_y), rng=rng)
        mark_x.append(mx); mark_y.append(my); mark_z.append(mz); mark_i.append(mi); mark_rgb.append(mrgb)

    stop_x = np.arange(-3.2, 3.2, 0.1)
    mx, my, mz, mi, mrgb = _painted_marking_points(np.full_like(stop_x, 311.0), stop_x, rng=rng)
    mark_x.append(mx); mark_y.append(my); mark_z.append(mz); mark_i.append(mi); mark_rgb.append(mrgb)

    mark_x = np.concatenate(mark_x); mark_y = np.concatenate(mark_y)
    mark_z = np.concatenate(mark_z); mark_i = np.concatenate(mark_i); mark_rgb = np.concatenate(mark_rgb)
    parts["markings"] = {"points": len(mark_x), "note": "dashed centerline, edge lines, crosswalk, stop line"}

    # ---- utility poles with crossarms + overhead conductors ---------------------
    pole_x_all, pole_y_all, pole_z_all, pole_i_all, pole_rgb_all = [], [], [], [], []
    conductor_x, conductor_y, conductor_z, conductor_i, conductor_rgb = [], [], [], [], []
    pole_positions = [
        (60.0, -7.5),
        (120.0, 7.5),
        (180.0, -7.5),
        (240.0, 7.5),
        (300.0, -7.5),
        (360.0, 7.5),
    ]
    for (px, py) in pole_positions:
        zs = np.arange(0.1, 10.8, 0.14)
        px_, py_, pz_, pi_, prgb_ = _cylinder_points(px, py, 0.16, zs, rng, "utility_pole", "utility_pole")
        pole_x_all.append(px_); pole_y_all.append(py_); pole_z_all.append(pz_)
        pole_i_all.append(pi_); pole_rgb_all.append(prgb_)
        ax_, ay_, az_, ai_, argb_ = _crossarm_points(px, py, rng)
        pole_x_all.append(ax_); pole_y_all.append(ay_); pole_z_all.append(az_)
        pole_i_all.append(ai_); pole_rgb_all.append(argb_)
    for (px1, py1), (px2, py2) in zip(pole_positions, pole_positions[1:]):
        cx_, cy_, cz_, ci_, crgb_ = _conductor_points((px1, py1), (px2, py2), rng)
        conductor_x.append(cx_); conductor_y.append(cy_); conductor_z.append(cz_)
        conductor_i.append(ci_); conductor_rgb.append(crgb_)
    pole_x = np.concatenate(pole_x_all); pole_y = np.concatenate(pole_y_all); pole_z = np.concatenate(pole_z_all)
    pole_i = np.concatenate(pole_i_all); pole_rgb = np.concatenate(pole_rgb_all)
    parts["utility_poles"] = {"points": len(pole_x), "count": len(pole_positions), "note": "10.8 m poles, 0.32 m dia, crossarms"}
    con_x = np.concatenate(conductor_x); con_y = np.concatenate(conductor_y)
    con_z = np.concatenate(conductor_z); con_i = np.concatenate(conductor_i)
    con_rgb = np.concatenate(conductor_rgb)
    parts["overhead_conductors"] = {"points": len(con_x), "count": len(pole_positions) - 1, "note": "sagging catenary wires"}

    # ---- traffic signs ----------------------------------------------------------
    sign_x_all, sign_y_all, sign_z_all, sign_i_all, sign_rgb_all = [], [], [], [], []
    sx, sy, sz, si, srgb = _panel_mesh(90.0, -5.6, 0.9, 0.9, 2.1, rng, "sign_panel_green")
    sign_x_all.append(sx); sign_y_all.append(sy); sign_z_all.append(sz); sign_i_all.append(si); sign_rgb_all.append(srgb)
    sx, sy, sz, si, srgb = _post_mesh(90.0, -5.6, 2.1, rng)
    sign_x_all.append(sx); sign_y_all.append(sy); sign_z_all.append(sz); sign_i_all.append(si); sign_rgb_all.append(srgb)

    sx, sy, sz, si, srgb = _panel_mesh(150.0, 5.6, 0.6, 0.9, 2.0, rng, "sign_panel_white")
    sign_x_all.append(sx); sign_y_all.append(sy); sign_z_all.append(sz); sign_i_all.append(si); sign_rgb_all.append(srgb)
    sx, sy, sz, si, srgb = _post_mesh(150.0, 5.6, 2.0, rng)
    sign_x_all.append(sx); sign_y_all.append(sy); sign_z_all.append(sz); sign_i_all.append(si); sign_rgb_all.append(srgb)

    sx, sy, sz, si, srgb = _panel_mesh(255.0, -5.8, 0.6, 0.6, 2.0, rng, "sign_panel_yellow")
    sign_x_all.append(sx); sign_y_all.append(sy); sign_z_all.append(sz); sign_i_all.append(si); sign_rgb_all.append(srgb)
    sx, sy, sz, si, srgb = _post_mesh(255.0, -5.8, 2.0, rng)
    sign_x_all.append(sx); sign_y_all.append(sy); sign_z_all.append(sz); sign_i_all.append(si); sign_rgb_all.append(srgb)

    sign_x = np.concatenate(sign_x_all); sign_y = np.concatenate(sign_y_all)
    sign_z = np.concatenate(sign_z_all); sign_i = np.concatenate(sign_i_all)
    sign_rgb = np.concatenate(sign_rgb_all)
    parts["traffic_signs"] = {"points": len(sign_x), "count": 3, "note": "guide + warning panels with supports"}

    # ---- guardrails (both sides) + concrete barrier -----------------------------
    rail_x_all, rail_y_all, rail_z_all, rail_i_all, rail_rgb_all = [], [], [], [], []
    for side in (-1.0, 1.0):
        rx, ry, rz, ri, rrgb = _guardrail_mesh(20.0, 130.0, side, rng)
        rail_x_all.append(rx); rail_y_all.append(ry); rail_z_all.append(rz)
        rail_i_all.append(ri); rail_rgb_all.append(rrgb)
    rail_x = np.concatenate(rail_x_all); rail_y = np.concatenate(rail_y_all)
    rail_z = np.concatenate(rail_z_all); rail_i = np.concatenate(rail_i_all)
    rail_rgb = np.concatenate(rail_rgb_all)
    parts["guardrails"] = {"points": len(rail_x), "count": 2, "note": "110 m w-beam rails with posts"}

    bxx, byy, bzz, bi, brgb = _concrete_barrier_mesh(340.0, 380.0, 5.5, rng)
    parts["concrete_barrier"] = {"points": len(bxx), "note": "40 m Jersey-style barrier"}

    # ---- assemble ----------------------------------------------------------------
    x = np.concatenate([ground_x, mark_x, pole_x, con_x, sign_x, rail_x, bxx])
    y = np.concatenate([ground_y, mark_y, pole_y, con_y, sign_y, rail_y, byy])
    z = np.concatenate([ground_z, mark_z, pole_z, con_z, sign_z, rail_z, bzz])
    intensity = np.concatenate([ground_i, mark_i, pole_i, con_i, sign_i, rail_i, bi])
    rgb = np.concatenate([ground_rgb, mark_rgb, pole_rgb, con_rgb, sign_rgb, rail_rgb, brgb]).astype(np.uint16)
    assert len(x) == len(y) == len(z) == len(intensity) == len(rgb), (
        f"length mismatch: x={len(x)} y={len(y)} z={len(z)} i={len(intensity)} rgb={len(rgb)}")
    y = y + 0.5
    point_source_id = np.where(y >= 0.5, 2, 1).astype(np.uint16)
    gps_time = _gps_time_for(x, pass_limit=length_m / 2)

    header = laspy.LasHeader(point_format=7, version="1.4")
    header.scales = [0.005, 0.005, 0.005]
    header.offsets = [2_500_000.0, 700_000.0, 250.0]
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

    # ---- ground truth object table (validation only) ----------------------------
    # Every placed object with its true class / center / extent. Detected assets
    # can be matched against this to compute precision, recall and positional
    # error - the ground truth is known because the scene is synthetic.
    ground_truth: list[Dict[str, Any]] = []
    for (gx_, gy_) in pole_positions:
        ground_truth.append({
            "gt_id": f"GT-POL-{len(ground_truth) + 1:03d}",
            "class": "utility_pole",
            "center": [gx_, gy_, 5.4],
            "dimensions_m": [0.32, 0.32, 10.8],
        })
    for (px1, py1), (px2, py2) in zip(pole_positions, pole_positions[1:]):
        ground_truth.append({
            "gt_id": f"GT-CON-{len(ground_truth) + 1:03d}",
            "class": "overhead_conductor",
            "center": [(px1 + px2) / 2, (py1 + py2) / 2, 9.7],
            "dimensions_m": [px2 - px1, 0.02, 2.2],
        })
    for (sx_, sy_, sz0_, sh_, sw_, scls) in [
        (90.0, -5.6, 2.1, 0.9, 0.9, "traffic_sign"),
        (150.0, 5.6, 2.0, 0.6, 0.9, "traffic_sign"),
        (255.0, -5.8, 2.0, 0.6, 0.6, "traffic_sign"),
    ]:
        ground_truth.append({
            "gt_id": f"GT-SGN-{len(ground_truth) + 1:03d}",
            "class": scls,
            "center": [sx_, sy_, sz0_ + sh_ / 2],
            "dimensions_m": [sw_, 0.1, sh_],
        })
    for side in (-1.0, 1.0):
        ground_truth.append({
            "gt_id": f"GT-GRD-{len(ground_truth) + 1:03d}",
            "class": "guardrail",
            "center": [75.0, side * 5.05, 0.5],
            "dimensions_m": [110.0, 0.6, 0.9],
        })
    ground_truth.append({
        "gt_id": f"GT-BAR-{len(ground_truth) + 1:03d}",
        "class": "safety_barrier",
        "center": [360.0, 5.5, 0.4],
        "dimensions_m": [40.0, 1.4, 0.8],
    })
    ground_truth.append({
        "gt_id": f"GT-PAV-{len(ground_truth) + 1:03d}",
        "class": "pavement",
        "center": [length_m / 2, 0.0, 0.0],
        "dimensions_m": [length_m, 7.3, 0.1],
    })
    ground_truth.append({
        "gt_id": f"GT-MRK-{len(ground_truth) + 1:03d}",
        "class": "pavement_marking",
        "center": [length_m / 2, 0.0, 0.02],
        "dimensions_m": [length_m, 6.0, 0.03],
    })

    return {
        "path": str(path),
        "points": len(x),
        "simulated": True,
        "scene": parts,
        "ground_truth": ground_truth,
        "bounds": [float(header.mins[i]) for i in range(3)] + [float(header.maxs[i]) for i in range(3)],
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": seed if seed is not None else "random",
    }


def build_synthetic_las(path: str | Path, ground_spacing: float = GROUND_SPACING, seed: int = 7) -> dict:
    """Write a small deterministic synthetic LAS and return a summary dict."""
    rng = np.random.default_rng(seed)

    ground_x, ground_y, ground_z, ground_i, ground_rgb = _road_surface_mesh(80.0, rng, road_half=3.65, shoulder=2.5, spacing=ground_spacing)
    ground_rgb = np.tile(np.array([8000, 8000, 8000]), (len(ground_x), 1)) + rng.integers(-300, 300, (len(ground_x), 3))

    line_x = np.arange(10.0, 60.0, 0.15)
    line_y = np.full_like(line_x, -1.65) + rng.normal(0.0, 0.02, line_x.shape)
    line_z = np.full_like(line_x, 0.02)
    line_i = rng.integers(18000, 22000, line_x.shape).astype(np.float64)
    line_rgb = np.tile(np.array([62000, 62000, 62000]), (len(line_x), 1))

    pole_x, pole_y, pole_z, pole_i, pole_rgb = _cylinder_points(25.0, 8.0, 0.12,
                                                                 np.arange(0.06, 7.2, 0.18), rng, "utility_pole", "utility_pole")

    panel_x, panel_y, panel_z, panel_i, panel_rgb = _panel_mesh(45.0, -10.0, 0.9, 0.8, 2.0, rng, "sign_panel_green")
    post_x, post_y, post_z, post_i, post_rgb = _post_mesh(45.0, -10.0, 2.0, rng)

    rail_x, rail_y, rail_z, rail_i, rail_rgb = _guardrail_mesh(10.0, 70.0, 1.0, rng)

    x = np.concatenate([ground_x, line_x, pole_x, panel_x, post_x, rail_x])
    y = np.concatenate([ground_y, line_y, pole_y, panel_y, post_y, rail_y])
    z = np.concatenate([ground_z, line_z, pole_z, panel_z, post_z, rail_z])
    intensity = np.concatenate([ground_i, line_i, pole_i, panel_i, post_i, rail_i])
    rgb = np.concatenate([ground_rgb, line_rgb, pole_rgb, panel_rgb, post_rgb, rail_rgb]).astype(np.uint16)

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
