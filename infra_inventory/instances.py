"""Instance extraction: grid connected components and PCA-based component metrics.

This is the layer that turns *points* into *instances*. The pipeline feeds each
detector a list of point indices per connected component, and detectors decide
whether a component matches an asset class from its measured metrics.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Dict, Iterable, List, Tuple

import numpy as np


def grid_components(
    x: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    resolution: float,
    min_cells: int = 1,
) -> List[np.ndarray]:
    """Connected components of ``mask`` points on an XY grid (8-connectivity).

    Returns lists of point indices, one per component with at least
    ``min_cells`` occupied cells.

    The point -> cell grouping is vectorized (sort by cell key, then split at
    group boundaries) while the connectivity BFS stays the same. The cell dict
    is inserted in original first-occurrence order and each cell's point list
    keeps original point order, so the returned components are identical to a
    pure-Python build.
    """
    selected = np.flatnonzero(mask)
    if not len(selected):
        return []
    cell_x = np.floor(x[selected] / resolution).astype(np.int64)
    cell_y = np.floor(y[selected] / resolution).astype(np.int64)
    # Sort by (cell_y, cell_x, original position): cells are contiguous AND each
    # cell's points stay in original order.
    order = np.lexsort((selected, cell_y, cell_x))
    scx, scy, ssel = cell_x[order], cell_y[order], selected[order]
    key = np.column_stack((scx, scy))
    starts = np.flatnonzero(np.concatenate(([True], (key[1:] != key[:-1]).any(axis=1))))
    ends = np.concatenate((starts[1:], [len(order)]))
    # Dict insertion order must match the old per-point loop (first occurrence
    # in original point order) so the BFS yields identical component order.
    first_original = ssel[starts]
    index_by_cell: Dict[Tuple[int, int], List[int]] = {}
    for position in np.argsort(first_original, kind="stable"):
        start, end = starts[position], ends[position]
        index_by_cell[(int(scx[start]), int(scy[start]))] = ssel[start:end].tolist()
    return [
        np.asarray([index for cell in group for index in index_by_cell[cell]], dtype=np.int64)
        for group in _connected_cells(index_by_cell)
        if len(group) >= min_cells
    ]


def _connected_cells(cells: Dict[Tuple[int, int], List[int]]) -> List[List[Tuple[int, int]]]:
    remaining = set(cells)
    groups: List[List[Tuple[int, int]]] = []
    while remaining:
        start = remaining.pop()
        group = [start]
        queue = deque([start])
        while queue:
            cell = queue.popleft()
            for dx, dy in (
                (1, 0), (-1, 0), (0, 1), (0, -1),
                (1, 1), (1, -1), (-1, 1), (-1, -1),
            ):
                candidate = (cell[0] + dx, cell[1] + dy)
                if candidate in remaining:
                    remaining.remove(candidate)
                    group.append(candidate)
                    queue.append(candidate)
        groups.append(group)
    return groups


def component_cross_section_m(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, indices: np.ndarray,
) -> float:
    """Cross-section thickness (2-sigma minor PCA axis) of a component.

    The axis-aligned bounding-box width is *orientation dependent*: a thin
    wire running at 14 deg to the tile axes has a bbox "width" of
    ``length * sin(angle)`` (metres) even though the wire itself is ~0.1 m
    thick, so detectors reject every diagonal conductor. Measuring the
    standard deviation around the dominant PCA axis gives the true
    cross-section in metres regardless of heading.

    Returns metres (``float("inf")`` for degenerate components).
    """
    points = np.column_stack((x[indices], y[indices], z[indices]))
    if len(points) < 3 or not np.isfinite(points).all():
        return float("inf")
    try:
        cov = np.cov(points, rowvar=False)
        eigenvalues = np.linalg.eigh(cov)[0]
        minor = max(float(eigenvalues[0]), 0.0)
        # ~95% of points sit within 2 sigma of the dominant axis for a clean
        # linear chain; a wire + dropout noise measures ~0.1-0.2 m.
        return 2.0 * math.sqrt(minor)
    except (np.linalg.LinAlgError, ValueError):  # pragma: no cover - defensive
        return float("inf")


def component_metrics(x: np.ndarray, y: np.ndarray, z: np.ndarray, indices: np.ndarray) -> Dict[str, float]:
    """Measured metrics for a component: bbox, dimensions, PCA geometry.

    Returns a plain dict of floats so detectors can stay pure functions:
      bounds        (min_x, min_y, min_z, max_x, max_y, max_z)
      length_m, width_m, height_m
      centroid_x/y/z
      orientation_deg   (azimuth of the dominant XY axis, 0-180)
      planarity, linearity, verticality  (eigenvalue ratios, 0-1)
    """
    points = np.column_stack((x[indices], y[indices], z[indices]))
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    bounds = (float(lo[0]), float(lo[1]), float(lo[2]), float(hi[0]), float(hi[1]), float(hi[2]))
    centroid = points.mean(axis=0)
    # Degenerate components (1-2 points, or zero variance along an axis) make
    # np.cov/np.linalg.eigh raise on real-world tiles (USGS 3DEP airborne data
    # is full of tiny fragments). Fall back to axis-aligned metrics instead of
    # crashing the whole run.
    planarity = linearity = verticality = columnarity = 0.0
    orientation = 0.0
    if len(points) >= 3 and np.isfinite(points).all():
        try:
            cov = np.cov(points, rowvar=False)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            eigenvalues = np.clip(np.nan_to_num(eigenvalues, nan=0.0), 0.0, None)
            if np.isfinite(eigenvalues).all():
                order = np.argsort(eigenvalues)
                e0, e1, e2 = eigenvalues[order[0]], eigenvalues[order[1]], eigenvalues[order[2]]
                # Dominant axis for elongated structures is the eigenvector of e2
                dominant = eigenvectors[:, order[2]]
                orientation = (math.degrees(math.atan2(dominant[1], dominant[0])) + 360) % 180
                planarity = float((e1 - e0) / e2) if e2 > 1e-12 else 0.0
                linearity = float((e2 - e1) / e2) if e2 > 1e-12 else 0.0
                # Verticality: how vertical the dominant eigenvector is (0 horizontal .. 1 vertical)
                verticality = float(abs(dominant[2])) if eigenvalues[order[2]] > 1e-12 else 0.0
                # Columnarity: cross-section symmetry (e0 ~ e1). A vertical *column* (pole)
                # has e0 ~ e1; a vertical *plane* (sign panel, wall) has e0 << e1.
                columnarity = float(e0 / e1) if e1 > 1e-12 else 0.0
        except (np.linalg.LinAlgError, ValueError):  # pragma: no cover - defensive
            pass
    return {
        "bounds": bounds,
        "length_m": float(hi[0] - lo[0]),
        "width_m": float(hi[1] - lo[1]),
        "height_m": float(hi[2] - lo[2]),
        "centroid_x": float(centroid[0]),
        "centroid_y": float(centroid[1]),
        "centroid_z": float(centroid[2]),
        "orientation_deg": float(orientation),
        "planarity": planarity,
        "linearity": linearity,
        "verticality": verticality,
        "columnarity": columnarity,
        "point_count": int(len(indices)),
    }


def longitudinal_lateral(metrics: Dict[str, float], orientation_deg: float) -> Tuple[float, float]:
    """Split bbox extents into longitudinal (along orientation) and lateral (across)."""
    length, width = metrics["length_m"], metrics["width_m"]
    if orientation_deg is not None and 30 <= orientation_deg <= 150:
        return max(length, width), min(length, width)
    return length, width


def merge_linear_pieces(
    assets: List[Any],
    class_name: str = "overhead_conductor",
    gap_m: float = 45.0,
    lateral_m: float = 3.0,
) -> List[Any]:
    """Re-join thin-linear assets that a tile boundary cut in half.

    Long thin objects (overhead conductors, wires) cross 40 m tiles at arbitrary
    angles; each tile piece is detected separately and usually fails the
    class's minimum-point/length rules on its own. Pieces of the *same* span
    are collinear continuations whose bounding boxes meet at the tile edge, so
    they are merged back into one asset (bbox union, point-weighted centroid,
    best-piece confidence). Pieces that meet at a utility pole belong to
    *different* spans and are never merged.
    """
    pieces = [a for a in assets if a.asset_class == class_name]
    if len(pieces) < 2:
        return assets
    poles = [a for a in assets if a.asset_class == "utility_pole"]
    rest = [a for a in assets if a.asset_class != class_name]

    used = [False] * len(pieces)
    merged: List[Any] = []
    for i, seed in enumerate(pieces):
        if used[i]:
            continue
        group = [seed]
        used[i] = True
        changed = True
        while changed:
            changed = False
            for j, candidate in enumerate(pieces):
                if used[j]:
                    continue
                if any(_continues_after(member, candidate, poles, gap_m, lateral_m) for member in group):
                    group.append(candidate)
                    used[j] = True
                    changed = True
        merged.append(group[0] if len(group) == 1 else _join_linear_pieces(group))
    return rest + merged


def _piece_line(a: Any) -> Optional[Tuple[float, float, float, float]]:
    """Best-fit line (cx, cy, dirx, diry) from an asset's highlight points.

    Returns None when the piece has no usable highlight points. The fitted line
    is what lets the merge distinguish *same-span continuations* (same line,
    gaps of metres from occlusion/tile cuts) from *parallel neighbouring
    wires* (parallel but offset by a metre or more) and from *different spans*
    that meet at a pole (different heading).
    """
    geom = getattr(a, "geometry", None) or {}
    pts = (geom or {}).get("highlight_points")
    if not pts or len(pts) < 5:
        return None
    arr = np.asarray(pts, dtype=np.float64)
    xy = arr[:, :2]
    if not np.isfinite(xy).all():
        return None
    center = xy.mean(axis=0)
    try:
        cov = np.cov(xy, rowvar=False)
        values, vectors = np.linalg.eigh(cov)
    except (np.linalg.LinAlgError, ValueError):  # pragma: no cover - defensive
        return None
    if values[-1] <= 1e-12:
        return None
    direction = vectors[:, -1]
    # Orient both directions consistently along +x so |dot| semantics are clean.
    if direction[0] < 0:
        direction = -direction
    return (float(center[0]), float(center[1]), float(direction[0]), float(direction[1]))


def _continues_after(
    a: Any, b: Any, poles: List[Any], gap_m: float, lateral_m: float,
) -> bool:
    """True when ``b`` looks like the next tile piece of the same span as ``a``."""
    ax0, ay0, az0, ax1, ay1, az1 = a.bounding_box
    bx0, by0, bz0, bx1, by1, bz1 = b.bounding_box
    acx, acy = (ax0 + ax1) / 2, (ay0 + ay1) / 2
    bcx, bcy = (bx0 + bx1) / 2, (by0 + by1) / 2
    dx, dy = bcx - acx, bcy - acy
    if max(abs(dx), abs(dy)) < 1e-6:
        return False
    if abs(dx) >= abs(dy):
        gap = max(0.0, max(ax0, bx0) - min(ax1, bx1))
        # Inter-piece band along the dominant axis (empty when the two pieces
        # overlap in x, i.e. their projections touch): pieces of the same span
        # can be separated by an occlusion gap of a few metres, but spans on
        # opposite sides of a pole are *always* separated by that pole - whose
        # claimed cells sit ~1.2 m outside each piece end. Fragments trimmed
        # asymmetrically put the pole anywhere inside the band, so refuse a
        # merge when a pole sits in the whole band (with margin), not just at
        # the junction midpoint.
        band_lo = min(ax1, bx1)
        band_hi = max(ax0, bx0)
        cross_lo = min(ay0, ay1, by0, by1)
        cross_hi = max(ay0, ay1, by0, by1)
    else:
        gap = max(0.0, max(ay0, by0) - min(ay1, by1))
        band_lo = min(ay1, by1)
        band_hi = max(ay0, by0)
        cross_lo = min(ax0, ax1, bx0, bx1)
        cross_hi = max(ax0, ax1, bx0, bx1)
    if gap > gap_m:
        return False
    if abs(az0 - bz0) > 3.0 and abs(az1 - bz1) > 3.0:
        return False
    # Same-span pieces lie on one straight line. Fit each piece's own line from
    # its highlight points: identical lines (cross-track offset ~0) merge;
    # parallel neighbouring wires sit a metre-plus apart; zigzag spans that
    # meet at a pole differ in heading by ~28 deg in a corridor.
    line_a = _piece_line(a)
    line_b = _piece_line(b)
    if line_a is not None and line_b is not None:
        _, _, adx, ady = line_a
        _, _, bdx, bdy = line_b
        if adx * bdx + ady * bdy < 0.90:  # headings differ by > ~26 deg
            return False
        # Perpendicular distance from b's centroid to a's fitted line. Same-wire
        # continuations sit within a wire diameter (~0.1-0.3 m); parallel
        # neighbouring phases are spaced a metre or more apart.
        cross_track = abs((line_b[0] - line_a[0]) * ady - (line_b[1] - line_a[1]) * adx)
        if cross_track > min(lateral_m, 1.2):
            return False
    # Different spans meet at the pole they share: never merge across a pole.
    for pole in poles:
        pc = pole.center
        px, py = float(pc["x"]), float(pc["y"])
        if abs(dx) >= abs(dy):
            in_band = band_lo - 2.0 <= px <= band_hi + 2.0
            in_cross = cross_lo - 2.0 <= py <= cross_hi + 2.0
        else:
            in_band = band_lo - 2.0 <= py <= band_hi + 2.0
            in_cross = cross_lo - 2.0 <= px <= cross_hi + 2.0
        if in_band and in_cross:
            return False
    return True


def _join_linear_pieces(group: List[Any]) -> Any:
    """Merge collinear pieces into one asset (largest piece keeps its identity)."""
    from .models import Asset

    best = max(group, key=lambda a: a.point_count)
    centers = [a.center for a in group]
    counts = [max(a.point_count, 1) for a in group]
    total = sum(counts)
    cx = sum(c["x"] * n for c, n in zip(centers, counts)) / total
    cy = sum(c["y"] * n for c, n in zip(centers, counts)) / total
    cz = sum(c["z"] * n for c, n in zip(centers, counts)) / total
    box = [
        min(a.bounding_box[0] for a in group), min(a.bounding_box[1] for a in group),
        min(a.bounding_box[2] for a in group), max(a.bounding_box[3] for a in group),
        max(a.bounding_box[4] for a in group), max(a.bounding_box[5] for a in group),
    ]
    x_span, y_span, z_span = box[3] - box[0], box[4] - box[1], box[5] - box[2]
    if x_span >= y_span:
        length_m, width_m = x_span, y_span
    else:
        length_m, width_m = y_span, x_span
    geometry = dict(best.geometry or {})
    ground_values = [a.geometry.get("ground_elevation_m") for a in group
                     if a.geometry and a.geometry.get("ground_elevation_m") is not None]
    if ground_values:
        geometry["ground_elevation_m"] = round(sum(ground_values) / len(ground_values), 4)
    source_tile = "+".join(sorted({a.source_tile for a in group}))
    return Asset(
        asset_id=best.asset_id,
        asset_class=best.asset_class,
        subclass=best.subclass,
        center={"x": round(cx, 4), "y": round(cy, 4), "z": round(cz, 4)},
        bounding_box=tuple(round(float(v), 4) for v in box),
        dimensions={"length_m": round(length_m, 3), "width_m": round(width_m, 3),
                    "height_m": round(z_span, 3)},
        point_count=sum(a.point_count for a in group),
        source_tile=source_tile,
        source_point_indices_sample=best.source_point_indices_sample,
        coordinate_reference_system=best.coordinate_reference_system,
        confidence=best.confidence,
        confidence_factors=best.confidence_factors,
        confidence_explanation=best.confidence_explanation + f" ({len(group)} tile pieces merged)",
        detection_method=best.detection_method + "-merged",
        intensity_stats=best.intensity_stats,
        rgb_stats=best.rgb_stats,
        orientation_deg=best.orientation_deg,
        source_run=best.source_run,
        source_scanner=best.source_scanner,
        source_point_source_id=best.source_point_source_id,
        model_prior_class=best.model_prior_class,
        model_confidence=best.model_confidence,
        processing_version=best.processing_version,
        geometry=geometry,
    )


def compactness(metrics: Dict[str, float]) -> float:
    """Occupied-volume fraction of the component's bounding box (0..1).

    Degenerate (zero-extent) bounding-box dimensions are clamped to a minimum so
    perfectly flat objects do not divide by zero.
    """
    volume = max(metrics["length_m"], 0.05) * max(metrics["width_m"], 0.05) * max(metrics["height_m"], 0.05)
    if volume <= 0:
        return 0.0
    points_estimate = metrics["point_count"] * 0.001  # rough per-point volume share
    return min(1.0, points_estimate / volume)


def group_line_segments(segments: List[Tuple[float, float, float, float, float, float]], gap_m: float = 1.5) -> List[List[Tuple[float, float, float, float, float, float]]]:
    """Group 3D line segments (x1,y1,z1,x2,y2,z2) into clusters by endpoint proximity.

    Used by the RoadMarkingExtraction DXF adapter to re-assemble vectorized
    segments into individual marking assets.
    """
    groups: List[List[Tuple[float, float, float, float, float, float]]] = []
    for segment in segments:
        placed = False
        for group in groups:
            for other in group:
                if _segment_near(segment, other, gap_m):
                    group.append(segment)
                    placed = True
                    break
            if placed:
                break
        if not placed:
            groups.append([segment])
    return groups


def _segment_near(a: Tuple[float, float, float, float, float, float], b: Tuple[float, float, float, float, float, float], gap_m: float) -> bool:
    a1 = np.asarray(a[:3], dtype=np.float64)
    a2 = np.asarray(a[3:], dtype=np.float64)
    b1 = np.asarray(b[:3], dtype=np.float64)
    b2 = np.asarray(b[3:], dtype=np.float64)
    return min(np.linalg.norm(a1 - b1), np.linalg.norm(a1 - b2), np.linalg.norm(a2 - b1), np.linalg.norm(a2 - b2)) <= gap_m