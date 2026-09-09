"""Instance extraction: grid connected components and PCA-based component metrics.

This is the layer that turns *points* into *instances*. The pipeline feeds each
detector a list of point indices per connected component, and detectors decide
whether a component matches an asset class from its measured metrics.
"""
from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence
from typing import Any, Dict, Iterable, List, Optional, Tuple

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
    cross_section = float("inf")
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
                # 2-sigma thickness across the dominant axis (same definition as
                # ``component_cross_section_m`` but computed from the covariance we
                # already factorized - one eigendecomposition, both values).
                cross_section = 2.0 * math.sqrt(max(float(e0), 0.0))
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
        "cross_section_m": cross_section,
        "point_count": int(len(indices)),
    }


def longitudinal_lateral(metrics: Dict[str, float], orientation_deg: float) -> Tuple[float, float]:
    """Split bbox extents into longitudinal (along orientation) and lateral (across)."""
    length, width = metrics["length_m"], metrics["width_m"]
    if orientation_deg is not None and 30 <= orientation_deg <= 150:
        return max(length, width), min(length, width)
    return length, width


def _aspect_from_orientation(length_m: float, width_m: float, orientation_deg: float) -> float:
    """Elongation ratio of a component (larger extent / smaller extent).

    For a truly horizontal long bar the bbox major axis is the length; for an
    axis of orientation_deg the major/minor split follows that bearing, so a
    similarly long bar rotated 15 deg off the axes still gets the right
    aspect instead of being penalized by the bbox diagonal effect.
    """
    if length_m <= 0 or width_m <= 0:
        return 1.0
    if orientation_deg is not None and 30 <= orientation_deg <= 150:
        major = max(length_m, width_m)
        minor = min(length_m, width_m)
    else:
        # Not clearly aligned with the XY bearings: use the raw ratio but cap
        # it so a degenerate/near-square component does not suddenly look
        # extremely elongated.
        major, minor = max(length_m, width_m), min(length_m, width_m)
    return major / max(minor, 1e-3)


def merge_linear_safety_pieces(
    assets: List[Any],
    guardrail_max_gap_m: float = 25.0,
    guardrail_lateral_m: float = 3.0,
    guardrail_linear_overlap_m: float = 3.0,
    guardrail_linear_aspect_ratio: float = 4.0,
    barrier_max_gap_m: float = 25.0,
    barrier_lateral_m: float = 3.0,
    barrier_linear_overlap_m: float = 3.0,
    barrier_linear_aspect_ratio: float = 4.0,
    classify_min_width_m: float = 0.35,
) -> List[Any]:
    """Re-join guardrail / safety_barrier pieces that tile boundaries cut.

    Both classes are pooled into one candidate set: a fragment of a rail can
    legitimately measure wider (or taller) than its neighbours — merged with a
    sign or a pole footprint — and get labeled ``safety_barrier`` while the
    rest of the span is ``guardrail`` (measured on QuickSim: one tile of a
    110 m rail reported 0.63 m cross-section instead of 0.16 m). Merging per
    class would strand that fragment as a stray barrier, so pieces of the
    *same continuous span* are joined regardless of which safety class each
    tile produced, and the joined span's class is resolved from the dominant
    (point-weighted) cross-section.

    ``classify_min_width_m`` is the boundary between the two classes for a
    mixed span (the same gate the detector uses for single-tile pieces).
    """
    pieces = [a for a in assets if a.asset_class in ("guardrail", "safety_barrier")]
    if len(pieces) < 2:
        return assets
    rest = [a for a in assets if a.asset_class not in ("guardrail", "safety_barrier")]
    gap_m = max(guardrail_max_gap_m, barrier_max_gap_m)
    lateral_m = max(guardrail_lateral_m, barrier_lateral_m)
    aspect_ratio = min(guardrail_linear_aspect_ratio, barrier_linear_aspect_ratio)
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
                if any(_continues_after_linear(member, candidate, gap_m, lateral_m, aspect_ratio) for member in group):
                    group.append(candidate)
                    used[j] = True
                    changed = True
        if len(group) == 1:
            merged.append(group[0])
        else:
            merged.append(_join_linear_safety_pieces(group, classify_min_width_m))
    return rest + merged


def _join_linear_safety_pieces(group: List[Any], classify_min_width_m: float = 0.35) -> Any:
    """Merge collinear safety pieces into one asset (largest piece keeps its identity)."""
    from .models import Asset

    # Span class: when the tile pieces disagree (guardrail vs barrier), the
    # *median* measured cross-section of the pieces decides. One wide fragment
    # (a rail tile merged with a sign footprint) must not reclassify an
    # otherwise narrow rail, nor one narrow stub a concrete barrier - the
    # median is robust to exactly one such contaminated piece.
    classes = {a.asset_class for a in group}
    if len(classes) == 1:
        asset_class = group[0].asset_class
    else:
        laterals: List[float] = []
        for a in group:
            metrics = getattr(a, "_metrics", None)
            if metrics is not None and isinstance(metrics, dict):
                lateral = min(float(metrics.get("length_m", 0.0)), float(metrics.get("width_m", 0.0)))
            else:
                bb = a.bounding_box
                lateral = min(float(bb[3] - bb[0]), float(bb[4] - bb[1]))
            laterals.append(lateral)
        laterals.sort()
        median_lateral = laterals[len(laterals) // 2]
        asset_class = "safety_barrier" if median_lateral >= classify_min_width_m else "guardrail"

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
    source_tile = "+".join(sorted({str(a.source_tile) for a in group}))
    return Asset(
        asset_id=best.asset_id,
        asset_class=asset_class,
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
        intensity_stats=getattr(best, "intensity_stats", None),
        rgb_stats=getattr(best, "rgb_stats", None),
        orientation_deg=getattr(best, "orientation_deg", None),
        source_run=getattr(best, "source_run", None),
        source_scanner=getattr(best, "source_scanner", None),
        source_point_source_id=getattr(best, "source_point_source_id", None),
        model_prior_class=getattr(best, "model_prior_class", None),
        model_confidence=getattr(best, "model_confidence", None),
        processing_version=best.processing_version,
        geometry=geometry,
    )


def _polygon_area_m2(vertices: Sequence[Tuple[float, float]]) -> float:
    """Shoelace area of a simple polygon in m² (positive for CCW input)."""
    n = len(vertices)
    if n < 3:
        return 0.0
    acc = 0.0
    for i in range(n):
        x0, y0 = vertices[i]
        x1, y1 = vertices[(i + 1) % n]
        acc += x0 * y1 - x1 * y0
    return abs(acc) * 0.5


def _xyhull_axes(box: Sequence[float], samples: Iterable[Sequence[float]], max_points: int = 4096) -> List[Tuple[float, float]]:
    """Convex hull vertices (x, y) of a footprint, from its box corners + samples.

    Falls back to the four box corners when there are no usable sample points
    (e.g. a cached asset whose point samples were not persisted).
    """
    pts = [
        (float(box[0]), float(box[1])), (float(box[0]), float(box[4])),
        (float(box[3]), float(box[1])), (float(box[3]), float(box[4])),
    ]
    stride = max(1, len(samples) // max_points)
    for p in samples[::stride]:
        try:
            x, y = float(p[0]), float(p[1])
        except (TypeError, ValueError, IndexError):
            continue
        if math.isfinite(x) and math.isfinite(y):
            pts.append((x, y))
    pts.sort()
    def _half(seq: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        out: List[Tuple[float, float]] = []
        for p in seq:
            while len(out) >= 2:
                (ox, oy), (px, py) = out[-2], out[-1]
                if (px - ox) * (p[1] - oy) - (py - oy) * (p[0] - ox) <= 1e-12:
                    out.pop()
                else:
                    break
            out.append(p)
        return out
    lower = _half(pts)
    upper = _half(pts[::-1])
    return lower[:-1] + upper[:-1]


def _surfaces_overlap_in_plane(a: Any, b: Any) -> bool:
    """True when two surface footprints actually intersect in the XY plane.

    Convex-hull intersection test (separating-axis over hull edges). Used so a
    stray pavement fragment one lane over is never glued onto the main deck.
    """
    ha = _xyhull_axes(a.bounding_box, (a.geometry or {}).get("highlight_points") or [])
    hb = _xyhull_axes(b.bounding_box, (b.geometry or {}).get("highlight_points") or [])
    return _convex_polygons_within(ha, hb, 0.0)


def _convex_polygons_within(
    poly_a: List[Tuple[float, float]], poly_b: List[Tuple[float, float]], proximity: float,
) -> bool:
    """True when two convex polygons intersect or come within ``proximity`` metres.

    Distance-aware separating-axis test: for each edge of one polygon the
    signed distances of the other polygon's vertices from the edge line are
    computed; if *all* of them lie more than ``proximity`` beyond the edge,
    the polygons are separated (exact for convex hulls). A zero proximity
    reproduces the classic SAT intersection test; a positive one additionally
    joins polygons whose footprints merely touch at a tile boundary — the
    common case with non-overlapping tiles, where a deck or painted line cut
    at x=40 sits 0.01-0.3 m from its neighbour at x=40.01.
    """
    for pa, pb in ((poly_a, poly_b), (poly_b, poly_a)):
        n = len(pa)
        for i in range(n):
            x0, y0 = pa[i]
            x1, y1 = pa[(i + 1) % n]
            ax, ay = x1 - x0, y1 - y0
            norm = math.hypot(ax, ay)
            if norm <= 1e-12:
                continue
            # Separating axis: the perpendicular of this edge. The polygons are
            # separated (beyond ``proximity``) iff their projections onto the
            # axis are disjoint with a gap larger than ``proximity``. Both
            # directions are covered by testing both sides of the inequality,
            # so the hull winding never matters.
            nx, ny = ay / norm, -ax / norm
            proj_a = [nx * px + ny * py for px, py in pa]
            proj_b = [nx * px + ny * py for px, py in pb]
            if max(proj_a) + proximity < min(proj_b):
                return False
            if max(proj_b) + proximity < min(proj_a):
                return False
    return True


def _join_surface_pieces(group: List[Any]) -> Any:
    """Union overlapping surface fragments into one asset (largest keeps its ID).

    Area/dimension math uses the union-of-hulls footprint so a full-corridor
    deck reports its real paved area instead of a fragment's sliver.
    """
    from .models import Asset

    best = max(group, key=lambda a: a.point_count)
    counts = [max(a.point_count, 1) for a in group]
    total = sum(counts)
    cx = sum(a.center["x"] * n for a, n in zip(group, counts)) / total
    cy = sum(a.center["y"] * n for a, n in zip(group, counts)) / total
    cz = sum(a.center["z"] * n for a, n in zip(group, counts)) / total
    box = [
        min(a.bounding_box[0] for a in group), min(a.bounding_box[1] for a in group),
        min(a.bounding_box[2] for a in group), max(a.bounding_box[3] for a in group),
        max(a.bounding_box[4] for a in group), max(a.bounding_box[5] for a in group),
    ]
    hull = _xyhull_axes(
        box,
        [p for a in group for p in ((a.geometry or {}).get("highlight_points") or [])],
    )
    union_area = _polygon_area_m2(hull)
    if union_area <= 0.0:
        union_area = max(1e-6, (box[3] - box[0]) * (box[4] - box[1]))
    geometry = dict(best.geometry or {})
    ground_values = [a.geometry.get("ground_elevation_m") for a in group
                     if a.geometry and a.geometry.get("ground_elevation_m") is not None]
    if ground_values:
        geometry["ground_elevation_m"] = round(sum(ground_values) / len(ground_values), 4)
    for key in ("area_m2",):
        if key in geometry or any(a.geometry and key in a.geometry for a in group):
            geometry[key] = round(union_area, 2)
    source_tile = "+".join(sorted({str(a.source_tile) for a in group}))
    return Asset(
        asset_id=best.asset_id,
        asset_class=best.asset_class,
        subclass=best.subclass,
        center={"x": round(cx, 4), "y": round(cy, 4), "z": round(cz, 4)},
        bounding_box=tuple(round(float(v), 4) for v in box),
        dimensions={
            "length_m": round(box[3] - box[0], 3),
            "width_m": round(box[4] - box[1], 3),
            "height_m": round(box[5] - box[2], 3),
        },
        point_count=sum(a.point_count for a in group),
        source_tile=source_tile,
        source_point_indices_sample=best.source_point_indices_sample,
        coordinate_reference_system=best.coordinate_reference_system,
        confidence=best.confidence,
        confidence_factors=best.confidence_factors,
        confidence_explanation=best.confidence_explanation + f" ({len(group)} tile fragments merged)",
        detection_method=best.detection_method + "-merged",
        intensity_stats=getattr(best, "intensity_stats", None),
        rgb_stats=getattr(best, "rgb_stats", None),
        orientation_deg=getattr(best, "orientation_deg", None),
        source_run=getattr(best, "source_run", None),
        source_scanner=getattr(best, "source_scanner", None),
        source_point_source_id=getattr(best, "source_point_source_id", None),
        model_prior_class=getattr(best, "model_prior_class", None),
        model_confidence=getattr(best, "model_confidence", None),
        processing_version=best.processing_version,
        geometry=geometry,
    )


def merge_surface_fragments(
    assets: List[Any],
    classes: Sequence[str] = ("pavement", "pavement_marking"),
    z_band_m: float = 0.25,
    proximity_m: float = 1.0,
) -> List[Any]:
    """Re-join surface assets (pavement deck, painted markings) split by tiles.

    Road surfaces are the one asset class where per-tile detection is *wrong by
    construction*: the ground/marking detectors emit one blob per tile, so a
    straight corridor reports dozens of fragments for what is physically one
    deck or one painted line. Two fragments are the same surface when their
    footprints intersect **or come within ``proximity_m``** in the XY plane
    (tiles are non-overlapping, so a surface cut at a tile boundary sits
    0.01-0.3 m from its neighbour), they sit in the same elevation band, and
    nothing is elevated between them. Merging keeps the largest fragment's
    identity, unions the point counts, and recomputes area/dimensions over the
    union footprint.

    Convex hulls are computed once per asset and cached: the closure loop
    tests every member-candidate pair repeatedly, and hull building dominates
    the merge runtime otherwise (measured 78% of total pipeline time on a
    210 k-point profile).
    """
    hull_cache: Dict[int, List[Tuple[float, float]]] = {}
    aabb_cache: Dict[int, Tuple[float, float, float, float]] = {}

    def _hull_of(asset: Any) -> List[Tuple[float, float]]:
        cached = hull_cache.get(id(asset))
        if cached is None:
            cached = _xyhull_axes(asset.bounding_box, (asset.geometry or {}).get("highlight_points") or [])
            hull_cache[id(asset)] = cached
        return cached

    def _aabb_of(asset: Any) -> Tuple[float, float, float, float]:
        """Axis-aligned bounds of the cached hull (x0, y0, x1, y1)."""
        cached = aabb_cache.get(id(asset))
        if cached is None:
            hull = _hull_of(asset)
            xs = [p[0] for p in hull]
            ys = [p[1] for p in hull]
            cached = (min(xs), min(ys), max(xs), max(ys))
            aabb_cache[id(asset)] = cached
        return cached

    for class_name in classes:
        pieces = [a for a in assets if a.asset_class == class_name]
        if len(pieces) < 2:
            continue
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
                sz0, sz1 = group[0].bounding_box[2], group[0].bounding_box[5]
                for j, candidate in enumerate(pieces):
                    if used[j]:
                        continue
                    cz0, cz1 = candidate.bounding_box[2], candidate.bounding_box[5]
                    if abs(sz0 - cz0) > z_band_m and abs(sz1 - cz1) > z_band_m:
                        continue
                    # AABB precheck: when *every* member's box is farther than
                    # ``proximity_m`` from the candidate's box in either axis,
                    # the hulls are certainly separated too — skip the SAT test.
                    # Far-apart pairs dominate the pairwise loop, and this is
                    # O(1) per pair with cached boxes (was the SAT over every
                    # pair, the hot spot of the merge on many-tile runs).
                    cx0, cy0, cx1, cy1 = _aabb_of(candidate)
                    close = False
                    for member in group:
                        mx0, my0, mx1, my1 = _aabb_of(member)
                        if not (
                            mx1 + proximity_m < cx0 or mx0 - proximity_m > cx1
                            or my1 + proximity_m < cy0 or my0 - proximity_m > cy1
                        ):
                            close = True
                            break
                    if not close:
                        continue
                    if any(
                        _convex_polygons_within(_hull_of(member), _hull_of(candidate), proximity_m)
                        for member in group
                    ):
                        group.append(candidate)
                        used[j] = True
                        changed = True
            merged.append(group[0] if len(group) == 1 else _join_surface_pieces(group))
        assets = rest + merged
    return assets


def merge_marking_segments(
    assets: List[Any],
    gap_m: float = 14.0,
    lateral_m: float = 1.2,
    aspect_ratio: float = 2.0,
) -> List[Any]:
    """Re-join collinear pavement-marking fragments into continuous lines.

    Painted markings fragment for two reasons: tile boundaries cut long edge
    lines (fixed by ``merge_surface_fragments``), and the paint itself is
    dashed — a centreline is a chain of 3 m blobs separated by ~9 m gaps.
    Fragments whose fitted axes are nearly parallel, laterally aligned (within
    ``lateral_m``), and within ``gap_m`` along their heading are one painted
    marking feature and are joined into a single asset (the largest fragment
    keeps its identity; area/dimensions come from the union footprint).
    """
    pieces = [a for a in assets if a.asset_class == "pavement_marking"]
    if len(pieces) < 2:
        return assets
    rest = [a for a in assets if a.asset_class != "pavement_marking"]
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
                if any(
                    _continues_after_linear(member, candidate, gap_m, lateral_m, aspect_ratio)
                    for member in group
                ):
                    group.append(candidate)
                    used[j] = True
                    changed = True
        merged.append(group[0] if len(group) == 1 else _join_surface_pieces(group))
    return rest + merged


def _pole_supersets(a: Any, b: Any) -> bool:
    """True when two detections describe the same vertical pole.

    Poles are thin and tall: near-identical plan position, overlapping height
    range. Cross-track center spread is capped at the pole diameter tolerance
    so two poles on opposite sides of the road are never deduplicated.
    """
    ax0, ay0, az0, ax1, ay1, az1 = a.bounding_box
    bx0, by0, bz0, bx1, by1, bz1 = b.bounding_box
    plan_a = max(ax1 - ax0, ay1 - ay0)
    plan_b = max(bx1 - bx0, by1 - by0)
    if plan_a > 2.0 or plan_b > 2.0:
        return False
    acx, acy = (ax0 + ax1) / 2, (ay0 + ay1) / 2
    bcx, bcy = (bx0 + bx1) / 2, (by0 + by1) / 2
    tol = max(0.6, 0.5 * (plan_a + plan_b))
    if math.hypot(bcx - acx, bcy - acy) > tol:
        return False
    return az0 < bz1 and bz0 < az1


def dedupe_pole_detections(assets: List[Any], classes: Sequence[str] = ("utility_pole",)) -> List[Any]:
    """Collapse duplicate detections of the same physical pole into one asset.

    Adjacent tiles both see the pole near their shared edge; without this pass
    the inventory double-counts every pole that straddles a boundary. The most
    confident detection wins; the loser's point count is folded into the kept
    asset so no evidence is discarded.
    """
    for class_name in classes:
        poles = [a for a in assets if a.asset_class == class_name]
        if len(poles) < 2:
            continue
        rest = [a for a in assets if a.asset_class != class_name]
        used = [False] * len(poles)
        kept: List[Any] = []
        for i, seed in enumerate(poles):
            if used[i]:
                continue
            group = [seed]
            used[i] = True
            for j in range(i + 1, len(poles)):
                if used[j]:
                    continue
                if any(_pole_supersets(member, poles[j]) for member in group):
                    group.append(poles[j])
                    used[j] = True
            if len(group) == 1:
                kept.append(seed)
                continue
            best = max(group, key=lambda a: (a.confidence, a.point_count))
            others = [a for a in group if a is not best]
            best.point_count += sum(a.point_count for a in others)
            tiles = {str(a.source_tile) for a in group}
            best.source_tile = "+".join(sorted(tiles))
            best.confidence_explanation = (
                best.confidence_explanation + f" (deduplicated: {len(group)} detections of the same pole)"
            )
            kept.append(best)
        assets = rest + kept
    return assets


def _linear_piece_line(a: Any, aspect_ratio: float) -> Optional[Tuple[float, float, float, float, float]]:
    """Fitted axis ``(cx, cy, dirx, diry, half_length)`` for a linear piece.

    Returns None when the piece is not elongated enough (aspect ratio below
    ``aspect_ratio``). The heading comes from the measured PCA orientation
    (``_metrics``, stored by the detector) when available, otherwise from the
    bounding-box major axis. ``half_length`` is half the along-heading extent,
    used to measure the along-track gap between two pieces.
    """
    box = a.bounding_box
    lx = float(box[3] - box[0])
    ly = float(box[4] - box[1])
    metrics = getattr(a, "_metrics", None)
    if metrics is not None and isinstance(metrics, dict) and metrics.get("orientation_deg") is not None:
        theta = math.radians(float(metrics["orientation_deg"]))
        direction = (math.cos(theta), math.sin(theta))
        along = max(float(metrics.get("length_m", lx)), float(metrics.get("width_m", ly)))
        cross = min(float(metrics.get("length_m", lx)), float(metrics.get("width_m", ly)))
    else:
        if max(lx, ly) <= 1e-6:
            return None
        direction = (1.0, 0.0) if lx >= ly else (0.0, 1.0)
        along, cross = max(lx, ly), min(lx, ly)
    if along / max(cross, 1e-3) < aspect_ratio:
        return None
    # PCA azimuth is modulo 180 degrees, so the same line can report 0.01 deg
    # (pointing +x) or 179.98 deg (pointing -x). Normalize the arrow so two
    # pieces of one span always agree on the heading direction.
    if direction[0] < 0.0 or (direction[0] == 0.0 and direction[1] < 0.0):
        direction = (-direction[0], -direction[1])
    cx = (box[0] + box[3]) / 2.0
    cy = (box[1] + box[4]) / 2.0
    return (cx, cy, direction[0], direction[1], max(along, 1e-3) / 2.0)


def _continues_after_linear(
    a: Any, b: Any, gap_m: float, lateral_m: float, aspect_ratio: float,
) -> bool:
    """True when ``b`` looks like the next piece of the same linear span as ``a``.

    Both pieces must be elongated and nearly parallel; the join is decided in
    the pieces' own frame: ``gap_along`` is the centre-to-centre offset
    projected onto the shared heading minus the two half-lengths, and
    ``cross_track`` is the perpendicular distance between the two piece axes.
    Two parallel roadside structures a few metres apart (rails on both sides
    of a corridor) are separated by their cross-track offset, while pieces of
    the same span - even fragments cut diagonally by a tile boundary - share
    a heading and a line, so they join regardless of which axis the tile
    grid split them on.

    Unlike conductor merging, no pole check applies: poles stand *beside*
    safety structures (measured 1.4-1.7 m off a rail), so a nearby pole is
    never evidence of a span boundary.
    """
    la = _linear_piece_line(a, aspect_ratio)
    lb = _linear_piece_line(b, aspect_ratio)
    if la is None or lb is None:
        return False
    ax0, ay0, az0, ax1, ay1, az1 = a.bounding_box
    bx0, by0, bz0, bx1, by1, bz1 = b.bounding_box
    # Same elevation corridor: both pieces' vertical ranges must broadly
    # overlap (a bridge deck 3 m above a roadside rail is never a continuation).
    if abs(az0 - bz0) > 3.0 and abs(az1 - bz1) > 3.0:
        return False
    acx, acy, adx, ady, ahalf = la
    bcx, bcy, bdx, bdy, bhalf = lb
    if adx * bdx + ady * bdy < 0.906:  # headings differ by > ~25 deg
        return False
    dx, dy = bcx - acx, bcy - acy
    gap_along = abs(dx * adx + dy * ady) - (ahalf + bhalf)
    if gap_along > gap_m:
        return False
    cross_track = abs(dx * ady - dy * adx)
    if cross_track > min(lateral_m, 1.5):
        return False
    return True


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