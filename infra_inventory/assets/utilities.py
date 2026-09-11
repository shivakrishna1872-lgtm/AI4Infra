"""Utility asset detection: poles, overhead conductors, and cabinets.

Rules are deliberately conservative and every rule is a *measured* geometric
property (footprint, vertical extent, linearity, density):

* Pole        - narrow horizontal footprint, multi-metre vertical extent
* Conductor   - elevated, thin, elongated linear chain (often near poles)
* Cabinet     - compact box near the ground with dense point support
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..instances import component_cross_section_m, component_metrics, grid_components
from ..models import Asset
from .common import TileContext, build_asset


def _true_height_in_footprint(
    ctx: TileContext, metrics: dict, height_cap: float,
    exclude: Optional[np.ndarray] = None,
) -> float:
    """Vertical extent of all tile points inside a component's XY footprint.

    The cabinet detector slices a height band (cabinet_min..cabinet_max), so a
    1.3 m cabinet's sliced component is only its top ~0.5 m — gating on the
    *sliced* height rejects every real enclosure (measured: cabinet recall
    0.0 on the QuickSim ground truth). The honest height is the footprint's
    full z-extent within the cabinet evidence window: above ground scatter,
    below the overhead-conductor band (height_cap + margin).

    ``exclude`` masks points already claimed by a detected pole, so a cabinet
    standing beside/under a pole line is not measured 2.8 m tall by the
    trunk's lower shaft (which would push the enclosure over the height gate
    and delete a real cabinet).
    """
    pad = 0.15
    bounds = metrics["bounds"]
    foot = (
        (ctx.x >= bounds[0] - pad) & (ctx.x <= bounds[3] + pad)
        & (ctx.y >= bounds[1] - pad) & (ctx.y <= bounds[4] + pad)
    )
    window = foot & (ctx.height > 0.12) & (ctx.height <= height_cap + 0.3)
    if exclude is not None:
        window &= ~exclude
    if not window.any():
        return float(metrics["height_m"])
    return float(ctx.height[window].max())


def detect_utilities(ctx: TileContext) -> List[Asset]:
    assets: List[Asset] = []
    settings = ctx.settings
    pole_points = np.zeros(len(ctx.x), dtype=bool)

    # ---- Poles ----------------------------------------------------------------
    # Exclude the ground surface itself (height < 0.3) so the pole stands as its
    # own component. Components routinely absorb neighbours at grid resolution
    # (a guardrail run, or the two wire spans that share a pole), so pole-ness is
    # measured from the *densest XY cell* - the shaft - not the whole component:
    # a merged guardrail inflates a component bbox to tens of metres and a merged
    # crossarm/wire pair collapses columnarity, even though the shaft is clean.
    pole_mask = ctx.height >= max(0.3, settings.pole_min_height_m * 0.12)
    for component in grid_components(ctx.x, ctx.y, pole_mask, settings.pole_resolution_m, min_cells=1):
        if len(component) < settings.pole_min_points:
            continue
        # Densest XY cell of the component = the trunk column. Guardrail/panel
        # neighbours sit metres away and never win the density vote. When a
        # cell is *tied* on density (a cabinet flush against a pole has more
        # points in its footprint cells than the shaft), the cell reaching
        # highest wins: a cabinet top stops at ~2.5 m, a pole trunk keeps
        # going, so the z-range tie-break picks the shaft, not the box.
        xs = ctx.x[component]
        ys = ctx.y[component]
        res = settings.pole_resolution_m
        ix = ((xs - xs.min()) / res).astype(np.int64)
        iy = ((ys - ys.min()) / res).astype(np.int64)
        x_cells = int(ix.max()) + 1
        keys = iy * x_cells + ix
        uniq, counts = np.unique(keys, return_counts=True)
        z_lo_cell = ctx.z[component].min()
        z_hi_cell = ctx.z[component].max()
        z_span_cell = max(z_hi_cell - z_lo_cell, 1e-6)
        z_max_in_cell = np.full(len(uniq), -np.inf)
        np.maximum.at(
            z_max_in_cell,
            np.searchsorted(uniq, keys),
            (ctx.z[component] - z_lo_cell) / z_span_cell,
        )
        best_key = uniq[int(np.lexsort((z_max_in_cell, -counts))[0])]  # density first, then highest reach
        cell_cx = xs.min() + (best_key % x_cells + 0.5) * res
        cell_cy = ys.min() + (best_key // x_cells + 0.5) * res
        # Near-axis points: the trunk plus attachments. The densest cell's centre
        # sits within half a cell diagonal of the pole axis and the shaft adds its
        # radius, so ~0.7 m always contains the whole trunk (0.45 m cells, poles
        # up to ~0.7 m diameter). The radius is also the *rejection* envelope:
        #  - too wide (1.2 m) leaks a concrete barrier's top edge (~1 m from the
        #    pole) into the lower-80% shaft, inflating the footprint past
        #    pole_max_footprint_m and collapsing columnarity (measured miss on
        #    the barrier-adjacent pole, seed 7 QuickSim);
        #  - too narrow (0.7 m) lets a sign panel merged with a guardrail rail
        #    (0.3 m offset) pass as a pole: the shaft footprint lands just under
        #    the gate (measured false positive, seed 7 QuickSim).
        # 0.9 m excludes the barrier (>1.0 m away) yet sweeps enough of a
        # parallel rail to push the sign+rail footprint over the gate.
        axis_mask = np.hypot(xs - cell_cx, ys - cell_cy) <= 0.9
        near_axis = component[axis_mask]
        if len(near_axis) < settings.pole_min_points:
            continue
        metrics = component_metrics(ctx.x, ctx.y, ctx.z, near_axis)
        # Lower 80% of the shaft only (excludes crossarm/wire points at the top)
        # for the narrow-footprint / columnarity gates.
        zs = ctx.z[near_axis]
        z_lo, z_hi = float(zs.min()), float(zs.max())
        shaft = near_axis[zs <= z_lo + 0.8 * max(z_hi - z_lo, 1e-3)]
        shaft_metrics = component_metrics(ctx.x, ctx.y, ctx.z, shaft)
        footprint = max(shaft_metrics["length_m"], shaft_metrics["width_m"])
        if (
            metrics["height_m"] >= settings.pole_min_height_m
            and footprint <= settings.pole_max_footprint_m
            and shaft_metrics["columnarity"] >= settings.pole_min_columnarity
        ):
            # Geometry: how pole-like is it? Dominant vertical extent + narrow base.
            vertical_ratio = metrics["height_m"] / max(footprint, 1e-3)
            geometry_score = min(0.95, 0.45 + min(vertical_ratio / 8.0, 0.35) + metrics["verticality"] * 0.15)
            explanation = (
                f"Narrow trunk footprint ({footprint:.2f} m) with {metrics['height_m']:.1f} m vertical "
                "extent satisfies utility-pole geometry rules."
            )
            asset = build_asset(
                ctx,
                asset_class="utility_pole",
                subclass="vertical_support",
                indices=near_axis,
                geometry_score=geometry_score,
                explanation=explanation,
                method="geometry-v1",
                model_target_classes=("utility_pole",),
            )
            if asset is not None:
                assets.append(asset)
                # Claim only the pole footprint + crossarm reach. Claiming the
                # whole connected component would swallow the attached conductor
                # run: at grid resolution the wire merges with the trunk at the
                # attachment, so a component-wide claim deletes the entire span
                # in every tile whose wire cells touch the pole.
                pole_points[component[axis_mask]] = True

    # ---- Overhead conductors ---------------------------------------------------
    # Poles and their crossarms sit in the same x,y cells the wires attach to;
    # including them merges trunk+crossarm+wire into one component whose vertical
    # extent (> conductor_max_height_extent_m) rejects the whole span. Exclude
    # points already claimed by a detected pole so each wire becomes its own
    # elevated thin chain.
    conductor_mask = ctx.height >= settings.conductor_min_height_m
    conductor_mask &= ~pole_points
    for component in grid_components(ctx.x, ctx.y, conductor_mask, settings.conductor_resolution_m, min_cells=3):
        if len(component) < settings.conductor_min_points:
            continue
        metrics = component_metrics(ctx.x, ctx.y, ctx.z, component)
        # Bounding-box width is orientation dependent (a 14-deg wire measures
        # metres wide), so the thickness gate uses the 2-sigma minor PCA axis.
        cross_section = component_cross_section_m(ctx.x, ctx.y, ctx.z, component)
        length = max(metrics["length_m"], metrics["width_m"])
        if (
            length >= settings.conductor_min_length_m
            and cross_section <= settings.conductor_max_width_m
            and metrics["height_m"] <= settings.conductor_max_height_extent_m
            and metrics["linearity"] >= 0.55
        ):
            geometry_score = min(0.7, 0.35 + metrics["linearity"] * 0.25 + min(length / 50.0, 0.2))
            explanation = (
                f"Elevated thin linear chain ({length:.1f} m long, ~{cross_section:.2f} m "
                "cross-section) is consistent with an overhead conductor; heuristic detection."
            )
            asset = build_asset(
                ctx,
                asset_class="overhead_conductor",
                subclass="conductor",
                indices=component,
                geometry_score=geometry_score,
                explanation=explanation,
                method="geometry-v2-heuristic",
                model_target_classes=(),
                min_points=settings.conductor_min_points,
            )
            if asset is not None:
                assets.append(asset)

    # ---- Cabinets ---------------------------------------------------------------
    # Pole trunks masquerade as compact near-ground boxes when sliced by the
    # height band; exclude points already claimed by a detected pole. The
    # component is built from the *full* cabinet evidence window (above ground
    # scatter, up to the overhead-conductor band) rather than the narrow
    # min..max band slice: a 0.95 m cabinet sliced at 0.8-2.5 m is a 0.15 m
    # sliver whose features look like background and got vetoed, while its
    # full box is exactly the enclosure shape the classifier was trained on.
    # The height gate then measures the component's true vertical extent.
    cabinet_window = (ctx.height > 0.12) & (ctx.height <= settings.cabinet_max_height_m + 0.3)
    cabinet_window &= ~pole_points
    for component in grid_components(ctx.x, ctx.y, cabinet_window, settings.cabinet_resolution_m, min_cells=2):
        if len(component) < settings.cabinet_min_points:
            continue
        metrics = component_metrics(ctx.x, ctx.y, ctx.z, component)
        side_x, side_y, height = metrics["length_m"], metrics["width_m"], metrics["height_m"]
        # Height gate uses the footprint's *full* vertical extent, not the
        # sliced band's (a 1.3 m cabinet sliced at 0.8-2.5 m measures 0.5 m
        # tall and would never pass a 0.8 m minimum on the sliced height).
        # Pole-claimed points are excluded so a cabinet under a pole line is
        # not measured 2.8 m tall by the trunk.
        true_height = _true_height_in_footprint(
            ctx, metrics, settings.cabinet_max_height_m, exclude=pole_points,
        )
        max_side = max(side_x, side_y)
        min_side = min(side_x, side_y)
        depth_ratio = min_side / max_side if max_side > 1e-6 else 0.0
        aspect_ratio = max_side / min_side if min_side > 1e-6 else 999.0
        # Strict box/cube profile: depth-to-width ratio 0.5-2.0
        if not (settings.cabinet_max_side_ratio >= depth_ratio >= (1.0 / settings.cabinet_max_side_ratio)):
            # Flat/elongated roadside structure -> reclassify as guardrail/barrier,
            # but only when the candidate is genuinely linear and low. A thin
            # vertical remnant (sign post + panel bottom merged at grid
            # resolution: ~0 m long, ~0.9 m wide, full-height) is neither a
            # cabinet nor a barrier and must be dropped (measured: 2 false
            # safety_barriers per QuickSim scene from sign posts).
            if max_side < 1.5 or true_height > settings.barrier_max_height_m:
                continue
            barrier_cls = "safety_barrier" if true_height > 1.0 else "guardrail"
            explanation = (
                f"Elongated roadside structure (aspect ratio {aspect_ratio:.1f}:1, "
                f"depth/width ratio {depth_ratio:.2f}, {max_side:.1f} m long, "
                f"{true_height:.2f} m tall) inconsistent with a utility cabinet; "
                f"reclassifying as {barrier_cls}."
            )
            asset = build_asset(
                ctx,
                asset_class=barrier_cls,
                subclass="barrier" if barrier_cls == "safety_barrier" else "guardrail_segment",  # validates against VALID_TAXONOMY
                indices=component,
                geometry_score=0.7,
                explanation=explanation,
                method="geometry-v1",
                model_target_classes=(barrier_cls,),
            )
            if asset is not None:
                assets.append(asset)
            continue
        if (
            settings.cabinet_min_side_m <= side_x <= settings.cabinet_max_side_m
            and settings.cabinet_min_side_m <= side_y <= settings.cabinet_max_side_m
            and settings.cabinet_min_height_m <= true_height <= settings.cabinet_max_height_m
        ):
            boxiness = 1.0 - min(1.0, abs(side_x - side_y) / max(max(side_x, side_y), 1e-3))
            geometry_score = min(0.85, 0.45 + boxiness * 0.25 + min(metrics["point_count"] / 2000.0, 0.2))
            explanation = (
                f"Compact near-ground box ({side_x:.2f} x {side_y:.2f} x {true_height:.2f} m, "
                f"depth/width {depth_ratio:.2f}) with dense point support is consistent "
                "with a utility cabinet."
            )
            asset = build_asset(
                ctx,
                asset_class="utility_cabinet",
                subclass="cabinet",
                indices=component,
                geometry_score=geometry_score,
                explanation=explanation,
                method="geometry-v1",
                model_target_classes=("utility_cabinet",),
            )
            if asset is not None:
                assets.append(asset)
    return assets