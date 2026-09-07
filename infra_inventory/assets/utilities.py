"""Utility asset detection: poles, overhead conductors, and cabinets.

Rules are deliberately conservative and every rule is a *measured* geometric
property (footprint, vertical extent, linearity, density):

* Pole        - narrow horizontal footprint, multi-metre vertical extent
* Conductor   - elevated, thin, elongated linear chain (often near poles)
* Cabinet     - compact box near the ground with dense point support
"""
from __future__ import annotations

from typing import List

import numpy as np

from ..instances import component_cross_section_m, component_metrics, grid_components
from ..models import Asset
from .common import TileContext, build_asset


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
        # neighbours sit metres away and never win the density vote.
        xs = ctx.x[component]
        ys = ctx.y[component]
        res = settings.pole_resolution_m
        ix = ((xs - xs.min()) / res).astype(np.int64)
        iy = ((ys - ys.min()) / res).astype(np.int64)
        x_cells = int(ix.max()) + 1
        keys = iy * x_cells + ix
        uniq, counts = np.unique(keys, return_counts=True)
        best_key = uniq[int(np.argmax(counts))]
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
    # height band; exclude points already claimed by a detected pole. Cabinets
    # must also sit with their base near the ground (not float at sign height).
    cabinet_mask = (ctx.height >= settings.cabinet_min_height_m) & (ctx.height <= settings.cabinet_max_height_m)
    cabinet_mask &= ~pole_points
    for component in grid_components(ctx.x, ctx.y, cabinet_mask, settings.cabinet_resolution_m, min_cells=2):
        if len(component) < settings.cabinet_min_points:
            continue
        metrics = component_metrics(ctx.x, ctx.y, ctx.z, component)
        side_x, side_y, height = metrics["length_m"], metrics["width_m"], metrics["height_m"]
        if (
            settings.cabinet_min_side_m <= side_x <= settings.cabinet_max_side_m
            and settings.cabinet_min_side_m <= side_y <= settings.cabinet_max_side_m
            and settings.cabinet_min_height_m <= height <= settings.cabinet_max_height_m
        ):
            boxiness = 1.0 - min(1.0, abs(side_x - side_y) / max(max(side_x, side_y), 1e-3))
            geometry_score = min(0.85, 0.45 + boxiness * 0.25 + min(metrics["point_count"] / 2000.0, 0.2))
            explanation = (
                f"Compact near-ground box ({side_x:.2f} x {side_y:.2f} x {height:.2f} m) with dense "
                "point support is consistent with a utility cabinet."
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