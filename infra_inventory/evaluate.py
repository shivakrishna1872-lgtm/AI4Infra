"""Ground-truth evaluation for the extraction pipeline.

``evaluate_against_ground_truth`` compares detected assets (``assets.json``
format) against a known object table. The Quick Simulation pipeline exports a
``ground_truth`` table inside ``simulation_meta`` because the synthetic scene is
fully known - that gives an honest, reproducible Precision / Recall / F1 /
positional-error measurement of the geometry detectors without any manual
labeling. The same function works against manually created ground-truth tables
for real data (see docs/VALIDATION.md).

Matching is class-aware greedy nearest-centroid within ``match_distance_m``.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _center_of(record: Dict[str, Any]) -> Optional[Tuple[float, float, float]]:
    center = record.get("center")
    if isinstance(center, dict):
        return (float(center["x"]), float(center["y"]), float(center["z"]))
    if isinstance(center, (list, tuple)) and len(center) == 3:
        return (float(center[0]), float(center[1]), float(center[2]))
    # Asset records store the centroid as {x, y, z} in ``center`` already, so a
    # missing center means we cannot evaluate the record.
    return None


def evaluate_against_ground_truth(
    assets: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]],
    match_distance_m: float | Dict[str, float] = 3.0,
    extent_match_classes: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Score detections against ground truth.

    ``assets`` are pipeline asset dicts (the ``assets.json`` schema, which
    exposes ``class`` and ``center``). ``ground_truth`` entries use the
    simulation table schema (``class`` + ``center`` as [x, y, z]).

    ``match_distance_m`` is the centroid match radius. Point assets (poles,
    signs) are matched at a few meters; long or area assets (guardrails,
    pavement, markings) are extracted per component/tile, so pass a per-class
    radius dict (e.g. ``{"utility_pole": 3.0, "pavement": 20.0}``) to keep
    those matches honest.

    ``extent_match_classes`` switches corridor classes (pavement, markings,
    guardrails, barriers) to **extent-aware** matching: a ground-truth object
    is an axis-aligned box inflated by its match radius, and every detection
    whose own bounding box overlaps that box belongs to the object. A 400 m
    painted line is one GT object; whether the pipeline delivers it as one
    merged asset or a handful of tile fragments, every piece on the line is a
    correct detection. Those classes report ``recall`` and ``fragmentation``
    (detections per matched GT object) instead of per-object precision.
    """

    extent_classes = set(extent_match_classes) if extent_match_classes else set()

    def _radius(cls: str) -> float:
        if isinstance(match_distance_m, dict):
            return float(match_distance_m.get(cls, 3.0))
        return float(match_distance_m)
    detected = [(asset, _center_of(asset)) for asset in assets]
    truths = [(gt, _center_of(gt)) for gt in ground_truth]

    # Index detected centroids per class.
    by_class: Dict[str, List[Tuple[Dict[str, Any], Tuple[float, float, float]]]] = {}
    for asset, center in detected:
        if center is None:
            continue
        by_class.setdefault(str(asset.get("class")), []).append((asset, center))

    matched_gt: set[int] = set()
    # Matched detections keyed by (class, index-within-class) so matches in one
    # class can never shadow candidates of another class.
    matched_det: set[Tuple[str, int]] = set()
    positional_errors: List[float] = []
    # Detections claimed per matched extent-matched GT (fragmentation source).
    detections_per_gt: Dict[int, int] = {}

    def _gt_box(gt: Dict[str, Any], radius: float) -> Tuple[float, float, float, float]:
        """GT object's XY footprint (centre + half-dimensions) inflated by radius."""
        cx, cy, _ = _center_of(gt)  # type: ignore[misc]
        dims = gt.get("dimensions_m") or [0.0, 0.0, 0.0]
        hx = float(dims[0]) / 2.0 if len(dims) > 0 else 0.0
        hy = float(dims[1]) / 2.0 if len(dims) > 1 else 0.0
        return (cx - hx - radius, cy - hy - radius, cx + hx + radius, cy + hy + radius)

    # Greedy nearest matching per class: walk ground truth in order and claim
    # the closest still-unclaimed detection within the match radius.
    for gt_index, (gt, gt_center) in enumerate(truths):
        if gt_center is None:
            continue
        cls = str(gt.get("class"))
        candidates = by_class.get(cls, [])
        if cls in extent_classes:
            # Extent-aware: every detection overlapping the inflated GT box is
            # a correct detection of this corridor object (fragments along a
            # 400 m line are the object, not false positives). When several
            # GT objects of the same class exist (two parallel rails), each
            # detection is claimed by the *nearest* GT whose box contains it -
            # a first-come assignment would give both rails to one GT.
            overlaps: List[Tuple[float, int, int]] = []  # (dist, gt_index, det_index)
            for gt2_index, (gt2, gt2_center) in enumerate(truths):
                if gt2_center is None or str(gt2.get("class")) != cls:
                    continue
                x0, y0, x1, y1 = _gt_box(gt2, _radius(cls))
                for det_index, (asset, det_center) in enumerate(candidates):
                    if (cls, det_index) in matched_det:
                        continue
                    bb = asset.get("bounding_box")
                    if not bb or len(bb) < 6:
                        continue
                    if float(bb[3]) >= x0 and float(bb[0]) <= x1 and float(bb[4]) >= y0 and float(bb[1]) <= y1:
                        overlaps.append((math.dist(gt2_center, det_center), gt2_index, det_index))
            claimed_by: Dict[int, int] = {}
            for _dist, gt2_index, det_index in sorted(overlaps):
                if (cls, det_index) in matched_det:
                    continue
                matched_det.add((cls, det_index))
                claimed_by[gt2_index] = claimed_by.get(gt2_index, 0) + 1
            for gt2_index, claimed in claimed_by.items():
                matched_gt.add(gt2_index)
                detections_per_gt[gt2_index] = claimed
            continue
        best: Optional[Tuple[float, int]] = None
        for det_index, (_asset, det_center) in enumerate(candidates):
            if (cls, det_index) in matched_det:
                continue
            dist = math.dist(gt_center, det_center)
            if dist <= _radius(cls) and (best is None or dist < best[0]):
                best = (dist, det_index)
        if best is not None:
            matched_gt.add(gt_index)
            matched_det.add((cls, best[1]))
            positional_errors.append(best[0])

    matched_count = len(matched_gt)
    precision = matched_count / len(detected) if detected else 0.0
    recall = matched_count / len(truths) if truths else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )
    positional_rmse = (
        math.sqrt(sum(e * e for e in positional_errors) / len(positional_errors))
        if positional_errors
        else None
    )

    # Per-class breakdown.
    per_class: Dict[str, Dict[str, Any]] = {}
    classes = sorted({str(gt.get("class")) for gt, _ in truths} | set(by_class.keys()))
    for cls in classes:
        gt_of_cls = [i for i, (gt, _) in enumerate(truths) if str(gt.get("class")) == cls]
        det_of_cls = by_class.get(cls, [])
        tp = sum(1 for i in gt_of_cls if i in matched_gt)
        fp = sum(1 for i, (_a, _c) in enumerate(det_of_cls) if (cls, i) not in matched_det)
        fn = len(gt_of_cls) - tp
        if cls in extent_classes:
            matched_det_count = sum(
                detections_per_gt.get(i, 0) for i in gt_of_cls if i in matched_gt
            )
            per_class[cls] = {
                "ground_truth": len(gt_of_cls),
                "detected": len(det_of_cls),
                "matched_gt": tp,
                "matched_detections": matched_det_count,
                "recall": tp / len(gt_of_cls) if gt_of_cls else 0.0,
                "fragmentation": (
                    matched_det_count / tp if tp else None
                ),
            }
            continue
        per_class[cls] = {
            "ground_truth": len(gt_of_cls),
            "detected": len(det_of_cls),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": tp / len(det_of_cls) if det_of_cls else 0.0,
            "recall": tp / len(gt_of_cls) if gt_of_cls else 0.0,
        }

    return {
        "match_distance_m": match_distance_m,
        "ground_truth_count": len(truths),
        "detected_count": len(detected),
        "matched_count": matched_count,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "positional_error_rmse_m": (
            round(positional_rmse, 4) if positional_rmse is not None else None
        ),
        "unmatched_ground_truth": [
            truths[i][0].get("gt_id", f"GT-{i}") for i in range(len(truths)) if i not in matched_gt
        ],
        "false_positive_asset_ids": [
            asset.get("asset_id", f"det-{i}")
            for cls, candidates in sorted(by_class.items())
            for i, (asset, _c) in enumerate(candidates)
            if (cls, i) not in matched_det
        ],
        "per_class": per_class,
    }