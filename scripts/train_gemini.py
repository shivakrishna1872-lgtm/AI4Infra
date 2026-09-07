#!/usr/bin/env python3
"""Measure the Gemini class-validation booster against QuickSim ground truth.

How the booster is "trained": the booster has no weights to fit — it is an
independent learned reviewer whose verdicts are folded into the transparent
confidence blend as the ``model`` factor. What this script measures is whether
that fold *improves the decision*: precision/recall/F1 of auto-accepted
compact assets at the calibrated 0.80 review cut, before and after the boost.

Two modes:

* default — **oracle simulation**: every asset receives the verdict a perfect
  reviewer would give (AGREE when the detected class matches the ground-truth
  object at its location, DISAGREE otherwise). This measures the mechanism
  (does a good learned reviewer lift the accepted set?) with zero API cost.
* ``--live`` — the real Google Gemini API (requires GEMINI_API_KEY, see
  docs/GEMINI.md); verdicts are cached in ``out/gemini_cache.json`` so
  re-runs of the same scenes do not re-bill.

Usage:
    python scripts/train_gemini.py                     # oracle simulation
    python scripts/train_gemini.py --live              # real Gemini API
    python scripts/train_gemini.py --report out/accuracy/gemini_report.json
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from infra_inventory.evaluate import evaluate_against_ground_truth  # noqa: E402
from infra_inventory.gemini import GeminiBooster, apply_verdict  # noqa: E402
from infra_inventory.models import Asset  # noqa: E402

#: Same match radii as scripts/calibrate_confidence.py (docs/VALIDATION.md).
MATCH_RADII = {
    "utility_pole": 3.0, "traffic_sign": 3.0, "utility_cabinet": 3.0,
    "rumble_strip": 10.0, "overhead_conductor": 15.0,
    "guardrail": 20.0, "safety_barrier": 20.0, "pavement": 25.0, "pavement_marking": 20.0,
}
COMPACT_CLASSES = {"utility_pole", "traffic_sign", "utility_cabinet", "rumble_strip"}
AREA_CLASSES = {"pavement", "pavement_marking", "guardrail", "safety_barrier"}
REVIEW_CUT = 0.80  # calibrated human-review threshold (configs/processing.yaml)


def _accepted(assets: list[dict]) -> list[dict]:
    """Assets that reach the inventory: not duplicate-flagged and above the cut."""
    return [a for a in assets if "LIKELY_DUPLICATE" not in a.get("qc_flags", []) and a["confidence"] >= REVIEW_CUT]


def _scores(assets: list[dict], ground_truth: list[dict]) -> dict:
    accepted = _accepted(assets)
    compact = [a for a in accepted if a["class"] in COMPACT_CLASSES]
    compact_gt = [g for g in ground_truth if g["class"] in COMPACT_CLASSES]
    result = evaluate_against_ground_truth(
        compact, compact_gt, match_distance_m={c: MATCH_RADII[c] for c in COMPACT_CLASSES}
    )
    gt_area = [g for g in ground_truth if g["class"] in AREA_CLASSES]
    area_recall = evaluate_against_ground_truth(
        [a for a in accepted if a["class"] in AREA_CLASSES], gt_area,
        match_distance_m={c: MATCH_RADII[c] for c in AREA_CLASSES},
    )["recall"] if gt_area else None
    return {
        "accepted": len(accepted),
        "compact_precision": round(result["precision"], 4),
        "compact_recall": round(result["recall"], 4),
        "compact_f1": round(result["f1"], 4),
        "area_recall": round(area_recall, 4) if area_recall is not None else None,
    }


def _center_xy(record: dict) -> tuple[float, float]:
    center = record["center"]
    if isinstance(center, dict):
        return float(center["x"]), float(center["y"])
    return float(center[0]), float(center[1])


def _nearest_gt(asset: dict, ground_truth: list[dict]) -> dict | None:
    ax, ay = _center_xy(asset)
    best, best_d = None, float("inf")
    for gt in ground_truth:
        gx, gy = _center_xy(gt)
        d = (gx - ax) ** 2 + (gy - ay) ** 2
        if d < best_d:
            best_d = d
            best = gt
    return best


def oracle_boost(assets: list[dict], ground_truth: list[dict]) -> list[Asset]:
    """Perfect-reviewer verdicts: AGREE iff a same-class GT object exists within
    the documented per-class match radius (the same rule evaluate.py uses), so
    per-tile area pieces are judged against the area they belong to — not the
    nearest unrelated object."""
    boosted: list[Asset] = []
    for record in assets:
        asset = _record_to_asset(record)
        ax, ay = _center_xy(record)
        agreed = False
        for gt in ground_truth:
            if gt["class"] != record["class"]:
                continue
            gx, gy = _center_xy(gt)
            if (gx - ax) ** 2 + (gy - ay) ** 2 <= (MATCH_RADII.get(record["class"], 15.0)) ** 2:
                agreed = True
                break
        if agreed:
            verdict = {"verdict": "AGREE", "plausibility": 0.95, "reason": "oracle: same-class ground truth within match radius"}
        else:
            verdict = {"verdict": "DISAGREE", "plausibility": 0.1, "reason": "oracle: no same-class ground truth within match radius"}
        apply_verdict(asset, verdict)
        boosted.append(asset)
    return boosted


def _record_to_asset(record: dict) -> Asset:
    """Build an Asset from an exported record (Asset.from_dict may not exist)."""
    return Asset(
        asset_id=record["asset_id"], asset_class=record["class"], subclass=record.get("subclass"),
        center=record["center"], bounding_box=tuple(record["bounding_box"]),
        dimensions=record.get("dimensions") or {}, point_count=record["point_count"],
        source_tile=record["source_tile"], source_point_indices_sample=record.get("source_point_indices_sample", []),
        coordinate_reference_system=record.get("coordinate_reference_system"),
        confidence=record["confidence"], confidence_factors=record.get("confidence_factors") or {},
        confidence_explanation=record.get("confidence_explanation", ""),
        detection_method=record.get("detection_method", ""),
        intensity_stats=record.get("intensity_stats"), rgb_stats=record.get("rgb_stats"),
        orientation_deg=record.get("orientation_deg"), source_run=record.get("source_run"),
        source_scanner=record.get("source_scanner"), model_prior_class=record.get("model_prior_class"),
        model_confidence=record.get("model_confidence"), processing_version=record.get("processing_version", "0.3.0"),
        geometry=record.get("geometry"), qc_flags=list(record.get("qc_flags") or []),
        flagged=bool(record.get("flagged")), condition=record.get("condition"),
        recommended_action=record.get("recommended_action"), review_required=bool(record.get("review_required")),
        assessment_reasoning=record.get("assessment_reasoning"),
    )


def build_scene(length_m: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Run the real pipeline on a QuickSim scene; return (assets, ground_truth)."""
    from infra_inventory.simulation import run_quick_simulation

    tmp = Path(tempfile.mkdtemp(prefix="ai4infra-gemini-"))
    project = run_quick_simulation(tmp, length_m=length_m, seed=seed)
    pipeline_dir = Path(project["output_dir"]) / "pipeline"
    assets = json.loads((pipeline_dir / "assets.json").read_text())
    ground_truth = project["simulation_meta"]["ground_truth"]
    return assets, ground_truth


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--live", action="store_true", help="Call the real Gemini API (needs GEMINI_API_KEY)")
    parser.add_argument("--api-key", help="Gemini API key (overrides GEMINI_API_KEY env)")
    parser.add_argument("--model", default="gemini-3.6-flash")
    parser.add_argument("--length", type=float, default=400.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[23, 42, 7])
    parser.add_argument("--report", type=Path, default=Path("out/accuracy/gemini_report.json"))
    args = parser.parse_args()

    rows = []
    for seed in args.seeds:
        print(f"[scene] seed {seed} ({args.length:.0f} m)...", flush=True)
        assets, ground_truth = build_scene(args.length, seed)
        baseline = _scores(assets, ground_truth)
        boosted: dict | None = None

        if args.live:
            # Cache verdicts on disk: re-runs of the same scenes cost zero API
            # calls (the free tier is 20 requests/day for flash models).
            booster = GeminiBooster(
                model=args.model, api_key=args.api_key,
                cache_path=str(args.report.parent / "gemini_cache.json"),
            )
            boosted_assets = [ _record_to_asset(a) for a in assets ]
            try:
                summary = booster.boost(boosted_assets)
            except Exception as exc:
                # Quota exhausted / network down: keep the run useful. Baseline
                # numbers are already recorded; the boosted numbers for this
                # seed come from whatever verdicts were cached so far.
                print(f"  live boost failed ({exc}); falling back to cached verdicts only")
                summary = {"cache_hits": 0, "api_calls": 0}
            applied = sum(1 for a in boosted_assets if a.model_confidence is not None)
            print(f"  live verdicts applied: {applied}/{len(boosted_assets)} "
                  f"(cache hits this run: {summary.get('cache_hits', 0)}, api calls: {summary.get('api_calls', 0)})")
        else:
            boosted_assets = oracle_boost(assets, ground_truth)

        boosted_records = [asset.to_dict() for asset in boosted_assets]
        boosted = _scores(boosted_records, ground_truth)
        rows.append({"seed": seed, "assets": len(assets), "baseline": baseline, "boosted": boosted})
        print(f"  baseline:  {baseline}")
        print(f"  boosted:   {boosted}")

    deltas = []
    for row in rows:
        b, k = row["baseline"], row["boosted"]
        deltas.append({
            "seed": row["seed"],
            "accepted_delta": k["accepted"] - b["accepted"],
            "compact_f1_delta": round(k["compact_f1"] - b["compact_f1"], 4),
            "area_recall_delta": (round(k["area_recall"] - b["area_recall"], 4)
                                  if b["area_recall"] is not None and k["area_recall"] is not None else None),
        })
    summary = {
        "mode": "live-gemini" if args.live else "oracle-simulation",
        "model": args.model if args.live else None,
        "review_cut": REVIEW_CUT,
        "note": "Oracle mode measures the booster mechanism with a perfect reviewer; "
                "live mode measures the real Gemini API. Boosted assets fold the verdict "
                "into the model confidence factor before the calibrated 0.80 review cut.",
        "rows": rows,
        "deltas": deltas,
        "mean_compact_f1_delta": round(float(np.mean([d["compact_f1_delta"] for d in deltas])), 4),
        "mean_accepted_delta": round(float(np.mean([d["accepted_delta"] for d in deltas])), 2),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())