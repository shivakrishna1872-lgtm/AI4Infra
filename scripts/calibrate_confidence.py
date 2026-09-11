#!/usr/bin/env python3
"""Calibrate the ALP confidence/review thresholds against ground truth.

Confidence in this project measures *how likely a detected asset is a true
positive*. The default human-review threshold is NOT chosen by intuition: this
script sweeps candidate thresholds over a processed scene whose ground truth is
known (Quick Simulation exports the exact placed objects) and reports measured
precision / recall / F1 at every cut. The threshold in
``configs/processing.yaml`` (``review_confidence_threshold``) and the condition
bands in ``infra_inventory/assessment.py`` should be chosen from this curve.

Usage:
    python scripts/calibrate_confidence.py                     # build + sweep a 400 m scene
    python scripts/calibrate_confidence.py --output out/sim    # sweep an existing run
    python scripts/calibrate_confidence.py --output out/sim --report out/calibration.json
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from infra_inventory.evaluate import evaluate_against_ground_truth  # noqa: E402

# Per-class match radii (object granularity differs per class; see docs/VALIDATION.md).
MATCH_RADII = {
    "utility_pole": 3.0,
    "traffic_sign": 3.0,
    "guardrail": 20.0,
    "safety_barrier": 20.0,
    "pavement": 25.0,
    "pavement_marking": 20.0,
    "utility_cabinet": 3.0,
    "rumble_strip": 10.0,
    "overhead_conductor": 15.0,
}


def load_run(output_dir: Path) -> tuple[list[dict], list[dict]]:
    """Load assets + ground truth from a Quick Simulation output directory.

    ``output_dir`` is the root passed to run_quick_simulation (the scene lives
    under output_dir/simulations/quick-*/pipeline/assets.json).
    """
    candidates = sorted((output_dir / "simulations").glob("quick-*/pipeline/assets.json"))
    if not candidates:
        raise SystemExit(f"No simulation output under {output_dir}/simulations — run a simulation first.")
    assets = json.loads(candidates[-1].read_text())
    pipeline_dir = candidates[-1].parent
    gt_file = pipeline_dir.parent / "ground_truth.json"
    if not gt_file.is_file():
        raise SystemExit(f"Missing {gt_file} — run the build path of this script.")
    ground_truth = json.loads(gt_file.read_text())
    return assets, ground_truth


#: Discrete objects whose instance identity is meaningful (a pole IS a pole).
#: Area/long classes (pavement, markings, guardrails, barriers) are extracted
#: per tile component, so their instance-level precision is meaningless under
#: fragmentation; they are scored by recall only (was any part of the GT object
#: detected), which stays honest. overhead_conductor is deliberately kept out
#: of the compact sweep: its confidence score uses the geometry-v2-heuristic
#: band (cross-section + linearity) rather than the class-consistent factors,
#: so a confidence cut on it would measure the sweep, not the review policy.
COMPACT_CLASSES = {"utility_pole", "traffic_sign", "utility_cabinet", "rumble_strip"}
AREA_CLASSES = {"pavement", "pavement_marking", "guardrail", "safety_barrier"}


def _recall_of(kept: list[dict], ground_truth: list[dict], classes: set[str]) -> float:
    """Fraction of ground-truth objects of these classes matched by a kept asset."""
    gt = [g for g in ground_truth if g["class"] in classes]
    if not gt:
        return 1.0
    result = evaluate_against_ground_truth(
        kept, gt, match_distance_m={c: MATCH_RADII[c] for c in classes if c in MATCH_RADII}
    )
    return round(result["recall"], 4)


def sweep(assets: list[dict], ground_truth: list[dict]) -> list[dict]:
    # QC LIKELY_DUPLICATE rows are routed to human review, not delivered to the
    # inventory (docs/VALIDATION.md §4) - the same exclusion train_thresholds.py
    # applies, so a tile-split object detected from two adjacent tiles never
    # counts against the accepted set's precision.
    assets = [a for a in assets if "LIKELY_DUPLICATE" not in a.get("qc_flags", [])]
    rows = []
    compact_gt = [g for g in ground_truth if g["class"] in COMPACT_CLASSES]
    for threshold in [round(0.30 + 0.05 * i, 2) for i in range(13)]:  # 0.30 .. 0.90
        kept = [a for a in assets if a["confidence"] >= threshold]
        compact = [a for a in kept if a["class"] in COMPACT_CLASSES]
        if not compact:
            rows.append({"threshold": threshold, "kept": len(kept), "precision": None,
                         "recall": None, "f1": None, "area_recall": _recall_of(kept, ground_truth, AREA_CLASSES)})
            continue
        result = evaluate_against_ground_truth(
            compact, compact_gt, match_distance_m={c: MATCH_RADII[c] for c in COMPACT_CLASSES if c in MATCH_RADII}
        )
        rows.append({
            "threshold": threshold,
            "kept": len(kept),
            "precision": round(result["precision"], 4),
            "recall": round(result["recall"], 4),
            "f1": round(result["f1"], 4),
            "area_recall": _recall_of(kept, ground_truth, AREA_CLASSES),
        })
    return rows


def recommend(rows: list[dict]) -> dict:
    """Pick a review threshold: the most selective cut whose auto-accepted
    compact objects are still essentially clean (precision >= 0.95) while
    compact recall stays >= 0.9 and area-class recall stays >= 0.8. Review
    routing is most valuable at the strictest threshold that does not lose true
    positives, so among the cuts meeting the bar we take the highest one (with
    a flat 1.000 curve that is 0.80, the shipped default). If no cut meets the
    bar, report the best-F1 cut and flag that detection quality (not the
    threshold) is the limiting factor."""
    candidates = [r for r in rows if r["precision"] is not None
                  and r["precision"] >= 0.95 and r["recall"] is not None and r["recall"] >= 0.9
                  and r["area_recall"] is not None and r["area_recall"] >= 0.8]
    if candidates:
        chosen = max(candidates, key=lambda r: r["threshold"])
        chosen["strategy"] = "strictest cut with compact precision >= 0.95, compact recall >= 0.9, area recall >= 0.8"
    else:
        valid = [r for r in rows if r["f1"] is not None]
        chosen = max(valid, key=lambda r: r["f1"]) if valid else rows[-1]
        chosen["strategy"] = "no cut met the precision bar — best compact F1 reported; improve detection, do not lower the bar"
    return chosen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Existing Quick Simulation output dir")
    parser.add_argument("--length", type=float, default=400.0, help="Scene length when building")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--report", type=Path, help="Write the sweep table to this JSON file")
    args = parser.parse_args()

    if args.output and (args.output / "simulations").is_dir() \
            and any((args.output / "simulations").glob("quick-*")):
        output_dir = args.output
        assets, ground_truth = load_run(output_dir)
        print(f"Using existing run: {output_dir}")
    else:
        from infra_inventory.simulation import run_quick_simulation

        tmp = Path(tempfile.mkdtemp(prefix="terra-point-calibrate-"))
        output_dir = tmp
        print(f"Building a {args.length:.0f} m Quick Simulation scene (this runs the real pipeline)...")
        project = run_quick_simulation(tmp, length_m=args.length, seed=args.seed)
        # Persist ground truth next to the pipeline outputs for the --output path.
        pipeline_dir = Path(project["output_dir"]) / "pipeline"
        (pipeline_dir.parent / "ground_truth.json").write_text(
            json.dumps(project["simulation_meta"]["ground_truth"], indent=2)
        )
        assets, ground_truth = load_run(output_dir)

    rows = sweep(assets, ground_truth)
    header = f"{'threshold':>10} {'kept':>7} {'precision':>10} {'recall':>7} {'f1':>7}"
    print(header)
    print("-" * len(header))
    for row in rows:
        def fmt(value):
            return "—" if value is None else f"{value:.3f}"
        print(f"{row['threshold']:>10.2f} {row['kept']:>7} {fmt(row['precision']):>10} "
              f"{fmt(row['recall']):>7} {fmt(row['f1']):>7}")

    chosen = recommend(rows)
    print("\nRecommended human-review threshold: "
          f"confidence < {chosen['threshold']:.2f} -> REVIEW "
          f"(kept {chosen['kept']} assets, precision {chosen['precision']}, "
          f"recall {chosen['recall']})")
    print("Set review_confidence_threshold in configs/processing.yaml to this value.")

    if args.report:
        args.report.write_text(json.dumps({
            "scene": str(output_dir),
            "sweep": rows,
            "recommended_threshold": chosen["threshold"],
            "review_condition": "REVIEW",
            "note": "Confidence measures true-positive likelihood; threshold set from "
                    "measured precision/recall, not intuition. See docs/ALP.md.",
        }, indent=2))
        print(f"Wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
