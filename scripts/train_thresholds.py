#!/usr/bin/env python3
"""Train / tune the geometry-detector thresholds against known ground truth.

The extraction "model" in this project is a set of measured-geometry detectors
(poles, conductors, cabinets, signs, guardrails, barriers, rumble strips,
pavement, markings) whose behaviour is governed by ~30 thresholds in
``ProcessingSettings``. There is no labelled real corpus in the repo, so the
training signal is the Quick Simulation ground truth: every synthetic scene is
a mobile-LiDAR corridor whose placed objects are known exactly, and the whole
real pipeline (tiling -> preprocessing -> detectors -> merge -> QC -> ALP) runs
over it. This script

  1. measures the current configuration across several scene seeds,
  2. runs a coordinate descent over the detector thresholds that gate the
     weakest classes, keeping a change only when the measured objective
     improves (never intuition),
  3. re-measures on held-out seeds that were not used for tuning, and
  4. writes a machine-readable report to ``--report``.

Objective (equal weights over the nine competition classes):

  compact classes  (utility_pole, overhead_conductor, utility_cabinet,
                    traffic_sign, rumble_strip):  instance F1
  linear/area classes (guardrail, safety_barrier, pavement,
                       pavement_marking):          recall
                   (per-instance precision on these is dominated by per-tile
                   fragmentation - see docs/VALIDATION.md - so recall + spatial
                   agreement is the honest measure; detections that QC flags as
                   LIKELY_DUPLICATE are excluded, because those rows go to human
                   review instead of the inventory.)

Usage:
    python scripts/train_thresholds.py                     # train + validate + report
    python scripts/train_thresholds.py --report out/accuracy/training.json
    python scripts/train_thresholds.py --seeds 11 23 --validate-seeds 7 5 99
    python scripts/train_thresholds.py --dry-run           # measure current config only
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from infra_inventory.evaluate import evaluate_against_ground_truth  # noqa: E402
from infra_inventory.models import ProcessingSettings  # noqa: E402
from infra_inventory.simulation import run_quick_simulation  # noqa: E402

# Per-class match radii: object granularity differs per class (see docs/VALIDATION.md).
MATCH_RADII: Dict[str, float] = {
    "utility_pole": 3.0,
    "traffic_sign": 3.0,
    "utility_cabinet": 3.0,
    "overhead_conductor": 12.0,
    "guardrail": 20.0,
    "safety_barrier": 20.0,
    "rumble_strip": 10.0,
    "pavement": 25.0,
    "pavement_marking": 20.0,
}
COMPACT = {"utility_pole", "overhead_conductor", "utility_cabinet", "traffic_sign", "rumble_strip"}
AREA = {"guardrail", "safety_barrier", "pavement", "pavement_marking"}

#: Corridor classes scored by extent-aware matching (docs/VALIDATION.md §2b):
#: these are extracted per tile piece on purpose, so one real object yields
#: several inventory rows matched to one GT object — per-instance precision
#: undercounts, and recall + fragmentation are the honest measures.
EXTENT_MATCH_CLASSES = tuple(sorted(AREA))

# (setting name, [candidate values]) for the coordinate descent. Values were
# picked around the shipped defaults; the descent only ever moves a knob when
# the measured objective improves on the train seeds.
KNOBS: Dict[str, List[float]] = {
    "pole_min_points": [30, 40, 50, 60],
    "pole_min_height_m": [2.0, 2.5, 3.0],
    "conductor_min_points": [10, 15, 20, 25],
    "conductor_min_length_m": [4.0, 5.0, 6.0],
    "conductor_max_width_m": [0.5, 0.6, 0.8],
    "sign_min_planarity": [0.45, 0.55, 0.65],
    "cabinet_min_points": [50, 60, 80],
    "rumble_min_band_points": [10, 12, 15],
}


def _objective(per_class: Dict[str, Dict[str, Any]]) -> float:
    """Equal-weight objective over compact-class F1 and area-class recall."""
    scores: List[float] = []
    for cls in sorted(COMPACT):
        m = per_class[cls]
        f = 2 * m["precision"] * m["recall"] / (m["precision"] + m["recall"]) if m["precision"] + m["recall"] else 0.0
        scores.append(f)
    for cls in sorted(AREA):
        m = per_class[cls]
        scores.append(m["recall"])
    return sum(scores) / len(scores) if scores else 0.0


def measure_one(
    root: Path, seed: int, settings: ProcessingSettings,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run one scene through the real pipeline and score it.

    Returns (evaluation dict, per-class dict). QC LIKELY_DUPLICATE rows are
    excluded: those are flagged for review, not delivered as inventory.
    """
    project = run_quick_simulation(root, length_m=400.0, seed=seed, settings=settings)
    assets = json.loads(
        (Path(project["output_dir"]) / "pipeline" / "assets.json").read_text(encoding="utf-8")
    )
    assets = [a for a in assets if "LIKELY_DUPLICATE" not in a.get("qc_flags", [])]
    ground_truth = project["simulation_meta"]["ground_truth"]
    result = evaluate_against_ground_truth(
        assets, ground_truth, match_distance_m=MATCH_RADII,
        extent_match_classes=EXTENT_MATCH_CLASSES,
    )
    return result, result["per_class"]


def measure(
    root: Path, seeds: List[int], settings: ProcessingSettings,
) -> Tuple[float, Dict[str, Dict[str, Any]]]:
    """Aggregate per-class counts over seeds and return (objective, per_class)."""
    agg: Dict[str, Dict[str, Any]] = {}
    for seed in seeds:
        _, per_class = measure_one(root, seed, settings)
        for cls, m in per_class.items():
            d = agg.setdefault(cls, {"gt": 0, "det": 0, "tp": 0, "fp": 0, "fn": 0,
                                     "matched_detections": 0})
            d["gt"] += m["ground_truth"]
            d["det"] += m["detected"]
            d["tp"] += m.get("true_positives", m.get("matched_gt", 0))
            d["fp"] += m.get("false_positives", 0)
            d["fn"] += m.get("false_negatives", 0)
            d["matched_detections"] += m.get("matched_detections", 0)
    table: Dict[str, Dict[str, Any]] = {}
    for cls in sorted(agg):
        d = agg[cls]
        p = d["tp"] / d["det"] if d["det"] else 0.0
        r = d["tp"] / d["gt"] if d["gt"] else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        row = {
            "ground_truth": d["gt"], "detected": d["det"],
            "true_positives": d["tp"], "false_positives": d["fp"], "false_negatives": d["fn"],
            "precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4),
        }
        if cls in AREA:
            # Extent-matched classes: recall is the score; report matched
            # detections and fragmentation (pieces per GT object) alongside.
            row["matched_detections"] = d["matched_detections"]
            row["fragmentation"] = (
                round(d["matched_detections"] / d["tp"], 2) if d["tp"] else None
            )
        table[cls] = row
    return _objective(table), table


def describe(settings: ProcessingSettings, changed: Dict[str, Any]) -> str:
    if not changed:
        return "shipped defaults"
    return ", ".join(f"{k}={settings.__dict__[k]}" for k in sorted(changed))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 42],
                        help="scene seeds used for tuning")
    parser.add_argument("--validate-seeds", type=int, nargs="+", default=[7, 5, 99],
                        help="held-out scene seeds used for the final re-measure")
    parser.add_argument("--report", type=Path, default=Path("out/accuracy/training_report.json"))
    parser.add_argument("--dry-run", action="store_true",
                        help="measure the shipped config only; skip the descent")
    args = parser.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="train_thresholds_"))
    try:
        settings = ProcessingSettings()
        baseline_obj, baseline = measure(tmp / "baseline", args.seeds, settings)
        print(f"baseline objective (train seeds {args.seeds}): {baseline_obj:.4f}")
        best_settings = deepcopy(settings)
        best_obj = baseline_obj
        history: List[Dict[str, Any]] = [{
            "stage": "baseline", "objective": round(baseline_obj, 4),
            "settings": "shipped defaults", "per_class": baseline,
        }]

        if not args.dry_run:
            for knob, candidates in KNOBS.items():
                current = float(getattr(best_settings, knob))
                best_for_knob: Tuple[float, Any] = (best_obj, None)
                for value in candidates:
                    if abs(value - current) < 1e-9:
                        continue
                    trial = deepcopy(best_settings)
                    setattr(trial, knob, type(getattr(best_settings, knob))(value))
                    obj, per_class = measure(tmp / f"{knob}={value}", args.seeds, trial)
                    print(f"  {knob}={value}: objective {obj:.4f}"
                          + ("  <-- better" if obj > best_obj + 1e-6 else ""))
                    if obj > best_for_knob[0] + 1e-6:
                        best_for_knob = (obj, value)
                if best_for_knob[1] is not None:
                    setattr(best_settings, knob, type(getattr(best_settings, knob))(best_for_knob[1]))
                    best_obj = best_for_knob[0]
                    _, per_class = measure(tmp / f"{knob}=best", args.seeds, best_settings)
                    history.append({
                        "stage": f"after {knob}", "objective": round(best_obj, 4),
                        "settings": describe(best_settings, {}),
                        "per_class": per_class,
                    })
                    print(f"  -> keep {knob}={getattr(best_settings, knob)} "
                          f"(objective {best_obj:.4f})")

        # Final: re-measure the best config on the held-out seeds.
        validate_obj, validate = measure(tmp / "validate", args.validate_seeds, best_settings)
        _, validate_baseline = measure(tmp / "validate_base", args.validate_seeds, ProcessingSettings())

        report = {
            "method": "coordinate descent on Quick Simulation ground truth",
            "objective_definition": {
                "compact_classes_instance_f1": sorted(COMPACT),
                "area_classes_recall": sorted(AREA),
                "note": "per-instance precision on area classes is dominated by "
                        "per-tile fragmentation; see docs/VALIDATION.md",
            },
            "train_seeds": args.seeds,
            "validate_seeds": args.validate_seeds,
            "baseline_objective": round(baseline_obj, 4),
            "tuned_objective": round(best_obj, 4),
            "tuned_settings_delta": {
                k: float(getattr(best_settings, k))
                for k in KNOBS if getattr(best_settings, k) != getattr(ProcessingSettings(), k)
            },
            "validate_baseline_objective": round(_objective(validate_baseline), 4),
            "validate_tuned_objective": round(validate_obj, 4),
            "history": history,
            "train_per_class_tuned": {cls: m for cls, m in measure(tmp / "train_final", args.seeds, best_settings)[1].items()},
            "validate_per_class": validate,
            "validate_per_class_baseline": validate_baseline,
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nreport: {args.report}")
        print(f"tuned objective {best_obj:.4f} (baseline {baseline_obj:.4f})")
        print(f"held-out validation: tuned {validate_obj:.4f} "
              f"vs shipped {_objective(validate_baseline):.4f}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
