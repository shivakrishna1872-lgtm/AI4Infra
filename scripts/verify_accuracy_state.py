#!/usr/bin/env python3
"""Offline accuracy + confidence verification harness.

Runs Quick Simulation scenes through the real pipeline, compares per-class
detection counts against ground truth, and re-computes the overall confidence
report twice:

* ``real``    — the scene's actual metadata (small sim, ~106k points)
* ``scenario``— production-scale metadata (112.8M points / 688 tiles) fed to
  the same confidence engine with the same assets, CRS and warnings.

Usage:
    python scripts/verify_accuracy_state.py            # seeds 9001-9003
    python scripts/verify_accuracy_state.py --seeds 7 23
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from infra_inventory.confidence import compute_confidence_report  # noqa: E402

CLASSES = [
    "utility_pole", "traffic_sign", "guardrail", "safety_barrier",
    "utility_cabinet", "pavement", "pavement_marking",
    "overhead_conductor", "rumble_strip",
]


def run_scene(seed: int, length_m: float = 400.0):
    from infra_inventory.simulation import run_quick_simulation

    with tempfile.TemporaryDirectory(prefix="qs_") as td:
        project = run_quick_simulation(Path(td), length_m=length_m, seed=seed)
        out_dir = Path(project["output_dir"])
        run_json = out_dir / "pipeline" / "run.json"
        assets_json = out_dir / "pipeline" / "assets.json"
        if not run_json.exists():
            for sub in out_dir.iterdir():
                if sub.is_dir() and (sub / "pipeline" / "run.json").exists():
                    run_json = sub / "pipeline" / "run.json"
                    assets_json = run_json.parent / "assets.json"
                    break
        if not run_json.exists():
            raise SystemExit(f"no run.json under {out_dir}")
        summary = json.loads(run_json.read_text())
        assets = json.loads(assets_json.read_text())
        gt = project["simulation_meta"]["ground_truth"]
    return summary, assets, gt


def score(point_count, bounds, tile_count, crs, point_format, warnings, assets, has_intensity):
    return compute_confidence_report(
        point_count=point_count,
        bounds=bounds,
        tile_count=tile_count,
        crs=crs,
        point_format=point_format,
        warnings=warnings,
        assets=assets,
        has_intensity=has_intensity,
        tile_size_m=40.0,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[9001, 9002, 9003])
    args = parser.parse_args()

    failures = 0
    for seed in args.seeds:
        summary, assets, gt = run_scene(seed)
        det = Counter(a["class"] for a in assets)
        gtc = Counter(g["class"] for g in gt)
        has_intensity = bool(summary.get("has_intensity", False))

        real = score(
            summary["point_count"], summary["bounds"], summary["tile_count"],
            summary["crs"], summary["point_format"], summary["warnings"],
            assets, has_intensity,
        )
        scenario = score(
            112_800_000, summary["bounds"], 688,
            summary["crs"], summary["point_format"], summary["warnings"],
            assets, has_intensity,
        )

        print(f"\nSEED {seed}")
        print(f"  real     : {real['overall_percent']:>3d}% ({real['grade']})  "
              f"density={real['components']['density_coverage']['percent']:.0f}% "
              f"crs={real['components']['crs']['percent']:.0f}% "
              f"intensity={real['components']['intensity_classification']['percent']:.0f}% "
              f"geom={real['components']['geometry_fit']['percent']:.0f}%")
        print(f"  scenario : {scenario['overall_percent']:>3d}% ({scenario['grade']})  "
              f"density={scenario['components']['density_coverage']['percent']:.0f}% "
              f"crs={scenario['components']['crs']['percent']:.0f}% "
              f"intensity={scenario['components']['intensity_classification']['percent']:.0f}% "
              f"geom={scenario['components']['geometry_fit']['percent']:.0f}%")
        for cls in CLASSES:
            d, g = det.get(cls, 0), gtc.get(cls, 0)
            marker = "" if g == 0 or d > 0 else "  <-- MISS"
            print(f"    {cls:22s} det={d:4d} gt={g:4d}{marker}")
            if g > 0 and d == 0:
                failures += 1

    print()
    if failures:
        print(f"RESULT: {failures} ground-truth classes with zero detections — NOT OK")
        return 1
    print("RESULT: every ground-truth class has at least one detection on all seeds — OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
