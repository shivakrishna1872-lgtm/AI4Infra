#!/usr/bin/env python3
"""Train the learned component classifier on labeled ground truth.

Trains the CPU component classifier (``infra_inventory/learned.py``) on
labeled connected-component features extracted from Quick Simulation scenes —
the project's ground truth — and evaluates it on held-out seeds. The learned
weights are saved to ``configs/learned_classifier.json`` and every later run
picks them up automatically (``--no-learned-prior`` opts out).

How it works
------------
1. For each training seed, run the Quick Simulation scene through the *real*
   pipeline with ``--collect-training``: every candidate component the
   detectors build is described by the same 14 measured features the
   classifier consumes, labeled with the detector's assigned class.
2. Relabel every example against the scene's exact ground-truth table
   (``simulation_meta.ground_truth``): a component is the class of the nearest
   ground-truth object within a class-appropriate radius (the same radii the
   evaluation uses), otherwise it is ``background``. This corrects detector
   mistakes instead of memorizing them.
3. Train the softmax classifier on train seeds; report accuracy / macro-F1 /
   per-class precision-recall on held-out validation seeds it never saw.

Usage::

    python scripts/train_classifier.py                       # train + validate + save
    python scripts/train_classifier.py --seeds 11 23 42      # custom train seeds
    python scripts/train_classifier.py --output my_model.json
    python scripts/train_classifier.py --dry-run             # collect + report only

The result is honest by construction: the reported accuracy is measured on
scenes whose placed objects are known exactly, and the saved model records the
measured validation metrics alongside the weights.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from infra_inventory.evaluate import evaluate_against_ground_truth  # noqa: E402
from infra_inventory.learned import (  # noqa: E402
    FEATURE_NAMES,
    LEARNED_CLASSES,
    FeatureNormalizer,
    component_features,
    train_softmax,
)
from infra_inventory.models import ProcessingSettings  # noqa: E402
from infra_inventory.pipeline import process_las  # noqa: E402
from infra_inventory.simulation import run_quick_simulation  # noqa: E402

#: Per-class match radii (same table as scripts/train_thresholds.py).
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

#: Classes scored per-object (each detection is one asset: poles, signs, ...).
COMPACT_CLASSES = ("utility_pole", "overhead_conductor", "utility_cabinet",
                   "traffic_sign", "rumble_strip")
#: Corridor classes scored by ground-truth coverage + fragmentation (a 400 m
#: rail is one GT object; per-object precision is meaningless there).
AREA_CLASSES = ("pavement", "pavement_marking", "guardrail", "safety_barrier")


def _class_index(name: str) -> int:
    return LEARNED_CLASSES.index(name) if name in LEARNED_CLASSES else LEARNED_CLASSES.index("background")


def collect_seed(root: Path, seed: int, length_m: float = 400.0) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Run one Quick Simulation scene; return (X, y, evaluation) for its components."""
    settings = ProcessingSettings()
    settings.collect_training = True
    project = run_quick_simulation(root / f"seed{seed}", length_m=length_m, seed=seed,
                                   settings=settings)
    out_dir = Path(project["output_dir"])
    assets = json.loads((out_dir / "pipeline" / "assets.json").read_text(encoding="utf-8"))
    ground_truth = project["simulation_meta"]["ground_truth"]

    features_by_tile: Dict[str, List[dict]] = {}
    # process_las writes under <sim>/pipeline/, so training JSONLs land there
    # (output_dir on the settings object is the pipeline output directory).
    training_dir = out_dir / "pipeline" / "training"
    if not training_dir.is_dir():
        training_dir = out_dir / "training"
    if training_dir.is_dir():
        for path in sorted(training_dir.glob("components_*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    record = json.loads(line)
                    features_by_tile.setdefault(record["tile"], []).append(record)

    # Relabel every collected component against exact ground truth. Matching is
    # extent-aware: each GT object is treated as an axis-aligned box inflated by
    # its class match radius, so a tile fragment 80 m along a 110 m guardrail
    # still matches that guardrail instead of being poisoned to ``background``
    # (a center-only match fails every corridor-scale object). When several GT
    # boxes contain a component (e.g. markings inside the pavement slab), the
    # detector's assigned class wins; otherwise the smallest box distance, then
    # the smallest volume (more specific class).
    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    for tile, records in features_by_tile.items():
        for record in records:
            cx, cy, cz = record["extras"]["center_xyz"]
            assigned = str(record.get("class", ""))
            candidates: List[Tuple[float, float, str]] = []  # (distance, volume, class)
            for gt in ground_truth:
                gx, gy, gz = (float(v) for v in gt["center"])
                dims = gt.get("dimensions_m") or [0.0, 0.0, 0.0]
                hx, hy, hz = (max(float(d) / 2.0, 0.0) for d in dims[:3])
                radius = MATCH_RADII.get(str(gt.get("class")), 3.0)
                ox = max(0.0, abs(cx - gx) - hx)
                oy = max(0.0, abs(cy - gy) - hy)
                oz = max(0.0, abs(cz - gz) - hz)
                dist = math.sqrt(ox * ox + oy * oy + oz * oz)
                if dist <= radius:
                    volume = max(hx * 2.0, 1e-3) * max(hy * 2.0, 1e-3) * max(hz * 2.0, 1e-3)
                    candidates.append((dist, volume, str(gt.get("class"))))
            if not candidates:
                label = "background"
            else:
                same = [c for c in candidates if c[2] == assigned]
                if same:
                    label = assigned
                else:
                    candidates.sort(key=lambda c: (c[0], c[1]))
                    label = candidates[0][2]
            X_list.append(np.asarray(record["features"], dtype=np.float64))
            y_list.append(_class_index(label))
    accepted = [a for a in assets if "LIKELY_DUPLICATE" not in a.get("qc_flags", [])]
    # Compact classes get honest per-object P/R/F1 (a pole is one asset). The
    # corridor classes (pavement, markings, guardrails, barriers) are one GT
    # object per span/deck; after the cross-tile merges they are reported by
    # ground-truth recall plus fragmentation (detections per matched GT) — the
    # same compact-vs-area split the accuracy report and train_gemini use.
    compact = evaluate_against_ground_truth(
        [a for a in accepted if a["class"] in COMPACT_CLASSES],
        [g for g in ground_truth if g["class"] in COMPACT_CLASSES],
        match_distance_m={c: MATCH_RADII[c] for c in COMPACT_CLASSES},
    )
    gt_area = [g for g in ground_truth if g["class"] in AREA_CLASSES]
    det_area = [a for a in accepted if a["class"] in AREA_CLASSES]
    if gt_area:
        area = evaluate_against_ground_truth(
            det_area, gt_area,
            match_distance_m={c: MATCH_RADII[c] for c in AREA_CLASSES},
            extent_match_classes=AREA_CLASSES,
        )
        matched = sum(
            area["per_class"].get(c, {}).get("matched_detections", 0) for c in AREA_CLASSES
        )
        fragmentation = (matched / area["matched_count"]) if area["matched_count"] else None
    else:
        area = {"recall": None, "matched_count": 0}
        fragmentation = None
    evaluation = {
        "compact_f1": compact["f1"],
        "compact_precision": compact["precision"],
        "compact_recall": compact["recall"],
        "area_recall": area["recall"],
        "fragmentation": round(fragmentation, 3) if fragmentation is not None else None,
        "per_class": {
            **compact["per_class"],
            "area_classes": {c: {"ground_truth": area["per_class"].get(c, {}).get("ground_truth", 0),
                                  "detected": area["per_class"].get(c, {}).get("detected", 0),
                                  "recall": area["per_class"].get(c, {}).get("recall", 0.0),
                                  "fragmentation": area["per_class"].get(c, {}).get("fragmentation")}
                              for c in AREA_CLASSES},
        },
    }
    return (
        np.vstack(X_list) if X_list else np.zeros((0, len(FEATURE_NAMES))),
        np.asarray(y_list, dtype=np.int64),
        evaluation,
    )


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    per_class: Dict[str, Dict[str, float]] = {}
    for index, name in enumerate(LEARNED_CLASSES):
        tp = int(np.sum((y_true == index) & (y_pred == index)))
        fp = int(np.sum((y_true != index) & (y_pred == index)))
        fn = int(np.sum((y_true == index) & (y_pred != index)))
        support = int(np.sum(y_true == index))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {
            "precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4), "support": support,
        }
    accuracy = float(np.mean(y_true == y_pred))
    f1s = [m["f1"] for name, m in per_class.items() if name != "background" and m["support"]]
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0
    return {"accuracy": round(accuracy, 4), "macro_f1_excluding_background": round(macro_f1, 4),
            "per_class": per_class}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 42],
                        help="Quick Simulation seeds used for training")
    parser.add_argument("--validate-seeds", type=int, nargs="+", default=[7, 5],
                        help="Held-out seeds used for the accuracy report")
    parser.add_argument("--length", type=float, default=400.0, help="Scene length (m)")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--output", default="configs/learned_classifier.json")
    parser.add_argument("--dry-run", action="store_true", help="Collect + evaluate only; do not save")
    args = parser.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="train_classifier_"))
    try:
        X_train_parts, y_train_parts = [], []
        pipeline_reports = []
        for seed in args.seeds:
            X, y, evaluation = collect_seed(tmp, seed, length_m=args.length)
            if len(X):
                X_train_parts.append(X)
                y_train_parts.append(y)
            pipeline_reports.append({
                "seed": seed,
                "compact_f1": evaluation["compact_f1"],
                "compact_recall": evaluation["compact_recall"],
                "area_recall": evaluation["area_recall"],
                "fragmentation": evaluation["fragmentation"],
            })
            print(f"seed {seed}: {len(X)} components collected", flush=True)
        if not X_train_parts:
            print("No training examples collected — nothing to train.", file=sys.stderr)
            return 1
        X_train = np.vstack(X_train_parts)
        y_train = np.concatenate(y_train_parts)

        # Validation scenes: same collection, seeds the model never trains on.
        X_val_parts, y_val_parts = [], []
        for seed in args.validate_seeds:
            X, y, _ = collect_seed(tmp, seed, length_m=args.length)
            if len(X):
                X_val_parts.append(X)
                y_val_parts.append(y)
        X_val = np.vstack(X_val_parts) if X_val_parts else np.zeros((0, len(FEATURE_NAMES)))
        y_val = np.concatenate(y_val_parts) if y_val_parts else np.zeros(0, dtype=np.int64)

        normalizer = FeatureNormalizer.fit(X_train)
        num_classes = len(LEARNED_CLASSES)
        W, b = train_softmax(normalizer.transform(X_train), y_train, num_classes,
                             epochs=args.epochs)

        def _predict(X: np.ndarray) -> np.ndarray:
            if not len(X):
                return np.zeros(0, dtype=np.int64)
            probs = _softmax(normalizer.transform(X) @ W + b)
            return np.argmax(probs, axis=1)

        train_metrics = _metrics(y_train, _predict(X_train))
        val_metrics = _metrics(y_val, _predict(X_val)) if len(X_val) else {"accuracy": None, "per_class": {}}

        print(f"\ntrain: n={len(y_train)} accuracy={train_metrics['accuracy']} "
              f"macro-F1(excl. background)={train_metrics['macro_f1_excluding_background']}")
        if len(X_val):
            print(f"held-out: n={len(y_val)} accuracy={val_metrics['accuracy']} "
                  f"macro-F1(excl. background)={val_metrics['macro_f1_excluding_background']}")
        for name in LEARNED_CLASSES:
            m = val_metrics["per_class"].get(name)
            if m and m["support"]:
                print(f"  {name:>20}: P={m['precision']:.2f} R={m['recall']:.2f} F1={m['f1']:.2f} (n={m['support']})")
        # End-to-end pipeline quality per train seed (compact per-object F1 +
        # area/linear coverage + fragmentation after the cross-tile merges).
        f1s = [r["compact_f1"] for r in pipeline_reports]
        recalls = [r["area_recall"] for r in pipeline_reports if r["area_recall"] is not None]
        frags = [r["fragmentation"] for r in pipeline_reports if r["fragmentation"] is not None]
        if f1s:
            print(f"pipeline: compact F1={sum(f1s) / len(f1s):.3f} (per-object, poles/signs/cabinets/conductors/rumble)")
        if recalls:
            print(f"pipeline: area/linear recall={sum(recalls) / len(recalls):.3f} "
                  f"(pavement/markings/guardrails/barriers), fragmentation="
                  f"{sum(frags) / len(frags):.2f} detections per GT object")

        if args.dry_run:
            print("\ndry run: model NOT saved")
            return 0

        from infra_inventory.learned import ComponentClassifier
        model = ComponentClassifier(
            weights=W, bias=b, normalizer=normalizer, version="learned-v1",
            training={
                "method": "softmax regression on Quick Simulation ground truth",
                "train_seeds": args.seeds,
                "validate_seeds": args.validate_seeds,
                "epochs": args.epochs,
                "train_examples": int(len(y_train)),
                "val_examples": int(len(y_val)),
                "train_accuracy": train_metrics["accuracy"],
                "val_accuracy": val_metrics.get("accuracy"),
                "val_macro_f1": val_metrics.get("macro_f1_excluding_background"),
                "pipeline_reports": pipeline_reports,
            },
        )
        out_path = Path(args.output)
        model.save(out_path)
        print(f"\nmodel saved: {out_path}")
        print("Every pipeline run now uses it automatically (disable with --no-learned-prior).")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _softmax(z: np.ndarray) -> np.ndarray:
    shifted = z - z.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


if __name__ == "__main__":
    raise SystemExit(main())
