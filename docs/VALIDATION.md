# Validation: extraction accuracy

This document defines how extraction accuracy is measured and how to reproduce
the numbers. The framework is code, not a promise: `infra_inventory.evaluate`
implements the matching and metrics, and the test suite runs it against the
synthetic scene's known ground truth on every CI pass.

---

## 1. Metrics

For a given asset class, matching detections to ground-truth objects yields:

- **TP** — ground-truth object with a matched detection.
- **FP** — detection with no ground-truth object (within match radius).
- **FN** — ground-truth object with no detection.

Then:

```
precision = TP / (TP + FP)          # of what we found, how much is real
recall    = TP / (TP + FN)          # of what is real, how much we found
F1        = 2·P·R / (P + R)
```

Positional error is the centroid distance of matched pairs:

```
RMSE = sqrt( mean( (dx² + dy² + dz²) ) )      # matched pairs only
```

## 2. Ground truth sources

### 2a. Simulation ground truth (automatic)

`python -m infra_inventory simulate` (or the web UI's Quick Simulation) places a
fully known scene — poles, conductors, signs, guardrails, barrier, pavement and
markings — and exports the object table in the project's
`simulation_meta.ground_truth`:

```json
{
  "gt_id": "GT-POL-001",
  "class": "utility_pole",
  "center": [60.0, -7.5, 5.4],
  "dimensions_m": [0.32, 0.32, 10.8]
}
```

Because the scene is synthetic, the table is exact — this is an honest
end-to-end accuracy check of the geometry detectors, with no manual labeling.

```python
import json
from pathlib import Path
from infra_inventory.evaluate import evaluate_against_ground_truth

assets = json.loads(Path("output/pipeline/assets.json").read_text())
ground_truth = json.loads(Path("project.json").read_text())["simulation_meta"]["ground_truth"]

result = evaluate_against_ground_truth(
    assets,
    ground_truth,
    match_distance_m={               # per-class match radii
        "utility_pole": 3.0,
        "traffic_sign": 3.0,
        "guardrail": 20.0,
        "safety_barrier": 20.0,
        "pavement": 25.0,
        "pavement_marking": 20.0,
    },
)
print(result["precision"], result["recall"], result["f1"], result["positional_error_rmse_m"])
```

### 2b. Manual ground truth for real data

For real LiDAR, build the same table by hand or with a point-cloud labeler
(`gt_id`, `class` matching the taxonomy, `center` in the **source CRS**), then
call the same function. This is the ground-truth comparison framework for the
competition dataset.

## 3. Matching rules

- Class-aware: a detection can only match a ground-truth object of the same class.
- Greedy nearest: each ground-truth object claims its closest still-unclaimed
  detection inside the class match radius.
- Radii are per class because object granularity differs: poles and signs are
  matched at a few meters; guardrails, barriers, pavement and markings are
  extracted **per tile component** (a 110 m rail becomes several 20–40 m pieces),
  so object-level ground truth is matched with class-appropriate radii and
  positional error is reported from those matches.

## 4. What the numbers mean (and don't)

- Recall is the headline metric for coverage: it answers "did we find the real
  objects?".
- Instance-level precision is **conservative** for long/area classes: per-tile
  splitting inflates FP counts (one real rail → several pieces, only one counts
  as TP). Use precision as a lower bound, and see
  [ENHANCEMENT.md](ENHANCEMENT.md) for the merge-across-tiles roadmap.
- `positional_error_rmse_m` reflects centroid agreement of matched objects;
  sub-meter values mean the extracted geometry is in the right place.

## 5. Automated checks (CI)

`tests/test_simulation.py::test_quick_simulation_scores_against_ground_truth`
runs the evaluator over a 400 m simulation and asserts: ≥ 12 matched objects,
recall ≥ 0.6, pole recall ≥ 0.5, sign precision ≥ 0.9, and a finite positional
RMSE. These thresholds regress the detectors — if a geometry change lowers
recall, the test fails on purpose.

## 6. Known gaps surfaced by validation

- `overhead_conductor`: the heuristic detector currently does not fire on the
  simulated sagged spans (0 detections). This class is the top priority for the
  learned PTv3 backend (see [LIMITATIONS.md](LIMITATIONS.md)).
- Marking pieces (per-dash components) outnumber object-level marking ground
  truth; recall is meaningful, precision is conservative.