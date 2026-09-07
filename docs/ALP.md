# ALP: Observation → Interpretation → Recommended Action

The **Action Logic Package** turns each detected asset from "a blob of points"
into a decision-ready record: an **observation** (what the pipeline measured),
an **interpretation** (what that measurement means, honestly scoped), and a
**recommended action** (what an infrastructure owner should do next), plus a
**human-review flag** for anything uncertain.

The ALP is deliberately grounded. Every interpretation cites a measured signal
(intensity, eigen-geometry, support, confidence, QC flags). An unmeasured cause
is never invented: a faded marking says *"measured retroreflectivity is low,
verify before restriping"*, never *"sun damage"*.

## 1. Trigger families (from what the scanner actually measures)

The trigger wording is mined from what crews and inspectors actually flag, then
mapped to signals this pipeline measures — there is no work-order text corpus in
a LiDAR challenge, so families are signal-defined, not phrase-matched:

| Family | Measured observation | Interpretation | Recommended action |
| --- | --- | --- | --- |
| Faded / low-retro marking | marking `intensity_stats.mean` < 12,000 (of 65,535) | marking retroreflectivity low | Restripe/refresh verification (§ MUTCD) |
| Faded / low-retro sign | sign panel `intensity_stats.mean` < 12,000 | panel retroreflectivity low | Verify retroreflectivity before replacement |
| Leaning pole | eigen-verticality → lean ≥ 8° from vertical | possible pole lean | Site stability/lean inspection before trusting the span |
| Sagging conductor | span vertical extent / length > 12% | possible clearance violation | Sag/clearance survey |
| Pavement patch with clutter/deviation | travelled-surface patch vertical spread > 0.4 m | patch includes curb, clutter or local deviation (sub-decimeter defects are *not* claimed at this resolution) | Field-inspect travelled surface |
| Fragmented rail / barrier | detected length below class minimum | gap or split detection | Verify continuity + end treatments |
| Sparse cabinet | point density < 5 pts/m² | incomplete capture | Verify enclosure at site |
| Short rumble strip | strip length < 2 m | fragment | Verify extent at site |
| Structural doubt (any class) | `LIKELY_DUPLICATE` / `INVALID_GEOMETRY` / `UNUSUAL_DIMENSIONS` / `BELOW_MIN_POINTS` | instance identity uncertain | Human review before downstream use |
| Low confidence (any class) | `confidence` < 0.80 | true-positive likelihood below the measured bar | Human review |

## 2. What confidence measures

`asset.confidence` is **the estimated likelihood that the detection is a true
positive** — that a real object of the reported class exists at the reported
location. It is a weighted blend of measurable factors (model prior / geometry
fit / point support / spatial context / class consistency), renormalized over
the factors actually present, and every factor value ships with the asset
(`confidence_factors`) so the number is auditable.

Confidence is **not** a condition rating. Condition is derived separately by the
ALP from class-specific signals; a perfectly confident faded marking is still
"POOR — restripe".

## 3. Calibration (measured, not intuition)

`scripts/calibrate_confidence.py` sweeps candidate confidence thresholds over a
scene with known ground truth (Quick Simulation exports the exact placed
objects), and reports measured precision / recall / F1 at every cut. Instance
precision is only meaningful for **compact, discrete objects** (poles, signs,
cabinets, rumble strips); area/long classes (pavement, markings, guardrails,
barriers) are extracted per tile component and are scored by **recall only**
(was any part of the real object detected). Overhead conductors are excluded
from the compact sweep: their confidence uses the geometry-v2-heuristic band
(cross-section + linearity), not the class-consistent factor set, so a
confidence cut on them would measure the sweep rather than the review policy.

Measured calibration run (400 m simulated corridor, seed 23, latest detectors;
tile-duplicate rows excluded, matching the train/eval methodology):

| threshold | kept assets | compact precision | compact recall | compact F1 |
| --- | --- | --- | --- | --- |
| 0.30–0.75 | 101 (99 at 0.70, 91 at 0.75) | 1.000 | 1.000 | 1.000 |
| **0.80** | **40** | **1.000** | **1.000** | **1.000** |
| 0.85 | 18 | 1.000 | 0.417 | 0.588 |
| 0.90 | 14 | 1.000 | 0.333 | 0.500 |

**Chosen default: review threshold = 0.80** (`review_confidence_threshold` in
`configs/processing.yaml`). The curve is flat at 1.000 from 0.30 up, so the
strictest cut that still keeps full compact recall (0.80, before recall drops
at 0.85) is the review edge: auto-accepted compact objects at that cut are
clean — measured precision 1.000 with recall 1.000 (the 61 detections below
0.80, incl. tile-duplicate and fragment strays, go to human review instead of
the inventory). Re-run the sweep whenever detectors change — the threshold must
follow the measurements, not the other way round.

Condition bands are the same calibrated edges: **GOOD ≥ 0.80**, **FAIR
0.60–0.80**, **POOR < 0.60**, **REVIEW** when structurally flagged.

## 4. Review routing

`review_required` is set when:

1. `confidence` < `review_confidence_threshold` (0.80 default), **or**
2. any structural QC flag is present (`LIKELY_DUPLICATE`, `INVALID_GEOMETRY`,
   `UNUSUAL_DIMENSIONS`, `BELOW_MIN_POINTS`).

Routing is a rule, not a model output: deterministic, explainable, and cheap to
re-audit. `reports/qc_report.json` lists exactly which assets need review and
why. Exports carry `condition`, `recommended_action`, `review_required`, and
`assessment_reasoning` per asset (JSON, CSV, GeoJSON).

## 5. The 50-insight audit

The brief asks for manual auditing of at least 50 insights before trusting
numbers. Procedure (documented so it is repeatable for the judges):

1. Run `python -m infra_inventory simulate --output output/sim --length 400 --seed 23`.
2. Load `output/sim/.../pipeline/assets.json` and
   `simulation_meta.ground_truth` from the project descriptor.
3. Audit the first 50 assets against the ground-truth table + the 3D viewer:
   - is the class right? is the location within the match radius?
   - does the ALP interpretation match the measurement it cites?
   - would you override condition/recommendation? note why.
4. Enter results in `reports/audit.csv` (`asset_id`, `class`, `audit_pass`,
   `audit_note`, `auditor`). The audit itself becomes ground truth for the next
   calibration pass.

On real data (no ground truth), audit against the source imagery/field records
the same way, using the reviewer report (`reports/qc_report.json` +
`summary.md` "Sent to human review" list) as the sampling frame.

## 6. Integrity rules

- Interpretations never claim an unmeasured cause; reasoning strings always name
  the measured signal(s) that fired.
- Confidence factors ship with every asset; nothing is post-hoc "AI confidence"
  with no explanation.
- Thresholds come from the calibration sweep output, recorded in this document
  and re-runnable via `scripts/calibrate_confidence.py`.
- Simulated data is labeled SIMULATED end-to-end and is never presented as
  competition results.
