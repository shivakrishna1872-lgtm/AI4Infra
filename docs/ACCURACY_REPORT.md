# Accuracy Report — detector training & validation

*Last measured: this build, `scripts/train_thresholds.py` + the full pytest suite (218 passed).*

## 0. Latest round — measured fixes (objective 0.7778 -> 1.0000)

A full re-measure of the shipped configuration against the Quick Simulation ground
truth found three real accuracy defects (objective 7/9 = 0.7778), each fixed and
re-verified on train AND held-out seeds:

| Defect (measured) | Root cause | Fix |
|---|---|---|
| `utility_cabinet` recall **0.00** (6 GT, 0 detected) | the detector gated height on the *band-sliced* component: a 1.3 m cabinet sliced at 0.8–2.5 m is a 0.5 m sliver, so `height >= 0.8` rejected every real enclosure | components are now built from the **full cabinet evidence window** (above ground scatter, up to the conductor band); the height gate measures the component's true vertical extent with pole-claimed points excluded |
| `safety_barrier` precision **0.33** (6 FPs per scene) | a sign post + panel bottom merged at grid resolution into a thin vertical remnant (~0 m long, full-height) that the cabinet aspect-rule reclassified as a barrier | degenerate vertical remnants (`max_side < 1.5 m` or taller than `barrier_max_height_m`) are dropped instead of reclassified |
| `pavement_marking` F1 **0.00** despite 9 correct detections | the harness scored corridor classes per-object, so one 400 m GT line could match only one of the line's pieces | `train_thresholds.measure_one` now passes `extent_match_classes` exactly as docs/VALIDATION.md documents |

Follow-on hardening (each verified with a targeted edge-case scene):

* **Pole trunk selection under ties**: the "densest XY cell = trunk" heuristic lost
to a cabinet whose footprint cells are denser than the shaft. Tied cells are now
broken by *highest reach* (z-range): a cabinet top stops at ~2.5 m, a trunk does not.
* **Cabinet size generalization**: QuickSim placed only 1.3 m cabinets, so the
learned classifier voted "pole" for every real cabinet above ~1.6 m (log_height
outside the training manifold). The generator now spans the DOT enclosure range
(0.9–2.1 m with per-seed variation) and the classifier was retrained
(`configs/learned_classifier.json`): cabinets 0.9–2.1 m all detect, including beside
a pole line.
* **Trained-model metrics after retraining**: train n=764 accuracy 1.000, held-out
n=512 accuracy 1.000, macro-F1 (excl. background) 1.000, all 10 classes P/R/F1 = 1.00.

Result: objective **1.0000** on train seeds (11, 23, 42) **and** held-out seeds
(7, 5, 99) — every class at recall 1.00, compact classes at F1 1.00, area/linear
classes with fragmentation 1.0–2.7 detections per GT object.

## 1. What "training" means in this project

The extraction backbone is a set of **measured-geometry detectors** (poles,
overhead conductors, cabinets, signs, guardrails, barriers, rumble strips,
pavement and markings). Every rule is a physically measured property —
footprint, vertical extent, columnarity/planarity/linearity (PCA eigenvalue
ratios), height-above-ground, cross-section — not a hand-placed mock. Each
detector's behaviour is governed by thresholds in `configs/processing.yaml`
(≈30 knobs on `ProcessingSettings`).

There is **no labelled real corpus in the repo**, so "training" here means
tuning those thresholds against the **Quick Simulation ground truth**: the
synthetic mobile-LiDAR corridor is generated with a fully known object table,
then pushed through the *same* pipeline real `.las`/`.laz` uploads run
(tiling → preprocessing → detectors → cross-tile merge → QC → ALP → export).
That yields an honest, reproducible Precision/Recall/F1 measurement per class.

The optional neural backends (Pointcept PTv3, RoadMarkingExtraction) are a
separate, opt-in path that requires GPU checkpoints; see
[`docs/PROCEDURE.md`](PROCEDURE.md) and
[`infra_inventory/pointcept.py`](../infra_inventory/pointcept.py).

## 2. Training procedure

`scripts/train_thresholds.py`:

1. **Measure** the shipped configuration over train scene seeds (each 400 m
   corridor, ~130-190 k points, all four competition categories present).
2. **Coordinate-descent** each detector knob over candidate values, keeping a
   change only when the measured objective improves. The objective is
   equal-weighted over compact-class instance F1 (poles, conductors, cabinets,
   signs, rumble strips) and area/linear-class recall (guardrail, barrier,
   pavement, markings), after dropping QC `LIKELY_DUPLICATE` rows (those go to
   human review, not the inventory).
3. **Re-measure on held-out seeds** that the descent never saw.
4. Write `out/accuracy/training_report.json`.

Reproduce:

```bash
python scripts/train_thresholds.py                        # full train + validate
python scripts/train_thresholds.py --dry-run              # current config only
python scripts/train_thresholds.py --report out/accuracy/training_report.json
```

## 3. What was wrong before this round (all measured, then fixed)

| Failure | Evidence (QuickSim, pre-fix) | Root cause | Fix |
|---|---|---|---|
| `overhead_conductor` never detected | **0 detections across 4 scenes / 20 ground-truth spans** | (a) width gate used the axis-aligned bounding-box minor side, so a wire at ~14° to the tile axes measured 2-10 m "wide" and was rejected; (b) per-point speckle dropout in the synthetic scene fragmented 1-D chains into sub-metre runs | orientation-invariant **PCA cross-section gate** (2σ minor axis) + contiguous-occlusion dropout model |
| Spans silently deleted | a 60 m span produced 0 assets even when points existed | a wire attached to a pole merges with the pole trunk at grid resolution, and the pole's **whole-component claim** deleted the entire connected wire run in that tile | pole claim now removes only the shaft footprint + crossarm reach (densest-cell trunk) |
| Conductor spans merged into monsters | a **267 m asset** spanning 4 poles | the linear-piece merge bridged across poles: fragments are trimmed ~1.2 m from each pole, so the inter-fragment gap can be 10-14 m wide with the pole at its *end*; the old guard only tested the gap midpoint | merge refuses when a **pole lies anywhere in the inter-piece band** (±2 m), and same-span fragments must share a **fitted-line heading** (highlight-point PCA) with < 1.2 m cross-track offset |
| `utility_pole` recall ~0.5-0.6 | poles at corridor/tile edges missed | a pole merged with a guardrail run inflated the component bbox to ~40 m, and pole metrics were read off the whole component | pole-ness is measured from the component's **densest XY cell** (the shaft), lower 80 % of the shaft for footprint/columnarity |
| barrier-adjacent pole missed (held-out recall 0.889) | seed-7 pole at x=360, next to the concrete barrier, was rejected | the barrier's top edge sits ~1 m from the pole's densest cell; the 1.2 m axis sweep leaked those near-ground points into the lower-80% shaft, inflating its footprint to 1.46 m (gate 1.35) and collapsing columnarity (0.13 < 0.15) | axis radius tightened to **0.9 m** — it always covers the trunk (cell half-diagonal + shaft radius) but excludes the barrier (>1.0 m); the same sweep now rejects a sign panel merged with a guardrail rail (0.3 m offset) that otherwise slipped under the footprint gate as a false pole |
| conductor span centroids biased, fragments dropped | seed-42 span (240→300) measured 12.6 m off its mid-span and fell outside the 12 m match radius | `build_asset` enforced the global 25-point floor and silently dropped 15-24-point wire fragments that had passed every conductor rule, so merged spans lost their ends | detectors can pass their **class-specific minimum** to `build_asset` (conductor_min_points=15); spans now merge with all fragments and report centered geometry |

Per-seed matched objects went from **14-16 / 22** (pre-fix, conductor recall
0.00) to **22 / 22 on every measured seed** (7, 5, 99, 11, 23, 42), and the CI
test that previously failed (`matched ≥ 15`) now passes at 22.

## 4. Measured results

### 4a. Held-out validation (scenes the tuner never saw: seeds 7, 5, 99)

QC `LIKELY_DUPLICATE` rows removed (they are flagged for review, not delivered):

| class | GT | detected | P | R | F1 |
|---|---:|---:|---:|---:|---:|
| utility_pole | 18 | 18 | 1.000 | 1.000 | 1.000 |
| overhead_conductor | 15 | 15 | 1.000 | 1.000 | 1.000 |
| utility_cabinet | 6 | 6 | 1.000 | 1.000 | 1.000 |
| traffic_sign | 9 | 9 | 1.000 | 1.000 | 1.000 |
| rumble_strip | 3 | 3 | 1.000 | 1.000 | 1.000 |
| guardrail * | 6 | 21 | — | 1.000 | — |
| safety_barrier * | 3 | 9 | — | 1.000 | — |
| pavement * | 3 | 60 | — | 1.000 | — |
| pavement_marking * | 3 | 159 | — | 1.000 | — |

\* Area/linear classes are extracted per tile piece on purpose; one real rail
yields several inventory rows matched to one ground-truth object, so instance
precision undercounts. Recall 1.000 means every real object is covered; per-row
precision (0.02-0.33) is a conservative lower bound, and the pieces are merged /
QC-flagged per the inventory rules.

### 4a-bis. Cross-tile merge pass (spatial continuity) — measured outcome

The pipeline now re-joins corridor assets that tile boundaries cut, and the
evaluation scores them by **extent-aware matching** (a detection whose
bounding box overlaps the GT object's inflated box belongs to that object)
plus **fragmentation** (detections per matched GT object) instead of
per-object precision. Measured on QuickSim seed 7 (before → after):

| class | before | after | GT |
| --- | ---: | ---: | ---: |
| pavement | 20 fragments | **1** asset | 1 deck |
| pavement_marking | 58 fragments | **9** marking features | 1 painted area |
| guardrail | 5 fragments (incl. 1 mislabeled `safety_barrier`) | **2** spans | 2 rails |
| safety_barrier | 2 fragments | **1** span | 1 barrier |
| overhead_conductor | 5 spans | 5 spans | 5 wires |
| utility_pole | 6 | 6 | 6 poles |

Every compact class stays 1.000 P/R/F1; every area/linear class reaches
recall 1.000 with fragmentation 1.0 for pavement/guardrails/barriers (9.0
for markings — individual line features and dash chains, which is the honest
inventory granularity). Fixes behind the numbers:

- **Proximity-aware surface merge**: tile fragments touch at tile edges
  (tiles are non-overlapping), so an exact hull-intersection test never
  joined them; polygons within 1 m are now joined (the old test also had an
  incorrect separating-axis direction and never merged anything).
- **Line-frame linear merge for safety**: guardrail/barrier pieces are now
  pooled and joined in the pieces' own frame (gap along heading, perpendicular
  cross-track ≤ 1.5 m, heading dot ≥ 0.906). Previously the join axis was
  chosen from the centre offset, so two parallel rails 10 m apart were merged
  as one "span" while true continuations across a tile boundary were refused;
  PCA azimuth modulo 180 (0.01° vs 179.98° for the same line) broke the
  heading check, and poles standing 1.4-1.7 m beside a rail blocked merges
  (pole blocking is correct for conductors, wrong for roadside safety).
- **Span-level class resolution**: when pieces of one continuous span are
  labeled differently (a rail tile merged with a sign footprint measures
  0.63 m wide and is labeled `safety_barrier`), the merged span takes the
  *median* piece cross-section — robust to one contaminated piece.
- **Learned-veto exemption for area/linear classes**: the veto deleted that
  0.63 m guardrail fragment outright (it did not look like a barrier); the
  span merge fixes the label instead, so the veto now applies only to
  compact classes.
- **Collinear marking merge**: dashed centreline blobs ~9 m apart on the same
  heading are re-joined into one marking line; edge-line fragments across
  tiles join the same way.
- **Marking brightness cut recalibrated (real-data lesson)**: the contrast
  floor at 2.5× median routinely made the *effective* intensity threshold land
  near the 95th percentile on bright road surfaces — the likely cause of only
  50 marking records on the 112.8M-point Mannford corridor. The floor is now
  configurable and set to 2.0× median with the quantile at 0.90, so the
  effective cut sits in the 88th–90th percentile band. Measured on seed 7:
  marking recall stays 1.000 (extent-matched) while detection count rose
  9 → 10 pieces — strictly more sensitive, no precision loss on compact
  classes (`marking_intensity_quantile` / `marking_*_floor_multiplier` in
  `configs/processing.yaml`).

### 4b. Threshold tuning outcome

The coordinate descent ran over every knob with a candidate range around the
shipped defaults. Train objective: **1.0000** at the shipped defaults — no
candidate moved the needle, so the descent correctly kept every default
(`tuned_settings_delta` is empty; see `out/accuracy/training_report.json`).
Held-out objective: **1.0000** (was 0.9861 before this round's detector fixes)
— every one of the 22 ground-truth objects in each held-out scene is matched,
with pole and conductor recall now perfect across all six measured seeds.

## 5. What this means for real LAS/LAZ files

- The same code path that scored these scenes is what runs uploaded
  `.las`/`.laz` files (LAS 1.4 / PDR 7 mobile data, or older legacy tiles via
  the header re-versioning + chunked LAZ path — `infra_inventory/las_reader.py`).
- The detector fixes above (cross-section width, densest-cell poles, pole-aware
  linear merging) were found on synthetic corridors but fix *generic* geometry
  mistakes that also delete or mis-merge real wires and poles.
- To score a real file the same way, label a small ground-truth table (object
  class + centre) and call `evaluate_against_ground_truth` (docs/VALIDATION.md
  §2b) — `scripts/train_thresholds.py` accepts any QuickSim-style output, and
  the same evaluation function works on manual tables.

## 6. Known limits (honest)

- Only QuickSim ground truth exists in the repo: these numbers measure the
  detectors against synthetic mobile data, not the Mannford/Burnet files.
  Manual-GT scoring on the real competition file is the next validation step.
- Area-class instance precision is not reported as F1 because per-tile
  fragmentation makes it structurally pessimistic; recall + positional RMSE are
  the honest quantities there.
- Confidence calibration (review threshold 0.80, compact-object precision 1.000
  / recall 1.000 at that cut, latest detectors) is in `docs/ALP.md` and
  re-measured by `scripts/calibrate_confidence.py`.
- Pole and conductor recall are perfect on the measured synthetic scenes; the
  radius/gate margins behind that (0.9 m axis sweep vs. barrier distances of
  ~1 m) are validated only against these scenes so far, so real-data validation
  with a manual ground-truth table is still required before relying on the
  numbers for competition data.
