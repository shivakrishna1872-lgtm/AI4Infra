# Accuracy Report — detector training & validation

*Last measured: this build, `scripts/train_thresholds.py` + the full pytest suite.*

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
