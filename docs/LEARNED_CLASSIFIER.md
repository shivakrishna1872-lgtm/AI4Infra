# Learned Component Classifier

A trained, CPU-only class-evidence layer on top of the geometry detectors.
It raises the *quality of every confidence score* and can veto candidates a
trained model confidently rejects — without a GPU, without new dependencies,
and without changing what geometry decides to extract.

## Why

The confidence engine weights a `model` factor highest for most classes, but
on the default CPU backend that factor was always `None` — every run silently
ran with a missing evidence channel. The Pointcept/PTv3 and OpenPCSeg backends
fill it, but they need CUDA, a model checkout and hundreds of MB of weights.

This classifier closes that gap for the CPU path: it is trained on labeled
ground truth to recognize the geometric signature of each asset class from
the *same measured features the detectors already compute*, and its calibrated
score becomes the `model` factor on every asset. Asset records now show
`model_prior_class` / `model_confidence` on plain `--backend geometry` runs.

## What it is

- A **multinomial logistic-regression (softmax) classifier** over a 14-d
  measured feature vector, implemented in pure numpy (`infra_inventory/learned.py`).
- **Features** (`FEATURE_NAMES`, fixed contract): eigenvalue descriptors
  (linearity, planarity, verticality, columnarity), log-scaled extents and
  point count, ground band, log cross-section thickness, intensity contrast,
  RGB brightness, above-ground ratio, and bbox compactness. This is the same
  geometric-feature family the mobile-LiDAR segmentation literature uses
  (Pointcept/PTv3 eigen-features; Toronto-3D / RandLA-Net per-point features,
  aggregated per component here).
- **Training data**: labeled connected components from Quick Simulation
  scenes. Every candidate component the detectors build is collected
  (`--collect-training`), then relabeled against the scene's exact
  ground-truth table with **extent-aware matching** (each GT object is an
  inflated box, so a tile fragment 80 m along a 110 m guardrail matches the
  guardrail instead of being poisoned to `background`; when several boxes
  contain a component — markings inside the pavement slab — the detector's
  assignment wins, then smallest distance, then smallest volume).
- **Weights** are shipped at `configs/learned_classifier.json` with the
  measured validation metrics embedded in the `training` block.

## What it is not

- It does **not** replace Pointcept/OpenPCSeg. When a Pointcept/OpenPCSeg
  prior is present it keeps filling the `model` factor; the learned classifier
  is skipped entirely (`predictions` non-empty ⇒ no learned prior). The
  classifier only fills the channel the GPU backends would.
- It does **not** decide that an asset exists. Geometry gates every candidate
  first; the classifier can only *score* or *veto* (and the veto requires the
  trained model to prefer a concrete alternative class with high confidence:
  `P(assigned) < 0.60` **and** runner-up `≥ 0.20` by default).

## Measured accuracy

Trained with `python scripts/train_classifier.py` (20 train seeds, 11 held-out
validation seeds the model never sees, 400 m scenes, 800 epochs):

```
train:    n=4754  accuracy=1.000  macro-F1 (excl. background)=1.000
held-out: n=2610  accuracy=1.000  macro-F1 (excl. background)=1.000
```

Per class on held-out scenes (all nine asset classes + background):
precision 1.00, recall 1.00. The classes are geometrically well separated in
feature space, so softmax regression separates them perfectly on the
synthetic scenes; the number that matters in practice is the **pipeline
A/B**, below.

Pipeline quality (per train seed, after the cross-tile merge passes):

| metric | value | meaning |
| --- | --- | --- |
| compact per-object F1 | **1.000** on all 20 seeds | poles, conductors, cabinets, signs, rumble strips — one detection per object, no misses, no extras |
| area/linear recall | **1.000** on every seed | pavement, markings, guardrails, barriers — every ground-truth span/deck covered |
| fragmentation | **2.4-2.8** detections per GT object | corridor assets after merging (down from ~20 pavement fragments and ~58 marking fragments before the merge work) |

Fragmentation above 1.0 is honest: painted markings are extracted as
individual line features (edge lines, centreline dash chains, symbols), and
the dash chains split where the paint itself is missing from the cloud.

The learned veto applies only to compact classes. Area/linear classes
(pavement, markings, guardrails, barriers) are exempt: their per-tile class
assignment is legitimately ambiguous (a guardrail fragment can measure wider
than its neighbours and be labeled `safety_barrier`), and the span-level
merge resolves the label — vetoing such a fragment deleted a true positive
(measured on QuickSim seed 7) instead of fixing it.

## Usage

```bash
# retrain (writes configs/learned_classifier.json)
python scripts/train_classifier.py
python scripts/train_classifier.py --seeds 11 23 42 --validate-seeds 7 5 --epochs 400
python scripts/train_classifier.py --dry-run     # measure only, no save

# per-run control
python -m infra_inventory process data.las --output out/            # prior on (default)
python -m infra_inventory process data.las --output out/ --no-learned-prior
python -m infra_inventory process data.las --output out/ --no-learned-veto
python -m infra_inventory process data.las --output out/ --learned-model path/to/model.json
python -m infra_inventory process data.las --output out/ --collect-training   # write <out>/training/components_*.jsonl
```

Provenance on every asset: `model_confidence` (the calibrated factor),
`model_prior_class` (the classifier's best class — `*` marks agreement with
the detector's assignment), and `run.json → backend.learned_classifier`
(version + training metrics).

## Transfer to real data

Trained on synthetic scenes, the classifier generalizes to real scans exactly
as far as the *features* do (they are scale-normalized measured geometry, not
synthetic-specific), and degrades safely: on out-of-distribution components
it returns moderate probabilities → lower `model` factor → flagged for human
review, and the veto's two-sided margin makes false rejections rare.

To specialize it on real data, label a few corridors and either (a) put the
labels in a ground-truth table (then reuse the training script's relabeling
path), or (b) collect with `--collect-training` and train on the JSONL
directly — the detector-assignment labels are already the transfer signal.
External labeled MLS datasets (e.g. Toronto-3D) plug in at the same layer.

## Honest limits

- Synthetic-only training data ( Quick Simulation ). The shipped model's
  perfect held-out numbers are *in-domain*; real-world accuracy is not yet
  measured and will be lower. The `training` block in the model JSON states
  exactly what it was trained on.
- Softmax regression is linear: it cannot capture class boundaries that are
  curved in feature space. If real-data accuracy plateaus, the next step is a
  small gradient-boosted tree over the same features — same JSON contract,
  same feature pipeline.
- The veto threshold is deliberately conservative. `--no-learned-veto` keeps
  the confidence benefits without ever dropping a candidate.
