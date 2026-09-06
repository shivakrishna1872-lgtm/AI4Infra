# Enhancement Procedure

This project is designed so that additional labeled data improves accuracy through
fine-tuning — without making the default pipeline depend on that data.

## 1. Adding labels / creating training data

1. Process a LAS to tiles: `python -m infra_inventory tile input.las --output tiles`.
2. Load the tiles in CloudCompare (or your labeling tool of choice) and label points
   with the competition taxonomy:
   `pavement, pavement_marking, utility_pole, overhead_conductor, utility_cabinet,
   traffic_sign, guardrail, safety_barrier, rumble_strip` (+ `background`).
3. Export labels per tile (e.g. `tile_<ix>_<iy>_label.npy` aligned with the tile's
   point order) and keep the class-name list in the same order everywhere.
4. Place labeled tiles under a `datasets/` directory and register a Pointcept dataset
   class that reads LAS + labels (see `configs/pointcept/infra_ptv3_nuscenes.py.example`).

## 2. Fine-tuning Pointcept / PTv3

```bash
python -m infra_inventory train \
  --pointcept-root /path/to/Pointcept \
  --pointcept-config /path/to/your_train_config.py
```

(or run Pointcept's `tools/train.py` directly — the CLI is a thin bridge).

* Start from the nuScenes checkpoint (`download-models`) so training is fine-tuning,
  not from-scratch.
* The PTv3 nuScenes config is the reference; keep the same architecture and adjust
  `num_classes` to your taxonomy. FlashAttention can stay off
  (`enable_flash=False`, smaller patch sizes) for more compatible GPUs.
* Train/eval splits: hold out whole tiles, not random points, to measure real
  generalization.

## 3. Adding infrastructure classes

1. Add the class to `VALID_TAXONOMY` (`infra_inventory/models.py`) and
   `CLASS_PREFIXES` (id prefix).
2. Add colors/weights to `configs/classes.yaml`.
3. Write a detector in `infra_inventory/assets/` following the existing pattern
   (mask → connected components → geometry rules → `build_asset`).
4. Register it in `assets/__init__.py::detect_all`.
5. Add dimension limits in `infra_inventory/qc.py` so QC flags nonsense instances.
6. Add a synthetic fixture to the test scene and a detector test.

## 4. Modifying class mappings

The Pointcept → infrastructure mapping lives in `configs/model.yaml`
(`class_mapping`). Each entry maps an upstream class to an infrastructure class with
a weight; `evidence_for` lists classes that receive weak evidence. Predictions are
imported only through this table — change it, not the model.

## 5. Evaluating precision / recall

With labeled tiles, evaluate per class:

* **Precision**: for each detected asset, does a labeled instance overlap it
  (e.g. ≥ 50% of the asset's points inside a label region)?
* **Recall**: for each labeled instance, was it detected?

A small script skeleton:

```python
# scripts/evaluate.py (skeleton)
# 1. run process on the labeled LAS
# 2. load per-tile labels
# 3. match assets to labels by point overlap
# 4. report per-class precision/recall and confusion across subclasses
```

## 6. Updating confidence thresholds

Confidence weights live in `configs/classes.yaml` (`confidence_weights`); the
low-confidence QC threshold is `low_confidence_threshold` in
`configs/processing.yaml`. After an evaluation pass, move the threshold to where
precision/recall trade off best per class.

## 7. Adding specialized detectors

* **RoadMarkingExtraction**: run the external subsystem via `--roadmarking-command`
  and consume its DXF output (already supported). To use its richer classifications,
  extend `roadmarking.parse_dxf_outputs` to read entity layers/classes.
* **New geometry detectors**: follow the existing detectors; every new rule must be a
  *measured* property with a documented threshold and confidence contribution.

## 8. Tuning knobs that matter most

| Knob | Effect |
| --- | --- |
| `tile_size_m` | smaller tiles = more boundary splits; larger = more memory per tile |
| `ground_bin_m` / `ground_quantile` | ground estimation robustness on slopes/vegetation |
| `marking_intensity_quantile` | how exclusive "bright" is for markings |
| `pole_min_height_m`, `pole_max_footprint_m` | pole recall vs false positives |
| `sign_min_planarity` | panel detection strictness |
| `guardrail_min_length_m` | guardrail recall vs clutter |
| `duplicate_distance_m` | cross-tile duplicate flagging |
| `low_confidence_threshold` | QC strictness |