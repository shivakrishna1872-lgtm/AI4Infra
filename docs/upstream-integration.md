# Upstream Integration Notes

This project separates an immediately runnable CPU pipeline from optional learned
backends, and never relabels an upstream benchmark taxonomy as a competition class
without an evidence-backed adapter.

## Pointcept and Point Transformer V3

Used (real, documented):

* **Entrypoint bridge** (`infra_inventory/pointcept.py`): invokes Pointcept's real
  `tools/test.py` with the upstream CLI contract (`--config-file`, `--num-gpus`,
  `--options save_path=... weight=...`), verified against the upstream source.
* **Dataset contract**: the pipeline exports per-tile LAS files that an adapted
  Pointcept dataset class consumes (template:
  `configs/pointcept/infra_ptv3_nuscenes.py.example`).
* **Model**: PTv3 nuScenes semantic-segmentation checkpoint from the official
  Hugging Face release (see `docs/MODEL_NOTES.md`), including the upstream
  documented option to run without FlashAttention.
* **Adaptation**: per-tile predictions are imported only through the documented
  `class_mapping` table (`configs/model.yaml`), which maps upstream classes to
  infrastructure evidence with explicit weights.

Not used:

* No Pointcept code is copied into this repository. Pointcept stays an external,
  versioned dependency used at runtime through `tools/test.py`.
* No fake or "mock" model outputs; if the Pointcept environment is missing, the
  backend reports it and the geometry pipeline still runs.

## RoadMarkingExtraction

Used (real, documented):

* **Adapter** (`infra_inventory/roadmarking.py`): stages per-tile LAS input, runs
  the user's configured RoadMarkingExtraction run script, and parses the resulting
  DXF vector output (LINE / LWPOLYLINE entities) into marking instances with
  `detection_method: roadmarkingextraction-v1`.
* The upstream build requirements (Eigen3, PCL, OpenCV, LibLas, DXFLib) are
  documented in the adapter and procedure docs; the subsystem stays external.

Not used:

* The portable `native-roadmarking-v1` detector in this repository is original code;
  it is a baseline, not a copy of the C++ project.
* No C++ sources are vendored.

## Original code in this repository

Everything under `infra_inventory/` (except the thin bridges above), the
configuration files, scripts, tests, and documentation are original to Terra Point.
This includes the streaming tile pipeline, the geometry detectors, the confidence
engine, QC, exports, and the 3D viewer.

## Competition-specific contribution

The inventory and attribution contract is the competition layer: object identity,
source-tile/point provenance, measured attributes, confidence factors, CRS
retention, and export artifacts. It is deliberately model-agnostic so that a valid
Pointcept fine-tuning run (docs/ENHANCEMENT.md) or a specialist pavement result
improves accuracy without weakening auditability.