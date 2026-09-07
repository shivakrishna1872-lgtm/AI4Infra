# AI4Infra Processing Procedure

This document is the repeatable, competition-ready procedure. Someone who has never
seen this project must be able to follow it and process another LiDAR mile.

---

## 1. Installation

```bash
git clone <your-repo-url> ai4infra
cd ai4infra
python -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'          # laspy, numpy, pyyaml, pytest
```

Python 3.9+ is required. No GPU or specialized hardware is needed for the default path.

## 2. Environment check

```bash
python -m infra_inventory backends
```

This reports whether the geometry backend (always available) or the Pointcept backend
(needs a Pointcept checkout + CUDA) can run on this machine.

## 3. Download model weights (optional, only for the Pointcept backend)

```bash
python -m infra_inventory download-models --output models
```

Downloads the documented PTv3 nuScenes semantic-segmentation checkpoint
(~530 MB) and its config from the upstream Hugging Face release, verifying the
published SHA-256. Sources and licensing: `docs/MODEL_NOTES.md`.

## 4. LAS preparation

* Expected: **LAS 1.4, Point Data Record Format 7** (Trimble MX9 mobile LiDAR, as
  provided by the competition).
* The file should carry intensity, RGB, `gps_time` (run separation) and
  `point_source_id` (scanner attribution) when available.
* If your LAS is an older version or a different point format, re-export with
  CloudCompare or laspy; the pipeline degrades gracefully but the competition source
  is LAS 1.4 / format 7.

Validate before processing:

```bash
python -m infra_inventory validate data/mannford.las
```

## 5. Run preprocessing (tiling)

```bash
python -m infra_inventory tile data/mannford.las --output output/mannford/tiles --tile-size 40
```

Tiles are written as LAS files (`tile_<ix>_<iy>.las`) in a single streaming pass;
the whole cloud is never loaded into memory.

## 6. Run inference

### Default (CPU-safe geometry backend)

```bash
python -m infra_inventory process data/mannford.las --output output/mannford
```

### With the Pointcept / PTv3 learned backend

```bash
python -m infra_inventory process data/mannford.las --output output/mannford \
  --backend pointcept \
  --pointcept-root /path/to/Pointcept \
  --pointcept-config configs/pointcept/infra_ptv3_nuscenes.py \
  --pointcept-weight models/ptv3_nuscenes_semseg.pth \
  --pointcept-class-names configs/pointcept/nuscenes_class_names.json
```

The pipeline writes tiles first, then runs Pointcept's real `tools/test.py` on them,
then imports per-tile predictions through the documented class-adaptation layer
(`configs/model.yaml`). GPU + Pointcept CUDA ops required; FlashAttention optional
(`enable_flash=False` + smaller patch sizes).

### With the specialized RoadMarkingExtraction subsystem

```bash
python -m infra_inventory process data/mannford.las --output output/mannford \
  --roadmarking-command /path/to/RoadMarkingExtraction/script/run_xxx.sh \
  --roadmarking-config /path/to/RoadMarkingExtraction/config/xxx.txt
```

The external C++ extractor runs per tile; its DXF vector output is parsed into
marking assets (`detection_method: roadmarkingextraction-v1`).

## 7. Asset extraction and classification

Extraction is automatic inside `process`:

1. Per tile: robust ground estimation → height above ground → features.
2. Grid connected components isolate *instances*.
3. Class-specific geometry rules decide class/subclass (see `assets/`):
   * **Pavement**: low-elevation continuous surface; markings are near-ground
     high-reflectance/bright components classified by shape.
   * **Utilities**: poles (narrow footprint, tall, columnar), conductors (elevated
     thin linear chains), cabinets (compact dense boxes).
   * **Signs**: planar panels (eigenvalue geometry) grouped with their support.
   * **Safety**: guardrails (long/low/narrow), barriers (wide cross-section),
     rumble strips (repeating transverse bands).
4. Pointcept predictions (when the backend runs) modulate the `model` confidence
   factor through the explicit mapping; geometry always gates detection.

## 8. Attribution

Every asset carries: source tile, representative source point indices,
intensity/RGB statistics, orientation, source run (GPS-time split) and scanner
(`point_source_id`) when measurable, CRS, detection method, processing version.

## 9. Confidence

Confidence is a weighted blend of measurable factors (model / geometry / support /
spatial context / class consistency), renormalized over the factors actually
present. The explanation string names the contributing factors and their values.

## 10. Exporting results

`process` writes:

```
output/
├── inventory.json          # run + QC summary + full inventory (agency-consumable)
├── assets.json             # every asset with all attributes
├── assets.csv              # tabular summary
├── assets.geojson          # point features at asset centroids
├── run.json                # provenance: CRS, bounds, warnings, backend
├── assets/                 # one JSON file per asset
├── tiles/                  # per-tile LAS + manifest
├── predictions/            # Pointcept exports (when that backend runs)
├── reports/                # summary.md + qc_report.json
└── viewer/                 # dependency-free 3D viewer
```

`viewer-data.json` is the only payload served to the browser and it is kept
slim on purpose: a decimated overview cloud (≤ 120k points) plus whitelisted
asset records (only the fields the 3D scene and inspector render, with capped
highlight-point evidence). Full provenance per asset stays in `assets.json`,
`inventory.json` and the per-asset files under `assets/`. The server also
gzip-compresses it, so a multi-GB scan becomes a small JSON download.

### 10b. MongoDB inventory mirror (optional)

Every asset can also be mirrored to MongoDB as a GeoJSON document with the
agency schema (`asset_id`, `category`, `subcategory`, `location` Point,
`attributes` incl. `height_m` / `lean_angle_deg` / `run_source` / `condition`),
plus confidence, QC flags, CRS and source provenance. A `2dsphere` index is
created for spatial queries. Enable via `MONGO_URI` env (server or CLI) or:

```bash
python -m infra_inventory process data/mannford.las --output output/mannford \
  --mongo-uri mongodb://localhost:27017
```

Requires `pip install -e '.[mongo]'`. The mirror never blocks the pipeline: a
failed export becomes a run warning, and prior documents of the same source
file are replaced so the collection mirrors the latest run. Full schema:
`docs/ASSET_SCHEMA.md` §5.

## 11. Visualization

```bash
python -m infra_inventory serve output/mannford
```

Open `http://127.0.0.1:8765`. The viewer is a single HTML file with WebGL:
orbit/zoom, elevation-colored cloud, class filters, asset list, and
click-to-inspect (metadata + highlighted source points).

## 12. Quality control

QC runs automatically: duplicate detections (`LIKELY_DUPLICATE`), implausible
dimensions (`UNUSUAL_DIMENSIONS`), low confidence (`LOW_CONFIDENCE`), and minimum
point support are flagged in each asset and summarized in `reports/qc_report.json`.
Inspect flagged assets before reporting results.

## 13. Working with the real competition LAS

```bash
python -m infra_inventory validate data/mannford.las
python -m infra_inventory process data/mannford.las --output output/mannford \
  --tile-size 40 --chunk-size 500000 --viewer-points 150000
python -m infra_inventory serve output/mannford
```

Tune `--tile-size` down (e.g. 25 m) if tiles get too dense for memory, or up if
instances keep splitting across boundaries. See `docs/ENHANCEMENT.md` for the full
tuning and fine-tuning guide.

## 14. Compressed LAZ input

`.laz` files are first-class input. The `lazrs` backend is installed with the
dev extras; without it, uploads fail with a clear "LAZ backend missing" error.

```bash
python -m infra_inventory validate data/corridor.laz
python -m infra_inventory process data/corridor.laz --output output/corridor
```

In the web app, drag the `.laz`/`.las` into the upload zone or use
`POST /api/projects/{id}/upload` + `POST /api/projects/{id}/process`. Jobs are
persisted to disk, so backend restarts never lose them.

## 14b. Legacy LAS versions (1.0 / 1.1) and very large tiles

The reader follows the approach of [LAStools](https://github.com/LAStools/LAStools)
for files its own tooling cannot write natively:

* **Chunked decompression**: `.laz` is decoded in streaming chunks through
  laspy's `lazrs` backend — the same LASzip codec LAStools uses — so a file is
  never held in memory whole.
* **Header re-versioning** (las2las analogue): laspy cannot *write* LAS 1.0
  headers, which crashed older USGS 3DEP tiles at the first 500k-point chunk
  (`FileVersionNotSupported: 1.0`). Tiles are now written with a writable LAS
  version while preserving scales, offsets, point format and every dimension;
  the input file is never modified and the re-version is recorded as a warning.
* **Uniform thinning** (`--max-input-points N`, the `las2las -thin` analogue):
  inputs larger than `N` points are uniformly decimated inside each streaming
  chunk, bounding memory and time for multi-hundred-million-point tiles while
  keeping the full spatial extent. Off for competition mobile data by default
  (0 = process every point); `summary.point_count` always reports the true
  input count and a warning states the thinning.

```bash
# a 2006 USGS 3DEP tile (LAS 1.0, PDR 1, ~16M points) — previously crashed:
python -m infra_inventory process TX_BurnetCo_2006_000014.las --output output/burnet
# same file, bounded to ~2M points for quick analysis:
python -m infra_inventory process TX_BurnetCo_2006_000014.las --output output/burnet --max-input-points 2000000
```

## 15. Simulation & automated validation

Quick Simulation (zero data) generates a synthetic mobile-LiDAR corridor and
runs it through the same pipeline — useful for demos, QA, and detector
regression checks:

```bash
python -m infra_inventory simulate --output output/sim --length 400 --seed 7
```

The project exports `simulation_meta.ground_truth` (the exact placed objects).
Score the run against it with the snippet in `docs/VALIDATION.md` §2a, which
reports Precision, Recall, F1, and positional RMSE per class.

## 15b. ALP assessment, confidence calibration and human review

Every asset is assessed by the ALP layer (`infra_inventory/assessment.py`): an
Observation → Interpretation → Recommended Action record plus a review flag,
exported as `condition`, `recommended_action`, `review_required`, and
`assessment_reasoning` (docs/ALP.md). `reports/summary.md` includes the
condition table and review count; `reports/qc_report.json` lists every asset
that needs a human look.

Calibrate the review threshold against ground truth instead of guessing:

```bash
python scripts/calibrate_confidence.py --length 400 --seed 23 --report reports/calibration.json
```

Set `review_confidence_threshold` in `configs/processing.yaml` to the
recommended value and record the sweep in docs/ALP.md.

## 15c. Disk space and very large files

A ~16M-point tile needs several GB for per-tile intermediates. Three lines of
defence:

1. **Web/API runs clean up**: jobs process with `save_tiles=false`, so per-tile
   LAS files are removed after processing (`run.json` records a warning).
   Flat exports omit per-point `highlight_points` to stay slim; full evidence
   lives in per-asset files and `viewer-data.json`.
2. **The API fails fast**: below 512 MB free, processing returns a clear 507
   message ("delete old projects…") instead of dying mid-write with
   `Errno 28`.
3. **Manual cleanup** when a run still fills the disk:

```bash
df -h                        # check free space
ls appdata/projects/         # each dir = one uploaded/processed project
rm -rf appdata/projects/<partial-or-failed-project-id>   # failed/partial runs
rm -f appdata/jobs/*.tmp     # leftover atomic-write temp files
```

Delete old projects from the web UI (project → delete) before reprocessing big
files. Never commit `.las`/`.laz` datasets.

## Known limitations (read before judging)

* Compact objects split by a tile boundary (e.g. a pole standing exactly on a
  tile edge) are detected twice and flagged `LIKELY_DUPLICATE` for human review;
  thin linear assets (overhead conductors) are re-joined across tiles by the
  pole-aware merge pass in `instances.merge_linear_pieces`.
* The default backend is geometry-based; Pointcept improves model evidence but the
  competition classes still require the adaptation layer and ideally fine-tuning.
* Measured extraction accuracy (Quick Simulation ground truth, held-out scenes),
  the tuning procedure and per-class P/R/F1: see
  [`docs/ACCURACY_REPORT.md`](ACCURACY_REPORT.md) and
  [`docs/VALIDATION.md`](VALIDATION.md).
* `source_point_indices_sample` are tile-local indices (see the tile manifest);
  full per-point provenance is preserved at tile level.
* GPU inference requires the Pointcept environment; nothing in the default path
  touches CUDA.