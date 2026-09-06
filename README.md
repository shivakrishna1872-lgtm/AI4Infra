# AI4Infra

**AI4Infra turns a real mobile-LiDAR LAS file into an auditable infrastructure asset inventory.**

The pipeline is **CLI-first**: it validates and streams LAS data, splits the cloud into
spatial tiles, runs geometric asset extraction (with an optional real Pointcept/PTv3
learned backend and the specialized RoadMarkingExtraction C++ subsystem), and produces
a structured inventory of **pavement, utilities, signs, and safety** assets — each with
measured geometry, source-point provenance, CRS, and a transparent confidence score.
A dependency-free 3D viewer is layered over the artifacts.

Nothing is faked: every asset is derived from real measured signals in the input LAS,
and every unavailable measurement is exported as `null`, never guessed.

## Architecture

```text
 LAS / LAZ ──► streaming reader (laspy + lazrs) ──► validation (LAS 1.4 / PDR 7 / CRS)
      │
      ▼
 spatial tiling (per-tile LAS) ──► [optional] Pointcept / PTv3 predictions
      │                               [optional] RoadMarkingExtraction DXF vectors
      ▼
 geometric instance extraction (pavement · utilities · signs · safety)
      │
      ▼
 attribution + confidence + QC ──► exports: assets.json/csv/geojson, inventory.json, run.json
      │
      ▼
 viewer-data.json ──► React / Three.js 3D digital twin (web UI + FastAPI server)

 Simulation:  synthetic mobile-LiDAR corridor ──► same pipeline ──► same viewer
 (Quick Simulation needs zero input files; Data Simulation runs the same path
  over an uploaded .las / .laz and is labeled EXTERNAL / SIMULATION, never
  COMPETITION DATA)
```

---

## Quick start

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'

# 1. Generate a synthetic test scene (TESTING ONLY - not competition data)
python -m infra_inventory demo-data --output data

# 2. Run the full pipeline (CPU-safe, no GPU required)
python -m infra_inventory process data/mannford_synthetic.las --output output/mannford

# 3. Open the 3D viewer
python -m infra_inventory serve output/mannford
```

Open `http://127.0.0.1:8765`. The viewer is local and uses no CDN or cloud service.

Verification:

```bash
pytest -q          # full test suite, including detectors on the synthetic scene
```

## Simulation & demo modes

The application ships a browser UI (`web/`) plus a FastAPI server that serves both
it and the processing API from one origin (`python -m infra_inventory app`). The UI
distinguishes three data classes — **COMPETITION DATA**, **EXTERNAL REAL DATA**, and
**SIMULATION** — and never mixes them.

### Quick Simulation (no data required)

Generates a synthetic mobile-LiDAR-style infrastructure corridor (road, painted
markings, utility poles with conductors, signs, guardrails) as a real LAS 1.4 /
Point Format 7 file, then runs it through the *same* pipeline as uploaded data.

```bash
# CLI
python -m infra_inventory simulate --output output/sim --length 400 --seed 7

# API (used by the web UI's "Open Simulation" button)
curl -X POST http://127.0.0.1:8766/api/simulate
```

Use it to demo the platform or iterate on the pipeline when no LiDAR file is at hand.
Generated projects are labeled `SIMULATED` end-to-end (project metadata, viewer
overlay, exports).

### Data Simulation (requires data)

Upload a **.las** or **.laz** file (drag-and-drop in the UI, or `POST /api/simulate/data`)
and the same extraction pipeline runs over it. Both `.las` and LAZ-compressed `.laz`
are supported — LAZ decoding is handled through laspy's `lazrs` backend.

### Learned backends (optional, GPU machines only)

* [Pointcept](https://github.com/Pointcept/Pointcept) / [Point Transformer V3](https://github.com/Pointcept/PointTransformerV3) — per-tile semantic priors via Pointcept's own `tools/test.py` (CUDA; FlashAttention optional).
* [RoadMarkingExtraction](https://github.com/YuePanEdward/RoadMarkingExtraction) — specialized marking vectorization adapter.

Neither is required to run the application: geometry-only detection is the default
and needs no GPU, which is deliberate so the whole project never depends on one
machine. See [`docs/upstream-integration.md`](docs/upstream-integration.md) and
`configs/pointcept/` for wiring details.

## What runs today (all real)

| Stage | Implementation |
| --- | --- |
| LAS ingestion | Streaming chunks through `laspy` (`lazrs` LASzip codec for `.laz`, LAStools-style); legacy LAS 1.0/1.1 inputs are re-versioned for tile writes, never modified; optional uniform thinning (`--max-input-points`) for huge airborne tiles |
| Validation | LAS version / point format / CRS / dimensions / bounds, with fix hints (`validate` command) |
| CRS | Read from LAS metadata (`EPSG:XXXX`); `CRS_UNRESOLVED` when absent — never assumed |
| Tiling | Spatial tiles persisted as per-tile LAS files (append mode, single pass) |
| Ground estimation | Robust per-tile ground from cell minima; height above ground per point |
| Pavement | Travelled surface (elevation + coverage) and painted markings (near-ground high-reflectance/brightness connected components), subclasses: lane/edge line, stop line, crosswalk, symbol |
| Utilities | Poles (narrow footprint + vertical extent + columnarity), overhead conductors (elevated thin linear chains, heuristic), cabinets (compact dense boxes) |
| Signs | Planar panels (eigenvalue geometry) grouped with their vertical support into one asset |
| Safety | Guardrails (long/low/narrow, roadside), concrete barriers (wide cross-section), rumble strips (repeating transverse bright bands, heuristic) |
| Instances | Grid connected components → individual assets with unique IDs |
| Attribution | Source tile, source point indices, run (GPS-time split), scanner (`point_source_id`), intensity/RGB stats, orientation, CRS |
| Confidence | Weighted blend of model / geometry / support / spatial-context / class-consistency factors, with an explanation string |
| Quality control | `LOW_CONFIDENCE`, `LIKELY_DUPLICATE`, `UNUSUAL_DIMENSIONS`, `INVALID_GEOMETRY`, `BELOW_MIN_POINTS` flags |
| Export | `inventory.json`, `assets.json/csv/geojson`, per-asset JSON, `run.json`, `reports/`, `tiles/` |
| 3D viewer | Dependency-free WebGL viewer: elevation-colored cloud, class filters, asset list, click-to-inspect with source-point highlight |

## Optional learned backend: Pointcept / PTv3

Pointcept's real inference entrypoint is `tools/test.py`; this application invokes it
exactly as the upstream project documents. The pipeline exports per-tile LAS files
first; your adapted Pointcept config consumes them and exports per-tile predictions,
which enter detection as a *model prior* through the documented adaptation layer
(`configs/model.yaml`). Geometry still decides whether an asset exists.

```bash
# download the documented PTv3 nuScenes weights (~530 MB, SHA-256 verified)
python -m infra_inventory download-models --output models

python -m infra_inventory process data/mannford.las --output output/mannford \
  --backend pointcept \
  --pointcept-root /path/to/Pointcept \
  --pointcept-config configs/pointcept/infra_ptv3_nuscenes.py \
  --pointcept-weight models/ptv3_nuscenes_semseg.pth \
  --pointcept-class-names configs/pointcept/nuscenes_class_names.json
```

GPU notes: PTv3 requires CUDA and Pointcept's CUDA ops. FlashAttention is **optional**
upstream — configure `enable_flash=False` and smaller patch sizes on non-compatible
GPUs. The default `geometry` backend never touches CUDA, so the project remains fully
CPU-safe until you opt in.

## Optional specialized subsystem: RoadMarkingExtraction

The external [RoadMarkingExtraction](https://github.com/YuePanEdward/RoadMarkingExtraction)
C++ project (PCL/OpenCV/LibLas/DXFLib) is a specialized pavement-marking extractor. The
adapter runs your configured run script per tile and ingests its DXF vector output as
marking assets with `detection_method: roadmarkingextraction-v1`:

```bash
python -m infra_inventory process data/mannford.las --output output/mannford \
  --roadmarking-command /path/to/RoadMarkingExtraction/script/run_xxx.sh \
  --roadmarking-config /path/to/RoadMarkingExtraction/config/xxx.txt
```

The portable `native-roadmarking-v1` detector runs by default and needs nothing extra.

## Project layout

```
infra_inventory/
├── cli.py                  # process / serve / validate / tile / backends / download-models / train / demo-data
├── pipeline.py             # orchestrator: streaming → tiles → detection → QC → export
├── las_reader.py           # streaming LAS/LAZ reader, CRS, GPS-run separation
├── validation.py           # LAS validation with fix hints
├── preprocessing.py        # tiling, ground, density, z-range features
├── instances.py            # connected components + PCA component metrics
├── assets/                 # pavement.py, utilities.py, signs.py, safety.py detectors
├── confidence.py           # transparent confidence engine
├── attribution.py          # provenance fields (run/scanner/stats/orientation)
├── qc.py                   # duplicate/dimension/confidence flags
├── export.py               # JSON/CSV/GeoJSON/per-asset/report writers
├── viewer.py               # dependency-free WebGL viewer
├── pointcept.py            # real tools/test.py bridge + prediction loader
├── roadmarking.py          # external C++ subsystem adapter + DXF parser
├── simulation.py           # Quick Simulation / Data Simulation pipelines
├── server.py               # FastAPI app: project API + hosted web UI
├── synthetic.py            # synthetic mobile-LiDAR scene generators
└── download_models.py      # verified pretrained-weight downloads
configs/                    # classes.yaml, model.yaml, processing.yaml, pointcept/ templates
scripts/                    # download_models.py, make_synthetic_las.py
tests/                      # full test suite incl. end-to-end, LAZ upload, and simulation paths
docs/                       # PROCEDURE, ASSET_SCHEMA, VALIDATION, LIMITATIONS, ENHANCEMENT, MODEL_NOTES, upstream-integration
```

Documentation: [`docs/PROCEDURE.md`](docs/PROCEDURE.md) (1-mile workflow),
[`docs/ASSET_SCHEMA.md`](docs/ASSET_SCHEMA.md) (inventory schema & competition
mapping), [`docs/VALIDATION.md`](docs/VALIDATION.md) (precision/recall/F1 &
positional-error framework), [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md)
(known edge cases), [`docs/ENHANCEMENT.md`](docs/ENHANCEMENT.md) (roadmap).

## Artifact contract

Each asset records: `asset_id`, `class`, `subclass`, XYZ center, bounding box,
dimensions, point count, representative source point indices, CRS, source tile,
run/scanner when measurable, detection method, confidence + confidence factors +
explanation, and QC flags. Missing values are `null` — never invented.

`source_point_indices_sample` are indices into the tile file named by `source_tile`
(the tiles are the processed units; see `output/tiles/manifest.json` for order).

## Upstream sources

- [Pointcept](https://github.com/Pointcept/Pointcept)
- [Point Transformer V3](https://github.com/Pointcept/PointTransformerV3)
- [RoadMarkingExtraction](https://github.com/YuePanEdward/RoadMarkingExtraction)

See `docs/upstream-integration.md` for exactly what is used from each, and
`docs/MODEL_NOTES.md` for pretrained-model sourcing and licensing.

## License

MIT (this project's code). Pretrained PTv3 nuScenes weights are released by the
upstream authors under their own license (CC-BY-NC-4.0 per the upstream repository) —
check `docs/MODEL_NOTES.md` before commercial use.