# AI4Infra

AI4Infra turns a real mobile-LiDAR LAS file into an auditable infrastructure asset inventory. The pipeline is intentionally **CLI-first**: it validates and streams LAS data, extracts measured geometry, produces inventory artifacts, then layers a dependency-free 3D viewer over those artifacts.

## What runs today

- Streams LAS/LAZ through `laspy`; no blind whole-cloud load.
- Validates LAS version/point format, captures CRS and source metadata, and spatially tiles data.
- Generates measured pavement, reflective pavement-marking, utility-pole, and guardrail candidates with source point samples, bounding boxes, dimensions, orientation, and transparent confidence factors.
- Writes `assets.json`, `assets.csv`, `assets.geojson`, `run.json`, and a WebGL viewer backed by actual sampled source points.
- Keeps Pointcept/PTv3 and RoadMarkingExtraction as explicit optional integrations rather than pretending their pretrained taxonomies already match competition classes.

## Quick start

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'

python -m infra_inventory process data/mannford.las --output output/mannford
python -m infra_inventory serve output/mannford
```

Open `http://127.0.0.1:8765`. The viewer is local and uses no CDN or cloud service.

## Pointcept / PTv3 is opt-in

Pointcept's real inference entrypoint is `tools/test.py`, driven by Pointcept configs, datasets, and matching model weights. The application can invoke that entrypoint only when you supply an adapted Pointcept dataset configuration and checkpoint:

```bash
python -m infra_inventory process data/mannford.las --output output/mannford \
  --backend pointcept \
  --pointcept-root /path/to/Pointcept \
  --pointcept-config /path/to/infra_ptv3_config.py \
  --pointcept-weight /path/to/model.pth
```

The app never maps a generic pretrained class to `utility_pole`, `traffic_sign`, or `guardrail` without a documented adaptation layer. PTv3 should be treated as an accelerator/backbone: it needs the Pointcept environment and normally CUDA extensions. FlashAttention is optional in the upstream project; configure PTv3 with `enable_flash=False` and smaller patch sizes for compatible GPU environments. The default `geometry` backend remains fully CPU-safe.

## Road markings

The default `native-roadmarking-v1` detector uses real near-ground intensity/RGB and connected-component measurements. It is a portable baseline, not a stand-in for learned segmentation. The external [RoadMarkingExtraction](https://github.com/YuePanEdward/RoadMarkingExtraction) project is a C++ LAS/PCD extraction, classification, and vectorization subsystem; integrate it as a separately versioned adapter once its PCL/OpenCV/LibLas environment is available.

## Artifact contract

Each asset records an ID, competition-oriented class/subclass, XYZ center, bounding box, dimensions, point count, representative source point indices, CRS, source tile, detection method, and a confidence breakdown. Missing fields are represented by `null`, never guessed.

## Verification

```bash
pytest -q
```

## Upstream sources

- [Pointcept](https://github.com/Pointcept/Pointcept)
- [Point Transformer V3](https://github.com/Pointcept/PointTransformerV3)
- [RoadMarkingExtraction](https://github.com/YuePanEdward/RoadMarkingExtraction)
