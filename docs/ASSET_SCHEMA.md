# Asset Inventory Schema

Every asset produced by the pipeline is an **individual instance** with measured
geometry, source provenance, and transparent confidence. Missing measurements are
exported as `null` — never guessed.

This document describes the on-disk schema (`assets.json`, `inventory.json`,
`run.json`) and its mapping to the Advanced Track competition schema.

---

## 1. Asset record (`assets.json` → array of records)

| Field | Type | Description |
| --- | --- | --- |
| `asset_id` | string | Unique instance id, e.g. `POL-00017`. Prefix per class: `PAV`, `MRK`, `POL`, `CON`, `CAB`, `SGN`, `GRD`, `BAR`, `RUM`. |
| `class` | string | Taxonomy class (below). |
| `subclass` | string \| null | Class-specific subtype (below). |
| `center` | object | `{ "x": float, "y": float, "z": float }` — centroid of the instance's source points, **in the source CRS** (no reprojection). |
| `bounding_box` | number[6] | `[min_x, min_y, min_z, max_x, max_y, max_z]`, source CRS. |
| `dimensions` | object | `{ "length_m", "width_m", "height_m" }` from eigen/axis-aligned metrics. |
| `point_count` | int | Number of source points supporting the instance. |
| `source_tile` | string | Tile file name the points were read from (`tiles/…`). |
| `source_point_indices_sample` | int[] | Up to `point_index_sample_limit` (256) **tile-local** point indices (see tile manifest for order). |
| `coordinate_reference_system` | string \| null | `EPSG:XXXX` read from the LAS header; `null` ⇒ `CRS: UNRESOLVED`. Never assumed. |
| `confidence` | float | 0..1 overall score. |
| `confidence_factors` | object | `model`, `geometry`, `support`, `spatial_context`, `class_consistency` (each float or `null`). |
| `confidence_explanation` | string | Human-readable account of the score. |
| `detection_method` | string | e.g. `geometry-v1`, `pointcept-geometry-v1`, `roadmarkingextraction-v1`. |
| `intensity_stats` | object \| null | `min/max/mean/std` of intensity on the instance points. |
| `rgb_stats` | object \| null | Same stats over RGB (16-bit values), `null` when the file has no color. |
| `orientation_deg` | float \| null | Bearing of the dominant axis (degrees). |
| `source_run` | string \| null | `Run 1` / `Run 2` … from GPS-time separation when separable. |
| `source_scanner` | string \| null | `Laser Left` / `Laser Right` … from `point_source_id`. |
| `source_point_source_id` | int \| null | Raw scanner id. |
| `model_prior_class` | string \| null | Class name from the learned backend when used. |
| `model_confidence` | float \| null | Model factor of the confidence blend. |
| `processing_version` | string | Pipeline version. |
| `geometry` | object \| null | `ground_elevation_m`, eigen metrics (`eigen_planarity/linearity/verticality`), and `highlight_points` — the exact 3D points (source CRS) that prove this instance in the viewer. |
| `qc_flags` | string[] | `LOW_CONFIDENCE`, `LIKELY_DUPLICATE`, `UNUSUAL_DIMENSIONS`, `INVALID_GEOMETRY`, `BELOW_MIN_POINTS`. |
| `flagged` | bool | `true` when any QC flag is set. |
| `condition` | string \| null | ALP condition: `GOOD` / `FAIR` / `POOR` / `REVIEW` — from calibrated confidence bands + class signals (see `docs/ALP.md`). |
| `recommended_action` | string \| null | ALP action for the owner (restripe / verify / inspect …); `null` before assessment. |
| `review_required` | bool | `true` when confidence < calibrated review threshold (0.80) or a structural QC flag is set — send to human review. |
| `assessment_reasoning` | string \| null | The Observation→Interpretation trail: which measured signals fired and why (never an unmeasured cause). |

### Taxonomy

| Class | Subclasses |
| --- | --- |
| `pavement` | `travelled_surface` |
| `pavement_marking` | `lane_line`, `edge_line`, `stop_line`, `crosswalk`, `symbol`, `other` |
| `utility_pole` | `vertical_support` |
| `overhead_conductor` | `conductor` |
| `utility_cabinet` | `cabinet` |
| `traffic_sign` | `panel_with_support`, `panel_only`, `sign_support` |
| `guardrail` | `roadside_barrier` |
| `safety_barrier` | `concrete_barrier` |
| `rumble_strip` | `rumble_strip` |

## 2. Mapping to the competition schema

| Competition field | This schema |
| --- | --- |
| `asset_id` | `asset_id` (same format, `UTIL-00017` ⇒ `POL-00017`) |
| `asset_class` | `class` |
| `asset_type` | `subclass` (see taxonomy) |
| `location {x,y,z}` | `center` |
| `crs` | `coordinate_reference_system` |
| `geometry.centroid` | `center` (as `[x,y,z]`) |
| `geometry.bbox` | `bounding_box` |
| `geometry.height/width/length` | `dimensions` |
| `point_count` | `point_count` |
| `attributes.intensity_mean` | `intensity_stats.mean` |
| `confidence` | `confidence` |
| `source.run` | `source_run` |
| `source.scanner` | `source_scanner` |
| `source.tile` | `source_tile` |
| `detection_method` | `detection_method` |
| `source_points` | `source_point_indices_sample` (sampled; full provenance at tile level) |

## 3. CRS rules

1. CRS is read from the LAS/LAZ header (`parse_crs`) and preserved verbatim through
   processing, the viewer payload, and every export.
2. Coordinates are **never** reprojected or silently converted. If a CRS is absent
   from the header, the record carries `null` and the UI/export show `CRS: UNRESOLVED`.
3. `run.json` records the same CRS once per run.

## 4. Other artifacts

- `inventory.json` — run-level document: project id, input file, point count,
  bounds, CRS, per-class counts, backend provenance, warnings, elapsed time.
  bounds, CRS, per-class counts, backend provenance, warnings, elapsed time.
- `run.json` — full run summary consumed by the web UI (`points`, `bounds`,
  `las_version`, `point_format`, `scanner_ids`, `run_count`, `tile_count`,
  `warnings`, `backend`, `processing_version`).
- `assets.csv` — flattened one-row-per-asset export (same fields).
- `assets.geojson` — `FeatureCollection`; each feature has `Point` geometry at the
  asset center and the full record as `properties` (CRS preserved in the
  record's `coordinate_reference_system` property).
- `viewer-data.json` — decimated point cloud (`points`, `point_rgb`,
  `point_intensity`, `point_class` + `point_class_names`) plus `assets` and
  `run`, served to the 3D viewer. The payload is slim by design: ≤ 120k
  overview points and whitelisted asset records (only what the scene and
  inspector render; highlight evidence capped at 24 points/asset) — heavy
  provenance (`source_point_indices_sample`, intensity/RGB stats, full
  evidence) stays in `assets.json` / per-asset files, never in the browser
  payload. The server serves it gzip-compressed.

## 5. MongoDB document schema

When `MONGO_URI` is set (or `--mongo-uri` passed), each asset is mirrored to
MongoDB as one GeoJSON-aware document (`terra_point.assets` by default; a
`2dsphere` index is created on `location`):

```json
{
  "asset_id": "POL-00017",
  "category": "Utilities",
  "subcategory": "Utility Pole",
  "location": {"type": "Point", "coordinates": [613004.21, 2447918.54, 318.42]},
  "attributes": {
    "height_m": 10.2,
    "width_m": 0.32,
    "length_m": 0.32,
    "lean_angle_deg": 3.6,
    "orientation_deg": 42.0,
    "run_source": "Run 1 Laser Left",
    "system_source": "Trimble MX9 - Run 1 Laser Left",
    "condition": "GOOD",
    "condition_flag": "GOOD",
    "recommended_action": "...",
    "point_count": 840,
    "avg_intensity": 138
  },
  "confidence": 0.94,
  "review_required": false,
  "confidence_factors": {},
  "detection_method": "geometry-v2-heuristic",
  "qc_flags": [],
  "geometry": {"bbox": [...], "ground_elevation_m": 310.1},
  "crs": "EPSG:2248",
  "source_run": "1",
  "source_scanner": "Laser Left",
  "source_tile": "tile_15325_61198",
  "source_file": "/data/mannford_run1_left.las",
  "processing_version": "0.3.0",
  "processed_at": "2026-09-06T12:00:00+00:00"
}
```

Rules:

1. `category` maps the taxonomy to the four competition categories:
   Pavement (pavement, pavement_marking), Utilities (utility_pole,
   overhead_conductor, utility_cabinet), Signs (traffic_sign), Safety
   (guardrail, safety_barrier, rumble_strip).
2. `location.coordinates` is `[x, y, z]` = `[easting, northing, elevation]`
   **in the source CRS** (preserved verbatim in `crs`; never reprojected).
3. `lean_angle_deg` is the ALP geometric proxy derived from
   `geometry.eigen_verticality` (0° = perfectly vertical); `null` when the
   eigen metrics are absent — never guessed.
4. `system_source` and `condition_flag` are the Trimble MX9 competition
   schema aliases for `run_source` and `condition` respectively (both are
   written so either consumer works); `avg_intensity` mirrors
   `intensity_stats.mean` when intensity is present.
4. Prior documents of the same `source_file` are replaced on export, so the
   collection always mirrors the latest run of each input.
5. The mirror is optional: export failures become run warnings and never
   affect the canonical JSON/CSV/GeoJSON exports.