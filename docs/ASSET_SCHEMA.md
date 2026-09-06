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
- `run.json` — full run summary consumed by the web UI (`points`, `bounds`,
  `las_version`, `point_format`, `scanner_ids`, `run_count`, `tile_count`,
  `warnings`, `backend`, `processing_version`).
- `assets.csv` — flattened one-row-per-asset export (same fields).
- `assets.geojson` — `FeatureCollection`; each feature has `Point` geometry at the
  asset center and the full record as `properties` (CRS preserved in the
  record's `coordinate_reference_system` property).
- `viewer-data.json` — decimated point cloud (`points`, `point_rgb`,
  `point_intensity`, `point_class` + `point_class_names`) plus `assets` and
  `run`, served to the 3D viewer.