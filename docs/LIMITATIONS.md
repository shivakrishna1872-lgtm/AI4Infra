# Limitations & edge cases

Honest inventory of what this pipeline does **not** do yet, why, and what is
planned (see [ENHANCEMENT.md](ENHANCEMENT.md) for the roadmap).

---

## Detection quality

| Limitation | Detail | Impact |
| --- | --- | --- |
| Overhead conductors | The heuristic chain detector does not fire on thin, sagged spans at mobile-LiDAR sampling (0 detections on the 400 m simulation; the ground truth table proves it). | `overhead_conductor` assets are effectively unavailable on the default backend. Needs the learned PTv3 prior or a dedicated catenary fitter. |
| Tile-boundary instances | Instances crossing tile boundaries split into two assets; `LIKELY_DUPLICATE` QC flags nearby pieces but no cross-tile merge exists yet. | Counts can over-report; precision is conservative. |
| Instance granularity | Long assets (guardrail, pavement, markings) are extracted per tile component, not as one object. | Object counts differ from "one rail = one asset" expectations; see [VALIDATION.md](VALIDATION.md) §4. |
| Small / partial objects | Objects below `min_asset_points` or heavily occluded are dropped (`BELOW_MIN_POINTS`). | Low-signal poles/signs can be missed. |
| Markings on uniform surfaces | Marking detection requires relative brightness/intensity contrast; a uniformly bright surface never triggers, and faded paint can be missed. | Older roads with low-retroreflective paint under-report. |

## Sensor & data

| Limitation | Detail |
| --- | --- |
| Airborne vs mobile | Public USGS 3DEP data is airborne: different density, geometry (roofs, vegetation), and no pole-face returns. Detectors tuned for mobile corridors will behave differently; results on 3DEP data must be labeled `EXTERNAL REAL DATA`, never competition data. |
| Occlusion | Shadows behind poles, vehicles, and vegetation cause missing points; the pipeline cannot hallucinate what the scanner never saw. |
| Thin wires & conductors | Sub-decimeter structures at 0.25–0.35 m sampling are near the detection floor even with learned backends. |
| Intensity scale drift | Intensity is sensor/age-dependent; detectors use relative contrast, not absolute thresholds, but cross-sensor generalization is not guaranteed. |
| Returns | Return-number features are read and preserved, but multi-return fusion is not yet used by the detectors. |
| Two-head/run fusion | `gps_time` run splitting and `point_source_id` scanner attribution are exported, but runs are not yet fused for cross-run verification (see ENHANCEMENT roadmap). |

## Software & environment

| Limitation | Detail |
| --- | --- |
| LAZ backend | LAZ reading requires the `lazrs` Python package (`pip install -e '.[dev]'` installs it). Without it, `.laz` uploads fail with a clear "LAZ backend missing" error instead of a crash. |
| GPU is optional | The default backend is pure geometry (CPU-safe). Pointcept/PTv3 needs CUDA + Pointcept's CUDA ops; FlashAttention is optional upstream (`enable_flash=False` + smaller patch sizes). |
| CRS | When the LAS header has no CRS, the pipeline says `CRS: UNRESOLVED` and never guesses. Georeferencing to a projected CRS is the user's responsibility before upload. |
| Viewer decimation | The web viewer renders up to `viewer_point_limit` (default 250k) sampled points; exports always contain full instance metadata, but the viewer is a decimated representation. |
| Server jobs | Processing jobs are persisted to disk, so a backend restart cannot orphan them; a job that stops updating for 90 s is reported as failed (restart mid-job) rather than hanging. |

## What is deliberately not faked

- No mock AI results: every asset is derived from measured points.
- No assumed CRS, intensity, or RGB: absent measurements are `null`.
- Simulation is always labeled `SIMULATION / DEMO DATA`; external data is labeled
  `EXTERNAL REAL DATA`; only the actual competition dataset may be labeled
  `COMPETITION DATA`.