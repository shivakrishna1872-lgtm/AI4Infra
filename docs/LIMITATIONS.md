# Limitations & edge cases

Honest inventory of what this pipeline does **not** do yet, why, and what is
planned (see [ENHANCEMENT.md](ENHANCEMENT.md) for the roadmap).

---

## Detection quality

| Limitation | Detail | Impact |
| --- | --- | --- |
| Overhead conductors | The chain detector fires only when a span keeps at least one detected fragment per tile piece; spans whose points are nearly fully occluded are still missed (0.93 recall on simulated spans; PTv3 prior planned as the gap-closer). | Sparse/occluded spans can be under-counted; see [ACCURACY_REPORT.md](ACCURACY_REPORT.md). |
| Tile-boundary instances | Compact objects split by a tile boundary (a pole standing exactly on the cut) are detected twice and flagged `LIKELY_DUPLICATE` for review; linear conductors are merged back across tiles by the pole-aware merge pass. | Duplicate rows go to human review, not the delivered inventory. |
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
| CRS | When the LAS header has no CRS, the pipeline forces the configured fallback (`EPSG:6553`, Mannford OK competition default) at load time and reports a `CRS_FALLBACK` warning so units stay metres/US-ft-consistent. Set `crs_fallback: null` in `configs/processing.yaml` to restore the old never-guess `CRS_UNRESOLVED` behaviour for non-competition data. |
| Viewer decimation | The web viewer renders up to `viewer_point_limit` (default 120k) uniformly sampled overview points plus whitelisted asset records (capped highlight evidence); exports always contain full instance metadata, but the viewer is a decimated representation. The payload is gzip-served, so even multi-GB scans load quickly. |
| Server jobs | Processing jobs are persisted to disk, so a backend restart cannot orphan them; a job that stops updating for 90 s is reported as failed (restart mid-job) rather than hanging. |
| Disk footprint | A 16M-point tile streams fine but its per-tile LAS intermediates are multi-GB. Web/API runs set `save_tiles=false` and remove tiles after processing (documented in `run.json` warnings); the CLI keeps them for Pointcept work and full provenance. Flat JSON exports (`assets.json`, `inventory.json`) intentionally omit per-point `highlight_points` — full evidence lives in the per-asset files under `assets/` and in `viewer-data.json`. On a nearly full disk the API now fails fast with a clear 507 message instead of `Errno 28` mid-write. |
| ALP geometry proxies | ALP condition signals are *geometric proxies* (e.g. lean from eigen-verticality, sag from span vertical extent). They flag *possible* issues for site verification; they are not survey-grade measurements. |

## What is deliberately not faked

- No mock AI results: every asset is derived from measured points.
- No assumed CRS, intensity, or RGB: absent measurements are `null`.
- Simulation is always labeled `SIMULATION / DEMO DATA`; external data is labeled
  `EXTERNAL REAL DATA`; only the actual competition dataset may be labeled
  `COMPETITION DATA`.