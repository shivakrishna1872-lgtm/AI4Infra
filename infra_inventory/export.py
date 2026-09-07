"""Export layer: JSON, CSV, GeoJSON, per-asset files, run metadata, and reports.

Output layout (mirrors the competition deliverable contract):

    output/
    ├── assets.json          # full inventory (all attributes)
    ├── assets.csv           # tabular summary
    ├── assets.geojson       # point features at asset centroids (CRS preserved)
    ├── run.json             # input provenance + warnings + backend info
    ├── inventory.json       # {run, qc, inventory} wrapper (consumable by agencies)
    ├── assets/              # one JSON file per asset
    ├── tiles/               # per-tile LAS files (the actual processed units)
    ├── predictions/         # Pointcept prediction exports (when the backend runs)
    ├── reports/             # summary.md + qc_report.json
    └── viewer/              # dependency-free 3D viewer + viewer-data.json
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .models import Asset, RunSummary

CSV_FIELDS = [
    "asset_id", "class", "subclass", "confidence", "point_count", "source_tile",
    "source_run", "source_scanner", "detection_method", "x", "y", "z",
    "length_m", "width_m", "height_m", "orientation_deg", "crs", "qc_flags",
    "condition", "recommended_action", "review_required",
]


def _cap_viewer_payload(viewer_data: Dict[str, object], max_bytes: int) -> Dict[str, object]:
    """Shrink the browser payload below ``max_bytes`` if needed.

    The pipeline already voxel-downsamples the background cloud and slims asset
    records, but a pathological input (huge extents, many assets) can still
    exceed the cap. This thins the cloud (and, as a last resort, drops the
    auxiliary RGB/intensity arrays and thins per-asset highlight evidence) until
    the serialized payload fits - the viewer must load in one fetch, never
    freeze on "Loading viewer data".
    """
    if max_bytes <= 0:
        return viewer_data
    serialized = json.dumps(viewer_data)
    if len(serialized.encode("utf-8")) <= max_bytes:
        return viewer_data
    points = viewer_data.get("points")
    if isinstance(points, list) and points:
        # Keep the originals so every pass re-thins from the full arrays with a
        # growing stride (thinning already-thinned lists would misalign them).
        originals = {"points": points}
        for key in ("point_colors", "point_rgb", "point_intensity", "point_class"):
            values = viewer_data.get(key)
            if isinstance(values, list) and len(values) == len(points):
                originals[key] = values
        stride = 2
        for _ in range(60):
            serialized = json.dumps(viewer_data)
            if len(serialized.encode("utf-8")) <= max_bytes:
                return viewer_data
            # Next stride from the measured ratio, so the loop converges in a
            # handful of passes instead of one overshoot per call.
            needed = len(serialized) / max(max_bytes, 1)
            stride = max(stride + 1, math.ceil(needed * 1.05))
            if stride > len(points):
                break
            for key, values in originals.items():
                viewer_data[key] = values[::stride]
    # Still over: drop the auxiliary per-point arrays (elevation colors are
    # always shipped), then thin per-asset highlight evidence.
    for key in ("point_rgb", "point_intensity", "point_class"):
        viewer_data.pop(key, None)
    serialized = json.dumps(viewer_data)
    if len(serialized.encode("utf-8")) <= max_bytes:
        return viewer_data
    assets = viewer_data.get("assets")
    if isinstance(assets, list) and assets:
        stride = 2
        while stride <= 64 and len(json.dumps(viewer_data).encode("utf-8")) > max_bytes:
            for asset in assets:
                geometry = asset.get("geometry")
                points_ = geometry.get("highlight_points") if isinstance(geometry, dict) else None
                if isinstance(points_, list) and points_:
                    geometry["highlight_points"] = points_[::stride]
            stride *= 2
    return viewer_data


def write_outputs(
    output: Path,
    summary: RunSummary,
    qc_report: List[dict],
    viewer: Dict[str, object],
    viewer_payload_max_bytes: int = 9_000_000,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    _ensure_subdirs(output)
    inventory = [asset.to_dict() for asset in summary.assets]
    # The flat JSON inventory files are the agency-consumable tabular artifact:
    # full per-point evidence (highlight_points) inflates them past 100 MB on
    # real tiles and duplicates what the per-asset files and viewer payload
    # already carry, so it is stripped here and kept there.
    flat_inventory = [_strip_evidence(asset) for asset in inventory]

    run = {
        "input_path": summary.input_path,
        "point_count": summary.point_count,
        "bounds": summary.bounds,
        "crs": summary.crs,
        "las_version": summary.las_version,
        "point_format": summary.point_format,
        "scanner_ids": summary.scanner_ids,
        "run_count": summary.run_count,
        "tile_count": summary.tile_count,
        "elapsed_seconds": round(summary.elapsed_seconds, 2),
        "warnings": summary.warnings,
        "backend": summary.backend,
        "processing_version": summary.processing_version,
    }
    qc_summary = {
        "total_assets": len(inventory),
        "flagged_assets": sum(1 for entry in qc_report if entry["flagged"]),
        "flags": _flag_counts(qc_report),
    }

    # --- Flat inventory files ------------------------------------------------
    (output / "assets.json").write_text(json.dumps(flat_inventory, indent=2), encoding="utf-8")
    _write_csv(output / "assets.csv", inventory)
    _write_geojson(output / "assets.geojson", inventory)
    (output / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    (output / "inventory.json").write_text(
        json.dumps({"run": run, "qc": qc_summary, "inventory": flat_inventory}, indent=2), encoding="utf-8"
    )

    # --- Per-asset files ------------------------------------------------------
    for asset in inventory:
        (output / "assets" / f"{asset['asset_id']}.json").write_text(
            json.dumps(asset, indent=2), encoding="utf-8"
        )

    # --- Reports ----------------------------------------------------------------
    (output / "reports").mkdir(parents=True, exist_ok=True)
    # QC report is a routing artifact, not a second full inventory: one line per
    # asset with its flags and ALP routing (asset_id, class, confidence,
    # condition, review). Built from the post-assessment inventory so condition
    # and review_required are present.
    slim_qc = [
        {"asset_id": entry.get("asset_id"), "class": entry.get("class"),
         "confidence": entry.get("confidence"), "condition": entry.get("condition"),
         "review_required": entry.get("review_required"), "qc_flags": entry.get("qc_flags", [])}
        for entry in inventory
    ]
    (output / "reports" / "qc_report.json").write_text(json.dumps(slim_qc, indent=2), encoding="utf-8")
    (output / "reports" / "summary.md").write_text(_summary_markdown(run, inventory, qc_report), encoding="utf-8")

    # --- Viewer data -------------------------------------------------------------
    # The browser payload must stay small: assets keep all metadata, but their
    # source-point evidence (highlight_points) is capped. Full evidence stays in
    # assets.json / inventory.json / per-asset files - the viewer only needs a
    # representative sample for highlighting.
    viewer_data = {
        "run": run,
        "assets": _viewer_assets(inventory, max_highlight_points=24, global_cap=120_000),
        "points": viewer.get("points", []),
        "point_colors": viewer.get("point_colors", []),
        "point_rgb": viewer.get("point_rgb"),
        "point_intensity": viewer.get("point_intensity"),
        "point_class": viewer.get("point_class"),
        "point_class_names": viewer.get("point_class_names", []),
    }
    viewer_data = _cap_viewer_payload(viewer_data, viewer_payload_max_bytes)
    (output / "viewer-data.json").write_text(json.dumps(viewer_data), encoding="utf-8")


def _strip_evidence(asset: dict) -> dict:
    """Return the asset without geometry.highlight_points (evidence lists).

    Flat exports keep every attribute, stat, source index and ID — only the
    per-point coordinate evidence is removed; full evidence remains available in
    the per-asset files under ``assets/`` and in ``viewer-data.json``.
    """
    copy = dict(asset)
    geometry = asset.get("geometry")
    if isinstance(geometry, dict) and geometry.get("highlight_points"):
        slim = dict(geometry)
        slim.pop("highlight_points", None)
        copy["geometry"] = slim
    return copy


#: Fields the 3D viewer + inspector actually render. Everything else (per-asset
#: source-point index samples, intensity/rgb stats, raw scanner ids, ...) stays in
#: assets.json / per-asset files - shipping it to the browser is what turned
#: viewer-data.json into tens of megabytes and stalled "Loading viewer data" on
#: real scans. Keep this list tight on purpose.
_VIEWER_ASSET_FIELDS = (
    "asset_id", "class", "subclass", "center", "bounding_box", "dimensions",
    "point_count", "source_tile", "coordinate_reference_system", "confidence",
    "confidence_factors", "confidence_explanation", "detection_method",
    "orientation_deg", "source_run", "source_scanner", "model_prior_class",
    "model_confidence", "processing_version", "geometry", "qc_flags", "flagged",
    "condition", "recommended_action", "review_required", "assessment_reasoning",
)


def _viewer_assets(inventory: List[dict], max_highlight_points: int, global_cap: int) -> List[dict]:
    """Assets for the viewer payload: whitelisted fields + bounded highlight evidence.

    The full record (including ``source_point_indices_sample`` - up to 256 tile
    indices per asset) stays in the flat exports and per-asset files; the viewer
    gets only what it draws and the inspector shows.
    """
    capped: List[tuple[dict, Optional[List[Any]]]] = []
    total = 0
    for asset in inventory:
        slim = {key: asset.get(key) for key in _VIEWER_ASSET_FIELDS}
        geometry = slim.get("geometry")
        points = geometry.get("highlight_points") if isinstance(geometry, dict) else None
        sample = None
        if points:
            stride = max(1, math.ceil(len(points) / max_highlight_points))
            sample = points[::stride][:max_highlight_points]
            total += len(sample)
        capped.append((slim, sample))
    # If the combined evidence still exceeds the payload budget, thin every
    # asset's sample uniformly (still representative, still ordered).
    global_stride = max(1, math.ceil(total / global_cap)) if total > global_cap else 1
    result: List[dict] = []
    for asset, sample in capped:
        if sample:
            geometry = dict(asset["geometry"])
            geometry["highlight_points"] = sample[::global_stride]
            asset["geometry"] = geometry
        result.append(asset)
    return result


def _ensure_subdirs(output: Path) -> None:
    for name in ("assets", "tiles", "predictions", "reports"):
        (output / name).mkdir(parents=True, exist_ok=True)


def _flag_counts(qc_report: List[dict]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for entry in qc_report:
        for flag in entry.get("qc_flags", []):
            counts[flag] = counts.get(flag, 0) + 1
    return dict(sorted(counts.items()))


def _write_csv(path: Path, inventory: List[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for asset in inventory:
            row = {key: asset.get(key) for key in CSV_FIELDS}
            center = asset.get("center") or {}
            row["x"], row["y"], row["z"] = center.get("x"), center.get("y"), center.get("z")
            dimensions = asset.get("dimensions") or {}
            row["length_m"] = dimensions.get("length_m")
            row["width_m"] = dimensions.get("width_m")
            row["height_m"] = dimensions.get("height_m")
            row["qc_flags"] = "|".join(asset.get("qc_flags") or [])
            writer.writerow(row)


def _write_geojson(path: Path, inventory: List[dict]) -> None:
    features = []
    for asset in inventory:
        properties = dict(asset)
        center = properties.pop("center")
        properties.pop("geometry", None)
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [center["x"], center["y"], center["z"]]},
            "properties": properties,
        })
    (path).write_text(
        json.dumps({"type": "FeatureCollection", "features": features}, indent=2), encoding="utf-8"
    )


def _summary_markdown(run: dict, inventory: List[dict], qc_report: List[dict]) -> str:
    lines = [
        "# AI4Infra processing report",
        "",
        f"- Input: `{run['input_path']}`",
        f"- Points: {run['point_count']:,}",
        f"- CRS: {run.get('crs') or 'CRS_UNRESOLVED'}",
        f"- LAS {run['las_version']} / Point format {run['point_format']}",
        f"- Tiles: {run['tile_count']}",
        f"- Scanners: {run.get('scanner_ids') or 'unknown'}",
        f"- Runs: {run.get('run_count') or 'unknown'}",
        f"- Backend: {run['backend'].get('name') if isinstance(run['backend'], dict) else run['backend']}",
        f"- Processing version: {run['processing_version']}",
        f"- Elapsed: {run['elapsed_seconds']}s",
        "",
        "## Assets",
        "",
        "| Class | Count |",
        "| --- | --- |",
    ]
    counts: Dict[str, int] = {}
    for asset in inventory:
        counts[asset["class"]] = counts.get(asset["class"], 0) + 1
    for asset_class in sorted(counts):
        lines.append(f"| {asset_class} | {counts[asset_class]} |")
    lines += ["", "## ALP assessment (condition / review routing)", ""]
    conditions: Dict[str, int] = {}
    reviews = 0
    for asset in inventory:
        condition = asset.get("condition") or "UNRATED"
        conditions[condition] = conditions.get(condition, 0) + 1
        if asset.get("review_required"):
            reviews += 1
    lines.append("| Condition | Count |")
    lines.append("| --- | --- |")
    for condition in sorted(conditions):
        lines.append(f"| {condition} | {conditions[condition]} |")
    lines.append(f"- Sent to human review: {reviews}")

    lines += ["", "## Quality control", ""]
    flagged = [entry for entry in qc_report if entry["flagged"]]
    lines.append(f"- Flagged assets: {len(flagged)} / {len(inventory)}")
    for entry in flagged:
        lines.append(
            f"- `{entry['asset_id']}` ({entry['class']}): "
            + ", ".join(entry.get("qc_flags") or [])
        )
    lines.append("")
    lines.append("_Generated by AI4Infra; every value is measured from the input LAS, never guessed._")
    return "\n".join(lines)


def write_warning_file(output: Path, message: str) -> None:
    (output / "reports" / "warnings.txt").write_text(message, encoding="utf-8")