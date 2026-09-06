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
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .models import Asset, RunSummary

CSV_FIELDS = [
    "asset_id", "class", "subclass", "confidence", "point_count", "source_tile",
    "source_run", "source_scanner", "detection_method", "x", "y", "z",
    "length_m", "width_m", "height_m", "orientation_deg", "crs", "qc_flags",
]


def write_outputs(
    output: Path,
    summary: RunSummary,
    qc_report: List[dict],
    points: List[Sequence[float]],
    point_colors: List[Sequence[float]],
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    _ensure_subdirs(output)
    inventory = [asset.to_dict() for asset in summary.assets]

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
    (output / "assets.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    _write_csv(output / "assets.csv", inventory)
    _write_geojson(output / "assets.geojson", inventory)
    (output / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    (output / "inventory.json").write_text(
        json.dumps({"run": run, "qc": qc_summary, "inventory": inventory}, indent=2), encoding="utf-8"
    )

    # --- Per-asset files ------------------------------------------------------
    for asset in inventory:
        (output / "assets" / f"{asset['asset_id']}.json").write_text(
            json.dumps(asset, indent=2), encoding="utf-8"
        )

    # --- Reports ----------------------------------------------------------------
    (output / "reports").mkdir(parents=True, exist_ok=True)
    (output / "reports" / "qc_report.json").write_text(json.dumps(qc_report, indent=2), encoding="utf-8")
    (output / "reports" / "summary.md").write_text(_summary_markdown(run, inventory, qc_report), encoding="utf-8")

    # --- Viewer data -------------------------------------------------------------
    viewer_data = {
        "run": run,
        "assets": inventory,
        "points": points,
        "point_colors": point_colors,
    }
    (output / "viewer-data.json").write_text(json.dumps(viewer_data), encoding="utf-8")


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