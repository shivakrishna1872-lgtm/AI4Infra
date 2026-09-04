from __future__ import annotations

import csv
import json
import math
import shutil
from collections import Counter, deque
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import laspy
import numpy as np

from .models import Asset, ProcessingSettings, RunSummary
from .pointcept import describe_backend, run_pointcept_test
from .viewer import write_viewer

VERSION = "0.1.0"


def _stats(values: np.ndarray) -> Optional[Dict[str, float]]:
    if not len(values):
        return None
    return {"min": float(values.min()), "mean": float(values.mean()), "max": float(values.max())}


def _bounds(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[float, float, float, float, float, float]:
    return (float(x.min()), float(y.min()), float(z.min()), float(x.max()), float(y.max()), float(z.max()))


def _orientation(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    if len(x) < 3:
        return None
    covariance = np.cov(np.column_stack((x, y)), rowvar=False)
    values, vectors = np.linalg.eigh(covariance)
    if values[-1] <= 0:
        return None
    vector = vectors[:, -1]
    return float((math.degrees(math.atan2(vector[1], vector[0])) + 360) % 180)


def _components(cells: Iterable[Tuple[int, int]]) -> List[List[Tuple[int, int]]]:
    remaining = set(cells)
    groups: List[List[Tuple[int, int]]] = []
    while remaining:
        start = remaining.pop()
        group = [start]
        queue = deque([start])
        while queue:
            cell = queue.popleft()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                candidate = (cell[0] + dx, cell[1] + dy)
                if candidate in remaining:
                    remaining.remove(candidate)
                    group.append(candidate)
                    queue.append(candidate)
        groups.append(group)
    return groups


def _cell_components(x: np.ndarray, y: np.ndarray, mask: np.ndarray, resolution: float, min_cells: int) -> List[np.ndarray]:
    selected = np.flatnonzero(mask)
    if not len(selected):
        return []
    coordinates = np.floor(np.column_stack((x[selected], y[selected])) / resolution).astype(np.int64)
    index_by_cell: Dict[Tuple[int, int], List[int]] = {}
    for index, cell in zip(selected.tolist(), coordinates.tolist()):
        index_by_cell.setdefault((cell[0], cell[1]), []).append(index)
    return [
        np.asarray([index for cell in group for index in index_by_cell[cell]], dtype=np.int64)
        for group in _components(index_by_cell)
        if len(group) >= min_cells
    ]


def _rgb(chunk: laspy.ScaleAwarePointRecord) -> Optional[np.ndarray]:
    names = set(chunk.point_format.dimension_names)
    if not {"red", "green", "blue"}.issubset(names):
        return None
    return np.column_stack((np.asarray(chunk.red), np.asarray(chunk.green), np.asarray(chunk.blue))).astype(np.float64)


def _asset(
    *, asset_id: str, asset_class: str, subclass: Optional[str], indices: np.ndarray, x: np.ndarray, y: np.ndarray,
    z: np.ndarray, intensity: np.ndarray, rgb: Optional[np.ndarray], source_indices: np.ndarray, tile: str,
    crs: Optional[str], geometry_score: float, support_score: float, explanation: str, method: str,
) -> Asset:
    box = _bounds(x[indices], y[indices], z[indices])
    dimensions = {"length_m": box[3] - box[0], "width_m": box[4] - box[1], "height_m": box[5] - box[2]}
    factors = {"model": None, "geometry": round(geometry_score, 3), "support": round(support_score, 3)}
    confidence = round(max(0.0, min(1.0, 0.55 * geometry_score + 0.45 * support_score)), 3)
    return Asset(
        asset_id=asset_id,
        asset_class=asset_class,
        subclass=subclass,
        center={"x": float(x[indices].mean()), "y": float(y[indices].mean()), "z": float(z[indices].mean())},
        bounding_box=box,
        dimensions=dimensions,
        point_count=int(len(indices)),
        source_tile=tile,
        source_point_indices_sample=[int(item) for item in source_indices[indices][:128]],
        coordinate_reference_system=crs,
        confidence=confidence,
        confidence_factors=factors,
        confidence_explanation=explanation,
        detection_method=method,
        intensity_stats=_stats(intensity[indices]),
        rgb_stats=({"red_mean": float(rgb[indices, 0].mean()), "green_mean": float(rgb[indices, 1].mean()), "blue_mean": float(rgb[indices, 2].mean())} if rgb is not None else None),
        orientation_deg=_orientation(x[indices], y[indices]),
        processing_version=VERSION,
    )


def _detect_tile(
    *, tile: str, x: np.ndarray, y: np.ndarray, z: np.ndarray, intensity: np.ndarray, rgb: Optional[np.ndarray],
    source_indices: np.ndarray, crs: Optional[str], id_counts: Counter, settings: ProcessingSettings,
) -> List[Asset]:
    if len(x) < 30:
        return []
    assets: List[Asset] = []
    ground = float(np.quantile(z, 0.12))
    height = z - ground

    pavement = height <= 0.28
    if int(pavement.sum()) >= settings.min_pavement_points:
        id_counts["PAV"] += 1
        assets.append(_asset(
            asset_id=f"PAV-{id_counts['PAV']:05d}", asset_class="pavement", subclass="travelled_surface",
            indices=np.flatnonzero(pavement), x=x, y=y, z=z, intensity=intensity, rgb=rgb, source_indices=source_indices,
            tile=tile, crs=crs, geometry_score=0.62, support_score=min(0.92, 0.45 + math.log10(pavement.sum()) / 8),
            explanation="Low-elevation surface has continuous point support within this spatial tile.", method="geometry-v1",
        ))

    bright_threshold = float(np.quantile(intensity, 0.88))
    bright = intensity >= bright_threshold
    if rgb is not None:
        brightness = rgb.mean(axis=1)
        bright |= brightness >= float(np.quantile(brightness, 0.88))
    marking_mask = pavement & bright
    for component in _cell_components(x, y, marking_mask, 0.35, 3):
        if len(component) < 20:
            continue
        id_counts["MRK"] += 1
        assets.append(_asset(
            asset_id=f"MRK-{id_counts['MRK']:05d}", asset_class="pavement_marking", subclass="reflective_marking",
            indices=component, x=x, y=y, z=z, intensity=intensity, rgb=rgb, source_indices=source_indices, tile=tile, crs=crs,
            geometry_score=0.62, support_score=min(0.95, 0.4 + math.log10(len(component)) / 7),
            explanation="Near-ground connected component has locally high reflectance or brightness.", method="native-roadmarking-v1",
        ))

    for component in _cell_components(x, y, height >= 0.0, 0.45, 1):
        if len(component) < 40:
            continue
        box = _bounds(x[component], y[component], z[component])
        dx, dy, dz = box[3] - box[0], box[4] - box[1], box[5] - box[2]
        if dz >= 3.0 and max(dx, dy) <= 1.35:
            id_counts["POL"] += 1
            assets.append(_asset(
                asset_id=f"POL-{id_counts['POL']:05d}", asset_class="utility_pole", subclass="vertical_support",
                indices=component, x=x, y=y, z=z, intensity=intensity, rgb=rgb, source_indices=source_indices, tile=tile, crs=crs,
                geometry_score=min(0.95, 0.58 + min(dz / 20, 0.25)), support_score=min(0.95, 0.4 + math.log10(len(component)) / 6),
                explanation="Narrow footprint and multi-metre vertical extent satisfy pole geometry rules.", method="geometry-v1",
            ))

    rail_mask = (height > 0.3) & (height < 1.7)
    for component in _cell_components(x, y, rail_mask, 0.5, 7):
        if len(component) < 60:
            continue
        box = _bounds(x[component], y[component], z[component])
        dx, dy, dz = box[3] - box[0], box[4] - box[1], box[5] - box[2]
        lateral = min(dx, dy)
        longitudinal = max(dx, dy)
        if longitudinal >= 6 and lateral <= 1.8 and dz <= 1.8:
            id_counts["GRD"] += 1
            assets.append(_asset(
                asset_id=f"GRD-{id_counts['GRD']:05d}", asset_class="guardrail", subclass="roadside_barrier",
                indices=component, x=x, y=y, z=z, intensity=intensity, rgb=rgb, source_indices=source_indices, tile=tile, crs=crs,
                geometry_score=0.7, support_score=min(0.92, 0.42 + math.log10(len(component)) / 7),
                explanation="Long, low, narrow connected structure is consistent with roadside barrier geometry.", method="geometry-v1",
            ))
    return assets


def _crs(reader: laspy.LasReader) -> Optional[str]:
    try:
        parsed = reader.header.parse_crs()
        return parsed.to_string() if parsed else None
    except Exception:
        return None


def _write_outputs(output: Path, summary: RunSummary, points: Sequence[Sequence[float]]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    inventory = [asset.to_dict() for asset in summary.assets]
    (output / "assets.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    run = {
        "input_path": summary.input_path, "point_count": summary.point_count, "bounds": summary.bounds, "crs": summary.crs,
        "las_version": summary.las_version, "point_format": summary.point_format, "warnings": summary.warnings,
        "backend": summary.backend, "processing_version": VERSION,
    }
    (output / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    (output / "viewer-data.json").write_text(json.dumps({"run": run, "assets": inventory, "points": points}), encoding="utf-8")
    features = []
    for asset in inventory:
        properties = dict(asset)
        center = properties.pop("center")
        properties.pop("geometry", None)
        features.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [center["x"], center["y"], center["z"]]}, "properties": properties})
    (output / "assets.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": features}), encoding="utf-8")
    fields = ["asset_id", "class", "subclass", "confidence", "point_count", "source_tile", "detection_method"]
    with (output / "assets.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for asset in inventory:
            writer.writerow({key: asset.get(key) for key in fields})
    write_viewer(output / "viewer")
    shutil.copy2(output / "viewer-data.json", output / "viewer" / "viewer-data.json")


def process_las(input_path: str | Path, output_dir: str | Path, settings: Optional[ProcessingSettings] = None) -> RunSummary:
    settings = settings or ProcessingSettings()
    path, output = Path(input_path).expanduser().resolve(), Path(output_dir).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    id_counts: Counter = Counter()
    visual_points: List[List[float]] = []
    warnings: List[str] = []
    with laspy.open(path) as reader:
        header = reader.header
        version = f"{header.version.major}.{header.version.minor}"
        if version != "1.4":
            message = f"LAS {version} supplied; pipeline supports it but the competition source is LAS 1.4."
            if settings.strict_las14:
                raise ValueError(message)
            warnings.append(message)
        if header.point_format.id != 7:
            warnings.append(f"Point Data Record Format {header.point_format.id}; format 7 is expected but supported dimensions will be used.")
        crs = _crs(reader)
        if not crs:
            warnings.append("CRS was not present or could not be parsed from LAS metadata.")
        estimated_points = int(header.point_count)
        sample_stride = max(1, math.ceil(max(estimated_points, 1) / settings.viewer_point_limit))
        global_offset = 0
        all_assets: List[Asset] = []
        for chunk_number, chunk in enumerate(reader.chunk_iterator(settings.chunk_size)):
            x, y, z = np.asarray(chunk.x, dtype=np.float64), np.asarray(chunk.y, dtype=np.float64), np.asarray(chunk.z, dtype=np.float64)
            intensity = np.asarray(chunk.intensity, dtype=np.float64) if "intensity" in chunk.point_format.dimension_names else np.zeros(len(x))
            rgb = _rgb(chunk)
            global_indices = np.arange(global_offset, global_offset + len(x), dtype=np.int64)
            viewer_mask = global_indices % sample_stride == 0
            visual_points.extend(np.column_stack((x[viewer_mask], y[viewer_mask], z[viewer_mask])).round(4).tolist())
            tile_x = np.floor(x / settings.tile_size_m).astype(np.int64)
            tile_y = np.floor(y / settings.tile_size_m).astype(np.int64)
            tile_keys = np.column_stack((tile_x, tile_y))
            for key in np.unique(tile_keys, axis=0):
                mask = (tile_x == key[0]) & (tile_y == key[1])
                tile = f"tile_{key[0]}_{key[1]}"
                all_assets.extend(_detect_tile(tile=tile, x=x[mask], y=y[mask], z=z[mask], intensity=intensity[mask], rgb=(rgb[mask] if rgb is not None else None), source_indices=global_indices[mask], crs=crs, id_counts=id_counts, settings=settings))
            global_offset += len(x)
        backend = describe_backend(settings.pointcept_root)
        if settings.backend == "pointcept":
            if not all((settings.pointcept_root, settings.pointcept_config, settings.pointcept_weight)):
                raise ValueError("Pointcept backend requires --pointcept-root, --pointcept-config, and --pointcept-weight. No class mapping is assumed.")
            backend = run_pointcept_test(root=settings.pointcept_root, config=settings.pointcept_config, weight=settings.pointcept_weight, num_gpus=settings.pointcept_num_gpus, save_path=output / "pointcept")
            warnings.append("Pointcept completed; import predictions only through a project-specific taxonomy adapter. Geometry inventory remains separately attributable.")
        summary = RunSummary(input_path=str(path), point_count=estimated_points, bounds=tuple(float(v) for v in header.mins) + tuple(float(v) for v in header.maxs), crs=crs, las_version=version, point_format=header.point_format.id, assets=all_assets, warnings=warnings, backend=backend)
    _write_outputs(output, summary, visual_points[:settings.viewer_point_limit])
    return summary
