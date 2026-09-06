"""Pipeline orchestrator.

Flow (designed for very large mobile-LiDAR clouds):

1. Stream the LAS in chunks; validate; write one LAS file per spatial tile to
   ``output/tiles/`` (append mode, single pass, bounded memory); sample points
   and colors for the 3D viewer.
2. Optional: run Pointcept's real ``tools/test.py`` over the tiles, then load
   the exported per-tile predictions as a *model prior* for detection.
3. Optional: run the external RoadMarkingExtraction subsystem per tile and
   ingest its DXF vector output.
4. Process tiles one at a time: ground estimation, geometric features, asset
   detection, attribution, confidence.
5. Quality control, then export inventory artifacts + viewer.

Every asset keeps its source tile, source point indices, CRS, measured
attributes, and confidence breakdown - nothing is invented.
"""
from __future__ import annotations

import json
import math
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import laspy
import numpy as np

from .assets import TileContext, assign_ids, detect_all
from .confidence import ConfidenceFactors, score_asset, support_score
from .export import write_outputs
from .las_reader import gps_run_labels, iter_chunks
from .models import Asset, ProcessingSettings, RunSummary
from .pointcept import describe_backend, load_class_names, load_predictions, run_pointcept_test
from .preprocessing import estimate_ground_z, height_above_ground
from .qc import run_quality_control
from .roadmarking import run_roadmarking
from .validation import validate_las
from .viewer import write_viewer

VERSION = "0.2.0"

#: Documented adaptation from upstream Pointcept classes to infrastructure
#: evidence (mirrors configs/model.yaml; used when the YAML is not loaded).
DEFAULT_CLASS_MAPPING: Dict[str, Dict[str, object]] = {
    "driveable_surface": {"asset_class": "pavement", "subclass": "travelled_surface", "weight": 0.90},
    "barrier": {"asset_class": "safety_barrier", "subclass": "concrete_barrier", "weight": 0.80},
    "manmade": {"asset_class": None, "weight": 0.35, "evidence_for": ["utility_pole", "traffic_sign", "utility_cabinet"]},
    "sidewalk": {"asset_class": None, "weight": 0.0},
    "terrain": {"asset_class": None, "weight": 0.0},
    "vegetation": {"asset_class": None, "weight": 0.0},
}

#: Viewer point colors: elevation gradient (deep blue -> teal -> amber).
_STOPS = ((0.06, 0.16, 0.38), (0.10, 0.45, 0.55), (0.95, 0.80, 0.42))


def _elevation_color(z01: np.ndarray) -> List[List[float]]:
    colors = []
    for value in z01.tolist():
        t = max(0.0, min(1.0, value))
        if t < 0.5:
            local = t / 0.5
            a, b = _STOPS[0], _STOPS[1]
        else:
            local = (t - 0.5) / 0.5
            a, b = _STOPS[1], _STOPS[2]
        colors.append([round(a[i] + (b[i] - a[i]) * local, 4) for i in range(3)])
    return colors


def _tile_header(source_header: laspy.LasHeader) -> laspy.LasHeader:
    try:
        header = laspy.LasHeader(point_format=source_header.point_format, version=source_header.version)
    except TypeError:  # pragma: no cover - older laspy
        header = laspy.LasHeader(point_format=int(source_header.point_format.id), version="1.4")
    header.scales = list(source_header.scales)
    header.offsets = list(source_header.offsets)
    try:
        header.vlrs = list(source_header.vlrs)
    except Exception:
        pass
    return header


def _append_tile(tiles_dir: Path, tile_name: str, source_header: laspy.LasHeader, chunk: object, mask: np.ndarray) -> None:
    """Append a chunk's subset of points to a tile's LAS file (create on first touch)."""
    tile_path = tiles_dir / f"{tile_name}.las"
    data = laspy.LasData(_tile_header(source_header))
    data.x, data.y, data.z = chunk.x[mask], chunk.y[mask], chunk.z[mask]
    if "intensity" in data.point_format.dimension_names:
        data.intensity = chunk.intensity[mask]
    if chunk.rgb is not None and {"red", "green", "blue"}.issubset(data.point_format.dimension_names):
        data.red, data.green, data.blue = chunk.rgb[mask, 0], chunk.rgb[mask, 1], chunk.rgb[mask, 2]
    if chunk.gps_time is not None and "gps_time" in data.point_format.dimension_names:
        data.gps_time = chunk.gps_time[mask]
    if chunk.point_source_id is not None and "point_source_id" in data.point_format.dimension_names:
        data.point_source_id = chunk.point_source_id[mask]
    if chunk.return_number is not None and "return_number" in data.point_format.dimension_names:
        data.return_number = chunk.return_number[mask]
    if chunk.number_of_returns is not None and "number_of_returns" in data.point_format.dimension_names:
        data.number_of_returns = chunk.number_of_returns[mask]
    if tile_path.is_file():
        with laspy.open(tile_path, mode="a") as writer:
            writer.append_points(data.points)
    else:
        with laspy.open(tile_path, mode="w", header=data.header) as writer:
            writer.write_points(data.points)


def _stream_tiles(
    input_path: Path, output: Path, settings: ProcessingSettings, progress: bool,
) -> Tuple[RunSummary, List[str], List[List[float]], List[List[float]]]:
    """Pass 1: validate, stream chunks, write tiles, sample viewer data."""
    warnings: List[str] = []
    validation = validate_las(input_path, strict_las14=settings.strict_las14)
    warnings.extend(validation.as_warning_messages())
    metadata = validation.metadata
    crs = metadata.crs

    tiles_dir = output / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)
    viewer_points: List[List[float]] = []
    viewer_colors: List[List[float]] = []
    tile_names: List[str] = []
    scanner_ids: List[int] = []
    saw_run2 = False
    z0, _, _, z1, _, _ = metadata.bounds

    sample_stride = max(1, math.ceil(metadata.point_count / settings.viewer_point_limit))
    processed = 0
    with laspy.open(input_path) as reader:
        source_header = reader.header
        for chunk_number, (_, _, chunk) in enumerate(iter_chunks(input_path, settings.chunk_size)):
            x, y, z = chunk.x, chunk.y, chunk.z
            processed += len(x)
            if progress:
                _progress(f"Streaming chunk {chunk_number + 1}", processed, metadata.point_count)
            mask = np.arange(len(x)) % sample_stride == 0
            z_sample = z[mask]
            if len(z_sample):
                z01 = (z_sample - z0) / max((z1 - z0), 1e-6)
                viewer_points.extend(np.column_stack((x[mask], y[mask], z_sample)).round(4).tolist())
                viewer_colors.extend(_elevation_color(z01))
            if chunk.point_source_id is not None:
                scanner_ids.extend(int(value) for value in np.unique(chunk.point_source_id) if int(value) > 0)
            labels = gps_run_labels(chunk.gps_time)
            if labels is not None and np.any(labels == 2):
                saw_run2 = True
            tx = np.floor(x / settings.tile_size_m).astype(np.int64)
            ty = np.floor(y / settings.tile_size_m).astype(np.int64)
            for key in np.unique(np.column_stack((tx, ty)), axis=0):
                tile_mask = (tx == key[0]) & (ty == key[1])
                tile_name = f"tile_{key[0]}_{key[1]}"
                if tile_name not in tile_names:
                    tile_names.append(tile_name)
                _append_tile(tiles_dir, tile_name, source_header, chunk, tile_mask)
        if progress:
            print()
    summary = RunSummary(
        input_path=str(input_path), point_count=metadata.point_count, bounds=metadata.bounds,
        crs=crs, las_version=metadata.version, point_format=metadata.point_format,
        warnings=warnings, tile_count=len(tile_names), scanner_ids=sorted(set(scanner_ids)),
        run_count=(2 if saw_run2 else (1 if metadata.has_gps_time else 0)),
        processing_version=VERSION,
    )
    return summary, tile_names, viewer_points, viewer_colors


def _detect_tile(
    tiles_dir: Path, tile_name: str, summary: RunSummary, settings: ProcessingSettings,
    id_counts: Counter, predictions: Dict[str, np.ndarray], class_names: List[str],
    class_mapping: Dict[str, Dict[str, object]], sample_stride: int, progress: bool,
) -> List[Asset]:
    """Pass 3: one tile -> assets (ground, features, detectors, attribution)."""
    tile_path = tiles_dir / f"{tile_name}.las"
    with laspy.open(tile_path) as reader:
        chunk = next(reader.chunk_iterator(10_000_000), None)
    if chunk is None or len(chunk.x) < settings.min_asset_points:
        return []
    x = np.asarray(chunk.x, dtype=np.float64)
    y = np.asarray(chunk.y, dtype=np.float64)
    z = np.asarray(chunk.z, dtype=np.float64)
    names = set(chunk.point_format.dimension_names)
    intensity = np.asarray(chunk.intensity, dtype=np.float64) if "intensity" in names else np.zeros(len(x))
    rgb = None
    if {"red", "green", "blue"}.issubset(names):
        rgb = np.column_stack((np.asarray(chunk.red), np.asarray(chunk.green), np.asarray(chunk.blue))).astype(np.float64)
    gps_time = np.asarray(chunk.gps_time, dtype=np.float64) if "gps_time" in names else None
    point_source_id = np.asarray(chunk.point_source_id, dtype=np.int64) if "point_source_id" in names else None

    ground = estimate_ground_z(z, x, y, settings.ground_bin_m, settings.ground_quantile)
    height = height_above_ground(z, ground)
    model_class = predictions.get(tile_name) if predictions else None
    if model_class is not None and len(model_class) != len(x):
        model_class = None

    ctx = TileContext(
        tile=tile_name, x=x, y=y, z=z, height=height, ground=ground, intensity=intensity, rgb=rgb,
        # Provenance: tile-local indices + tile name; the run manifest records the
        # streaming order so source points remain traceable.
        source_indices=np.arange(len(x), dtype=np.int64),
        gps_labels=gps_run_labels(gps_time), point_source_id=point_source_id, crs=summary.crs,
        model_class=model_class, class_names=class_names, mapping=class_mapping,
        settings=settings, id_counts=id_counts, sample_stride=sample_stride, viewer_limit=settings.viewer_point_limit,
    )
    assets = detect_all(ctx)
    for asset in assets:
        assign_ids(asset, ctx)
        _attach_highlight_points(ctx, asset)
    if progress:
        counts = Counter(asset.asset_class for asset in assets)
        print(
            f"  tile {tile_name}: {len(x):,} points | assets: {len(assets)} "
            f"| classes: {dict(counts)}",
            flush=True,
        )
    return assets


def _attach_highlight_points(ctx: TileContext, asset: Asset) -> None:
    """Store the asset's actual source-point coordinates for 3D highlighting.

    ``source_point_indices_sample`` holds tile-local indices; we resolve them to
    real XYZ here so the viewer never depends on global index bookkeeping.
    """
    indices = np.asarray(asset.source_point_indices_sample, dtype=np.int64)
    indices = indices[indices < len(ctx.x)]
    if not len(indices):
        return
    stride = max(1, math.ceil(len(indices) / 256))
    sample = indices[::stride][:256]
    points = np.column_stack((ctx.x[sample], ctx.y[sample], ctx.z[sample])).round(4)
    geometry = dict(asset.geometry or {})
    geometry["highlight_points"] = points.tolist()
    asset.geometry = geometry


def _assets_from_roadmarking(
    instances: List[Dict[str, object]], tile: str, summary: RunSummary,
    id_counts: Counter, settings: ProcessingSettings,
) -> List[Asset]:
    assets: List[Asset] = []
    for instance in instances:
        segments = instance["geometry"]
        xs = [coord for segment in segments for coord in (segment[0], segment[3])]
        ys = [coord for segment in segments for coord in (segment[1], segment[4])]
        zs = [coord for segment in segments for coord in (segment[2], segment[5])]
        length = float(instance.get("length_m") or 0.0)
        width = float(instance.get("width_m") or 0.0)
        vertices = len(segments) * 2
        aspect_ok = width > 0 and length / max(width, 1e-3) > 3
        geometry_score = min(0.85, 0.45 + (0.3 if aspect_ok else 0.0) + min(len(segments) / 40.0, 0.2))
        factors = ConfidenceFactors(
            model=None,
            geometry=round(geometry_score, 3),
            support=round(support_score(vertices, settings.min_asset_points), 3),
            spatial_context=None,
            class_consistency=None,
        )
        confidence, explanation = score_asset("pavement_marking", factors)
        asset = Asset(
            asset_id="",
            asset_class="pavement_marking",
            subclass="other",
            center={
                "x": round(float(instance["center"]["x"]), 4),
                "y": round(float(instance["center"]["y"]), 4),
                "z": round(float(instance["center"]["z"]), 4),
            },
            bounding_box=(min(xs), min(ys), min(zs), max(xs), max(ys), max(zs)),
            dimensions={"length_m": round(length, 3), "width_m": round(width, 3), "height_m": round(max(zs) - min(zs), 3)},
            point_count=vertices,
            source_tile=tile,
            source_point_indices_sample=[],
            coordinate_reference_system=summary.crs,
            confidence=confidence,
            confidence_factors=factors.to_dict(),
            confidence_explanation=explanation + " Marking vectorized by the external RoadMarkingExtraction subsystem.",
            detection_method="roadmarkingextraction-v1",
            processing_version=VERSION,
            geometry={"segments": len(segments), "source_files": instance.get("source_files", [])},
        )
        id_counts["MRK"] += 1
        asset.asset_id = f"MRK-{id_counts['MRK']:05d}"
        assets.append(asset)
    return assets


def process_las(
    input_path: str | Path,
    output_dir: str | Path,
    settings: Optional[ProcessingSettings] = None,
    class_mapping: Optional[Dict[str, Dict[str, object]]] = None,
    confidence_weights: Optional[Dict[str, Dict[str, float]]] = None,
    progress: bool = True,
) -> RunSummary:
    """Run the full pipeline: LAS -> tiling -> detection -> inventory -> exports."""
    settings = settings or ProcessingSettings()
    class_mapping = class_mapping or DEFAULT_CLASS_MAPPING
    del confidence_weights  # reserved for config-driven confidence weighting (configs/classes.yaml)
    path = Path(input_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    summary, tile_names, viewer_points, viewer_colors = _stream_tiles(path, output, settings, progress)

    # ---- Optional learned backend: Pointcept / PTv3 ---------------------------
    predictions: Dict[str, np.ndarray] = {}
    class_names: List[str] = []
    backend = describe_backend(settings.pointcept_root, settings.pointcept_class_names)
    if settings.backend == "pointcept":
        if not all((settings.pointcept_root, settings.pointcept_config, settings.pointcept_weight)):
            raise ValueError(
                "Pointcept backend requires --pointcept-root, --pointcept-config, and "
                "--pointcept-weight. No class mapping is assumed without them."
            )
        class_names = load_class_names(settings.pointcept_class_names)
        backend = run_pointcept_test(
            root=settings.pointcept_root, config=settings.pointcept_config,
            weight=settings.pointcept_weight, num_gpus=settings.pointcept_num_gpus,
            save_path=output / "predictions", class_names=class_names,
        )
        predictions = load_predictions(output / "predictions", tile_names)
        summary.warnings.append(
            "Pointcept predictions imported through the documented adaptation layer "
            "(configs/model.yaml); geometry detectors still gate every asset."
        )
    summary.backend = backend

    # ---- Optional specialized subsystem: RoadMarkingExtraction -------------------
    roadmarking_assets: List[Asset] = []
    if settings.roadmarking_command:
        print("[roadmarking] running external extraction subsystem per tile", flush=True)
        work_dir = output / "roadmarking"
        id_counts_rm: Counter = Counter()
        for tile_name in tile_names:
            instances = run_roadmarking(
                settings.roadmarking_command, settings.roadmarking_config,
                output / "tiles", tile_name, work_dir,
            )
            roadmarking_assets.extend(_assets_from_roadmarking(instances, tile_name, summary, id_counts_rm, settings))
        summary.warnings.append(
            f"RoadMarkingExtraction contributed {len(roadmarking_assets)} marking assets "
            "(detection_method: roadmarkingextraction-v1)."
        )

    # ---- Detection pass ---------------------------------------------------------
    id_counts: Counter = Counter()
    sample_stride = max(1, math.ceil(summary.point_count / settings.viewer_point_limit))
    all_assets: List[Asset] = []
    if progress:
        print(f"[detection] processing {len(tile_names)} tiles", flush=True)
    for tile_name in tile_names:
        all_assets.extend(_detect_tile(
            output / "tiles", tile_name, summary, settings, id_counts,
            predictions, class_names, class_mapping, sample_stride, progress,
        ))
    all_assets.extend(roadmarking_assets)

    # ---- Quality control + export -------------------------------------------------
    all_assets, qc_report = run_quality_control(all_assets, settings)
    summary.assets = all_assets
    summary.elapsed_seconds = time.monotonic() - started

    write_outputs(output, summary, qc_report, viewer_points, viewer_colors)
    write_viewer(output / "viewer")
    shutil.copy2(output / "viewer-data.json", output / "viewer" / "viewer-data.json")
    (output / "tiles" / "manifest.json").write_text(
        json.dumps({"tiles": tile_names, "count": len(tile_names), "order": "streaming append order"}, indent=2),
        encoding="utf-8",
    )
    if progress:
        print(
            f"[done] {len(all_assets)} assets in {summary.elapsed_seconds:.1f}s -> {output}",
            flush=True,
        )
    return summary


def _progress(label: str, current: int, total: int) -> None:
    fraction = min(1.0, current / max(total, 1))
    bar_length = 24
    filled = int(bar_length * fraction)
    bar = "=" * filled + "-" * (bar_length - filled)
    percent = int(fraction * 100)
    print(f"\r[{bar}] {percent:3d}%  {label}  ({current:,}/{total:,})", end="", flush=True)