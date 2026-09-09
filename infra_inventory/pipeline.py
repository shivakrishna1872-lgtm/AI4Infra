"""Pipeline orchestrator.

Flow (designed for very large mobile-LiDAR clouds):

1. Stream the LAS in chunks; validate; write one LAS file per spatial tile to
   ``output/tiles/`` (append mode, single pass, bounded memory); sample points,
   colors, RGB and intensity for the 3D viewer.
2. Optional: run Pointcept's real ``tools/test.py`` over the tiles, then load
   the exported per-tile predictions as a *model prior* for detection.
3. Optional: run the external RoadMarkingExtraction subsystem per tile and
   ingest its DXF vector output.
4. Process tiles one at a time: ground estimation, geometric features, asset
   detection, attribution, confidence.
5. Quality control, then export inventory artifacts + viewer payload (including
   per-point class labels derived from the detected assets' source points).

Every asset keeps its source tile, source point indices, CRS, measured
attributes, and confidence breakdown - nothing is invented.
"""
from __future__ import annotations

import hashlib
import json
import math
import multiprocessing
import os
import shutil
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import laspy
import numpy as np

from .assets import TileContext, detect_all
from .instances import (
    dedupe_pole_detections,
    merge_linear_pieces,
    merge_linear_safety_pieces,
    merge_marking_segments,
    merge_surface_fragments,
)
from .assessment import assess_all
from .confidence import ConfidenceFactors, compute_confidence_report, score_asset, support_score
from .export import write_outputs
from .las_reader import gps_run_labels, iter_chunks
from .models import Asset, ProcessingSettings, RunSummary
from .mongo_export import export_to_mongo
from .pointcept import describe_backend, load_class_names, load_predictions, run_pointcept_test
from .preprocessing import estimate_ground_z, height_above_ground, voxel_downsample
from .qc import run_quality_control
from .roadmarking import run_roadmarking
from .validation import validate_las
from .viewer import write_viewer

VERSION = "0.3.0"

#: OpenPCSeg upstream taxonomies (verified against the upstream READMEs;
#: see configs/model.yaml for the full documentation).
#: Toronto-3D (WeikaiTan/Toronto-3D): unclassified 0, Road 1, Road marking 2,
#: Natural 3, Building 4, Utility line 5, Pole 6, Car 7, Fence 8.
TORONTO3D_CLASS_NAMES = [
    "unclassified", "road", "road_marking", "natural", "building",
    "utility_line", "pole", "car", "fence",
]
#: SemanticKITTI (OpenPCSeg model zoo; IGNORE_LABEL = 0).
SEMKITTI_CLASS_NAMES = [
    "car_c", "bicycle", "motorcycle", "truck", "other-vehicle", "person",
    "bicyclist", "motorcyclist", "road", "parking", "sidewalk",
    "other-ground", "building", "fence", "vegetation", "trunk", "terrain",
    "pole", "traffic-sign", "other-object",
]
OPENPCSEG_TAXONOMIES = {
    "toronto3d": TORONTO3D_CLASS_NAMES,
    "semantickitti": SEMKITTI_CLASS_NAMES,
}

#: Documented adaptation from upstream OpenPCSeg classes to infrastructure
#: evidence (mirrors configs/model.yaml openpcseg.class_mapping; shared by
#: both taxonomies — the superset of names resolves for either checkpoint).
DEFAULT_OPENPCSEG_CLASS_MAPPING: Dict[str, Dict[str, object]] = {
    "road": {"asset_class": "pavement", "subclass": "travelled_surface", "weight": 0.90},
    "road_marking": {"asset_class": "pavement_marking", "weight": 0.85},
    "utility_line": {"asset_class": "overhead_conductor", "weight": 0.85},
    "pole": {"asset_class": "utility_pole", "weight": 0.85},
    "fence": {"asset_class": "safety_barrier", "weight": 0.55, "evidence_for": ["guardrail"]},
    "building": {"asset_class": "utility_cabinet", "weight": 0.10},
    "traffic-sign": {"asset_class": "traffic_sign", "weight": 0.85},
    "trunk": {"asset_class": None, "weight": 0.2, "evidence_for": ["utility_pole"]},
    "terrain": {"asset_class": None, "weight": 0.1},
    "parking": {"asset_class": None, "weight": 0.05},
    "sidewalk": {"asset_class": None, "weight": 0.05},
    "natural": {"asset_class": None, "weight": 0.0},
    "car": {"asset_class": None, "weight": 0.0},
    "vegetation": {"asset_class": None, "weight": 0.0},
}

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

#: Stage names reported through the progress callback.
STAGES = ("validating", "streaming", "pointcept", "openpcseg", "roadmarking", "detecting", "exporting", "done")


def _elevation_color(z01: np.ndarray) -> List[List[float]]:
    """Elevation-gradient RGB colors, vectorized (per-point Python loop measured
    at ~20% of pipeline runtime on a 210 k-point corridor)."""
    z = np.clip(np.asarray(z01, dtype=np.float64), 0.0, 1.0)
    below = z < 0.5
    local = np.where(below, z / 0.5, (z - 0.5) / 0.5)
    stops = np.asarray(_STOPS, dtype=np.float64)
    segment = np.where(below, 0, 1)
    a = stops[segment]
    b = stops[segment + 1]
    colors = np.round(a + (b - a) * local[:, None], 4)
    return colors.tolist()


def _tile_header(source_header: laspy.LasHeader) -> laspy.LasHeader:
    """Build a header for an internal tile file, upgrading the LAS version when
    the source version cannot be written.

    laspy cannot *create* headers for LAS 1.0 (and some 1.1) files even though
    it reads them fine - old USGS 3DEP tiles (e.g. Burnet County 2006) are often
    LAS 1.0 / point format 1. Without this upgrade, tiling dies on the first
    chunk with ``FileVersionNotSupported: 1.0`` (surfacing to users as the
    cryptic "Processing failed: 1.0"). Tile files are internal artifacts, so
    upgrading to the oldest writable version that supports the point format is
    lossless: scales, offsets, dimensions and VLRs are preserved.
    """
    for version in (str(source_header.version), "1.2", "1.3", "1.4"):
        try:
            header = laspy.LasHeader(
                point_format=source_header.point_format, version=version
            )
            break
        except Exception:  # pragma: no cover - laspy version quirks
            continue
    else:  # pragma: no cover - defensive; 1.4 covers every PDR
        header = laspy.LasHeader(point_format=int(source_header.point_format.id), version="1.4")
    header.scales = list(source_header.scales)
    header.offsets = list(source_header.offsets)
    try:
        header.vlrs = list(source_header.vlrs)
    except Exception:
        pass
    # Carry extra-bytes dimensions (LAS 1.4 only — older versions cannot
    # declare them) so tiling preserves *every* dimension, not just the core
    # ones. Extra dims with a scale/offset are skipped: writing them without
    # rescaling would corrupt the stored integers, so it is safer to omit
    # than to write wrong values.
    try:
        if str(source_header.version) >= "1.4":
            for dim in source_header.point_format.extra_dimensions:
                if getattr(dim, "scale", 1.0) in (1.0, None) and getattr(dim, "offset", 0.0) in (0.0, None):
                    header.add_extra_dim(laspy.ExtraBytesParams(name=dim.name, type=dim.type))
    except Exception:
        pass
    return header


#: Writable-tile-header cache: one entry per distinct source header (per run
#: there is exactly one source, so this is one entry). Building a LasHeader +
#: copying its VLR list costs real time; doing it for every (chunk x tile)
#: append was measurable overhead on large files. Keyed by the header's own
#: identity fingerprint, never by file path (tiles are internal artifacts).
_TILE_HEADER_CACHE: Dict[tuple, laspy.LasHeader] = {}


def _tile_header_cached(source_header: laspy.LasHeader) -> laspy.LasHeader:
    """Return the writable tile header for a source header, cached per header."""
    key = (
        str(source_header.version),
        int(source_header.point_format.id),
        tuple(float(v) for v in source_header.scales),
        tuple(float(v) for v in source_header.offsets),
        tuple(source_header.point_format.extra_dimension_names),
    )
    cached = _TILE_HEADER_CACHE.get(key)
    if cached is not None:
        return cached
    built = _tile_header(source_header)
    _TILE_HEADER_CACHE[key] = built
    return built


def _append_tile(
    tiles_dir: Path,
    tile_name: str,
    source_header: laspy.LasHeader,
    chunk: object,
    mask: np.ndarray,
    writers: Optional[Dict[str, laspy.LasWriter]] = None,
) -> None:
    """Append a chunk's subset of points to a tile's LAS file (create on first touch).

    Every source dimension present in the tile format is copied, so tiling is
    lossless: X/Y/Z, intensity, RGB, NIR, GPS time, returns, point source,
    classification, scan angle, user data, scanner channel, flag bits, and
    extra-bytes dims. ``writers`` optionally holds open per-tile writers for
    the streaming pass so a chunk never re-opens the same tile file.
    """
    tile_path = tiles_dir / f"{tile_name}.las"
    data = laspy.LasData(_tile_header_cached(source_header))
    data.x, data.y, data.z = chunk.x[mask], chunk.y[mask], chunk.z[mask]
    if "intensity" in data.point_format.dimension_names:
        data.intensity = chunk.intensity[mask]
    if chunk.rgb is not None and {"red", "green", "blue"}.issubset(data.point_format.dimension_names):
        data.red, data.green, data.blue = chunk.rgb[mask, 0], chunk.rgb[mask, 1], chunk.rgb[mask, 2]
    if chunk.nir is not None and "nir" in data.point_format.dimension_names:
        data.nir = chunk.nir[mask]
    if chunk.gps_time is not None and "gps_time" in data.point_format.dimension_names:
        data.gps_time = chunk.gps_time[mask]
    if chunk.point_source_id is not None and "point_source_id" in data.point_format.dimension_names:
        data.point_source_id = chunk.point_source_id[mask]
    if chunk.return_number is not None and "return_number" in data.point_format.dimension_names:
        data.return_number = chunk.return_number[mask]
    if chunk.number_of_returns is not None and "number_of_returns" in data.point_format.dimension_names:
        data.number_of_returns = chunk.number_of_returns[mask]
    if chunk.classification is not None and "classification" in data.point_format.dimension_names:
        data.classification = chunk.classification[mask]
    if chunk.scan_angle is not None and "scan_angle" in data.point_format.dimension_names:
        data.scan_angle = chunk.scan_angle[mask]
    elif chunk.scan_angle is not None and "scan_angle_rank" in data.point_format.dimension_names:
        data.scan_angle_rank = chunk.scan_angle[mask]
    if chunk.user_data is not None and "user_data" in data.point_format.dimension_names:
        data.user_data = chunk.user_data[mask]
    if chunk.scanner_channel is not None and "scanner_channel" in data.point_format.dimension_names:
        data.scanner_channel = chunk.scanner_channel[mask]
    for flag in ("synthetic", "key_point", "withheld", "overlap", "scan_direction_flag", "edge_of_flight_line"):
        value = getattr(chunk, flag, None)
        if value is not None and flag in data.point_format.dimension_names:
            setattr(data, flag, value[mask])
    if chunk.extra:
        for name, values in chunk.extra.items():
            if name in data.point_format.dimension_names:
                data[name] = values[mask]
    if writers is not None:
        entry = writers.get(tile_name)
        if entry is None:
            if tile_path.is_file():
                entry = ("append", laspy.open(tile_path, mode="a"))
            else:
                entry = ("write", laspy.open(tile_path, mode="w", header=data.header))
            writers[tile_name] = entry
        mode, writer = entry
        if mode == "append":
            writer.append_points(data.points)
        else:
            writer.write_points(data.points)
    elif tile_path.is_file():
        with laspy.open(tile_path, mode="a") as writer:
            writer.append_points(data.points)
    else:
        with laspy.open(tile_path, mode="w", header=data.header) as writer:
            writer.write_points(data.points)


def _stream_tiles(
    input_path: Path, output: Path, settings: ProcessingSettings, progress: bool,
    progress_callback: Optional[Callable[[dict], None]],
) -> Tuple[RunSummary, List[str], dict]:
    """Pass 1: validate, stream chunks, write tiles, sample viewer data."""
    warnings: List[str] = []
    validation = validate_las(
        input_path, strict_las14=settings.strict_las14, crs_fallback=settings.crs_fallback,
    )
    warnings.extend(validation.as_warning_messages())
    metadata = validation.metadata
    crs = metadata.crs

    tiles_dir = output / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)
    viewer: dict = {"points": [], "point_colors": [], "point_rgb": None, "point_intensity": None}
    if metadata.has_rgb:
        viewer["point_rgb"] = []
    if metadata.has_intensity:
        viewer["point_intensity"] = []
    # LOD background cloud: pool a bounded candidate set during streaming, then
    # voxel-downsample to the viewer budget. A strided sample alone ships the
    # densest areas over-represented and can still be tens of MB; the voxel grid
    # guarantees one point per cell of space at bounded size (docs/GEMINI.md,
    # "viewer payload"). The hard byte cap is enforced in export.write_outputs.
    pool_limit = min(settings.viewer_point_limit * 4, 500_000)
    pool_x: List[float] = []
    pool_y: List[float] = []
    pool_z: List[float] = []
    pool_rgb: List[List[float]] = []
    pool_intensity: List[float] = []
    pool_stride = max(1, math.ceil(metadata.point_count / pool_limit))
    tile_names: List[str] = []
    scanner_ids: List[int] = []
    saw_run2 = False
    z0, _, _, z1, _, _ = metadata.bounds

    input_stride = 1
    if settings.max_input_points and metadata.point_count > settings.max_input_points:
        input_stride = max(2, math.ceil(metadata.point_count / settings.max_input_points))
        thinned = metadata.point_count // input_stride
        warnings.append(
            f"Input thinned uniformly to ~{thinned:,} points (max_input_points="
            f"{settings.max_input_points:,}, stride {input_stride}). The inventory is "
            "measured from the thinned cloud; source point counts are approximate."
        )
    processed = 0
    writers: Dict[str, laspy.LasWriter] = {}
    try:
        with laspy.open(input_path) as reader:
            tile_header = _tile_header(reader.header)
            if str(reader.header.version) != str(tile_header.version):
                warnings.append(
                    f"Input LAS {reader.header.version} is not writable by laspy; tiles are "
                    f"written as LAS {tile_header.version} (scales/offsets/dimensions preserved). "
                    "This only affects internal tile files, never the input."
                )
            source_header = reader.header
            for chunk_number, (_, _, chunk) in enumerate(iter_chunks(input_path, settings.chunk_size, stride=input_stride)):
                x, y, z = chunk.x, chunk.y, chunk.z
                processed += len(x)
                if chunk_number == 0:
                    # Dead radiometric channels: some exporters keep the RGB/intensity
                    # dimensions but never fill them ("no-color" LAS 1.4 fmt 7
                    # exports). An all-zero channel must not silently feed the
                    # marking detector as "everything is bright" - report it so the
                    # run.json warnings explain exactly what happened (and the
                    # detector itself also guards per-tile in bright_near_ground_mask).
                    if metadata.has_intensity and len(chunk.intensity):
                        finite_intensity = chunk.intensity[np.isfinite(chunk.intensity)]
                        if not len(finite_intensity) or float(finite_intensity.max()) <= 0.0:
                            warnings.append(
                                "[INTENSITY_DEAD] The intensity channel is all zeros; "
                                "marking detection falls back to RGB/NIR brightness or geometry only."
                            )
                    if chunk.rgb is not None and len(chunk.rgb):
                        finite_rgb = chunk.rgb[np.isfinite(chunk.rgb)]
                        if not len(finite_rgb) or float(finite_rgb.max()) <= 0.0:
                            warnings.append(
                                "[RGB_DEAD] The RGB channel is all zeros (a no-color export); "
                                "color-based marking evidence is unavailable."
                            )
                if progress:
                    _progress(f"Streaming chunk {chunk_number + 1}", processed, metadata.point_count)
                if progress_callback:
                    progress_callback({"stage": "streaming", "points_processed": processed, "point_count": metadata.point_count})
                mask = np.arange(len(x)) % pool_stride == 0
                if mask.any():
                    pool_x.extend(x[mask].tolist())
                    pool_y.extend(y[mask].tolist())
                    pool_z.extend(z[mask].tolist())
                    if metadata.has_rgb and chunk.rgb is not None:
                        pool_rgb.extend(np.clip(chunk.rgb[mask] / 65535.0, 0.0, 1.0).round(4).tolist())
                    if metadata.has_intensity:
                        values = chunk.intensity[mask]
                        if values.max() > 0:
                            pool_intensity.extend(np.clip(values / float(values.max()), 0.0, 1.0).round(4).tolist())
                if chunk.point_source_id is not None:
                    scanner_ids.extend(int(value) for value in np.unique(chunk.point_source_id) if int(value) > 0)
                labels = gps_run_labels(chunk.gps_time)
                if labels is not None and np.any(labels == 2):
                    saw_run2 = True
                tx = np.floor(x / settings.tile_size_m).astype(np.int64)
                ty = np.floor(y / settings.tile_size_m).astype(np.int64)
                # Pack (tile_x, tile_y) into one uint64 per point and run a single
                # 1-D unique. This is provably injective (both offsets are
                # subtracted first, so each lane fits in 32 bits) and avoids the
                # 2-D column_stack + lexsort that np.unique(..., axis=0) needs —
                # the hottest single spot in the streaming pass on large files.
                tx_min = int(tx.min())
                ty_min = int(ty.min())
                keys = ((tx - tx_min).astype(np.uint64) << 32) | (ty - ty_min).astype(np.uint64)
                for key in np.unique(keys):
                    kx = int(key >> 32) + tx_min
                    ky = int(key & 0xFFFFFFFF) + ty_min
                    tile_mask = (tx == kx) & (ty == ky)
                    tile_name = f"tile_{kx}_{ky}"
                    if tile_name not in tile_names:
                        tile_names.append(tile_name)
                    _append_tile(tiles_dir, tile_name, source_header, chunk, tile_mask, writers)
            if progress:
                print()
    finally:
        # Flush every pooled writer so each tile's header point count and
        # every dimension are durable on disk before detection reads them.
        for _, writer in writers.values():
            writer.close()
    # Voxel LOD pass: one point per occupied cell, ~viewer_point_limit cells.
    if pool_x:
        px = np.asarray(pool_x, dtype=np.float64)
        py = np.asarray(pool_y, dtype=np.float64)
        pz = np.asarray(pool_z, dtype=np.float64)
        keep = voxel_downsample(px, py, pz, settings.viewer_point_limit)
        viewer["points"] = np.column_stack((px[keep], py[keep], pz[keep])).round(4).tolist()
        z01 = (pz[keep] - z0) / max((z1 - z0), 1e-6)
        viewer["point_colors"] = _elevation_color(z01)
        if metadata.has_rgb and pool_rgb:
            viewer["point_rgb"] = [pool_rgb[i] for i in keep.tolist()]
        if metadata.has_intensity and pool_intensity:
            viewer["point_intensity"] = [pool_intensity[i] for i in keep.tolist()]
    if not viewer["point_rgb"]:
        viewer["point_rgb"] = None
    if not viewer["point_intensity"]:
        viewer["point_intensity"] = None
    summary = RunSummary(
        input_path=str(input_path), point_count=metadata.point_count, bounds=metadata.bounds,
        crs=crs, las_version=metadata.version, point_format=metadata.point_format,
        warnings=warnings, tile_count=len(tile_names), scanner_ids=sorted(set(scanner_ids)),
        run_count=(2 if saw_run2 else (1 if metadata.has_gps_time else 0)),
        processing_version=VERSION,
    )
    return summary, tile_names, viewer


def _available_workers() -> int:
    """Cores available to this process, honoring cgroup CPU quotas.

    The detectors are GIL-bound Python (numpy releases the GIL but the
    per-component loops do not), so worker processes - never threads - are used
    for the tile pass. Oversubscribing hurts (context-switch overhead), so the
    worker count follows the cgroup quota (e.g. 200000/100000 = 2 CPUs) when
    one is set, capped at 16.
    """
    try:
        quota = Path("/sys/fs/cgroup/cpu.max").read_text(encoding="utf-8").strip()
        if quota and quota != "max":
            maximum_text, period_text = quota.split()  # format: <quota> <period>
            period, maximum = int(period_text), int(maximum_text)
            if period > 0 and maximum > 0:
                return max(1, min(16, maximum // period))
    except (OSError, ValueError):
        pass
    try:
        quota_us = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text(encoding="utf-8").strip())
        period_us = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text(encoding="utf-8").strip())
        if quota_us > 0 and period_us > 0:
            return max(1, min(16, -(-quota_us // period_us)))
    except (OSError, ValueError):
        pass
    return max(1, min(16, (os.cpu_count() or 4)))


def _file_sha256(path: Path) -> str:
    """SHA-256 of a file (streamed, bounded memory)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_tile_manifest(tiles_dir: Path, tile_names: List[str], input_sha256: str) -> None:
    """Persist the tile manifest immediately after streaming.

    Written *before* detection so a crashed run can be resumed from the tiles on
    disk (the SHA-256 binds the tiles to the exact input file).
    """
    tiles_dir.mkdir(parents=True, exist_ok=True)
    (tiles_dir / "manifest.json").write_text(
        json.dumps({
            "tiles": tile_names,
            "count": len(tile_names),
            "order": "streaming append order",
            "input_sha256": input_sha256,
        }, indent=2),
        encoding="utf-8",
    )


def _pool_viewer_from_tiles(
    tiles_dir: Path, tile_names: List[str], settings: ProcessingSettings, metadata: object,
) -> Tuple[dict, List[int], int]:
    """Rebuild the viewer LOD pool (and run/scanner provenance) from existing tiles.

    Resume path: the streaming pass is skipped, so the bounded viewer pool that
    was collected while streaming must be re-derived from the tiles themselves.
    Mirrors the pooling logic in ``_stream_tiles`` exactly (strided pool then
    voxel downsample), so the exported viewer payload is identical to a fresh run.
    """
    viewer: dict = {"points": [], "point_colors": [], "point_rgb": None, "point_intensity": None}
    if metadata.has_rgb:
        viewer["point_rgb"] = []
    if metadata.has_intensity:
        viewer["point_intensity"] = []
    pool_limit = min(settings.viewer_point_limit * 4, 500_000)
    pool_stride = max(1, math.ceil(metadata.point_count / pool_limit))
    pool_x: List[float] = []
    pool_y: List[float] = []
    pool_z: List[float] = []
    pool_rgb: List[List[float]] = []
    pool_intensity: List[float] = []
    scanner_ids: List[int] = []
    saw_run2 = False
    for tile_name in tile_names:
        with laspy.open(tiles_dir / f"{tile_name}.las") as reader:
            chunk = next(reader.chunk_iterator(10_000_000), None)
        if chunk is None or not len(chunk.x):
            continue
        x = np.asarray(chunk.x, dtype=np.float64)
        y = np.asarray(chunk.y, dtype=np.float64)
        z = np.asarray(chunk.z, dtype=np.float64)
        names = set(chunk.point_format.dimension_names)
        mask = np.arange(len(x)) % pool_stride == 0
        if mask.any():
            pool_x.extend(x[mask].tolist())
            pool_y.extend(y[mask].tolist())
            pool_z.extend(z[mask].tolist())
            if metadata.has_rgb and {"red", "green", "blue"}.issubset(names):
                rgb = np.column_stack((np.asarray(chunk.red), np.asarray(chunk.green), np.asarray(chunk.blue)))
                pool_rgb.extend(np.clip(rgb[mask] / 65535.0, 0.0, 1.0).round(4).tolist())
            if metadata.has_intensity and "intensity" in names:
                values = np.asarray(chunk.intensity)[mask]
                if values.max() > 0:
                    pool_intensity.extend(np.clip(values / float(values.max()), 0.0, 1.0).round(4).tolist())
        if "point_source_id" in names:
            scanner_ids.extend(int(v) for v in np.unique(chunk.point_source_id) if int(v) > 0)
        if metadata.has_gps_time and "gps_time" in names:
            labels = gps_run_labels(chunk.gps_time)
            if labels is not None and np.any(labels == 2):
                saw_run2 = True
    z0, _, _, z1, _, _ = metadata.bounds
    if pool_x:
        px = np.asarray(pool_x, dtype=np.float64)
        py = np.asarray(pool_y, dtype=np.float64)
        pz = np.asarray(pool_z, dtype=np.float64)
        keep = voxel_downsample(px, py, pz, settings.viewer_point_limit)
        viewer["points"] = np.column_stack((px[keep], py[keep], pz[keep])).round(4).tolist()
        z01 = (pz[keep] - z0) / max((z1 - z0), 1e-6)
        viewer["point_colors"] = _elevation_color(z01)
        if metadata.has_rgb and pool_rgb:
            viewer["point_rgb"] = [pool_rgb[i] for i in keep.tolist()]
        if metadata.has_intensity and pool_intensity:
            viewer["point_intensity"] = [pool_intensity[i] for i in keep.tolist()]
    if not viewer["point_rgb"]:
        viewer["point_rgb"] = None
    if not viewer["point_intensity"]:
        viewer["point_intensity"] = None
    return viewer, sorted(set(scanner_ids)), (2 if saw_run2 else (1 if metadata.has_gps_time else 0))


def _asset_from_dict(record: dict) -> Asset:
    """Rebuild an Asset from its exported record (cache round-trip)."""
    return Asset(
        asset_id=record["asset_id"], asset_class=record["class"], subclass=record.get("subclass"),
        center=record["center"], bounding_box=tuple(record["bounding_box"]),
        dimensions=record.get("dimensions") or {}, point_count=record["point_count"],
        source_tile=record["source_tile"],
        source_point_indices_sample=record.get("source_point_indices_sample", []),
        coordinate_reference_system=record.get("coordinate_reference_system"),
        confidence=record["confidence"], confidence_factors=record.get("confidence_factors") or {},
        confidence_explanation=record.get("confidence_explanation", ""),
        detection_method=record.get("detection_method", ""),
        intensity_stats=record.get("intensity_stats"), rgb_stats=record.get("rgb_stats"),
        orientation_deg=record.get("orientation_deg"), source_run=record.get("source_run"),
        source_scanner=record.get("source_scanner"),
        source_point_source_id=record.get("source_point_source_id"),
        model_prior_class=record.get("model_prior_class"),
        model_confidence=record.get("model_confidence"),
        processing_version=record.get("processing_version", "0.3.0"),
        geometry=record.get("geometry"), qc_flags=list(record.get("qc_flags") or []),
        flagged=bool(record.get("flagged")), condition=record.get("condition"),
        recommended_action=record.get("recommended_action"),
        review_required=bool(record.get("review_required")),
        assessment_reasoning=record.get("assessment_reasoning"),
    )


def _write_tile_cache(cache_dir: Path, tile_name: str, tile_assets: List[Asset], point_count: int) -> None:
    """Persist one tile's extracted assets (atomic write)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{tile_name}.json"
    tmp = cache_dir / f"{tile_name}.json.tmp"
    tmp.write_text(
        json.dumps({"point_count": point_count, "assets": [a.to_dict() for a in tile_assets]}),
        encoding="utf-8",
    )
    tmp.replace(path)


def _read_tile_cache(cache_dir: Path, tile_name: str) -> Optional[Tuple[List[Asset], int]]:
    """Load a tile's cached extraction, or None when absent/corrupt."""
    path = cache_dir / f"{tile_name}.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return [_asset_from_dict(a) for a in data["assets"]], int(data["point_count"])
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None


def _extract_tile_assets(
    tiles_dir: Path, tile_name: str, summary: RunSummary, settings: ProcessingSettings,
    predictions: Dict[str, np.ndarray], class_names: List[str],
    class_mapping: Dict[str, Dict[str, object]], sample_stride: int,
    classifier: object = None,
) -> Tuple[List[Asset], int]:
    """Pass 3 (worker): one tile -> assets (ground, features, detectors, attribution).

    Thread-safe by construction: no shared mutable state (a per-worker
    throwaway Counter is used for the TileContext, and asset IDs are assigned
    afterwards, sequentially, so outputs are byte-identical to a single-threaded
    run). Returns ``(assets, point_count)``.
    """
    tile_path = tiles_dir / f"{tile_name}.las"
    with laspy.open(tile_path) as reader:
        chunk = next(reader.chunk_iterator(10_000_000), None)
    if chunk is None or len(chunk.x) < settings.min_asset_points:
        return [], 0
    x = np.asarray(chunk.x, dtype=np.float64)
    y = np.asarray(chunk.y, dtype=np.float64)
    z = np.asarray(chunk.z, dtype=np.float64)
    names = set(chunk.point_format.dimension_names)
    intensity = np.asarray(chunk.intensity, dtype=np.float64) if "intensity" in names else np.zeros(len(x))
    rgb = None
    if {"red", "green", "blue"}.issubset(names):
        rgb = np.column_stack((np.asarray(chunk.red), np.asarray(chunk.green), np.asarray(chunk.blue))).astype(np.float64)
    nir = np.asarray(chunk.nir, dtype=np.float64) if "nir" in names else None
    gps_time = np.asarray(chunk.gps_time, dtype=np.float64) if "gps_time" in names else None
    point_source_id = np.asarray(chunk.point_source_id, dtype=np.int64) if "point_source_id" in names else None

    ground = estimate_ground_z(z, x, y, settings.ground_bin_m, settings.ground_quantile)
    height = height_above_ground(z, ground)
    model_class = predictions.get(tile_name) if predictions else None
    if model_class is not None and len(model_class) != len(x):
        model_class = None

    ctx = TileContext(
        tile=tile_name, x=x, y=y, z=z, height=height, ground=ground, intensity=intensity, rgb=rgb,
        nir=nir,
        # Provenance: tile-local indices + tile name; the run manifest records the
        # streaming order so source points remain traceable.
        source_indices=np.arange(len(x), dtype=np.int64),
        gps_labels=gps_run_labels(gps_time), point_source_id=point_source_id, crs=summary.crs,
        model_class=model_class, class_names=class_names, mapping=class_mapping,
        settings=settings, id_counts=Counter(), sample_stride=sample_stride,
        viewer_limit=settings.viewer_point_limit, classifier=classifier,
    )
    assets = detect_all(ctx)
    if getattr(settings, "collect_training", False):
        from .learned import flush_training_buffer
        flush_training_buffer(ctx)
    for asset in assets:
        _attach_highlight_points(ctx, asset)
    return assets, int(len(x))


def _assign_ids_sequential(all_assets: List[Asset], id_counts: Counter) -> None:
    """Assign deterministic asset IDs in pipeline order (threads never share the counter)."""
    from .models import CLASS_PREFIXES

    for asset in all_assets:
        prefix = CLASS_PREFIXES[asset.asset_class]
        id_counts[prefix] += 1
        asset.asset_id = f"{prefix}-{id_counts[prefix]:05d}"


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


def _compute_point_classes(viewer_points: List[List[float]], assets: List[Asset], class_names: List[str]) -> List[int]:
    """Label each viewer point with the class of the nearest asset source point.

    Labels are derived from the *actual* detection points (``highlight_points``)
    via a spatial grid, so the AI DETECTION coloring is real evidence, not paint.
    Class ids start at 1; 0 = no detected asset nearby. Vectorized: highlight
    points are sorted by their 1 m grid cell and each viewer point looks up the
    (up to) 27 neighbouring cells with two ``searchsorted`` calls per offset,
    so a 250 k-point cloud labels in well under a second instead of a Python
    triple loop.
    """
    count = len(viewer_points)
    if not viewer_points or not assets:
        return [0] * count
    cell = 1.0
    max_label_points_per_asset = 64  # 1 m match radius covers 0.5 m stride
    rows: List[Tuple[float, float, float, float]] = []
    for class_id, name in enumerate(class_names, start=1):
        for asset in assets:
            if asset.asset_class != name:
                continue
            points = (asset.geometry or {}).get("highlight_points", [])
            stride = max(1, math.ceil(len(points) / max_label_points_per_asset))
            for point in points[::stride]:
                rows.append((class_id, point[0], point[1], point[2]))
    if not rows:
        return [0] * count
    highlight = np.asarray(rows, dtype=np.float64)  # (M, 4): class, x, y, z
    hx = np.floor(highlight[:, 1] / cell).astype(np.int64)
    hy = np.floor(highlight[:, 2] / cell).astype(np.int64)
    hz = np.floor(highlight[:, 3] / cell).astype(np.int64)
    order = np.lexsort((hz, hy, hx))  # cell-sorted: each cell's points are contiguous
    hx, hy, hz, highlight = hx[order], hy[order], hz[order], highlight[order]
    cell_keys = np.column_stack((hx, hy, hz)).view(
        np.dtype([("a", "i8"), ("b", "i8"), ("c", "i8")])
    ).ravel()

    points = np.asarray(viewer_points, dtype=np.float64)  # (N, 3)
    vx, vy, vz = points[:, 0], points[:, 1], points[:, 2]
    vcx = np.floor(vx / cell).astype(np.int64)
    vcy = np.floor(vy / cell).astype(np.int64)
    vcz = np.floor(vz / cell).astype(np.int64)
    radius2 = 1.0  # metres (squared)
    best = np.full(count, radius2 * radius2, dtype=np.float64)
    best_class = np.zeros(count, dtype=np.int64)
    index_range = np.arange(count, dtype=np.int64)
    # Pre-filter (lossless): a viewer point can only match a highlight point if
    # its cell lies within Chebyshev distance 1 of an occupied highlight cell.
    # Build the occupied *neighborhood* set from the highlight side (M cells,
    # small), then one membership test for all N points. On real corridors most
    # viewer points are far from any asset, so the 27-offset search below then
    # runs over a small candidate subset instead of the whole cloud.
    structured = np.dtype([("a", "i8"), ("b", "i8"), ("c", "i8")])
    neighbor_parts = [
        np.column_stack((hx + dx, hy + dy, hz + dz)).view(structured).ravel()
        for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
    ]
    # One sort of the concatenated neighborhood keys (no per-offset unique):
    # membership is decided by a binary search + equality against the sorted
    # array, duplicates are harmless for an existence test.
    neighbor_cells = np.sort(np.concatenate(neighbor_parts))
    point_keys = np.column_stack((vcx, vcy, vcz)).view(structured).ravel()
    pos = np.searchsorted(neighbor_cells, point_keys, side="left")
    pos_clipped = np.minimum(pos, len(neighbor_cells) - 1)
    occupied = neighbor_cells[pos_clipped] == point_keys
    candidates = np.flatnonzero(occupied)
    if not len(candidates):
        return [0] * count
    # Search only the candidate subset; remap results back to point indices.
    sub_vcx, sub_vcy, sub_vcz = vcx[candidates], vcy[candidates], vcz[candidates]
    sub_vx, sub_vy, sub_vz = vx[candidates], vy[candidates], vz[candidates]
    sub_best = best[candidates]
    sub_best_class = best_class[candidates]
    sub_range = np.arange(len(candidates), dtype=np.int64)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                target = np.column_stack((sub_vcx + dx, sub_vcy + dy, sub_vcz + dz)).view(structured).ravel()
                start = np.searchsorted(cell_keys, target, side="left")
                end = np.searchsorted(cell_keys, target, side="right")
                lengths = end - start
                total = int(lengths.sum())
                if total == 0:
                    continue
                owner = np.repeat(sub_range, lengths)
                segment_begin = np.repeat(np.cumsum(lengths) - lengths, lengths)
                hl = start[owner] + (np.arange(total, dtype=np.int64) - segment_begin)
                d2 = (
                    (sub_vx[owner] - highlight[hl, 1]) ** 2
                    + (sub_vy[owner] - highlight[hl, 2]) ** 2
                    + (sub_vz[owner] - highlight[hl, 3]) ** 2
                )
                improved = d2 < sub_best[owner]
                updated = owner[improved]
                sub_best[updated] = d2[improved]
                sub_best_class[updated] = highlight[hl[improved], 0].astype(np.int64)
    best_class[candidates] = sub_best_class
    return best_class.tolist()


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
    progress_callback: Optional[Callable[[dict], None]] = None,
    mongo_uri: Optional[str] = None,
    mongo_database: str = "ai4infra",
    mongo_collection: str = "assets",
) -> RunSummary:
    """Run the full pipeline: LAS -> tiling -> detection -> inventory -> exports.

    ``progress_callback`` receives stage dicts:
      {"stage": "validating"|"streaming"|"pointcept"|"roadmarking"|"detecting"|"exporting"|"done",
       "points_processed": int, "point_count": int, "tiles_done": int, "tiles_total": int,
       "assets": int, "elapsed_seconds": float, "message": str}
    """
    settings = settings or ProcessingSettings()
    class_mapping = class_mapping or DEFAULT_CLASS_MAPPING
    del confidence_weights  # reserved for config-driven confidence weighting (configs/classes.yaml)
    path = Path(input_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    def report(stage: str, **extra: object) -> None:
        if progress_callback:
            progress_callback({
                "stage": stage,
                "points_processed": 0,
                "point_count": 0,
                "tiles_done": 0,
                "tiles_total": 0,
                "assets": 0,
                "elapsed_seconds": round(time.monotonic() - started, 2),
                "message": "",
                **extra,
            })

    report("validating", message="Validating LAS metadata")
    # Resume path: a crashed/interrupted run left its per-tile LAS files and
    # manifest on disk; when the manifest's input SHA-256 matches this exact
    # input file, skip the streaming pass (which would re-decompress and
    # re-write multi-GB tiles, and can hit 507 on a nearly full disk). The
    # viewer pool and run/scanner provenance are re-derived from the tiles, so
    # the exported artifacts are identical to a fresh run.
    summary: Optional[RunSummary] = None
    tile_names: List[str] = []
    viewer: dict = {}
    tiles_dir = output / "tiles"
    if settings.resume_from_tiles:
        manifest_path = tiles_dir / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            existing = [t for t in manifest.get("tiles", [])
                        if (tiles_dir / f"{t}.las").is_file()]
            if existing and manifest.get("input_sha256") == _file_sha256(path):
                validation = validate_las(
                    path, strict_las14=settings.strict_las14, crs_fallback=settings.crs_fallback,
                )
                metadata = validation.metadata
                resumed_summary = RunSummary(
                    input_path=str(path), point_count=metadata.point_count,
                    bounds=metadata.bounds, crs=metadata.crs, las_version=metadata.version,
                    point_format=metadata.point_format,
                    warnings=list(validation.as_warning_messages()),
                    tile_count=len(existing), scanner_ids=[], run_count=0,
                    processing_version=VERSION,
                )
                viewer, scanner_ids, run_count = _pool_viewer_from_tiles(
                    tiles_dir, existing, settings, metadata)
                resumed_summary.scanner_ids = scanner_ids
                resumed_summary.run_count = run_count
                resumed_summary.warnings.append(
                    f"Resumed from {len(existing)} existing tiles (streaming pass skipped; "
                    "input verified by SHA-256)."
                )
                summary = resumed_summary
                tile_names = existing
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            summary = None
    if summary is None:
        summary, tile_names, viewer = _stream_tiles(path, output, settings, progress, progress_callback)
        _write_tile_manifest(tiles_dir, tile_names, _file_sha256(path))

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
        report("pointcept", message="Running Pointcept / PTv3 inference on tiles")
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
    elif settings.backend == "openpcseg":
        from .openpcseg import describe_openpcseg, load_indexed_predictions, run_openpcseg_infer

        if not all((settings.openpcseg_root, settings.openpcseg_config, settings.openpcseg_weight)):
            raise ValueError(
                "OpenPCSeg backend requires --openpcseg-root, --openpcseg-config, and "
                "--openpcseg-weight. No class mapping is assumed without them."
            )
        if settings.openpcseg_taxonomy not in OPENPCSEG_TAXONOMIES:
            raise ValueError(
                f"Unknown openpcseg taxonomy '{settings.openpcseg_taxonomy}'. "
                f"Available: {', '.join(sorted(OPENPCSEG_TAXONOMIES))}"
            )
        report("openpcseg", message="Running OpenPCSeg inference on tensor tiles")
        from .tensor_export import export_tiles_to_tensors

        tensor_manifest = export_tiles_to_tensors(output / "tiles", output / "tensors", tile_names)
        backend = describe_openpcseg(settings.openpcseg_root)
        backend = run_openpcseg_infer(
            root=settings.openpcseg_root, cfg_file=settings.openpcseg_config,
            checkpoint=settings.openpcseg_weight, num_gpus=settings.openpcseg_num_gpus,
            prediction_dir=output / "predictions",
            extra_options={"DATA.DATA_PATH": str((output / "tensors").resolve())},
        )
        backend["name"] = "openpcseg"
        backend["taxonomy"] = settings.openpcseg_taxonomy
        backend["tensor_manifest"] = tensor_manifest
        class_names = list(OPENPCSEG_TAXONOMIES[settings.openpcseg_taxonomy])
        predictions = load_indexed_predictions(output / "predictions", tile_names)
        # A caller-supplied mapping wins; the documented default applies otherwise.
        class_mapping = class_mapping or dict(DEFAULT_OPENPCSEG_CLASS_MAPPING)
        matched = sum(1 for tile in tile_names if len(predictions.get(tile, [])) > 0)
        summary.warnings.append(
            f"OpenPCSeg predictions imported for {matched}/{len(tile_names)} tiles through "
            "the documented adaptation layer (configs/model.yaml); geometry detectors "
            "still gate every asset."
        )
    summary.backend = backend

    # ---- Optional specialized subsystem: RoadMarkingExtraction -------------------
    roadmarking_assets: List[Asset] = []
    if settings.roadmarking_command:
        report("roadmarking", message="Running RoadMarkingExtraction subsystem per tile")
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

    # ---- Detection pass (parallel across cores) ----------------------------------
    # Tiles are independent units: each worker reads its own tile and runs the
    # detectors on it. The detectors are GIL-bound Python, so a *process* pool
    # is used (threads measured *slower* than one worker). Worker count follows
    # the cgroup CPU quota. Outputs stay byte-identical to a sequential run:
    # results are reassembled in tile order and asset IDs are assigned
    # afterwards, never inside the workers.
    id_counts: Counter = Counter()
    sample_stride = max(1, math.ceil(summary.point_count / settings.viewer_point_limit))
    all_assets: List[Asset] = []
    # ---- Learned component classifier (CPU prior) ----------------------------
    # Trained on labeled ground truth (scripts/train_classifier.py, Quick
    # Simulation scenes; the same feature conventions as the Toronto-3D/PTv3
    # MLS literature). Fills the `model` confidence factor when no GPU prior
    # (Pointcept/OpenPCSeg) is present, and vetoes candidates the trained
    # model confidently rejects. Absent file -> None -> geometry-only path.
    classifier = None
    if settings.learned_prior_enabled and not predictions:
        from .learned import load_classifier
        classifier = load_classifier(
            Path(settings.learned_classifier_path) if settings.learned_classifier_path else None
        )
        if classifier is not None:
            summary.warnings.append(
                f"Learned component classifier '{classifier.version}' active: trained class "
                "evidence feeds the model confidence factor (CPU prior; geometry still gates "
                "every asset)."
            )
            # Same dict object as summary.backend — provenance lands in run.json.
            backend["learned_classifier"] = {
                "version": classifier.version,
                "training": classifier.training,
            }
        if getattr(settings, "collect_training", False):
            settings.output_dir = str(output)
    elif getattr(settings, "collect_training", False):
        settings.output_dir = str(output)
    # Per-tile extraction cache: a run killed mid-detection resumes from the
    # tiles it already finished instead of redoing them (each tile's assets are
    # deterministic, so a cached tile is identical to a fresh extraction).
    # Cleared on fresh runs; survives only within the resume flow.
    cache_dir = output / "tile_cache"
    if not settings.resume_from_tiles:
        shutil.rmtree(cache_dir, ignore_errors=True)
    workers = _available_workers()
    if progress:
        print(f"[detection] processing {len(tile_names)} tiles on {workers} worker processes", flush=True)
    results: Dict[int, Tuple[List[Asset], int]] = {}
    completed = 0
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("fork")
    ) as pool:
        futures = {}
        for index, tile_name in enumerate(tile_names):
            cached = _read_tile_cache(cache_dir, tile_name)
            if cached is not None:
                results[index] = cached
                continue
            futures[pool.submit(_extract_tile_assets, output / "tiles", tile_name, summary, settings,
                               predictions, class_names, class_mapping, sample_stride, classifier)] = index
        try:
            for future in as_completed(futures):
                index = futures[future]
                tile_assets, point_count = future.result()
                results[index] = (tile_assets, point_count)
                _write_tile_cache(cache_dir, tile_names[index], tile_assets, point_count)
                completed += 1
                report("detecting", tiles_done=completed, tiles_total=len(tile_names),
                       message=f"Detecting assets in {tile_names[index]}")
        except Exception:
            for future in futures:
                future.cancel()
            raise
    for tile_index, tile_name in enumerate(tile_names):
        tile_assets, point_count = results[tile_index]
        if progress:
            counts = Counter(asset.asset_class for asset in tile_assets)
            print(
                f"  tile {tile_name}: {point_count:,} points | assets: {len(tile_assets)} "
                f"| classes: {dict(counts)}",
                flush=True,
            )
        _assign_ids_sequential(tile_assets, id_counts)
        all_assets.extend(tile_assets)
    all_assets.extend(roadmarking_assets)
    # Thin linear assets (overhead conductors) are cut by tile boundaries; join
    # the collinear pieces back into spans before QC/assessment so each span is
    # one asset (merges across shared poles are explicitly refused).
    all_assets = merge_linear_pieces(all_assets)
    # Linear safety assets (guardrails, concrete barriers) are also fragmented by
    # tile boundaries; re-join collinear pieces into one continuous roadside asset.
    all_assets = merge_linear_safety_pieces(all_assets)
    # Ground/marking detectors emit one blob per tile, so flat surfaces must be
    # re-joined by footprint overlap or a corridor reports dozens of pavement
    # "assets". Poles seen from both sides of a tile edge are deduplicated the
    # same way, before QC counts the inventory.
    all_assets = merge_surface_fragments(all_assets)
    # Painted markings fragment *within* tiles too (dashed centreline blobs ~9 m
    # apart on the same heading); join collinear fragments into one marking line.
    all_assets = merge_marking_segments(all_assets)
    all_assets = dedupe_pole_detections(all_assets)

    # ---- Optional learned backend: Gemini class-validation booster ----------------------
    # Geometry decided every asset; Gemini independently validates the assigned
    # class from the measured evidence and folds the verdict into the ``model``
    # confidence factor. Any failure (no key, network, parse) degrades to the
    # geometry-only confidence and records a warning - never a broken run, and
    # never a half-boosted inventory (docs/GEMINI.md).
    gemini_summary: Optional[dict] = None
    if settings.backend == "gemini":
        try:
            from .gemini import GeminiBooster

            report("detecting", message="Gemini class-validation pass")
            booster = GeminiBooster(
                model=settings.gemini_model,
                api_key=settings.gemini_api_key,
                cache_path=str(output / "gemini_cache.json"),
            )
            gemini_summary = booster.boost(all_assets)
            summary.backend = {
                "name": "gemini",
                "available": True,
                "gpu_required": False,
                "reason": f"Gemini class-validation booster ({settings.gemini_model}); "
                           "geometry detectors remain the asset gate.",
            }
            summary.warnings.append(
                f"Gemini validated {gemini_summary['applied']}/{gemini_summary['eligible']} "
                f"assets (model {settings.gemini_model}, {gemini_summary['api_calls']} API calls, "
                f"{gemini_summary['cache_hits']} cache hits, "
                f"{gemini_summary['disagreements']} disagreements routed to review)."
            )
        except Exception as exc:  # the booster is optional; never fail the run for it
            summary.warnings.append(f"Gemini validation skipped: {exc}")
            settings.backend = "geometry"

    # ---- Quality control + ALP assessment + export -----------------------------------
    report("exporting", assets=len(all_assets), message="Quality control, assessment and export")
    all_assets, qc_report = run_quality_control(all_assets, settings)
    assess_all(all_assets, settings.review_confidence_threshold)
    summary.assets = all_assets
    # Honesty gate: painted markings / rumble strips are detected from brightness
    # (RGB when present, intensity otherwise). A no-color export with flat
    # intensity (measured on the Trimble MX9 "no-color" variants: p99/median
    # ratio ~1.2, paint indistinguishable from asphalt) silently collapses the
    # marking count by an order of magnitude. Say it out loud instead of
    # reporting a misleadingly thin inventory.
    try:
        meta = validate_las(path, strict_las14=settings.strict_las14).metadata
    except Exception:
        meta = None
    if meta is not None and hasattr(meta, "has_rgb") and not meta.has_rgb:
        marking_count = sum(1 for a in all_assets if a.asset_class == "pavement_marking")
        rumble_count = sum(1 for a in all_assets if a.asset_class == "rumble_strip")
        signal = (
            "the intensity channel is flat (paint is not separable from asphalt)"
            if getattr(meta, "has_intensity", False)
            else "there is no intensity channel either"
        )
        summary.warnings.append(
            f"Input has NO color (RGB) channel: pavement markings and rumble strips are "
            f"detected from brightness, and here {signal}, so only high-retroreflectivity "
            f"paint was found ({marking_count} markings, {rumble_count} rumble strips). "
            f"Upload the color (RGB) export of the same scan for full paint detection."
        )
    # Overall dataset confidence (density/coverage, CRS, radiometrics, geometry
    # fit) — computed once per run, embedded in every export and the UI. Runs
    # after the honesty gate so the score sees the same warnings the report
    # carries; a scoring failure must never fail the run.
    try:
        summary.confidence_report = compute_confidence_report(
            point_count=summary.point_count,
            bounds=summary.bounds,
            tile_count=summary.tile_count,
            crs=summary.crs,
            point_format=summary.point_format,
            warnings=summary.warnings,
            assets=all_assets,
            has_intensity=(meta.has_intensity if meta is not None else None),
            tile_size_m=settings.tile_size_m,
        )
    except Exception as exc:  # the score must never fail a run
        summary.warnings.append(f"Confidence scoring skipped: {exc}")
    summary.elapsed_seconds = time.monotonic() - started

    # ---- Optional MongoDB mirror -------------------------------------------------------
    # Off by default; enabled by MONGO_URI (server/CLI env) or --mongo-uri.
    # The mirror must never break the pipeline: any failure is a warning, and
    # the canonical JSON/CSV/GeoJSON exports are unaffected.
    mongo_uri = mongo_uri or os.environ.get("MONGO_URI")
    if mongo_uri:
        try:
            mongo_result = export_to_mongo(
                [asset.to_dict() for asset in all_assets],
                {"input_path": str(path), "point_count": summary.point_count,
                 "crs": summary.crs, "processing_version": VERSION},
                mongo_uri, database=mongo_database, collection=mongo_collection,
            )
            summary.warnings.append(
                f"MongoDB mirror updated: {mongo_result['inserted']} assets -> "
                f"{mongo_result['database']}.{mongo_result['collection']}."
            )
        except Exception as exc:  # mirror is optional; never fail the run for it
            summary.warnings.append(f"MongoDB export skipped: {exc}")

    asset_classes = sorted({asset.asset_class for asset in all_assets})
    viewer["point_class"] = _compute_point_classes(viewer["points"], all_assets, asset_classes)
    viewer["point_class_names"] = asset_classes
    # Web/API runs do not need to keep multi-GB per-tile LAS files after
    # processing (the viewer + exports carry everything the app serves); the CLI
    # keeps them for Pointcept/roadmarking work and full point provenance. The
    # warning is recorded before export so it lands in run.json / inventory.json.
    if not settings.save_tiles:
        summary.warnings.append(
            "Per-tile LAS files were removed after processing (save_tiles=false); "
            "source_tile names and point indices are preserved in every asset, and "
            "representative evidence points ship in the per-asset exports."
        )
    write_outputs(output, summary, qc_report, viewer)
    write_viewer(output / "viewer")
    shutil.copy2(output / "viewer-data.json", output / "viewer" / "viewer-data.json")
    if not settings.save_tiles:
        shutil.rmtree(tiles_dir, ignore_errors=True)
        shutil.rmtree(cache_dir, ignore_errors=True)
    if progress:
        print(
            f"[done] {len(all_assets)} assets in {summary.elapsed_seconds:.1f}s -> {output}",
            flush=True,
        )
    report("done", assets=len(all_assets), message="Processing complete",
           elapsed_seconds=round(time.monotonic() - started, 2))
    return summary


def _progress(label: str, current: int, total: int) -> None:
    fraction = min(1.0, current / max(total, 1))
    bar_length = 24
    filled = int(bar_length * fraction)
    bar = "=" * filled + "-" * (bar_length - filled)
    percent = int(fraction * 100)
    print(f"\r[{bar}] {percent:3d}%  {label}  ({current:,}/{total:,})", end="", flush=True)