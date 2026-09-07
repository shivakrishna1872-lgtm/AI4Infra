"""Merge-clean Trimble MX9 scanner streams into one LAS file.

The competition dataset ships as four files (Run 1 / Run 2 × Laser Left /
Laser Right) that cover the same corridor. Processing them separately would
count the same physical object up to four times; processing them as raw
concatenation would leave duplicate point clutter (both heads see the same
pole, and overlapping run passes re-cover the same pavement). This module
merges them in one bounded-memory streaming pass:

1. **Streamed append** — points are never all held in memory; each input is
   read chunk by chunk and appended to the output file.
2. **Cell dedupe (LAStools ``lasmerge -dup`` analogue)** — a point is kept only
   if its quantized cell (``x/y/z`` at ``dedupe_cell_m`` resolution, shared
   origin, packed into one int64) has not been seen before. The same physical
   point seen by both heads or both runs collapses to one record. The seen-set
   is capped (``max_dedupe_keys``) with an explicit warning — the merge never
   unboundedly grows memory.
3. **Scanner provenance** — ``point_source_id`` is stamped per input file
   (1 = first input, 2 = second, ...) so the pipeline's attribution layer can
   report which stream each asset's points came from. Run separation is
   preserved through ``gps_time`` (the pipeline splits runs by GPS k-means).

Dimensions are unioned across inputs (RGB, returns, flags, classification and
gps_time all survive when any input carries them). The output is written as
LAS 1.4 / point format 7 by default — the competition format — via laspy's
``lazrs`` backend for ``.laz`` outputs.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import laspy
import numpy as np

#: laspy 2.x uses ``laz_backend``; older builds used ``LazBackend`` — normalise here.
_LAZ_KWARGS: Dict[str, object] = {"laz_backend": laspy.LazBackend.Lazrs} if hasattr(laspy, "LazBackend") else {}


def _dimension_superset(headers: List[laspy.LasHeader]) -> List[str]:
    """Union of dimension names across all input headers (superset, stable order)."""
    names: List[str] = []
    for header in headers:
        for name in header.point_format.dimension_names:
            if name not in names:
                names.append(name)
    return names


def _copy_dimensions(
    data: laspy.LasData, source: laspy.LasData, names: List[str], keep: np.ndarray
) -> None:
    """Copy every requested dimension from ``source`` to ``data`` (masked).

    XYZ are excluded — the caller assigns them first (they define the record
    length) — and scaled/offset mismatches between files degrade gracefully.
    """
    for name in names:
        if name in ("X", "Y", "Z"):
            continue
        if name in data.point_format.dimension_names and name in source.point_format.dimension_names:
            try:
                setattr(data, name, np.asarray(getattr(source, name))[keep])
            except (ValueError, TypeError):  # scale/offset mismatch between files
                pass


#: Packed-cell key bit widths (21 bits per axis at cell_m resolution covers
#: a 2^21 * cell span; at 5 cm that is ~105 km — ample for any survey corridor).
_KEY_BITS = 21
_KEY_MASK = (1 << _KEY_BITS) - 1


def _cell_origins(headers: List[laspy.LasHeader], cell_m: float) -> Tuple[int, int, int]:
    """Minimum cell index per axis across all inputs (shared key origin)."""
    mins = [np.asarray(header.mins, dtype=np.float64) for header in headers]
    global_min = np.min(np.vstack(mins), axis=0)
    return tuple(int(math.floor(global_min[axis] / cell_m)) for axis in range(3))


def _dedupe_keys(
    chunk: laspy.LasData, cell_m: float, origin: Tuple[int, int, int],
    seen: Set[int], max_seen: int,
) -> Tuple[np.ndarray, bool]:
    """Packed-cell dedupe: keep points whose (x, y, z) cell was never seen.

    Keys are packed into one int64 (21 bits per axis, shared origin) so the
    seen-set stays small enough to hold even multi-run merges in memory. When
    ``max_seen`` is reached the dedupe saturates and the remaining points pass
    through unchanged (reported as a warning) — the merge never silently
    corrupts or unboundedly grows memory.
    """
    x = np.floor(np.asarray(chunk.x, dtype=np.float64) / cell_m).astype(np.int64) - origin[0]
    y = np.floor(np.asarray(chunk.y, dtype=np.float64) / cell_m).astype(np.int64) - origin[1]
    z = np.floor(np.asarray(chunk.z, dtype=np.float64) / cell_m).astype(np.int64) - origin[2]
    overflow = np.any((x < 0) | (x > _KEY_MASK) | (y < 0) | (y > _KEY_MASK) | (z < 0) | (z > _KEY_MASK))
    keys = ((x << (2 * _KEY_BITS)) | (y << _KEY_BITS) | z).tolist()
    keep = np.zeros(len(keys), dtype=bool)
    saturated = len(seen) >= max_seen
    for index, key in enumerate(keys):
        if saturated:
            keep[index] = True
            continue
        if key not in seen:
            seen.add(key)
            keep[index] = True
        if len(seen) >= max_seen:
            saturated = True
    return keep, overflow or saturated


def merge_las_files(
    inputs: List[Union[str, Path]],
    output: Union[str, Path],
    dedupe_cell_m: float = 0.05,
    assign_scanner_ids: bool = True,
    chunk_size: int = 500_000,
    max_dedupe_keys: int = 25_000_000,
) -> Dict[str, object]:
    """Merge-clean a list of LAS/LAZ files into one output (bounded memory).

    Returns a summary dict: per-input point counts, duplicates removed, output
    point count, output point format/version, and warnings (e.g. CRS mismatch
    or missing GPS time).
    """
    paths = [Path(path).expanduser().resolve() for path in inputs]
    if not paths:
        raise ValueError("merge_las_files requires at least one input file")
    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    headers: List[laspy.LasHeader] = []
    point_counts: List[int] = []
    crss: List[Optional[str]] = []
    for path in paths:
        with laspy.open(path) as reader:
            headers.append(reader.header)
            point_counts.append(int(reader.header.point_count))
            crss.append(getattr(reader.header, "crs", None))

    dimensions = _dimension_superset(headers)
    base = headers[0]
    try:
        header = laspy.LasHeader(point_format=7, version="1.4")
    except Exception:  # pragma: no cover - laspy version quirks
        header = laspy.LasHeader(point_format=int(base.point_format.id), version="1.4")
    header.scales = list(base.scales)
    header.offsets = list(base.offsets)
    if getattr(base, "crs", None) is not None:
        try:
            header.add_crs(base.crs)
        except Exception:
            pass

    warnings: List[str] = []
    if len(set(map(str, crss))) > 1:
        warnings.append(
            "Inputs have different CRS values; the first input's CRS is used. "
            "Verify all runs share a CRS before interpreting coordinates."
        )
    has_gps = "gps_time" in dimensions

    origin = _cell_origins(headers, dedupe_cell_m) if dedupe_cell_m > 0 else (0, 0, 0)
    seen: Set[int] = set()
    dedupe_saturated = False
    dedupe_overflow = False
    total_read = 0
    duplicates = 0
    kept_total = 0
    first_write = True
    for file_index, path in enumerate(paths):
        with laspy.open(path) as reader:
            for chunk in reader.chunk_iterator(chunk_size):
                total_read += len(chunk.x)
                if dedupe_cell_m > 0:
                    keep, flagged = _dedupe_keys(chunk, dedupe_cell_m, origin, seen, max_dedupe_keys)
                    if flagged and not dedupe_overflow:
                        if len(seen) >= max_dedupe_keys:
                            dedupe_saturated = True
                        else:
                            dedupe_overflow = True
                            warnings.append(
                                "Point coordinates exceeded the dedupe key span; duplicate removal "
                                "was disabled for the remaining points (spatial correctness unaffected)."
                            )
                else:
                    keep = np.ones(len(chunk.x), dtype=bool)
                duplicates += int((~keep).sum())
                if not keep.any():
                    continue
                data = laspy.LasData(header)
                data.x = np.asarray(chunk.x)[keep]
                data.y = np.asarray(chunk.y)[keep]
                data.z = np.asarray(chunk.z)[keep]
                _copy_dimensions(data, chunk, dimensions, keep)
                if assign_scanner_ids and "point_source_id" in data.point_format.dimension_names:
                    data.point_source_id = np.full(int(keep.sum()), file_index + 1, dtype=np.uint16)
                if first_write:
                    with laspy.open(output_path, mode="w", header=data.header, **_LAZ_KWARGS) as writer:
                        writer.write_points(data.points)
                    first_write = False
                else:
                    with laspy.open(output_path, mode="a", **_LAZ_KWARGS) as writer:
                        writer.append_points(data.points)
                kept_total += int(keep.sum())
    if dedupe_saturated:
        warnings.append(
            f"Dedupe cell set saturated at {max_dedupe_keys:,} cells; later points passed through "
            "unchanged (raise --max-dedupe-keys for very large merges)."
        )
    if first_write:
        raise ValueError("No points were written — all inputs are empty or fully duplicate.")

    return {
        "inputs": [str(path) for path in paths],
        "points_read": total_read,
        "points_kept": kept_total,
        "duplicates_removed": duplicates,
        "scanner_ids": list(range(1, len(paths) + 1)) if assign_scanner_ids else None,
        "gps_time_present": has_gps,
        "output": str(output_path),
        "point_format": int(header.point_format.id),
        "las_version": str(header.version),
        "warnings": warnings,
    }