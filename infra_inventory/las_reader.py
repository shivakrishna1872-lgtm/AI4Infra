"""Streaming LAS/LAZ access with metadata extraction.

Never loads an entire LAS file into memory: points are consumed in chunks of
``chunk_size`` through laspy's chunk iterator. The competition source is LAS 1.4
/ Point Data Record Format 7 (Trimble MX9, two scanner heads), so the reader
also surfaces ``gps_time`` (run separation) and ``point_source_id`` (scanner
attribution) whenever the file exposes them.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

import laspy
import numpy as np

from .errors import (
    EmptyPointCloudError,
    InvalidCoordinateDataError,
    InputNotFoundError,
    LazBackendMissingError,
    MalformedLasError,
)


def _laz_backend_error(exc: Exception) -> bool:
    """True when the exception is laspy complaining a LAZ backend is missing."""
    message = str(exc).lower()
    return "laz" in message and ("backend" in message or "lazrs" in message or "laszip" in message)

LAS14 = "1.4"
EXPECTED_POINT_FORMAT = 7


@dataclass
class LasMetadata:
    path: str
    version: str
    point_format: int
    point_count: int
    crs: Optional[str]
    scales: Tuple[float, float, float]
    offsets: Tuple[float, float, float]
    has_intensity: bool
    has_rgb: bool
    has_returns: bool
    has_gps_time: bool
    has_point_source_id: bool
    has_classification: bool
    bounds: Tuple[float, float, float, float, float, float]

    def to_dict(self) -> Dict[str, object]:
        return {
            "path": self.path,
            "version": self.version,
            "point_format": self.point_format,
            "point_count": self.point_count,
            "crs": self.crs,
            "scales": list(self.scales),
            "offsets": list(self.offsets),
            "has_intensity": self.has_intensity,
            "has_rgb": self.has_rgb,
            "has_returns": self.has_returns,
            "has_gps_time": self.has_gps_time,
            "has_point_source_id": self.has_point_source_id,
            "has_classification": self.has_classification,
            "bounds": list(self.bounds),
        }


@dataclass
class LasChunk:
    """One streaming chunk of point data, already extracted into numpy arrays."""

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    intensity: np.ndarray
    rgb: Optional[np.ndarray]  # N x 3 float64 in 0..65535
    return_number: Optional[np.ndarray]
    number_of_returns: Optional[np.ndarray]
    gps_time: Optional[np.ndarray]
    point_source_id: Optional[np.ndarray]
    classification: Optional[np.ndarray]
    global_offset: int  # first global point index of this chunk


def parse_crs(reader: laspy.LasReader) -> Optional[str]:
    """Return the CRS as a string (e.g. EPSG:XXXX) or None when unresolved."""
    try:
        parsed = reader.header.parse_crs()
        if parsed is None:
            return None
        text = parsed.to_string()
        return text or None
    except Exception:
        return None


def read_metadata(path: str | Path) -> LasMetadata:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise InputNotFoundError(str(path))
    try:
        with laspy.open(path) as reader:
            header = reader.header
            names = set(header.point_format.dimension_names)
            version = f"{header.version.major}.{header.version.minor}"
            bounds = (
                float(header.mins[0]), float(header.mins[1]), float(header.mins[2]),
                float(header.maxs[0]), float(header.maxs[1]), float(header.maxs[2]),
            )
            return LasMetadata(
                path=str(path),
                version=version,
                point_format=int(header.point_format.id),
                point_count=int(header.point_count),
                crs=parse_crs(reader),
                scales=tuple(float(v) for v in header.scales),
                offsets=tuple(float(v) for v in header.offsets),
                has_intensity="intensity" in names,
                has_rgb={"red", "green", "blue"}.issubset(names),
                has_returns={"return_number", "number_of_returns"}.issubset(names),
                has_gps_time="gps_time" in names,
                has_point_source_id="point_source_id" in names,
                has_classification="classification" in names,
                bounds=bounds,
            )
    except EmptyPointCloudError:
        raise
    except LazBackendMissingError:
        raise
    except Exception as exc:  # laspy raises generic OSErrors/ValueErrors on malformed files
        if _laz_backend_error(exc):
            raise LazBackendMissingError() from exc
        raise MalformedLasError(str(exc)) from exc


def iter_chunks(path: str | Path, chunk_size: int, stride: int = 1) -> Iterator[Tuple[LasMetadata, laspy.LasReader, LasChunk]]:
    """Yield (metadata, reader, chunk) tuples without buffering the whole cloud.

    Yields the metadata once per chunk so callers can stream without keeping
    state; the reader is yielded so scale-aware dimensions are cheap.

    ``stride > 1`` applies uniform per-chunk thinning (the streaming analogue of
    LAStools' ``las2las -thin``): every ``stride``-th point of each decompressed
    chunk is kept, so arbitrarily large files can be processed with bounded
    memory and time while preserving the full spatial extent and per-chunk
    global point offsets (``LasChunk.global_offset`` still counts source points).
    """
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise InputNotFoundError(str(path))
    try:
        with laspy.open(path) as reader:
            metadata = read_metadata(path)
            if metadata.point_count == 0:
                raise EmptyPointCloudError()
            global_offset = 0
            for raw_chunk in reader.chunk_iterator(chunk_size):
                chunk_length = len(raw_chunk)
                if stride > 1:
                    # LAStools-style uniform thinning inside the chunk. Slicing the
                    # point record keeps scale-aware dimensions intact; memory stays
                    # bounded by chunk_size / stride points.
                    raw_chunk = raw_chunk[::stride]
                names = set(raw_chunk.point_format.dimension_names)
                x = np.asarray(raw_chunk.x, dtype=np.float64)
                y = np.asarray(raw_chunk.y, dtype=np.float64)
                z = np.asarray(raw_chunk.z, dtype=np.float64)
                if not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isfinite(z).all():
                    raise InvalidCoordinateDataError("non-finite X/Y/Z values found in chunk")
                intensity = (
                    np.asarray(raw_chunk.intensity, dtype=np.float64)
                    if "intensity" in names
                    else np.zeros(len(x), dtype=np.float64)
                )
                rgb = None
                if {"red", "green", "blue"}.issubset(names):
                    rgb = np.column_stack((
                        np.asarray(raw_chunk.red, dtype=np.float64),
                        np.asarray(raw_chunk.green, dtype=np.float64),
                        np.asarray(raw_chunk.blue, dtype=np.float64),
                    ))
                chunk = LasChunk(
                    x=x,
                    y=y,
                    z=z,
                    intensity=intensity,
                    rgb=rgb,
                    return_number=(np.asarray(raw_chunk.return_number, dtype=np.int64) if "return_number" in names else None),
                    number_of_returns=(np.asarray(raw_chunk.number_of_returns, dtype=np.int64) if "number_of_returns" in names else None),
                    gps_time=(np.asarray(raw_chunk.gps_time, dtype=np.float64) if "gps_time" in names else None),
                    point_source_id=(np.asarray(raw_chunk.point_source_id, dtype=np.int64) if "point_source_id" in names else None),
                    classification=(np.asarray(raw_chunk.classification, dtype=np.int64) if "classification" in names else None),
                    global_offset=global_offset,
                )
                global_offset += chunk_length
                yield metadata, reader, chunk
    except EmptyPointCloudError:
        raise
    except InvalidCoordinateDataError:
        raise
    except Exception as exc:
        if _laz_backend_error(exc):
            raise LazBackendMissingError() from exc
        raise MalformedLasError(str(exc)) from exc


def gps_run_labels(gps_time: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Split a GPS-time trace into run labels (Run 1 / Run 2 ...) using 1D k-means.

    Returns ``None`` when the signal cannot be separated into at least two
    populations with a meaningful share of points each.
    """
    if gps_time is None or len(gps_time) < 1000:
        return None
    values = np.asarray(gps_time, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if len(finite) < 1000:
        return None
    low, high = float(finite.min()), float(finite.max())
    if high - low < 1e-3:
        return None
    # 1D k-means with k=2, deterministic initial split at the midpoint
    center_a, center_b = low, high
    labels = np.empty(len(values), dtype=np.int64)
    for _ in range(40):
        labels = np.where(np.abs(values - center_a) <= np.abs(values - center_b), 0, 1)
        new_a = float(values[labels == 0].mean()) if np.any(labels == 0) else center_a
        new_b = float(values[labels == 1].mean()) if np.any(labels == 1) else center_b
        if abs(new_a - center_a) < 1e-9 and abs(new_b - center_b) < 1e-9:
            center_a, center_b = new_a, new_b
            break
        center_a, center_b = new_a, new_b
    share = min(float((labels == 0).mean()), float((labels == 1).mean()))
    if share < 0.05:
        return None
    # True bimodality check: the two pass populations must be separated by much
    # more than their own spread (a uniform distribution splits mid-range too).
    cluster_a, cluster_b = values[labels == 0], values[labels == 1]
    separation = abs(center_a - center_b)
    spread = max(float(cluster_a.std()), float(cluster_b.std()), 1e-9)
    # Two real survey passes are separated by far more than their own spread
    # (uniform noise splits mid-range at a ratio around 3.5)
    if separation <= 6.0 * spread:
        return None
    ordered = 0 if center_a < center_b else 1
    labels = np.where(labels == ordered, 1, 2)
    return labels