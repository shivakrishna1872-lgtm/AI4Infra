"""LAS validation: structured issues with severity, message, and fix hints.

Validation never silently fails. Issues are either ``info``, ``warning``
(processing continues, the issue is recorded in ``run.json``) or ``error``
(processing stops with a clear explanation). The competition source is LAS 1.4 /
Point Data Record Format 7, so those expectations are validated explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

from .errors import (
    EmptyPointCloudError,
    InvalidCoordinateDataError,
    MalformedLasError,
    MissingDimensionError,
    UnsupportedLasVersionError,
)
from .las_reader import DEFAULT_CRS_FALLBACK, EXPECTED_POINT_FORMAT, LAS14, read_metadata


@dataclass
class Issue:
    severity: str  # info | warning | error
    code: str
    message: str
    hint: str = ""

    def to_dict(self) -> dict:
        return {"severity": self.severity, "code": self.code, "message": self.message, "hint": self.hint}


@dataclass
class ValidationResult:
    metadata: object
    issues: List[Issue]
    point_count: int

    @property
    def errors(self) -> List[Issue]:
        return [issue for issue in self.issues if issue.severity == "error"]

    @property
    def warnings(self) -> List[Issue]:
        return [issue for issue in self.issues if issue.severity == "warning"]

    def as_warning_messages(self) -> List[str]:
        return [f"[{issue.code}] {issue.message}" for issue in self.warnings]

    def raise_if_invalid(self) -> None:
        if self.errors:
            first = self.errors[0]
            raise RuntimeError(f"{first.message} Suggested fix: {first.hint}")


# ---------------------------------------------------------------------------
# Bounded LAS header sanity check (upload gate)
# ---------------------------------------------------------------------------
#
# laspy.open() can spin forever on a text-mangled / truncated file: it loops
# `for _ in range(number_of_vlrs / number_of_evlrs)` reading from the stream,
# and once the stream is at EOF every read returns b"" with zero-length
# payloads, so the loop never terminates. A corrupt upload must therefore be
# rejected with hard structural bounds BEFORE laspy ever opens the file.
# Every check below is O(1) per header field and O(vlr-count) total (bounded
# by file size / minimum record size), so a hostile file cannot stall it.

_VLR_HEADER_BYTES = 54  # 2 reserved + 16 user_id + 2 record_id + 2 len + 32 desc
_EVLR_HEADER_BYTES = 60  # 2 reserved + 16 user_id + 2 record_id + 8 len + 32 desc
_MAX_VLRS = 100_000
_MAX_EVLRS = 1_000_000


def _u16(data: bytes, offset: int) -> int:
    return int.from_bytes(data[offset : offset + 2], "little")


def _u32(data: bytes, offset: int) -> int:
    return int.from_bytes(data[offset : offset + 4], "little")


def _u64(data: bytes, offset: int) -> int:
    return int.from_bytes(data[offset : offset + 8], "little")


def check_las_header_bounded(path: str | Path) -> Tuple[bool, str]:
    """Structurally validate a LAS header without trusting any count field.

    Returns ``(ok, reason)``. When ``ok`` is False, ``reason`` explains why
    the file cannot be a valid LAS (mangled header, truncated body, or
    unbounded VLR/EVLR counts). Cheap enough to run on every upload.
    """
    path = Path(path)
    try:
        size = path.stat().st_size
    except OSError as exc:
        return False, f"cannot stat file: {exc}"
    if size < 227:
        return False, f"file is only {size} bytes (a LAS header alone is 227+)"
    with path.open("rb") as handle:
        head = handle.read(435)  # 375-byte LAS 1.4 header + EVLR fields
    if len(head) < 227:
        return False, "file too short to hold a LAS header"
    if head[:4] != b"LASF":
        return False, "missing the LASF file signature (not a LAS file)"

    major, minor = head[24], head[25]
    if major == 1 and minor > 4:
        return False, f"unsupported LAS version 1.{minor}"
    if major == 2 and minor > 4:
        return False, f"unsupported LAS version 2.{minor}"
    if major not in (1, 2):
        return False, f"unrecognized LAS major version {major}"

    header_size = _u16(head, 94)
    if not 227 <= header_size <= 375:
        return False, f"header size {header_size} is outside the valid 227..375 range"
    # NOTE: parse the layout exactly the way laspy does, or the gate will
    # reject files laspy reads fine (and vice versa). laspy reads the offset
    # as a u32 at byte 96 and the VLR count as a u32 at byte 100 — a Trimble
    # MX9 export carries its real (small) offset and VLR count in those
    # legacy fields even though header_size is 375. Reading them as a u64
    # collides num_vlrs into the offset's high bytes and looks "huge".
    offset = _u32(head, 96)
    num_vlrs = _u32(head, 100)
    if num_vlrs > _MAX_VLRS:
        return False, f"{num_vlrs} VLRs declared — implausible (max {_MAX_VLRS})"

    # Walk the VLR table the way laspy does: each VLR is a 54-byte header
    # FOLLOWED BY ITS PAYLOAD, so headers are not contiguous. A laszip VLR
    # ("laszip encoded") marks a LAZ file, whose point-data offset and point
    # counts are sentinels/unset by design — those checks only apply to
    # uncompressed LAS. The walk is bounded: every VLR advances at least 54
    # bytes and the table is capped at 64 MiB (real files use < a few MiB).
    is_laz = False
    pos = header_size
    with path.open("rb") as handle:
        for _ in range(num_vlrs):
            if pos + _VLR_HEADER_BYTES > size:
                return False, "VLR table truncated — corrupted file"
            if pos - header_size > 64 * 1024 * 1024:
                return False, "VLR table implausibly large — corrupted header"
            handle.seek(pos)
            vlr_head = handle.read(_VLR_HEADER_BYTES)
            if len(vlr_head) < _VLR_HEADER_BYTES:
                return False, "VLR table truncated — corrupted file"
            if vlr_head[2:18].rstrip(b"\0") == b"laszip encoded":
                is_laz = True
            payload = _u16(vlr_head, 20)
            pos += _VLR_HEADER_BYTES + payload
    if pos > size:
        return False, "VLR table exceeds the file size — corrupted file"

    if not is_laz:
        # Uncompressed LAS: the offset must be a real position inside the file
        # and the VLR table must end before it.
        if offset < header_size:
            return False, f"point-data offset {offset} is before the header end ({header_size})"
        if offset > size:
            return False, f"point-data offset {offset} exceeds the file size ({size}) — truncated or mangled header"
        if pos > offset:
            return False, "VLR table overruns the point-data offset — corrupted header"

    point_format = head[104]
    if is_laz:
        # laszip writers set the high bit (0x80) as a compressed-data flag on
        # the format byte; laspy masks it before interpreting the format.
        point_format &= 0x7F
    if point_format > 10:
        return False, f"point data record format {point_format} is invalid (0..10)"
    record_length = _u16(head, 105)
    if not 20 <= record_length <= 1024:
        return False, f"point data record length {record_length} is outside 20..1024"

    if not is_laz:
        # laspy reads the u64 point count at byte 247 for minor >= 4 (the
        # legacy u32 at 107 is 0 on such files), else the u32 at 107.
        if minor >= 4:
            count = _u64(head, 247)
        else:
            count = _u32(head, 107)
        if count and offset + count * record_length > size + record_length:
            return False, (
                f"header declares {count} points × {record_length}-byte records = "
                f"{offset + count * record_length} bytes, but the file is only {size} "
                "bytes — truncated or mangled file"
            )

    if not is_laz and minor >= 4:
        # laspy-compatible offsets: waveform record at 227, start of first
        # EVLR at 235, number of EVLRs at 243.
        num_evlrs = _u32(head, 243)
        start_evlr = _u64(head, 235)
        if num_evlrs:
            if num_evlrs > _MAX_EVLRS:
                return False, f"{num_evlrs} EVLRs declared — implausible (max {_MAX_EVLRS})"
            if start_evlr < offset:
                return False, "EVLR start precedes the point data — corrupted header"
            if start_evlr > size or start_evlr + num_evlrs * _EVLR_HEADER_BYTES > size:
                return False, "EVLR table exceeds the file size — corrupted header"
            pos = start_evlr
            for _ in range(num_evlrs):
                if pos + _EVLR_HEADER_BYTES > size:
                    return False, "EVLR table overruns the file size — corrupted header"
                with path.open("rb") as handle:
                    handle.seek(pos)
                    evlr_head = handle.read(_EVLR_HEADER_BYTES)
                if len(evlr_head) < _EVLR_HEADER_BYTES:
                    return False, "EVLR header truncated — corrupted file"
                payload = _u64(evlr_head, 20)
                if pos + _EVLR_HEADER_BYTES + payload > size:
                    return False, "an EVLR payload overruns the file size — corrupted header"
                pos += _EVLR_HEADER_BYTES + payload
    return True, ""


def validate_las(
    path: str | Path, strict_las14: bool = False,
    crs_fallback: str | None = DEFAULT_CRS_FALLBACK,
) -> ValidationResult:
    """Validate a LAS file against the competition expectations.

    ``crs_fallback`` (e.g. ``EPSG:6553``, the Mannford OK default) is forced
    at load time when the header carries no resolvable CRS — the laspy
    analogue of PDAL's ``override_srs`` — and reported as a CRS_FALLBACK
    warning instead of CRS_UNRESOLVED. ``None`` keeps the old never-assume
    behaviour.

    Raises :class:`infra_inventory.errors.InfraError` subclasses for hard
    failures; returns structured issues otherwise.
    """
    path = Path(path).expanduser().resolve()
    metadata = read_metadata(path, crs_fallback=crs_fallback)
    issues: List[Issue] = []

    if metadata.version != LAS14:
        message = f"LAS {metadata.version} supplied; the competition source is LAS 1.4."
        if strict_las14:
            raise UnsupportedLasVersionError(metadata.version, strict=True)
        issues.append(Issue(
            "warning", "LAS_VERSION",
            message,
            "If accuracy matters, re-export as LAS 1.4 (CloudCompare: 'Save as' -> LAS 1.4).",
        ))
    else:
        issues.append(Issue("info", "LAS_VERSION", "LAS 1.4 confirmed."))

    if metadata.point_format != EXPECTED_POINT_FORMAT:
        issues.append(Issue(
            "warning", "POINT_FORMAT",
            f"Point Data Record Format {metadata.point_format}; format 7 (RGB + returns) is expected.",
            "Supported dimensions are used; RGB/return-based stages degrade gracefully.",
        ))
    else:
        issues.append(Issue("info", "POINT_FORMAT", "Point Data Record Format 7 confirmed."))

    if metadata.point_count == 0:
        raise EmptyPointCloudError()

    if metadata.crs is None:
        issues.append(Issue(
            "warning", "CRS_UNRESOLVED",
            "The coordinate reference system could not be resolved from LAS metadata. Outputs will report CRS_UNRESOLVED.",
            "Re-export the LAS with CRS/VLR metadata, or document the expected EPSG code manually.",
        ))
    elif metadata.crs_fallback_used:
        issues.append(Issue(
            "warning", "CRS_FALLBACK",
            f"No CRS in the LAS header; assumed {metadata.crs} (configured fallback, "
            "laspy override_srs analogue). Distance units follow that EPSG.",
            "Verify this is the capture's real zone; override configs/processing.yaml "
            "crs_fallback if not.",
        ))
    else:
        issues.append(Issue("info", "CRS", f"CRS resolved: {metadata.crs}"))

    if not metadata.has_intensity:
        issues.append(Issue(
            "warning", "NO_INTENSITY",
            "The LAS has no intensity dimension; reflective-marking detection is disabled.",
            "Re-export with intensity enabled if the scanner recorded it.",
        ))
    if not metadata.has_rgb:
        issues.append(Issue(
            "warning", "NO_RGB",
            "The LAS has no RGB dimensions; brightness-based evidence is disabled.",
            "Re-export with RGB (format 7) if color was captured.",
        ))
    if metadata.has_nir:
        issues.append(Issue(
            "info", "NIR",
            "Infrared (NIR) channel detected (LAS 1.4 point format 8); NIR is fused "
            "into pavement-marking brightness evidence alongside intensity/RGB.",
        ))
    if not metadata.has_returns:
        issues.append(Issue(
            "info", "NO_RETURNS",
            "Return information is absent; multi-return features are skipped.",
            "The pipeline works without returns; nothing to fix.",
        ))
    if not metadata.has_gps_time:
        issues.append(Issue(
            "info", "NO_GPS_TIME",
            "gps_time is absent; run attribution falls back to 'unknown'.",
            "Re-export with gps_time to enable Run 1/Run 2 attribution.",
        ))
    if not metadata.has_point_source_id:
        issues.append(Issue(
            "info", "NO_POINT_SOURCE_ID",
            "point_source_id is absent; scanner attribution falls back to 'unknown'.",
            "Re-export with point_source_id to attribute Laser Left/Laser Right.",
        ))

    x0, y0, z0, x1, y1, z1 = metadata.bounds
    if not all(np.isfinite([x0, y0, z0, x1, y1, z1])) or (x1 <= x0 and y1 <= y0 and z1 <= z0):
        raise InvalidCoordinateDataError("spatial bounds are empty or non-finite")

    issues.append(Issue("info", "POINT_COUNT", f"{metadata.point_count:,} points."))
    issues.append(Issue(
        "info", "BOUNDS",
        f"Bounds X [{x0:.2f}, {x1:.2f}] Y [{y0:.2f}, {y1:.2f}] Z [{z0:.2f}, {z1:.2f}]",
    ))
    return ValidationResult(metadata=metadata, issues=issues, point_count=metadata.point_count)


def ensure_dimension(metadata: object, name: str, requested_by: str) -> None:
    if name == "intensity" and not metadata.has_intensity:
        raise MissingDimensionError(name, requested_by)
    if name == "rgb" and not metadata.has_rgb:
        raise MissingDimensionError(name, requested_by)
    if name == "gps_time" and not metadata.has_gps_time:
        raise MissingDimensionError(name, requested_by)