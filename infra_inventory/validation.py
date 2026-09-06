"""LAS validation: structured issues with severity, message, and fix hints.

Validation never silently fails. Issues are either ``info``, ``warning``
(processing continues, the issue is recorded in ``run.json``) or ``error``
(processing stops with a clear explanation). The competition source is LAS 1.4 /
Point Data Record Format 7, so those expectations are validated explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

from .errors import (
    EmptyPointCloudError,
    InvalidCoordinateDataError,
    MalformedLasError,
    MissingDimensionError,
    UnsupportedLasVersionError,
)
from .las_reader import EXPECTED_POINT_FORMAT, LAS14, read_metadata


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


def validate_las(path: str | Path, strict_las14: bool = False) -> ValidationResult:
    """Validate a LAS file against the competition expectations.

    Raises :class:`infra_inventory.errors.InfraError` subclasses for hard
    failures; returns structured issues otherwise.
    """
    path = Path(path).expanduser().resolve()
    metadata = read_metadata(path)
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