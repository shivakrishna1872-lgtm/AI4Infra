from __future__ import annotations

from pathlib import Path

import laspy
import numpy as np
import pytest

from infra_inventory.errors import EmptyPointCloudError, UnsupportedLasVersionError
from infra_inventory.validation import validate_las


def _write_las(path: Path, version: str, point_format: int, points: int = 100) -> None:
    header = laspy.LasHeader(point_format=point_format, version=version)
    header.scales = [0.01, 0.01, 0.01]
    cloud = laspy.LasData(header)
    cloud.x = np.linspace(0, 10, points)
    cloud.y = np.linspace(0, 10, points)
    cloud.z = np.zeros(points)
    cloud.write(path)


def test_valid_las14_format7(synthetic_las: Path) -> None:
    result = validate_las(synthetic_las)
    assert result.point_count == read_count(synthetic_las)
    assert not result.errors
    codes = {issue.code for issue in result.issues}
    assert "LAS_VERSION" in codes
    assert "POINT_FORMAT" in codes


def read_count(path: Path) -> int:
    import laspy

    with laspy.open(path) as reader:
        return int(reader.header.point_count)


def test_las12_warns_and_strict_fails(tmp_path: Path) -> None:
    path = tmp_path / "old.las"
    _write_las(path, "1.2", 3)
    result = validate_las(path)
    assert any(issue.code == "LAS_VERSION" and issue.severity == "warning" for issue in result.issues)
    with pytest.raises(UnsupportedLasVersionError):
        validate_las(path, strict_las14=True)


def test_crs_unresolved_warning(synthetic_las: Path) -> None:
    result = validate_las(synthetic_las)
    assert any(issue.code == "CRS_UNRESOLVED" for issue in result.issues)


def test_empty_cloud_raises(tmp_path: Path) -> None:
    path = tmp_path / "empty.las"
    _write_las(path, "1.4", 7, points=0)
    with pytest.raises(EmptyPointCloudError):
        validate_las(path)


def test_missing_file_raises(tmp_path: Path) -> None:
    from infra_inventory.errors import InputNotFoundError

    with pytest.raises(InputNotFoundError):
        validate_las(tmp_path / "missing.las")


def test_warning_messages_have_hints(synthetic_las: Path) -> None:
    result = validate_las(synthetic_las)
    for issue in result.warnings:
        assert issue.hint, f"warning {issue.code} lacks a hint"