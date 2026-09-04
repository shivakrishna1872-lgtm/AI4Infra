from pathlib import Path

import laspy
import numpy as np

from infra_inventory.pipeline import process_las


def test_processes_las_and_writes_inventory(tmp_path: Path) -> None:
    points = 900
    header = laspy.LasHeader(point_format=7, version="1.4")
    header.scales = [0.01, 0.01, 0.01]
    cloud = laspy.LasData(header)
    grid = np.linspace(0, 25, points)
    cloud.x, cloud.y = grid, np.mod(grid * 1.7, 12)
    cloud.z = np.where(np.arange(points) % 10 == 0, 0.4, 0.0)
    cloud.intensity = np.where(np.arange(points) % 10 == 0, 2000, 60)
    cloud.red = np.full(points, 55000, dtype=np.uint16)
    cloud.green = np.full(points, 55000, dtype=np.uint16)
    cloud.blue = np.full(points, 55000, dtype=np.uint16)
    source = tmp_path / "sample.las"
    cloud.write(source)

    result = process_las(source, tmp_path / "out")

    assert result.point_count == points
    assert (tmp_path / "out" / "assets.json").is_file()
    assert (tmp_path / "out" / "assets.geojson").is_file()
    assert (tmp_path / "out" / "viewer" / "index.html").is_file()
    assert any(asset.asset_class == "pavement" for asset in result.assets)
