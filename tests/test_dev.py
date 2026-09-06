"""Dev-facing simulation tests: CLI simulate, server endpoints, LAZ upload support."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from infra_inventory.simulation import run_quick_simulation, run_small_synthetic


def _run_cli(argv: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    env = dict(__import__("os").environ)
    env["PYTHONPATH"] = str(cwd or Path.cwd())
    return subprocess.run(
        [sys.executable, "-m", "infra_inventory", *argv],
        cwd=str(cwd) if cwd else None,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )


def test_cli_simulate_runs_quick_pipeline(tmp_path: Path) -> None:
    result = _run_cli(["simulate", "--output", str(tmp_path), "--length", "100.0", "--seed", "11"])
    assert result.returncode == 0, result.stderr + result.stdout
    parsed = json.loads(result.stdout.split("\n\n")[0])
    assert parsed["simulated"] is True
    assert parsed["points"] > 0
    assert parsed["assets"] > 0
    assert parsed["scene_parts"]["utility_poles"]["count"] == 6
    assert parsed["scene_parts"]["traffic_signs"]["count"] == 3


def test_small_synthetic_yields_expected_assets(tmp_path: Path) -> None:
    project = run_small_synthetic(tmp_path / "tiny", seed=7)
    assets = json.loads(
        (Path(project["output_dir"]) / "pipeline" / "assets.json").read_text(encoding="utf-8")
    )
    classes = {asset["class"] for asset in assets}
    assert "pavement" in classes
    assert "utility_pole" in classes
    assert "guardrail" in classes


def _server_client(tmp_path: Path):
    from fastapi.testclient import TestClient

    from infra_inventory import server

    server.DATA_DIR = tmp_path / "appdata"
    return TestClient(server.app)


def _wait_done(client, job_id, timeout=180):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        response = client.get(f"/api/jobs/{job_id}")
        assert response.status_code == 200, response.text
        last = response.json()
        if last["stage"] in ("done", "error"):
            return last
        time.sleep(0.15)
    raise AssertionError(f"job never finished: {last}")


def test_server_quick_simulation_is_processed(tmp_path: Path) -> None:
    client = _server_client(tmp_path)
    response = client.post("/api/simulate")
    assert response.status_code == 200, response.text
    project = response.json()
    assert project["simulated"] is True
    assert project["processed"] is True
    assert project["input_file"].endswith(".las")
    assert project["point_count"] > 0
    assert project["asset_count"] > 0
    assert project.get("scene")
    assert project["summary"]["simulation_note"].startswith("SIMULATION")

    # viewer data available at the standard endpoint immediately
    viewer = client.get(f"/api/projects/{project['id']}/viewer-data")
    assert viewer.status_code == 200
    payload = viewer.json()
    assert len(payload["points"]) > 0
    assert len(payload["assets"]) == project["asset_count"]
    assert payload["run"]["point_count"] == project["point_count"]

    # project shows in the list as processed
    listing = {p["id"]: p for p in client.get("/api/projects").json()}
    assert listing[project["id"]]["processed"] is True


def test_server_upload_laz_process_roundtrip(tmp_path: Path) -> None:
    """The 502 regression: uploading a .laz must validate and process cleanly."""
    import laspy
    import numpy as np

    src = tmp_path / "source.laz"
    header = laspy.LasHeader(point_format=7, version="1.4")
    header.scales = [0.01, 0.01, 0.01]
    data = laspy.LasData(header)
    n = 400
    data.x = np.linspace(0, 30, n)
    data.y = np.linspace(-6, 6, n)
    data.z = np.zeros(n)
    data.intensity = np.full(n, 1200, np.uint16)
    data.red = np.full(n, 9000, np.uint16)
    data.green = np.full(n, 9000, np.uint16)
    data.blue = np.full(n, 10000, np.uint16)
    data.write(src, laz_backend=laspy.LazBackend.Lazrs)

    client = _server_client(tmp_path)
    created = client.post("/api/projects", params={"name": "laz test"}).json()
    with src.open("rb") as handle:
        upload = client.post(
            f"/api/projects/{created['id']}/upload",
            files={"file": ("corridor.laz", handle, "application/octet-stream")},
        )
    assert upload.status_code == 200, upload.text
    assert upload.json()["input_file"].endswith(".laz")

    proc = client.post(f"/api/projects/{created['id']}/process", json={})
    assert proc.status_code == 200
    status = _wait_done(client, proc.json()["job_id"])
    assert status["stage"] == "done", status
    assert status["points_processed"] == n
    assert client.get(f"/api/projects/{created['id']}/viewer-data").status_code == 200


def test_server_rejects_unsupported_upload_file(tmp_path: Path) -> None:
    client = _server_client(tmp_path)
    created = client.post("/api/projects", params={"name": "bad upload"}).json()
    response = client.post(
        f"/api/projects/{created['id']}/upload",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 400
    assert "las" in response.json()["detail"].lower()


def test_server_quick_simulation_list_and_export(tmp_path: Path) -> None:
    client = _server_client(tmp_path)
    project = client.post("/api/simulate").json()
    pid = project["id"]
    for name in ("assets.json", "assets.csv", "assets.geojson", "inventory.json", "run.json"):
        assert client.get(f"/api/projects/{pid}/exports/{name}").status_code == 200, name
    assert client.delete(f"/api/projects/{pid}").status_code == 200
