"""End-to-end tests for the FastAPI inspection platform.

Covers: project lifecycle, simulation, async processing job progress,
viewer-data payload shape, and export endpoints. Uses the simulation
endpoint so the test is fast and deterministic.
"""
import time

import pytest

from infra_inventory import server

pytestmark = pytest.mark.usefixtures("clean_appdata")


def wait_done(client, job_id, timeout=60.0):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        response = client.get(f"/api/jobs/{job_id}")
        assert response.status_code == 200
        last = response.json()
        if last["stage"] in ("done", "error"):
            return last
        time.sleep(0.1)
    raise AssertionError(f"job never finished: {last}")


def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "version" in body


def test_simulate_process_viewer_export_roundtrip(client):
    # 1. simulate a scene
    response = client.post("/api/simulate")
    assert response.status_code == 200
    project = response.json()
    assert project["simulated"] is True
    assert project["input_file"].endswith(".las")

    # 2. process it as a background job
    response = client.post(f"/api/projects/{project['id']}/process", json={})
    assert response.status_code == 200
    job_id = response.json()["job_id"]

    # 3. progress updates arrive, and the job completes
    status = wait_done(client, job_id)
    assert status["stage"] == "done", status
    assert status["assets"] > 0
    assert status["points_processed"] > 0
    assert status["tiles_total"] > 0

    # 4. project is now marked processed with real counts
    response = client.get(f"/api/projects/{project['id']}")
    assert response.status_code == 200
    info = response.json()
    assert info["processed"] is True
    assert info["asset_count"] == status["assets"]
    assert info["point_count"] == status["points_processed"]

    # 5. viewer-data is the real pipeline payload
    response = client.get(f"/api/projects/{project['id']}/viewer-data")
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["points"]) > 0
    assert len(payload["points"][0]) == 3
    assert len(payload["assets"]) == status["assets"]
    assert payload["run"]["point_count"] == status["points_processed"]
    assert "point_class" in payload
    assert payload["point_class_names"]
    # every asset has attribution fields the UI depends on
    for asset in payload["assets"]:
        assert asset["asset_id"].startswith(("PAV-", "MRK-", "POL-", "SGN-", "GRD-", "BAR-", "RUM-", "CON-", "CAB-"))
        assert 0.0 <= asset["confidence"] <= 1.0
        assert "confidence_factors" in asset
        assert "center" in asset
        assert "bounding_box" in asset
        assert asset["source_tile"]
        assert asset["detection_method"]

    # 6. exports exist and carry the same inventory
    for name in ("assets.json", "assets.csv", "assets.geojson", "inventory.json", "run.json"):
        response = client.get(f"/api/projects/{project['id']}/exports/{name}")
        assert response.status_code == 200, name
    response = client.get(f"/api/projects/{project['id']}/exports/nope.txt")
    assert response.status_code == 404

    # 7. cleanup
    assert client.delete(f"/api/projects/{project['id']}").status_code == 200


def test_upload_rejects_non_las(client):
    project = client.post("/api/projects", params={"name": "bad"}).json()
    response = client.post(
        f"/api/projects/{project['id']}/upload",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 400
    assert "las" in response.json()["detail"].lower()
    client.delete(f"/api/projects/{project['id']}")


def test_process_requires_upload(client):
    project = client.post("/api/projects", params={"name": "empty"}).json()
    response = client.post(f"/api/projects/{project['id']}/process", json={})
    assert response.status_code == 400
    client.delete(f"/api/projects/{project['id']}")


def test_missing_project_404(client):
    assert client.get("/api/projects/deadbeef").status_code == 404
    assert client.get("/api/projects/deadbeef/viewer-data").status_code == 404


def test_job_survives_in_memory_registry_reset(client, tmp_path):
    """The 'Job not found' regression: jobs must survive a server restart."""
    from infra_inventory.synthetic import build_synthetic_las

    las = tmp_path / "input.las"
    build_synthetic_las(las)
    project = client.post("/api/projects", params={"name": "job persist"}).json()
    with las.open("rb") as handle:
        upload = client.post(
            f"/api/projects/{project['id']}/upload",
            files={"file": ("input.las", handle, "application/octet-stream")},
        )
    assert upload.status_code == 200
    started = client.post(f"/api/projects/{project['id']}/process", json={}).json()
    job_id = started["job_id"]
    status = wait_done(client, job_id)
    assert status["stage"] == "done"

    # Simulate the server process restarting: wipe the in-memory registry.
    server._JOBS.clear()
    recovered = client.get(f"/api/jobs/{job_id}")
    assert recovered.status_code == 200
    assert recovered.json()["stage"] == "done"


def test_stale_job_file_reported_as_error_after_restart(client):
    """A mid-flight job whose worker died surfaces a clear error, not 404."""
    import time as _time

    job_id = "dead-worker-0001"
    server._write_job({
        "id": job_id,
        "project_id": "nope",
        "stage": "streaming",
        "message": "Streaming read + spatial tiling",
        "points_processed": 10,
        "updated_at": _time.time() - 9999.0,
    })
    server._JOBS.clear()
    response = client.get(f"/api/jobs/{job_id}")
    assert response.status_code == 200
    body = response.json()
    assert body["stage"] == "error"
    assert "restarted" in body["message"]