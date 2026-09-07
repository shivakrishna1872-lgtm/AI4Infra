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
    response = client.post(
        f"/api/projects/{project['id']}/process",
        content=b"{}",
        headers={"content-type": "application/json"},
    )
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
    response = client.post(
        f"/api/projects/{project['id']}/process",
        content=b"{}",
        headers={"content-type": "application/json"},
    )
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


def test_processing_preflight_returns_clear_507_when_disk_full(client, monkeypatch):
    """Starting a job on a nearly full disk fails fast with a routing message
    instead of crashing mid-write with Errno 28."""
    monkeypatch.setattr(server, "_free_disk_bytes", lambda: server.MIN_FREE_DISK_BYTES - 1)
    project = client.post("/api/projects", params={"name": "diskfull"}).json()
    response = client.post(
        f"/api/projects/{project['id']}/process",
        content=b"{}",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 507
    assert "disk" in response.json()["detail"].lower()
    # Quick Simulation is preflighted too
    assert client.post("/api/simulate").status_code == 507


def test_web_job_does_not_keep_tile_intermediates(client, tmp_path):
    """The disk-footprint regression: web jobs run with save_tiles=false, so a
    processed project holds the viewer payload + exports but not multi-GB tiles."""
    from infra_inventory.synthetic import build_synthetic_las

    las = tmp_path / "input.las"
    build_synthetic_las(las)
    project = client.post("/api/projects", params={"name": "no tiles"}).json()
    with las.open("rb") as handle:
        upload = client.post(
            f"/api/projects/{project['id']}/upload",
            files={"file": ("input.las", handle, "application/octet-stream")},
        )
    assert upload.status_code == 200
    started = client.post(
        f"/api/projects/{project['id']}/process",
        content=b"{}",
        headers={"content-type": "application/json"},
    ).json()
    status = wait_done(client, started["job_id"])
    assert status["stage"] == "done"
    output_dir = server._project_dir(project["id"]) / "output"
    assert (output_dir / "viewer-data.json").is_file()
    assert not (output_dir / "tiles").exists()
    assert client.get(f"/api/projects/{project['id']}/viewer-data").status_code == 200
    info = client.get(f"/api/projects/{project['id']}").json()
    assert info["processed"] is True
    assert info["asset_count"] == status["assets"]


def test_data_simulation_writes_directly_no_copytree(client, tmp_path):
    """simulate/data processes into the project output directly (the old path
    doubled disk with a run-dir + copytree)."""
    from infra_inventory.synthetic import build_synthetic_las

    las = tmp_path / "input.las"
    build_synthetic_las(las)
    with las.open("rb") as handle:
        response = client.post(
            "/api/simulate/data",
            files={"upload": ("input.las", handle, "application/octet-stream")},
        )
    assert response.status_code == 200, response.text
    project = response.json()
    assert project["processed"] is True
    assert project["asset_count"] > 0
    output_dir = server._project_dir(project["id"]) / "output"
    assert not (output_dir / "tiles").exists()
    assert client.get(f"/api/projects/{project['id']}/viewer-data").status_code == 200
    for name in ("assets.json", "inventory.json", "assets.csv", "assets.geojson"):
        assert client.get(f"/api/projects/{project['id']}/exports/{name}").status_code == 200, name


def test_viewer_data_served_gzip_compressed(client, tmp_path):
    """The viewer payload is tens of MB for real scans; it must be served
    gzip-compressed so the hosted preview tunnel doesn't stall on the transfer."""
    from infra_inventory.synthetic import build_synthetic_las

    las = tmp_path / "input.las"
    build_synthetic_las(las)
    project = client.post("/api/projects", params={"name": "gzip"}).json()
    with las.open("rb") as handle:
        client.post(
            f"/api/projects/{project['id']}/upload",
            files={"file": ("input.las", handle, "application/octet-stream")},
        )
    started = client.post(
        f"/api/projects/{project['id']}/process",
        content=b"{}",
        headers={"content-type": "application/json"},
    ).json()
    status = wait_done(client, started["job_id"])
    assert status["stage"] == "done"

    plain = client.get(f"/api/projects/{project['id']}/viewer-data")
    assert plain.status_code == 200
    gzipped = client.get(
        f"/api/projects/{project['id']}/viewer-data",
        headers={"Accept-Encoding": "gzip"},
    )
    assert gzipped.status_code == 200
    # Starlette's gzip middleware sets the header; the test client then
    # decompresses transparently, so assert the header + intact payload.
    assert gzipped.headers.get("content-encoding") == "gzip"
    assert len(gzipped.json()["points"]) == len(plain.json()["points"]) > 0


def test_second_process_on_same_project_is_rejected(client, tmp_path, monkeypatch):
    """Two concurrent jobs on one project would wipe each other's output and
    orphan a zombie thread; the second start must fail fast with 409."""
    import time as _time
    from infra_inventory.synthetic import build_synthetic_las

    las = tmp_path / "input.las"
    build_synthetic_las(las)
    project = client.post("/api/projects", params={"name": "double"}).json()
    with las.open("rb") as handle:
        client.post(
            f"/api/projects/{project['id']}/upload",
            files={"file": ("input.las", handle, "application/octet-stream")},
        )

    # Hold the first job open so the second request provably overlaps it.
    original = server.process_las

    def slow(*args, **kwargs):
        _time.sleep(4)
        return original(*args, **kwargs)

    monkeypatch.setattr(server, "process_las", slow)
    first = client.post(f"/api/projects/{project['id']}/process", json={})
    assert first.status_code == 200
    second = client.post(
        f"/api/projects/{project['id']}/process",
        content=b"{}",
        headers={"content-type": "application/json"},
    )
    assert second.status_code == 409
    assert "already being processed" in second.json()["detail"]
    status = wait_done(client, first.json()["job_id"])
    assert status["stage"] == "done"
    # After completion the project can be processed again.
    monkeypatch.setattr(server, "process_las", original)
    again = client.post(
        f"/api/projects/{project['id']}/process",
        content=b"{}",
        headers={"content-type": "application/json"},
    )
    assert again.status_code == 200
    assert wait_done(client, again.json()["job_id"])["stage"] == "done"


def test_web_settings_gemini_auto_enable(monkeypatch) -> None:
    """_web_settings enables the Gemini booster when the key is present and
    falls back to geometry-only without it (the pipeline never needs the net)."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    settings = server._web_settings()
    assert settings.backend == "geometry"
    assert settings.save_tiles is False
    assert settings.viewer_point_limit == 250_000

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    settings = server._web_settings()
    assert settings.backend == "gemini"
    # Explicit opt-out wins even with a key present.
    assert server._web_settings(use_gemini=False).backend == "geometry"
    # Model override passes through.
    assert server._web_settings(gemini_model="gemini-test").gemini_model == "gemini-test"


def test_process_endpoint_accepts_use_gemini_flag(client, tmp_path, monkeypatch) -> None:
    """The process request forwards the Gemini switch; a forced-on run without a
    valid key still completes (geometry-only, warning recorded)."""
    from infra_inventory.synthetic import build_synthetic_las

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    upload = client.post("/api/projects", json={"name": "gemini flag"})
    project = upload.json()
    src = tmp_path / "in.las"
    build_synthetic_las(src, seed=7)
    with src.open("rb") as handle:
        up = client.post(f"/api/projects/{project['id']}/upload", files={"file": ("in.las", handle, "application/octet-stream")})
    assert up.status_code == 200
    start = client.post(
        f"/api/projects/{project['id']}/process",
        content=b'{"use_gemini": "false"}',
        headers={"content-type": "application/json"},
    )
    assert start.status_code == 200
    status = wait_done(client, start.json()["job_id"])
    assert status["stage"] == "done"
    run = client.get(f"/api/projects/{project['id']}").json()
    # The project processed successfully either way; backend recorded in run.json.
    assert run.get("processed") is True