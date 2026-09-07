"""Resumable chunked upload tests (local mode + S3 config detection).

Covers: session create/resume, part storage, resume-with-missing-part
rejection, atomic assembly, abort cleanup, stale-session pruning, and the
/api/simulate/data size guard that routes big files to the chunked path.
"""
import json
import shutil
import uuid

import pytest
from fastapi.testclient import TestClient

from infra_inventory import server, upload_store

pytestmark = pytest.mark.usefixtures("clean_appdata")


@pytest.fixture(autouse=True)
def _wipe_upload_sessions():
    """Upload sessions live outside project dirs; keep tests independent."""
    shutil.rmtree(upload_store.DATA_DIR / "uploads", ignore_errors=True)
    yield
    shutil.rmtree(upload_store.DATA_DIR / "uploads", ignore_errors=True)


def _uid() -> str:
    return uuid.uuid4().hex[:16]


def _make_las_bytes(total: int = 1_000_000) -> bytes:
    """A real (tiny) LAS file padded with a payload blob to a target size."""
    import io

    import laspy
    import numpy as np

    header = laspy.LasHeader(point_format=7, version="1.4")
    header.scales = [0.001] * 3
    las = laspy.LasData(header)
    n = 50
    las.x = np.arange(n) * 0.1
    las.y = np.zeros(n)
    las.z = np.ones(n)
    buf = io.BytesIO()
    las.write(buf)
    base = buf.getvalue()
    return base + b"\x00" * max(0, total - len(base))


def _init_session(client: TestClient, project_id: str, size: int, uid: str) -> dict:
    response = client.post(
        f"/api/projects/{project_id}/uploads",
        json={"filename": "scan.las", "total_size": size, "upload_id": uid},
    )
    assert response.status_code == 200
    return response.json()


def test_chunked_upload_roundtrip_assembles_exact_bytes(client: TestClient):
    created = client.post("/api/projects?name=demo")
    assert created.status_code == 200
    project_id = created.json()["id"]

    payload = _make_las_bytes(2_000_000)
    upload_id = _uid()
    manifest = _init_session(client, project_id, len(payload), upload_id)
    assert manifest["storage"] == "local"
    assert manifest["upload_id"] == upload_id
    chunk = manifest["chunk_size"]
    assert manifest["chunk_count"] == -(-len(payload) // chunk)

    # Send all parts.
    for part in range(1, manifest["chunk_count"] + 1):
        blob = payload[(part - 1) * chunk: part * chunk]
        response = client.put(
            f"/api/projects/{project_id}/uploads/{upload_id}/part/{part}",
            files={"file": ("part", blob, "application/octet-stream")},
        )
        assert response.status_code == 200, response.text
        assert (part - 1) in response.json()["parts_received"] or part in response.json()["parts_received"]

    status = client.get(f"/api/projects/{project_id}/uploads/{upload_id}")
    assert status.status_code == 200
    assert status.json()["parts_received"] == list(range(1, manifest["chunk_count"] + 1))

    done = client.post(
        f"/api/projects/{project_id}/uploads/{upload_id}/complete",
        json={},
    )
    assert done.status_code == 200, done.text
    body = done.json()
    assert body["processed"] is False  # upload never auto-processes

    target = server._project_dir(project_id) / "input.las"
    assert target.read_bytes() == payload

    meta = json.loads((server._project_dir(project_id) / "project.json").read_text())
    assert meta["input_sha256"]
    assert meta["upload_storage"] == "local"

    # Session cleaned up after completion.
    assert client.get(f"/api/projects/{project_id}/uploads/{upload_id}").status_code == 404


def test_complete_rejects_missing_parts_then_resume_succeeds(client: TestClient):
    created = client.post("/api/projects?name=demo")
    project_id = created.json()["id"]

    payload = _make_las_bytes(3 * 1024 * 1024 + 11)

    # Force a multi-part scenario by shrinking the chunk via env override
    # (chunk_size()/chunk_count() read the env at call time, no reload needed).
    import os

    old = upload_store.chunk_size()
    upload_id = _uid()
    try:
        os.environ["UPLOAD_CHUNK_BYTES"] = str(1024 * 1024)
        manifest = _init_session(client, project_id, len(payload), upload_id)
        chunk = manifest["chunk_size"]
        parts = manifest["chunk_count"]
        assert parts >= 2

        # Send all but the last part.
        for part in range(1, parts):
            blob = payload[(part - 1) * chunk: part * chunk]
            response = client.put(
                f"/api/projects/{project_id}/uploads/{upload_id}/part/{part}",
                files={"file": ("part", blob, "application/octet-stream")},
            )
            assert response.status_code == 200

        incomplete = client.post(
            f"/api/projects/{project_id}/uploads/{upload_id}/complete", json={}
        )
        assert incomplete.status_code == 400
        assert "missing" in incomplete.json()["detail"].lower()

        # Resume: send the missing part, then complete.
        last = parts
        blob = payload[(last - 1) * chunk:]
        response = client.put(
            f"/api/projects/{project_id}/uploads/{upload_id}/part/{last}",
            files={"file": ("part", blob, "application/octet-stream")},
        )
        assert response.status_code == 200

        done = client.post(
            f"/api/projects/{project_id}/uploads/{upload_id}/complete", json={}
        )
        assert done.status_code == 200
        target = server._project_dir(project_id) / "input.las"
        assert target.read_bytes() == payload
    finally:
        import os

        os.environ.pop("UPLOAD_CHUNK_BYTES", None)
        assert upload_store.chunk_size() == old


def test_init_is_idempotent_for_resume(client: TestClient):
    created = client.post("/api/projects?name=demo")
    project_id = created.json()["id"]

    upload_id = _uid()
    first = _init_session(client, project_id, 10_000_000, upload_id)
    assert first["resumed"] is False
    second = _init_session(client, project_id, 10_000_000, upload_id)
    assert second["resumed"] is True
    assert second["upload_id"] == first["upload_id"]


def test_abort_drops_session(client: TestClient):
    created = client.post("/api/projects?name=demo")
    project_id = created.json()["id"]
    upload_id = _uid()
    _init_session(client, project_id, 10_000_000, upload_id)
    response = client.delete(f"/api/projects/{project_id}/uploads/{upload_id}")
    assert response.status_code == 200
    assert client.get(f"/api/projects/{project_id}/uploads/{upload_id}").status_code == 404


def test_upload_id_validation(client: TestClient):
    created = client.post("/api/projects?name=demo")
    project_id = created.json()["id"]
    # Malformed ids are rejected before touching the filesystem.
    response = client.post(
        f"/api/projects/{project_id}/uploads",
        json={"filename": "scan.las", "total_size": 1000, "upload_id": "../evil"},
    )
    assert response.status_code == 400


def test_wrong_extension_rejected(client: TestClient):
    created = client.post("/api/projects?name=demo")
    project_id = created.json()["id"]
    response = client.post(
        f"/api/projects/{project_id}/uploads",
        json={"filename": "scan.exe", "total_size": 1000},
    )
    assert response.status_code == 400


def test_prune_stale_sessions(tmp_path, monkeypatch):
    shutil.rmtree(upload_store.DATA_DIR / "uploads", ignore_errors=True)
    # Build a session dir with an ancient manifest.
    uploads = upload_store.DATA_DIR / "uploads"
    stale_id, fresh_id = _uid(), _uid()
    session = uploads / stale_id
    (session / "parts").mkdir(parents=True, exist_ok=True)
    (session / "manifest.json").write_text(
        json.dumps(
            {
                "upload_id": stale_id,
                "project_id": "x" * 12,
                "storage": "local",
                "created_at": 0.0,
            }
        )
    )
    removed = upload_store.prune_stale_sessions()
    assert removed >= 1
    assert not session.exists()

    # Recent sessions survive.
    fresh = uploads / fresh_id
    fresh.mkdir(parents=True, exist_ok=True)
    import time as _time

    (fresh / "manifest.json").write_text(
        json.dumps({"upload_id": fresh_id, "created_at": _time.time()})
    )
    assert upload_store.prune_stale_sessions() == 0
    assert fresh.exists()


def test_simulate_data_routes_big_files_to_chunked_path(client: TestClient, monkeypatch):
    monkeypatch.setattr(server, "SIMULATE_SYNC_MAX_BYTES", 10)
    payload = _make_las_bytes(40_000)
    response = client.post(
        "/api/simulate/data",
        files={"upload": ("scan.las", payload, "application/octet-stream")},
    )
    assert response.status_code == 413
    detail = response.json()["detail"]
    assert "chunked" in detail.lower() or "uploads" in detail.lower()


def test_s3_config_detection(monkeypatch):
    # Clean env -> local mode.
    for key in (
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_REGION",
        "S3_BUCKET",
        "S3_ENDPOINT_URL",
        "STORAGE_BACKEND",
    ):
        monkeypatch.delenv(key, raising=False)
    assert upload_store.storage_mode() == "local"

    # Credentials + bucket -> s3 mode (no network is touched here).
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.setenv("S3_BUCKET", "test-bucket")
    assert upload_store.storage_mode() == "s3"

    # STORAGE_BACKEND=local pins local mode even with credentials.
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    assert upload_store.storage_mode() == "local"


def test_chunk_size_env_bounds(monkeypatch):
    monkeypatch.delenv("UPLOAD_CHUNK_BYTES", raising=False)
    default = upload_store.chunk_size()
    assert default == upload_store.DEFAULT_CHUNK_BYTES
    monkeypatch.setenv("UPLOAD_CHUNK_BYTES", "999")  # below 1 MiB floor -> default
    assert upload_store.chunk_size() == default
    monkeypatch.setenv("UPLOAD_CHUNK_BYTES", str(16 * 1024 * 1024))
    assert upload_store.chunk_size() == 16 * 1024 * 1024
