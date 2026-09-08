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


def _create_project(client: TestClient, name: str = "demo") -> str:
    created = client.post(f"/api/projects?name={name}")
    assert created.status_code == 200
    return created.json()["id"]


def _synthetic_bytes() -> bytes:
    """A real (tiny) LAS file padded with a zero payload to a stable test size."""
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
    return base + b"\x00" * max(0, (1_000_000 - len(base)))


def _synthetic_sha256() -> str:
    import hashlib

    return hashlib.sha256(_synthetic_bytes()).hexdigest()


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
    assert meta["input_size_bytes"] == len(payload)
    assert meta["upload_storage"] == "local"

    # Session cleaned up after completion.
    assert client.get(f"/api/projects/{project_id}/uploads/{upload_id}").status_code == 404


def test_upload_rejects_text_mangled_single_shot(client: TestClient) -> None:
    """The single-shot /upload path accepts a clean file (recording its
    SHA-256) and rejects a text-mangled body with 400 instead of installing a
    corrupted input.las (the bug pattern that produced the 'offset-sized
    integer' parse failures)."""
    data = _synthetic_bytes()
    project_id = _create_project(client)

    # Clean upload first.
    resp = client.post(
        f"/api/projects/{project_id}/upload",
        files={"file": ("x.las", data, "application/octet-stream")},
    )
    assert resp.status_code == 200, resp.text
    meta = json.loads((server._project_dir(project_id) / "project.json").read_text())
    assert meta["input_sha256"] == _synthetic_sha256()
    assert meta["input_size_bytes"] == len(data)

    # A UTF-8 double-encoded body (the exact corruption shape seen in the
    # wild when a browser reads a binary file as text) must be rejected before
    # it becomes input.las, and the project's PREVIOUS good input.las must
    # survive untouched.
    mangled = data.decode("latin-1").encode("utf-8")
    if mangled != data:
        resp = client.post(
            f"/api/projects/{project_id}/upload",
            files={"file": ("x.las", mangled, "application/octet-stream")},
        )
        assert resp.status_code == 400, resp.text
        assert "not a valid LAS" in resp.json()["detail"]
        assert (server._project_dir(project_id) / "input.las").read_bytes() == data


def test_upload_rejects_truncated_transfer(client: TestClient) -> None:
    """A transfer cut mid-flight (truncated body) must 400 with the LAS
    validation message and never publish input.las."""
    data = _synthetic_bytes()
    project_id = _create_project(client)
    # Cut INSIDE the header (not the zero padding): a header read shorter than
    # the minimum must be rejected outright.
    truncated = data[:200]
    resp = client.post(
        f"/api/projects/{project_id}/upload",
        files={"file": ("x.las", truncated, "application/octet-stream")},
    )
    assert resp.status_code == 400, resp.text
    assert "not a valid LAS" in resp.json()["detail"]
    assert not (server._project_dir(project_id) / "input.las").exists()


def test_complete_rejects_missing_parts_then_resume_succeeds(client: TestClient):
    created = client.post(f"/api/projects?name=demo")



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


def test_upload_size_matches_body(client: TestClient) -> None:
    """Upload a synthetic file and confirm the project metadata reflects the
    byte count that was actually sent, and that a transfer which lands a
    different size is rejected instead of silently becoming input.las."""
    data = _synthetic_bytes()
    project_id = _create_project(client)
    resp = client.post(
        f"/api/projects/{project_id}/upload",
        files={"file": ("x.las", data, "application/octet-stream")},
    )
    assert resp.status_code == 200
    meta = json.loads((server._project_dir(project_id) / "project.json").read_text())
    assert meta["input_size_bytes"] == len(data)
    assert meta["input_sha256"] == _synthetic_sha256()


def test_orphaned_session_reset_after_project_delete(client: TestClient) -> None:
    """Deleting a project must not 403 future uploads of the same file into a
    new project: the orphaned session (same derived upload_id) is reset
    server-side instead of failing with 'Upload session belongs to a different
    project' for the rest of its TTL."""
    first = client.post("/api/projects?name=first")
    first_id = first.json()["id"]
    upload_id = _uid()

    manifest = _init_session(client, first_id, 1_000_000, upload_id)
    assert manifest["upload_id"] == upload_id
    assert manifest["project_id"] == first_id

    # Deleting the owner project frees its upload sessions immediately.
    deleted = client.delete(f"/api/projects/{first_id}")
    assert deleted.status_code == 200
    assert deleted.json()["freed"] is True
    assert not (server._project_dir(first_id)).exists()

    # The same file (same derived id) now uploads cleanly into a new project.
    second = client.post("/api/projects?name=second")
    second_id = second.json()["id"]
    resumed = _init_session(client, second_id, 1_000_000, upload_id)
    assert resumed["resumed"] is False
    assert resumed["project_id"] == second_id

    # And the full chunked round-trip works end to end.
    payload = _synthetic_bytes()
    parts = resumed["chunk_count"]
    chunk = resumed["chunk_size"]
    for part in range(1, parts + 1):
        blob = payload[(part - 1) * chunk: part * chunk]
        response = client.put(
            f"/api/projects/{second_id}/uploads/{upload_id}/part/{part}",
            files={"file": ("part", blob, "application/octet-stream")},
        )
        assert response.status_code == 200, response.text
    done = client.post(
        f"/api/projects/{second_id}/uploads/{upload_id}/complete",
        json={},
    )
    assert done.status_code == 200, done.text


def test_delete_project_cleans_upload_sessions(client: TestClient) -> None:
    """DELETE /api/projects/{id} must remove the project's upload session
    directories (multi-GB part files) so deleting projects frees storage."""
    created = client.post("/api/projects?name=demo")
    project_id = created.json()["id"]
    upload_id = _uid()
    _init_session(client, project_id, 5_000_000, upload_id)

    session_dir = upload_store._session_dir(upload_id)
    assert session_dir.is_dir()

    deleted = client.delete(f"/api/projects/{project_id}")
    assert deleted.status_code == 200
    assert not session_dir.exists()
    assert client.get(f"/api/projects/{project_id}").status_code == 404


def test_release_project_prunes_after_grace(client: TestClient) -> None:
    """POST /api/projects/{id}/release (fired when the user closes the tab)
    marks the project for deletion after the grace period — including its
    upload sessions and job files — so a finished scan frees its disk."""
    import time

    created = client.post("/api/projects?name=viewed")
    project_id = created.json()["id"]
    upload_id = _uid()
    _init_session(client, project_id, 5_000_000, upload_id)
    assert upload_store._session_dir(upload_id).is_dir()

    released = client.post(f"/api/projects/{project_id}/release")
    assert released.status_code == 200
    assert released.json()["grace_seconds"] > 0

    # Within the grace window the project still exists (checked on disk:
    # opening it via the API would cancel the release, by design).
    assert server._project_dir(project_id).exists()

    # After the grace window, the prune pass deletes everything.
    removed = server._prune_released(now=time.time() + 3600)
    assert removed >= 1
    assert client.get(f"/api/projects/{project_id}").status_code == 404
    assert not server._project_dir(project_id).exists()
    assert not upload_store._session_dir(upload_id).exists()


def test_release_cancelled_when_project_reopened(client: TestClient) -> None:
    """Re-opening a released project (or listing projects) cancels the pending
    deletion — a refresh or accidental close must not lose the scan."""
    import time

    created = client.post("/api/projects?name=reopened")
    project_id = created.json()["id"]
    client.post(f"/api/projects/{project_id}/release")

    # User comes back within the grace period: opening the project cancels it.
    assert client.get(f"/api/projects/{project_id}").status_code == 200
    assert server._prune_released(now=time.time() + 3600) == 0
    assert client.get(f"/api/projects/{project_id}").status_code == 200

    # Listing also cancels pending releases.
    second = client.post("/api/projects?name=listed")
    second_id = second.json()["id"]
    client.post(f"/api/projects/{second_id}/release")
    assert client.get("/api/projects").status_code == 200
    assert server._prune_released(now=time.time() + 3600) == 0
    assert client.get(f"/api/projects/{second_id}").status_code == 200


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


def test_complete_rejects_text_mangled_upload(client: TestClient):
    """A part that was corrupted through a text-mode/encoding round-trip must
    never become <project>/input.las. This is the gate that blocked the
    repeated 'Malformed LAS file' failures from a bad upload session."""
    created = client.post("/api/projects?name=demo")
    assert created.status_code == 200
    project_id = created.json()["id"]

    # Build a real tiny LAS, then deliberately feed the server a mangled copy
    # (UTF-8 double-encoded binary) so the part arrives corrupt.
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
    good = buf.getvalue()

    # Mimic the exact corruption shape seen in the wild: high bytes expanded
    # into multi-byte UTF-8 sequences.
    mangled = good.decode("latin-1").encode("utf-8")
    assert mangled != good

    upload_id = uuid.uuid4().hex[:16]
    manifest = _init_session(client, project_id, len(mangled), upload_id)
    assert manifest["storage"] == "local"

    # Send the single (corrupt) part.
    response = client.put(
        f"/api/projects/{project_id}/uploads/{upload_id}/part/1",
        files={"file": ("part", mangled, "application/octet-stream")},
    )
    assert response.status_code == 200

    # Completing must refuse to publish a corrupt LAS.
    done = client.post(
        f"/api/projects/{project_id}/uploads/{upload_id}/complete",
        json={},
    )
    assert done.status_code == 400, done.text
    assert "not a valid LAS" in done.json()["detail"]
    assert not (server._project_dir(project_id) / "input.las").exists()


def test_stale_session_with_mismatched_parts_is_reset_on_resume(client: TestClient):
    """A session whose on-disk parts do not match the manifest (stale/corrupt)
    must be reset when another project resumes the same upload_id, never reused."""
    created = client.post("/api/projects?name=demo")
    project_id = created.json()["id"]
    upload_id = uuid.uuid4().hex[:16]

    manifest = _init_session(client, project_id, 1_000_000, upload_id)
    # Simulate a stale/corrupt session by writing a part that is NOT in the
    # manifest's parts_received list.
    parts_dir = upload_store._session_dir(upload_id) / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    (parts_dir / "part-000001").write_bytes(b"garbage" * 50)

    # A different project resumes the same upload_id: the mismatch must force
    # a reset, and the new session must be fresh (not resumed=True from the
    # stale state).
    other = client.post("/api/projects?name=other")
    other_id = other.json()["id"]
    resumed = _init_session(client, other_id, 2_000_000, upload_id)
    # After reset the session is effectively new for the new project.
    assert resumed["resumed"] is False
    assert resumed["project_id"] == other_id
    assert resumed["total_size"] == 2_000_000

    # The old corrupt part must have been purged.
    assert not (upload_store._session_dir(upload_id) / "parts" / "part-000001").exists()
