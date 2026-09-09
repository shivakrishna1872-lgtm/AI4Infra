"""FastAPI server for the AI4Infra inspection platform.

Exposes the real pipeline over HTTP:

* projects            - create / list / get projects (filesystem-backed)
* upload              - accept a LAS file for a project
* process             - start an async processing job (threaded, stage progress)
* jobs/<id>           - poll job state (stage, points, tiles, assets, elapsed)
* viewer-data         - the 3D viewer payload (real pipeline output)
* exports/<name>      - CSV / JSON / GeoJSON download of the inventory
* simulate            - generate a labeled simulated scene (SIMULATION ONLY)

Everything served here is produced by ``infra_inventory.pipeline.process_las``;
the frontend never invents infrastructure.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import laspy

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import os

from . import __version__
from .models import ProcessingSettings
from .pipeline import process_las
from .simulation import run_quick_simulation, run_data_simulation, run_small_synthetic
from . import upload_store
from .validation import check_las_header_bounded

DATA_DIR = Path(__file__).resolve().parent.parent / "appdata"

#: Built React frontend (web-dist); served at "/" when present so the API and
#: the 3D digital twin ship from the same origin.
WEB_DIST = Path(__file__).resolve().parent.parent / "web-dist"

#: In-memory job registry: job_id -> status dict (guarded by the lock).
#: Every job is ALSO persisted to DATA_DIR/jobs/<id>.json on each update, so a
#: server restart or a multi-worker deployment can never turn a running job
#: into a "Job not found" for the polling frontend.
_JOBS: Dict[str, dict] = {}
_JOBS_LOCK = threading.Lock()


def _job_file(job_id: str) -> Path:
    return DATA_DIR / "jobs" / f"{job_id}.json"


def _safe_replace(src: Path, dst: Path, *, retries: int = 8, base_delay: float = 0.05) -> None:
    """Retry atomic rename on Windows/OneDrive PermissionError (WinError 5)."""
    import time as _time
    last_exc = None
    for attempt in range(retries):
        try:
            os.replace(src, dst)
            return
        except PermissionError as exc:
            last_exc = exc
            _time.sleep(base_delay * (2 ** attempt))
        except OSError:
            raise
    raise PermissionError(
        f"_safe_replace: failed after {retries} retries ({src} -> {dst})"
    ) from last_exc


def _write_job(job: dict) -> None:
    """Atomically persist a job record so it survives process restarts."""
    path = _job_file(job["id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(job, indent=2), encoding="utf-8")
        _safe_replace(tmp, path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


#: Minimum free bytes required before a processing job will start. Jobs stream
#: multi-GB tiles to DATA_DIR, so starting one on a nearly full disk just turns
#: into a crash mid-way; we fail fast with a clear message instead.
MIN_FREE_DISK_BYTES = 512 * 1024 * 1024
# Slimmest possible prompt data the viewer needs. Keeping this low leaves room
# for real .las/.laz uploads while still letting the 3D scene open.
MIN_VIEWER_BYTES = 64 * 1024 * 1024


def _free_disk_bytes() -> int:
    try:
        return shutil.disk_usage(DATA_DIR).free
    except OSError:
        return MIN_FREE_DISK_BYTES  # unknown -> let the run proceed and surface real errors


def _cleanup_stale_job_files() -> None:
    """Remove leftover atomic-write .tmp files and stale job json at startup."""
    jobs_dir = DATA_DIR / "jobs"
    if not jobs_dir.is_dir():
        return
    for path in jobs_dir.glob("*.tmp"):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


_TERMINAL_STAGES = {"done", "error"}

#: /api/simulate/data processes inline inside the request; above this size the
#: client must use the resumable chunked upload + async process job instead.
SIMULATE_SYNC_MAX_BYTES = 128 * 1024 * 1024


def _active_job_for_project(project_id: str) -> Optional[str]:
    """Return the id of a running (non-terminal) job for a project, if any.

    Checks both the in-memory registry and the persisted job files so a
    restarted server still refuses to double-process a project whose previous
    run is alive. A stale record that stopped updating is treated as dead.
    """
    now = time.time()
    with _JOBS_LOCK:
        for job_id, job in _JOBS.items():
            if job.get("project_id") == project_id and job.get("stage") not in _TERMINAL_STAGES:
                return job_id
    jobs_dir = DATA_DIR / "jobs"
    if jobs_dir.is_dir():
        for path in jobs_dir.glob("*.json"):
            try:
                job = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if job.get("project_id") != project_id:
                continue
            if job.get("stage") in _TERMINAL_STAGES:
                continue
            age = now - float(job.get("updated_at") or 0.0)
            if age < 90.0:
                return path.stem
    return None


def _disk_full_message() -> str:
    free_mb = _free_disk_bytes() // (1024 * 1024)
    return (
        "Out of disk space while processing (only ~"
        f"{free_mb} MB free). Delete old projects from the Projects list "
        "(or remove appdata/projects/* on the server), then retry."
    )


def _ensure_disk_headroom(min_free: int = MIN_FREE_DISK_BYTES) -> None:
    """Fail fast with a clear 507 instead of crashing mid-write with Errno 28.

    Projects the user has finished viewing ("released" via the frontend beacon
    when they close the tab) are pruned first, so an upload/process retry
    frees its own space instead of hitting 507.
    """
    _prune_released()
    free = _free_disk_bytes()
    if free < min_free:
        raise HTTPException(
            status_code=507,
            detail=_disk_full_message(),
        )


#: How long a released project survives before auto-deletion. Short enough
#: that "finish viewing, close the tab" frees the disk, long enough that a
#: refresh or accidental close re-opens the project before it is gone.
RELEASE_GRACE_SECONDS = 60


def _release_meta(meta: dict) -> bool:
    """Cancel a pending release (the user came back). True when changed."""
    if "release_at" in meta:
        meta.pop("release_at", None)
        return True
    return False


def _prune_released(now: Optional[float] = None) -> int:
    """Delete projects whose release grace period has passed.

    Called at startup, before disk-headroom checks, and on every project
    listing. Projects with a live job are never pruned (a release beacon can
    only come from a project the user has already viewed, but guard anyway).
    Returns the number of projects removed.
    """
    now = time.time() if now is None else now
    root = DATA_DIR / "projects"
    if not root.is_dir():
        return 0
    removed = 0
    for directory in root.iterdir():
        meta_path = directory / "project.json"
        if not meta_path.is_file():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        release_at = meta.get("release_at")
        if release_at is None or float(release_at) > now:
            continue
        project_id = meta.get("id", directory.name)
        if _active_job_for_project(project_id) is not None:
            continue
        upload_store.abort_project_sessions(project_id)
        jobs_dir = DATA_DIR / "jobs"
        if jobs_dir.is_dir():
            for path in jobs_dir.glob("*.json"):
                try:
                    job = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if job.get("project_id") == project_id:
                    path.unlink(missing_ok=True)
        shutil.rmtree(directory, ignore_errors=True)
        removed += 1
    return removed


app = FastAPI(
    title="AI4Infra",
    description="AI-powered mobile LiDAR infrastructure asset extraction",
    version=__version__,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
# The viewer payload for real scans is tens of MB of JSON; compress it so the
# hosted preview tunnel (and slow clients) don't stall on a 44 MB transfer.
app.add_middleware(GZipMiddleware, minimum_size=1000)


@app.on_event("startup")
def _startup() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    _cleanup_stale_job_files()
    pruned = _prune_released()
    if pruned:
        print(f"released projects pruned: {pruned}", flush=True)
    removed = upload_store.prune_stale_sessions()
    if removed:
        print(f"upload sessions pruned: {removed}", flush=True)


STAGES = [
    "validating",
    "streaming",
    "pointcept",
    "openpcseg",
    "roadmarking",
    "detecting",
    "exporting",
    "done",
]


class ProjectInfo(BaseModel):
    id: str
    name: str
    created_at: str
    input_file: Optional[str] = None
    simulated: bool = False
    processed: bool = False
    point_count: Optional[int] = None
    crs: Optional[str] = None
    las_version: Optional[str] = None
    point_format: Optional[int] = None
    asset_count: Optional[int] = None
    scene: Optional[dict] = None
    summary: Optional[dict] = None
    # Overall dataset confidence from the processed run (0-100 + grade).
    confidence_percent: Optional[int] = None
    confidence_grade: Optional[str] = None


class ProcessRequest(BaseModel):
    tile_size_m: Optional[float] = None
    viewer_point_limit: Optional[int] = None
    #: "auto" (default) uses Gemini validation when GEMINI_API_KEY is set;
    #: True/False force it on/off regardless of the environment.
    use_gemini: Optional[str] = None


def _web_settings(
    use_gemini: Optional[bool] = None, gemini_model: Optional[str] = None
) -> ProcessingSettings:
    """Settings shared by every web processing path (upload/process/simulate).

    Web jobs always: drop multi-GB tile intermediates, raise the viewer LOD
    budget, and enable the Gemini class-validation booster when the API key is
    present in the environment (the booster degrades to geometry-only with a
    recorded warning when the key is missing or invalid — the pipeline never
    depends on the network)."""
    settings = ProcessingSettings()
    settings.save_tiles = False  # never keep multi-GB tile intermediates in web runs
    settings.viewer_point_limit = 250_000
    key = os.environ.get("GEMINI_API_KEY")
    settings.backend = "gemini" if (key and use_gemini is not False) else "geometry"
    if gemini_model:
        settings.gemini_model = gemini_model
    return settings


def _project_dir(project_id: str) -> Path:
    return DATA_DIR / "projects" / project_id


def _read_project(project_id: str) -> dict:
    directory = _project_dir(project_id)
    if not directory.is_dir():
        raise HTTPException(status_code=404, detail=f"Project {project_id} not found")
    meta_path = directory / "project.json"
    if not meta_path.is_file():
        raise HTTPException(status_code=404, detail=f"Project {project_id} has no metadata")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _save_project(project_id: str, meta: dict) -> None:
    directory = _project_dir(project_id)
    directory.mkdir(parents=True, exist_ok=True)
    # Atomic write: a kill/restart mid-write must never leave a truncated
    # project.json behind (that made projects unreadable and requests 502/500).
    tmp = directory / "project.json.tmp"
    tmp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    _safe_replace(tmp, directory / "project.json")


def _project_info(meta: dict) -> ProjectInfo:
    output = _project_dir(meta["id"]) / "output"
    processed = (output / "assets.json").is_file()
    info = ProjectInfo(
        id=meta["id"],
        name=meta["name"],
        created_at=meta["created_at"],
        input_file=meta.get("input_file"),
        simulated=meta.get("simulated", False),
        processed=processed,
    )
    if processed:
        run_path = output / "run.json"
        if run_path.is_file():
            run = json.loads(run_path.read_text(encoding="utf-8"))
            info.point_count = run.get("point_count")
            info.crs = run.get("crs")
            info.las_version = run.get("las_version")
            info.point_format = run.get("point_format")
        assets_path = output / "assets.json"
        if assets_path.is_file():
            info.asset_count = len(json.loads(assets_path.read_text(encoding="utf-8")))
        confidence = run.get("confidence_report")
        if isinstance(confidence, dict):
            info.confidence_percent = confidence.get("overall_percent")
            info.confidence_grade = confidence.get("grade")
        info.summary = meta
    info.scene = meta.get("scene")
    return info


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "version": __version__, "jobs": len(_JOBS)}


@app.get("/api/projects")
def list_projects() -> List[ProjectInfo]:
    _prune_released()
    root = DATA_DIR / "projects"
    if not root.is_dir():
        return []
    result = []
    for directory in sorted(root.iterdir()):
        meta_path = directory / "project.json"
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                # Opening the list cancels any pending release: the user is
                # back, so a closed-tab project survives its grace period.
                if _release_meta(meta):
                    _save_project(meta["id"], meta)
                result.append(_project_info(meta))
            except Exception:
                continue
    result.sort(key=lambda item: item.created_at, reverse=True)
    return result


@app.post("/api/projects", response_model=ProjectInfo)
def create_project(name: str = "Untitled scan") -> ProjectInfo:
    project_id = uuid.uuid4().hex[:12]
    meta = {
        "id": project_id,
        "name": name,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "simulated": False,
    }
    _save_project(project_id, meta)
    return _project_info(meta)


@app.get("/api/projects/{project_id}", response_model=ProjectInfo)
def get_project(project_id: str) -> ProjectInfo:
    meta = _read_project(project_id)
    if _release_meta(meta):  # user reopened the project: cancel the release
        _save_project(project_id, meta)
    return _project_info(meta)


@app.post("/api/projects/{project_id}/release")
def release_project(project_id: str) -> dict:
    """Signal that the user finished viewing this project and left.

    The frontend fires this via navigator.sendBeacon when the tab closes or
    the user navigates back to the landing page. The project (input.las,
    output, upload sessions, job records) is deleted after a short grace
    period unless the user re-opens it first — that keeps a processed scan
    from hogging disk until the next upload.
    """
    meta = _read_project(project_id)
    meta["release_at"] = time.time() + RELEASE_GRACE_SECONDS
    _save_project(project_id, meta)
    return {"released": project_id, "grace_seconds": RELEASE_GRACE_SECONDS}


@app.post("/api/projects/{project_id}/upload", response_model=ProjectInfo)
async def upload_las(project_id: str, file: UploadFile = File(...)) -> ProjectInfo:
    meta = _read_project(project_id)
    directory = _project_dir(project_id)
    if not file.filename or not file.filename.lower().endswith((".las", ".laz")):
        raise HTTPException(status_code=400, detail="Only .las / .laz files are accepted")
    # Stage to a temp name so a rejected upload never wipes a project's
    # previous good input.las; publish with an atomic rename only on success.
    target = directory / "input.las"
    staged = directory / "input.las.uploading"
    digest = hashlib.sha256()
    landed_size = 0
    with staged.open("wb") as handle:
        while True:
            block = file.file.read(1024 * 1024)
            if not block:
                break
            landed_size += len(block)
            digest.update(block)
            handle.write(block)
    if landed_size == 0:
        staged.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Upload was empty - no bytes received.")
    # Integrity gate: a text-mangled / truncated / non-LAS body must never
    # become <project>/input.las (that historically shipped the "Malformed LAS
    # file: cannot fit 'int' into an offset-sized integer" processing failures).
    # The bounded structural check runs first because laspy.open() can hang
    # forever on a garbage VLR/EVLR count in a mangled file.
    ok, reason = check_las_header_bounded(staged)
    if not ok:
        staged.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail=(
                f"Uploaded file is not a valid LAS ({reason}). The upload was "
                "rejected before it could corrupt the project; please re-upload "
                "the original file."
            ),
        )
    try:
        with laspy.open(staged) as _:
            pass
    except Exception as exc:
        staged.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail=(
                f"Uploaded file is not a valid LAS ({exc}). The upload was rejected "
                "before it could corrupt the project; please re-upload the original file."
            ),
        ) from exc
    os.replace(staged, target)  # atomic publish: the previous file is intact on rejection
    meta["input_file"] = file.filename
    meta["input_size_bytes"] = landed_size
    meta["input_sha256"] = digest.hexdigest()
    meta["uploaded_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    meta.pop("processed", None)
    _save_project(project_id, meta)
    return _project_info(meta)


# ---------------------------------------------------------------------------
# Resumable chunked uploads (3.8-5 GB LAS/LAZ never rides a single request).
#
#   POST /api/projects/{id}/uploads            -> create/resume a session
#   PUT  /api/projects/{id}/uploads/{uid}/part/{n}  -> send one chunk (local)
#   POST /api/projects/{id}/uploads/{uid}/presign   -> direct-to-S3 part URLs
#   GET  /api/projects/{id}/uploads/{uid}           -> which parts arrived
#   POST /api/projects/{id}/uploads/{uid}/complete  -> assemble input.las
#   DELETE /api/projects/{id}/uploads/{uid}         -> cancel session
#
# Each PUT is one ~8 MiB request: fast enough to finish inside any proxy's
# body timeout, small enough that a stalled chunk costs one retry, not 5 GB.
# With S3 configured the browser instead PUTs presigned URLs directly to the
# bucket, so the app server never streams file bytes at all.
# ---------------------------------------------------------------------------


def _upload_error(exc: "upload_store.UploadError") -> HTTPException:
    return HTTPException(status_code=exc.status_code, detail=str(exc))


class UploadStartRequest(BaseModel):
    filename: str
    total_size: int
    upload_id: Optional[str] = None  # client-generated id enables resume
    mode: str = "multipart"  # "multipart" (resumable parts) | "put" (one presigned PUT)


class PresignRequest(BaseModel):
    part_numbers: List[int]


class CompleteUploadRequest(BaseModel):
    # s3 mode: the browser reports [{PartNumber, ETag}, ...]; local mode ignores it.
    parts: Optional[List[Dict[str, Any]]] = None
    # Kick off the processing job in the same response (still async: the job
    # runs in the background worker; the HTTP call only queues it).
    auto_process: Optional[bool] = None


@app.post("/api/projects/{project_id}/uploads")
def create_upload(project_id: str, request: UploadStartRequest) -> dict:
    if request.total_size <= 0:
        raise HTTPException(status_code=400, detail="total_size must be positive")
    try:
        manifest = upload_store.init_upload(
            project_id=project_id,
            filename=request.filename,
            total_size=request.total_size,
            upload_id=request.upload_id or uuid.uuid4().hex[:16],
        )
    except upload_store.UploadError as exc:
        raise _upload_error(exc) from exc
    # "put" mode: hand back ONE presigned PUT URL so the browser uploads the
    # whole object straight to storage (the <1s URL request -> direct PUT
    # architecture). The S3 event notification then triggers processing.
    if request.mode == "put" and manifest["storage"] == "s3":
        try:
            presigned = upload_store.presign_put(project_id, manifest["upload_id"])
        except upload_store.UploadError as exc:
            raise _upload_error(exc) from exc
        return {**manifest, **presigned}
    return manifest


@app.put("/api/projects/{project_id}/uploads/{upload_id}/part/{part_number}")
async def upload_part(project_id: str, upload_id: str, part_number: int, file: UploadFile = File(...)) -> dict:
    """Receive one chunk (local-storage mode). Kept async so Starlette streams
    it to disk without buffering the chunk twice in memory."""
    try:
        return await upload_store.store_part_stream_async(project_id, upload_id, part_number, file.file)
    except upload_store.UploadError as exc:
        raise _upload_error(exc) from exc


@app.post("/api/projects/{project_id}/uploads/{upload_id}/presign")
def presign_parts(project_id: str, upload_id: str, request: PresignRequest) -> dict:
    try:
        urls = upload_store.presign_parts(project_id, upload_id, request.part_numbers)
        return {"urls": {str(n): u for n, u in urls.items()}}
    except upload_store.UploadError as exc:
        raise _upload_error(exc) from exc


@app.get("/api/projects/{project_id}/uploads/{upload_id}/put-url")
def presign_put(project_id: str, upload_id: str) -> dict:
    """Single presigned PUT for the whole object (two-step direct upload)."""
    try:
        return upload_store.presign_put(project_id, upload_id)
    except upload_store.UploadError as exc:
        raise _upload_error(exc) from exc


@app.get("/api/projects/{project_id}/uploads/{upload_id}")
def upload_session_status(project_id: str, upload_id: str) -> dict:
    try:
        return upload_store.upload_status(project_id, upload_id)
    except upload_store.UploadError as exc:
        raise _upload_error(exc) from exc


@app.post("/api/projects/{project_id}/uploads/{upload_id}/complete")
def complete_upload(project_id: str, upload_id: str, request: CompleteUploadRequest) -> ProjectInfo:
    """Assemble the uploaded object into input.las. Processing stays a separate
    async job (202-style): this call returns as soon as the file is in place."""
    _ensure_disk_headroom()
    try:
        result = upload_store.complete_upload(
            project_id,
            upload_id,
            _project_dir(project_id) / "input.las",
            parts=request.parts,
        )
    except upload_store.UploadError as exc:
        raise _upload_error(exc) from exc
    meta = _read_project(project_id)
    meta["input_file"] = result["filename"]
    meta["input_size_bytes"] = result["size"]
    meta["uploaded_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    meta["upload_storage"] = result["storage"]
    if result.get("sha256"):
        meta["input_sha256"] = result["sha256"]
    if result.get("external"):
        # events mode: the object stays in the bucket; the processing worker
        # fetches it at job start (server-side transfer, inside the DC).
        meta["input_external"] = {
            "bucket": os.environ.get("S3_BUCKET"),
            "key": result.get("s3_key"),
        }
    meta.pop("processed", None)
    _save_project(project_id, meta)
    job_id = None
    if request.auto_process:
        settings = _web_settings()
        try:
            job_id = _launch_job(project_id, settings)
        except HTTPException:
            pass  # active job / missing input: surface via /process endpoints
    info = _project_info(meta)
    if job_id:
        info.summary = {**(info.summary or {}), "job_id": job_id}
    return info


@app.delete("/api/projects/{project_id}/uploads/{upload_id}")
def abort_upload_session(project_id: str, upload_id: str) -> dict:
    try:
        upload_store.abort_upload(project_id, upload_id)
        return {"aborted": upload_id}
    except upload_store.UploadError as exc:
        raise _upload_error(exc) from exc


# ---------------------------------------------------------------------------
# Storage event webhook (requirement 3): after the browser uploads directly to
# S3, an S3 Event Notification (or any webhook bridge) POSTs here and the
# project becomes processable — with zero file bytes through the app server.
#
# S3 console: bucket -> Properties -> Event notifications -> "All object create
# events" -> destination: this endpoint via SNS topic (HTTP subscription) or an
# EventBridge -> API destination. See docs/UPLOADS.md for the click-path.
# ---------------------------------------------------------------------------


def _iter_storage_events(payload: Dict[str, Any]):
    """Yield (bucket, key, size) from SNS-wrapped, raw S3, or simplified JSON."""
    message = payload.get("Message")  # SNS envelope carries a JSON string
    if isinstance(message, str):
        try:
            payload = json.loads(message)
        except ValueError:
            pass
    records = payload.get("Records") or []
    if not records and payload.get("bucket") and payload.get("key"):
        records = [{"s3": {"bucket": {"name": payload["bucket"]}, "object": {"key": payload["key"], "size": payload.get("size")}}}]
    for record in records:
        s3 = record.get("s3") or {}
        bucket = (s3.get("bucket") or {}).get("name")
        key = (s3.get("object") or {}).get("key")
        if not bucket or not key:
            continue
        from urllib.parse import unquote

        yield bucket, unquote(key), (s3.get("object") or {}).get("size")


@app.post("/api/storage/events")
async def storage_event_webhook(request: Request) -> dict:
    """Accept S3/SNS ObjectCreated notifications and ingest matched sessions.

    Unknown keys return 200 with matched=0 (shared buckets fire events for
    unrelated objects; failing them would just trigger delivery retries).
    Set ACTIVE_PROCESSING=1 to auto-start the processing job on every ingest.
    """
    body = await request.body()
    try:
        payload = json.loads(body or b"{}")
    except ValueError:
        raise HTTPException(status_code=400, detail="Event body must be JSON")
    matched, ignored = [], 0
    bucket_env = os.environ.get("S3_BUCKET")
    for bucket, key, size in _iter_storage_events(payload):
        if bucket_env and bucket != bucket_env:
            ignored += 1
            continue
        try:
            summary = upload_store.ingest_external_object(bucket, key)
        except upload_store.UploadError:
            ignored += 1
            continue
        if size:
            summary["total_size"] = size
        project_id = summary["project_id"]
        meta = _read_project(project_id)
        meta["input_file"] = summary["filename"]
        meta["input_size_bytes"] = size or summary.get("total_size")
        meta["uploaded_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        meta["upload_storage"] = "s3"
        meta["input_external"] = {"bucket": bucket, "key": key}
        meta.pop("processed", None)
        _save_project(project_id, meta)
        job_id = None
        if os.environ.get("ACTIVE_PROCESSING", "").lower() in ("1", "true", "yes"):
            try:
                job_id = _launch_job(project_id, _web_settings())
            except HTTPException:
                job_id = None
        summary["job_id"] = job_id
        matched.append(summary)
    return {"matched": len(matched), "ingested": matched, "ignored": ignored}


def _run_job(project_id: str, job_id: str, settings: ProcessingSettings) -> None:
    directory = _project_dir(project_id)
    output = directory / "output"
    input_path = directory / "input.las"
    # Resume a crashed run: when the previous attempt already wrote per-tile LAS
    # files and their manifest matches this exact input file (SHA-256), keep
    # them and skip the streaming pass. Re-streaming a multi-GB file rewrites
    # the tiles and can 507 on a nearly full disk; resume also makes reprocess
    # dramatically faster. Never append into a stale/partial output otherwise.
    resume = False
    manifest = output / "tiles" / "manifest.json"
    if manifest.is_file():
        try:
            expected = json.loads(manifest.read_text(encoding="utf-8")).get("input_sha256")
            digest = hashlib.sha256()
            with open(input_path, "rb") as handle:
                for block in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(block)
            resume = bool(expected) and expected == digest.hexdigest()
        except Exception:
            resume = False
    if not resume and output.exists():
        shutil.rmtree(output, ignore_errors=True)
    settings.resume_from_tiles = resume

    def report(payload: dict) -> None:
        with _JOBS_LOCK:
            _JOBS[job_id].update(payload)
            _JOBS[job_id]["updated_at"] = time.time()
            snapshot = dict(_JOBS[job_id])
        _write_job(snapshot)

    try:
        report({"stage": "validating", "message": "Validating LAS metadata"})
        summary = process_las(
            input_path,
            output,
            settings,
            progress=False,
            progress_callback=report,
        )
        with _JOBS_LOCK:
            _JOBS[job_id].update({
                "stage": "done",
                "message": "Processing complete",
                "assets": len(summary.assets),
                "points_processed": summary.point_count,
                "point_count": summary.point_count,
                "tiles_total": summary.tile_count,
                "tiles_done": summary.tile_count,
                "elapsed_seconds": round(summary.elapsed_seconds, 2),
            })
            snapshot = dict(_JOBS[job_id])
        _write_job(snapshot)
    except OSError as exc:  # pragma: no cover - defensive
        message = _disk_full_message() if exc.errno == 28 else str(exc)
        with _JOBS_LOCK:
            _JOBS[job_id].update({"stage": "error", "message": message})
            snapshot = dict(_JOBS[job_id])
        _write_job(snapshot)
    except Exception as exc:  # pragma: no cover - defensive
        with _JOBS_LOCK:
            _JOBS[job_id].update({"stage": "error", "message": str(exc)})
            snapshot = dict(_JOBS[job_id])
        _write_job(snapshot)


@app.post("/api/projects/{project_id}/process")
def start_process(project_id: str, request: ProcessRequest) -> dict:
    _ensure_disk_headroom()
    meta = _read_project(project_id)
    directory = _project_dir(project_id)
    input_path = directory / "input.las"
    if not input_path.is_file():
        raise HTTPException(status_code=400, detail="Upload a LAS file before processing")
    settings = _web_settings(
        use_gemini=None if request.use_gemini is None else request.use_gemini not in ("false", "0", "off", "False"),
    )
    if request.tile_size_m:
        settings.tile_size_m = request.tile_size_m
    if request.viewer_point_limit:
        settings.viewer_point_limit = request.viewer_point_limit
    # Refuse to run two jobs on the same project: the second run rmtree's the
    # first one's output dir out from under it, orphaning a zombie thread that
    # spins forever on missing tiles and steals CPU from real jobs.
    active = _active_job_for_project(project_id)
    if active is not None:
        raise HTTPException(
            status_code=409,
            detail=f"This project is already being processed (job {active}). Wait for it to finish or delete the project.",
        )
    job_id = uuid.uuid4().hex[:12]
    with _JOBS_LOCK:
        _JOBS[job_id] = {
            "id": job_id,
            "project_id": project_id,
            "stage": "queued",
            "message": "Queued",
            "points_processed": 0,
            "point_count": 0,
            "tiles_done": 0,
            "tiles_total": 0,
            "assets": 0,
            "elapsed_seconds": 0.0,
            "started_at": time.time(),
        }
    _write_job(_JOBS[job_id])
    thread = threading.Thread(target=_run_job, args=(project_id, job_id, settings), daemon=True)
    thread.start()
    return {"job_id": job_id}


@app.post("/api/simulate/data", response_model=ProjectInfo)
def simulate_data(upload: UploadFile = File(...)) -> ProjectInfo:
    """Data-dependent simulation: upload a real LAS/LAZ, process it, get a project."""
    if not upload.filename or not upload.filename.lower().endswith((".las", ".laz")):
        raise HTTPException(status_code=400, detail="Only .las / .laz files are accepted")
    _ensure_disk_headroom()
    # Large captures must not ride this single synchronous request: a 3.8-5 GB
    # body risks the proxy body-timeout ("connection stalled"). Route them to
    # the resumable chunked path instead (identical processing downstream).
    upload.file.seek(0, 2)
    upload_size = upload.file.tell()
    upload.file.seek(0)
    if upload_size > SIMULATE_SYNC_MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"File is ~{upload_size // (1024 * 1024)} MiB - too large for the "
                "synchronous data-simulation request (proxies kill long uploads). "
                "Create a project, upload via POST /api/projects/{id}/uploads "
                "(chunked + resumable), then POST /api/projects/{id}/process. "
                "The pipeline result is identical."
            ),
        )
    project_id = uuid.uuid4().hex[:12]
    directory = _project_dir(project_id)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "input.las"
    with target.open("wb") as handle:
        shutil.copyfileobj(upload.file, handle)
    landed_size = target.stat().st_size if target.is_file() else 0
    landed_sha256 = hashlib.sha256(target.read_bytes()).hexdigest() if target.is_file() else ""
    if landed_size != upload_size:
        target.unlink(missing_ok=True)
        raise HTTPException(
            status_code=500,
            detail=(
                f"Data simulation write landed {landed_size} bytes but {upload_size} were sent. "
                "The upload was corrupted; retry the upload."
            ),
        )
    # Integrity gate: never run the pipeline on a text-mangled / truncated
    # body (the "Malformed LAS" failures came from exactly that). The bounded
    # structural check runs first because laspy.open() can hang forever on a
    # garbage VLR/EVLR count in a mangled file.
    ok, reason = check_las_header_bounded(target)
    if not ok:
        target.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail=(
                f"Uploaded file is not a valid LAS ({reason}). The upload was "
                "rejected before processing; please re-upload the original file."
            ),
        )
    try:
        with laspy.open(target) as _:
            pass
    except Exception as exc:
        target.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail=(
                f"Uploaded file is not a valid LAS ({exc}). The upload was rejected "
                "before processing; please re-upload the original file."
            ),
        ) from exc
    settings = _web_settings()
    output = directory / "output"
    try:
        # Process directly into the project output (no run-dir + copytree, which
        # used to double the disk footprint of large uploads).
        from .validation import validate_las

        summary = process_las(target, output, settings, progress=False)
        metadata = validate_las(target).metadata
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"Data simulation failed: {exc}")
    meta = {
        "id": project_id,
        "name": f"Data Simulation · {upload.filename}",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "simulated": True,
        "input_file": upload.filename,
        "input_size_bytes": upload_size,
        "input_sha256": landed_sha256,
        "point_count": summary.point_count,
        "crs": summary.crs,
        "las_version": summary.las_version,
        "point_format": summary.point_format,
        "asset_count": len(summary.assets),
        "backend": summary.backend,
        "run_count": summary.run_count,
        "scanner_ids": summary.scanner_ids,
        "source_kind": "Data Simulation",
        "processed": True,
        "output_dir": str(output),
        "simulation_source_file": str(target),
        "warnings": summary.warnings,
        "processing_version": summary.processing_version,
        "elapsed_seconds": round(summary.elapsed_seconds, 2),
        "scene": None,
        "simulation_note": "DATA SIMULATION - uploaded LiDAR processed through the pipeline.",
        "simulation_meta": {
            "source_file": str(target),
            "crs": metadata.crs,
            "las_version": metadata.version,
            "point_format": metadata.point_format,
            "point_count": metadata.point_count,
            "note": "DATA SIMULATION - uploaded LiDAR processed through the pipeline.",
        },
    }
    _save_project(project_id, meta)
    return _project_info(meta)


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    if job is not None:
        return job
    # In-memory record gone (restart / multi-worker): fall back to the persisted
    # job file. A record that is neither done nor error and stopped updating is
    # reported as an error so the UI can fail cleanly instead of polling forever.
    path = _job_file(job_id)
    if path.is_file():
        try:
            stale = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            raise HTTPException(status_code=404, detail="Job not found")
        if stale.get("stage") not in ("done", "error"):
            age = time.time() - float(stale.get("updated_at") or 0.0)
            if age > 90.0:
                stale.update({
                    "stage": "error",
                    "message": "The server restarted while this job was running — please run the pipeline again.",
                    "updated_at": time.time(),
                })
                _write_job(stale)
        return stale
    raise HTTPException(status_code=404, detail="Job not found")


@app.get("/api/projects/{project_id}/viewer-data")
def viewer_data(project_id: str):
    output = _project_dir(project_id) / "output"
    path = output / "viewer-data.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="No processed viewer data yet")
    return FileResponse(path, media_type="application/json")


@app.get("/api/projects/{project_id}/viewer")
def viewer_index(project_id: str):
    output = _project_dir(project_id) / "output" / "viewer"
    index = output / "index.html"
    if not index.is_file():
        raise HTTPException(status_code=404, detail="No viewer built yet")
    return FileResponse(index, media_type="text/html")


#: Download route mapping: kind -> (relative output file, media type). The
#: browser-facing downloads go through /download/{kind} so the UI can fetch a
#: typed blob; the raw /exports/{name} route stays for programmatic access.
_DOWNLOADS = {
    "json": ("assets.json", "application/json"),
    "geojson": ("assets.geojson", "application/geo+json"),
    "csv": ("assets.csv", "text/csv"),
    "inventory": ("inventory.json", "application/json"),
    "run": ("run.json", "application/json"),
    "report": ("reports/summary.md", "text/markdown"),
}


@app.get("/api/projects/{project_id}/download/{kind}")
def download_export(project_id: str, kind: str):
    """Stream a processed artifact as an attachment download.

    Explicit per-kind route (``json``, ``geojson``, ``csv``, ``inventory``,
    ``run``, ``report``) with the correct MIME type and a
    ``Content-Disposition: attachment`` filename so the browser always saves
    the file instead of rendering it inline.
    """
    entry = _DOWNLOADS.get(kind)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Unknown download kind {kind!r}")
    name, media = entry
    output = _project_dir(project_id) / "output"
    path = output / name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Export not available yet — process the project first")
    # Friendlier filename: <project-slug>-<kind>.<ext> (fall back to the
    # canonical artifact name when the project has no name).
    meta = _read_project(project_id)
    slug = re.sub(r"-+", "-", "".join(ch if ch.isalnum() else "-" for ch in (meta.get("name") or "").lower())).strip("-")
    prefix = f"{slug or 'ai4infra'}-{kind}"
    ext = name.rsplit(".", 1)[-1]
    filename = f"{prefix}.{ext}"
    return FileResponse(path, media_type=media, filename=filename)


@app.get("/api/projects/{project_id}/exports/{name}")
def export_file(project_id: str, name: str):
    output = _project_dir(project_id) / "output"
    allowed = {"assets.json", "assets.csv", "assets.geojson", "inventory.json", "run.json"}
    if name not in allowed:
        raise HTTPException(status_code=404, detail=f"Unknown export {name!r}")
    path = output / name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Export not available yet")
    media = {
        "assets.csv": "text/csv",
        "assets.geojson": "application/geo+json",
    }.get(name, "application/json")
    return FileResponse(path, media_type=media, filename=name)


@app.post("/api/simulate", response_model=ProjectInfo)
def simulate() -> ProjectInfo:
    """Quick Simulation: generate a synthetic mobile-LiDAR scene and process it
    through the real pipeline - no uploaded file required (SIMULATION ONLY).
    """
    _ensure_disk_headroom()
    project_id = uuid.uuid4().hex[:12]
    directory = _project_dir(project_id)
    directory.mkdir(parents=True, exist_ok=True)
    settings = _web_settings()
    try:
        project = run_quick_simulation(directory, settings=settings)
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"Quick simulation failed: {exc}")

    # Stage the run into the standard project layout so the project reads as
    # processed through every API endpoint (viewer-data, exports, list, get).
    generated_las = Path(project["simulation_meta"]["generated_file"])
    if generated_las.is_file():
        shutil.copy2(generated_las, directory / "input.las")
    run_dir = Path(project["output_dir"]) / "pipeline"
    output = directory / "output"
    if run_dir.is_dir():
        if output.exists():
            shutil.rmtree(output)
        shutil.copytree(run_dir, output)

    meta = {
        "id": project_id,
        "name": project["name"],
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "simulated": True,
        "input_file": project.get("input_file"),
        "input_size_bytes": project.get("input_size_bytes"),
        "point_count": project["point_count"],
        "crs": project["crs"],
        "las_version": project["las_version"],
        "point_format": project["point_format"],
        "asset_count": project["asset_count"],
        "backend": project.get("backend", {}),
        "run_count": project.get("run_count", 0),
        "scanner_ids": project.get("scanner_ids", []),
        "source_kind": project.get("source_kind", "Quick Simulation"),
        "processed": True,
        "output_dir": str(output),
        "warnings": project.get("warnings", []),
        "processing_version": project.get("processing_version", "0.3.0"),
        "elapsed_seconds": project.get("elapsed_seconds", 0.0),
        "scene": project.get("scene_summary", {}),
        "simulation_note": "SIMULATION / DEMO DATA - not real competition data",
        "simulation_meta": project.get("simulation_meta"),
    }
    _save_project(project_id, meta)
    return _project_info(meta)


@app.delete("/api/projects/{project_id}")
def delete_project(project_id: str) -> dict:
    directory = _project_dir(project_id)
    if not directory.is_dir():
        raise HTTPException(status_code=404, detail="Project not found")
    # Free the project's data AND its upload sessions (multi-GB part files)
    # plus job records, so "delete to free storage" actually frees storage.
    upload_store.abort_project_sessions(project_id)
    jobs_dir = DATA_DIR / "jobs"
    if jobs_dir.is_dir():
        for path in jobs_dir.glob("*.json"):
            try:
                job = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if job.get("project_id") == project_id:
                path.unlink(missing_ok=True)
    shutil.rmtree(directory)
    return {"deleted": project_id, "freed": True}


# Serve the built frontend last so /api/* routes always win.
if WEB_DIST.is_dir():
    app.mount("/", StaticFiles(directory=str(WEB_DIST), html=True), name="web")


def main(argv: Optional[List[str]] = None) -> int:
    """Run the inspection platform with uvicorn (if installed).

    Binds to 0.0.0.0 and honors the injected PORT (cloud previews), but falls
    back to :8766 so a local `python -m infra_inventory app` still starts even
    when PORT is unset. On a crash we print the traceback and exit with 3 so
    the caller knows the server died mid-request rather than appearing to be
    fine and then returning 502 on every request.
    """
    import os
    import sys

    try:
        import uvicorn  # type: ignore[import-not-found]
    except ImportError:
        print(
            "The API server needs `uvicorn` and `fastapi`: pip install -e '.[server]'",
            file=sys.stderr,
        )
        return 2
    host = "0.0.0.0"
    port = 8766
    injected = os.environ.get("PORT")
    if injected:
        try:
            port = int(injected)
        except ValueError:
            pass
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"AI4Infra API: http://{host}:{port}  (data: {DATA_DIR})", flush=True)
    try:
        uvicorn.run(app, host=host, port=port, log_level="warning")
    except Exception as exc:  # noqa: BLE001 - last-resort crash reporter
        print(f"AI4Infra API crashed: {exc}", file=sys.stderr, flush=True)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())