"""Resumable chunked uploads for multi-GB LAS/LAZ files.

Problem this solves: a 3.8-5 GB point cloud cannot survive a single HTTP
request through the app server. Any proxy or gateway with a 60-second body
timeout kills the connection mid-transfer ("connection stalled") and the whole
upload restarts from zero.

Design (two interchangeable backends behind one API):

* ``local`` (default, zero external dependencies): the browser sends the file
  in ~8 MiB chunks to ``POST /api/projects/{id}/uploads/{uid}/part/{n}``.
  Chunks land in ``DATA_DIR/uploads/<upload_id>/`` and are concatenated
  exactly once on ``complete``. Any interrupted upload resumes from the last
  acknowledged part (see ``status``).
* ``s3`` (enabled automatically when AWS credentials + bucket are configured):
  the backend only *presigns* S3 ``PutObject`` URLs for each part; the browser
  uploads directly to S3, so the app server never streams file bytes. The
  multipart upload is finalized server-side (``CompleteMultipartUpload``) and
  the finished object is streamed down to the project's ``input.las``.

Session manifests are JSON under ``DATA_DIR/uploads/<upload_id>/manifest.json``
so both modes survive a server restart (the browser re-queries ``status`` and
continues where it left off).

The pipeline contract is unchanged: every path ends with a complete
``<project>/input.las`` on local disk, so processing, viewer export, and the
MongoDB mirror need no modification.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = [
    "UploadError",
    "chunk_size",
    "chunk_count",
    "storage_mode",
    "s3_configured",
    "init_upload",
    "store_part",
    "presign_parts",
    "upload_status",
    "complete_upload",
    "ingest_external_object",
    "fetch_external_object",
    "presign_put",
    "abort_upload",
    "prune_stale_sessions",
]

# Same appdata root as server.py (kept here so upload_store has no import
# cycle with the FastAPI module; both resolve the identical path).
DATA_DIR = Path(__file__).resolve().parent.parent / "appdata"

_UPLOADS_DIR = DATA_DIR / "uploads"

#: 8 MiB chunks: small enough to retry cheaply on a flaky connection, large
#: enough that a 5 GB file needs only ~640 requests (S3 min part size is 5 MiB,
#: max 10 000 parts -> supports files up to ~80 TB at this chunk size).
DEFAULT_CHUNK_BYTES = 8 * 1024 * 1024

#: Sessions older than this are pruned at startup (crashed/abandoned uploads).
SESSION_TTL_SECONDS = 48 * 3600

_UPLOAD_ID_RE = re.compile(r"^[a-f0-9]{12,32}$")
_PROJECT_ID_RE = re.compile(r"^[a-f0-9]{12,32}$")
_LAS_SUFFIXES = (".las", ".laz")

_LOCK = threading.Lock()


class UploadError(Exception):
    """User-facing upload error (HTTP 400/404 mapping is done by the server)."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


# ---------------------------------------------------------------------------
# Storage backend selection
# ---------------------------------------------------------------------------

_S3_ENV_KEYS = ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION")


def s3_configured() -> bool:
    """True when AWS credentials + bucket are available in the environment."""
    if os.environ.get("STORAGE_BACKEND", "").lower() == "local":
        return False
    if not all(os.environ.get(k) for k in _S3_ENV_KEYS):
        return False
    return bool(os.environ.get("S3_BUCKET"))


def storage_mode() -> str:
    return "s3" if s3_configured() else "local"


def chunk_size() -> int:
    raw = os.environ.get("UPLOAD_CHUNK_BYTES")
    if raw and raw.isdigit() and 1024 * 1024 <= int(raw) <= 256 * 1024 * 1024:
        return int(raw)
    return DEFAULT_CHUNK_BYTES


def chunk_count(total_size: int) -> int:
    if total_size <= 0:
        raise UploadError("File size must be a positive number of bytes")
    return max(1, -(-total_size // chunk_size()))  # ceil division


def _s3():
    """Lazily build the boto3 client (import cost paid only in s3 mode)."""
    try:
        import boto3  # type: ignore[import-not-found]
        from botocore.config import Config  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - guarded by s3_configured
        raise UploadError(
            "STORAGE_BACKEND=s3 requires boto3: pip install -e '.[s3]'"
        ) from exc
    return boto3.client(
        "s3",
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        region_name=os.environ.get("AWS_REGION"),
        endpoint_url=os.environ.get("S3_ENDPOINT_URL") or None,
        config=Config(signature_version="s3v4", retries={"max_attempts": 5}),
    )


def _bucket() -> str:
    bucket = os.environ.get("S3_BUCKET")
    if not bucket:  # pragma: no cover - guarded by s3_configured
        raise UploadError("S3_BUCKET is not configured", 500)
    return bucket


def _s3_key(upload_id: str, filename: str) -> str:
    safe = filename.replace("\\", "_").replace("/", "_")
    return f"lidar-uploads/{upload_id}/{safe}"


def _parse_s3_key(key: str) -> Optional[Dict[str, str]]:
    """Reverse of _s3_key: parse a notification key back to (upload_id, filename).

    Returns None for keys outside the lidar-uploads/ namespace so unrelated
    bucket activity never matches a session.
    """
    parts = key.split("/")
    if len(parts) != 3 or parts[0] != "lidar-uploads":
        return None
    return {"upload_id": parts[1], "filename": parts[2]}


# ---------------------------------------------------------------------------
# Manifest handling
# ---------------------------------------------------------------------------

def _session_dir(upload_id: str) -> Path:
    if not _UPLOAD_ID_RE.match(upload_id):
        raise UploadError("Malformed upload id")
    return _UPLOADS_DIR / upload_id


def _manifest_path(upload_id: str) -> Path:
    return _session_dir(upload_id) / "manifest.json"


def _read_manifest(upload_id: str) -> Dict[str, Any]:
    path = _manifest_path(upload_id)
    if not path.is_file():
        raise UploadError(f"Unknown upload session: {upload_id}", 404)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise UploadError(f"Corrupt upload session {upload_id}: {exc}", 500) from exc


def _write_manifest(manifest: Dict[str, Any]) -> None:
    path = _manifest_path(manifest["upload_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    tmp.replace(path)


def _validate_ids(project_id: str, upload_id: str) -> None:
    if not _PROJECT_ID_RE.match(project_id):
        raise UploadError("Malformed project id")
    if not _UPLOAD_ID_RE.match(upload_id):
        raise UploadError("Malformed upload id")


def _check_session_owner(manifest: Dict[str, Any], project_id: str) -> None:
    if manifest.get("project_id") != project_id:
        raise UploadError("Upload session belongs to a different project", 403)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def init_upload(
    project_id: str,
    filename: str,
    total_size: int,
    upload_id: str,
) -> Dict[str, Any]:
    """Create (or resume) an upload session and return its client contract."""
    if not filename.lower().endswith(_LAS_SUFFIXES):
        raise UploadError("Only .las / .laz files are accepted")
    _validate_ids(project_id, upload_id)

    mode = storage_mode()
    csize = chunk_size()
    ccount = chunk_count(total_size)

    existing_path = _manifest_path(upload_id)
    if existing_path.is_file():
        manifest = _read_manifest(upload_id)
        _check_session_owner(manifest, project_id)
        # Idempotent resume: same upload_id -> hand back the same contract.
        manifest["resumed"] = True
        _write_manifest(manifest)
        return manifest

    manifest: Dict[str, Any] = {
        "upload_id": upload_id,
        "project_id": project_id,
        "filename": filename,
        "total_size": int(total_size),
        "chunk_size": csize,
        "chunk_count": ccount,
        "storage": mode,
        "created_at": time.time(),
        "parts_received": [],
        "resumed": False,
        # How /complete finalizes the remote object:
        #   copy   - (removed: kept the pipeline local-file contract simple)
        #   stream - server downloads the object (default; works with any creds)
        #   events - the file never flows through this server: an S3 event
        #            notification/webhook calls ingest_external_object() and
        #            the processing worker reads straight from the bucket.
        "finalize_mode": "stream",
    }
    if mode == "s3":
        key = _s3_key(upload_id, filename)
        try:
            created = _s3().create_multipart_upload(
                Bucket=_bucket(),
                Key=key,
                ContentType="application/octet-stream",
            )
        except Exception as exc:
            raise UploadError(f"S3 CreateMultipartUpload failed: {exc}", 502) from exc
        manifest["s3_key"] = key
        manifest["s3_upload_id"] = created["UploadId"]
    else:
        _session_dir(upload_id).mkdir(parents=True, exist_ok=True)

    _write_manifest(manifest)
    return manifest


def store_part(project_id: str, upload_id: str, part_number: int, data: bytes) -> Dict[str, Any]:
    """Store one chunk of the file on the server (local mode)."""
    manifest = _read_manifest(upload_id)
    _check_session_owner(manifest, project_id)
    if manifest["storage"] != "local":
        raise UploadError("This session uploads parts directly to S3 (presigned URLs)", 409)
    if not 1 <= part_number <= manifest["chunk_count"]:
        raise UploadError(
            f"Part number {part_number} outside 1..{manifest['chunk_count']}"
        )

    part_dir = _session_dir(upload_id) / "parts"
    part_dir.mkdir(parents=True, exist_ok=True)
    part_path = part_dir / f"part-{part_number:06d}"
    tmp = part_path.with_suffix(".tmp")
    tmp.write_bytes(data)
    tmp.replace(part_path)  # atomic: a half-written part is never acknowledged

    with _LOCK:
        if part_number not in manifest["parts_received"]:
            manifest["parts_received"].append(part_number)
            manifest["parts_received"].sort()
        _write_manifest(manifest)
    return {"received": part_number, "parts_received": manifest["parts_received"]}


def presign_parts(project_id: str, upload_id: str, part_numbers: List[int]) -> Dict[int, str]:
    """Presign direct-to-S3 PUT URLs for the requested part numbers (s3 mode)."""
    manifest = _read_manifest(upload_id)
    _check_session_owner(manifest, project_id)
    if manifest["storage"] != "s3":
        raise UploadError("This session uploads parts to the API (local mode)", 409)
    if not part_numbers:
        raise UploadError("No part numbers requested")
    if min(part_numbers) < 1 or max(part_numbers) > manifest["chunk_count"]:
        raise UploadError(
            f"Part numbers outside 1..{manifest['chunk_count']}"
        )
    client = _s3()
    urls: Dict[int, str] = {}
    for n in part_numbers:
        try:
            urls[n] = client.generate_presigned_url(
                "upload_part",
                Params={
                    "Bucket": _bucket(),
                    "Key": manifest["s3_key"],
                    "UploadId": manifest["s3_upload_id"],
                    "PartNumber": n,
                },
                ExpiresIn=int(os.environ.get("S3_PRESIGN_TTL", "3600")),
            )
        except Exception as exc:
            raise UploadError(f"S3 presign failed for part {n}: {exc}", 502) from exc
    return urls


def upload_status(project_id: str, upload_id: str) -> Dict[str, Any]:
    """Acknowledge which parts the server/storage already holds (resume support)."""
    manifest = _read_manifest(upload_id)
    _check_session_owner(manifest, project_id)
    received = sorted(manifest.get("parts_received", []))
    out: Dict[str, Any] = {
        "upload_id": upload_id,
        "storage": manifest["storage"],
        "chunk_size": manifest["chunk_size"],
        "chunk_count": manifest["chunk_count"],
        "parts_received": received,
    }
    if manifest["storage"] == "s3":
        out["s3_parts"] = _s3_received_parts(manifest)
    return out


def _s3_received_parts(manifest: Dict[str, Any]) -> List[int]:
    """List parts S3 has acknowledged (authoritative after browser crashes)."""
    client = _s3()
    parts: List[int] = []
    token = None
    while True:
        kwargs: Dict[str, Any] = {
            "Bucket": _bucket(),
            "Key": manifest["s3_key"],
            "UploadId": manifest["s3_upload_id"],
        }
        if token:
            kwargs["PartNumberMarker"] = token
        resp = client.list_parts(**kwargs)
        parts.extend(int(p["PartNumber"]) for p in resp.get("Parts", []))
        token = resp.get("NextPartNumberMarker")
        if not resp.get("IsTruncated"):
            break
    return sorted(parts)


def complete_upload(
    project_id: str,
    upload_id: str,
    input_target: Path,
    parts: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Assemble the uploaded file into ``input_target`` (atomic) and clean up.

    Returns ``{"size": ..., "sha256": ..., "storage": ...}``.
    """
    manifest = _read_manifest(upload_id)
    _check_session_owner(manifest, project_id)
    input_target.parent.mkdir(parents=True, exist_ok=True)
    tmp_target = input_target.with_name(input_target.name + ".assembling")

    if manifest["storage"] == "s3":
        parts = parts or [
            {"PartNumber": n, "ETag": ""}
            for n in _s3_received_parts(manifest)
        ]
        client = _s3()
        if parts and not parts[-1].get("ETag"):
            missing = [p["PartNumber"] for p in parts if not p.get("ETag")]
            raise UploadError(
                f"Missing ETags for part(s) {missing[:8]}. Re-send them with the "
                "presigned flow (the browser reads ETags from the PUT response; "
                "the bucket CORS must expose the ETag header)."
            )
        client.complete_multipart_upload(
            Bucket=_bucket(),
            Key=manifest["s3_key"],
            UploadId=manifest["s3_upload_id"],
            MultipartUpload={"Parts": parts},
        )
        mode = manifest.get("finalize_mode", "stream")
        if mode == "events":
            # The object stays in the bucket: the S3 event notification (or the
            # POST /api/storage/events webhook) triggers processing directly.
            # Nothing is copied or downloaded here.
            return {
                "size": int(manifest["total_size"]),
                "sha256": None,
                "storage": "s3",
                "filename": manifest["filename"],
                "external": True,
                "s3_key": manifest["s3_key"],
            }
        # finalize_mode == "stream" (default): bring the object down to local
        # disk so the existing pipeline contract (<project>/input.las) holds.
        client.download_file(_bucket(), manifest["s3_key"], str(tmp_target))
        # Best-effort cleanup of the finished object + any stale session parts.
        try:
            client.delete_object(Bucket=_bucket(), Key=manifest["s3_key"])
        except Exception:
            pass
    else:
        received = sorted(manifest.get("parts_received", []))
        if received != list(range(1, manifest["chunk_count"] + 1)):
            missing = sorted(set(range(1, manifest["chunk_count"] + 1)) - set(received))
            raise UploadError(
                f"Upload incomplete: {len(missing)} part(s) missing "
                f"(first few: {missing[:8]}). Resume before completing."
            )
        part_dir = _session_dir(upload_id) / "parts"
        digest = hashlib.sha256()
        size = 0
        with tmp_target.open("wb") as out:
            for n in received:
                part_path = part_dir / f"part-{n:06d}"
                with part_path.open("rb") as chunk:
                    while True:
                        block = chunk.read(4 * 1024 * 1024)
                        if not block:
                            break
                        size += len(block)
                        digest.update(block)
                        out.write(block)
        if size != manifest["total_size"]:
            tmp_target.unlink(missing_ok=True)
            raise UploadError(
                f"Assembled size {size} != declared size {manifest['total_size']}"
            )

    os.replace(tmp_target, input_target)  # atomic publish
    result = {
        "size": input_target.stat().st_size,
        "sha256": None,
        "storage": manifest["storage"],
        "filename": manifest["filename"],
    }
    if manifest["storage"] == "local":
        # Cheap integrity marker for the ops runbook (computed during concat).
        result["sha256"] = digest.hexdigest()
    abort_upload(project_id, upload_id, ignore_errors=True)
    return result


def fetch_external_object(bucket: str, key: str, target: Path) -> Dict[str, Any]:
    """Worker-side fetch for event-ingested objects: streams s3://bucket/key
    to ``target`` (atomic) so the pipeline's local-file contract holds. This
    transfer runs inside the datacenter — the browser's 5 GB upload never
    touches the app server.
    """
    client = _s3()
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".assembling")
    client.download_file(bucket, key, str(tmp))
    os.replace(tmp, target)
    return {"path": str(target), "size": target.stat().st_size}


def presign_put(project_id: str, upload_id: str) -> Dict[str, Any]:
    """Two-step direct upload (no multipart): return ONE presigned PUT URL for
    the whole object. The browser uploads the entire file straight to storage;
    the S3 event notification then hits POST /api/storage/events and processing
    starts without the client ever calling /complete.

    Prefer the multipart flow for resumability; this is the simplest correct
    variant of "ask for URL (<1s) then PUT directly to S3".
    """
    manifest = _read_manifest(upload_id)
    _check_session_owner(manifest, project_id)
    if manifest["storage"] != "s3":
        raise UploadError("Whole-object presigned PUT needs s3 mode", 409)
    url = _s3().generate_presigned_url(
        "put_object",
        Params={
            "Bucket": _bucket(),
            "Key": manifest["s3_key"],
            "ContentType": "application/octet-stream",
        },
        ExpiresIn=int(os.environ.get("S3_PRESIGN_TTL", "3600")),
    )
    return {"url": url, "s3_key": manifest["s3_key"], "method": "PUT"}


def ingest_external_object(bucket: str, key: str) -> Dict[str, Any]:
    """Event-driven ingest (requirement 3): an S3 event notification / webhook
    reports that the browser uploaded directly to storage — no app-server byte
    ever flowed for this file.

    Flow: the client first created a session via POST /uploads (in s3 mode but
    WITHOUT going through presign/complete), then the frontend PUT the whole
    object to a presigned single-shot URL, or to the key derived from the
    session (lidar-uploads/<upload_id>/<filename>). When the storage platform
    fires its ObjectCreated event at POST /api/storage/events, the server
    matches the key to the session, records the object pointer, and the project
    is ready to process (the worker streams from the bucket).

    Returns the ingest summary; raises UploadError(404) when no session matches
    (callers should then decide whether the event is from an unknown client).
    """
    parsed = _parse_s3_key(key)
    if parsed is None:
        raise UploadError(f"Key outside the upload namespace: {key}", 404)
    upload_id = parsed["upload_id"]
    try:
        manifest = _read_manifest(upload_id)
    except UploadError:
        raise UploadError(
            f"No upload session matches s3://{bucket}/{key} — create the session "
            "via POST /api/projects/{id}/uploads before uploading to storage",
            404,
        )
    if manifest.get("storage") != "s3":
        raise UploadError("Session is not in s3 mode", 409)
    manifest["s3_key"] = manifest.get("s3_key") or key
    manifest["finalize_mode"] = "events"
    manifest["event_ingested_at"] = time.time()
    with _LOCK:
        _write_manifest(manifest)
    return {
        "upload_id": upload_id,
        "project_id": manifest["project_id"],
        "s3_key": manifest["s3_key"],
        "filename": manifest["filename"],
        "total_size": manifest["total_size"],
    }


def abort_upload(project_id: str, upload_id: str, ignore_errors: bool = False) -> None:
    """Cancel a session: drop local chunks / abort the S3 multipart upload."""
    try:
        manifest = _read_manifest(upload_id)
        _check_session_owner(manifest, project_id)
    except UploadError:
        if ignore_errors:
            return
        raise
    if manifest["storage"] == "s3":
        try:
            _s3().abort_multipart_upload(
                Bucket=_bucket(),
                Key=manifest["s3_key"],
                UploadId=manifest["s3_upload_id"],
            )
        except Exception:
            if not ignore_errors:
                raise
    shutil.rmtree(_session_dir(upload_id), ignore_errors=True)


def prune_stale_sessions(now: Optional[float] = None) -> int:
    """Delete expired local session directories (call once at startup)."""
    if not _UPLOADS_DIR.is_dir():
        return 0
    now = now if now is not None else time.time()
    removed = 0
    for session in _UPLOADS_DIR.iterdir():
        if not session.is_dir():
            continue
        try:
            manifest = json.loads(
                (session / "manifest.json").read_text(encoding="utf-8")
            )
            age = now - float(manifest.get("created_at") or 0.0)
            storage = manifest.get("storage", "local")
        except (OSError, ValueError):
            age = SESSION_TTL_SECONDS + 1  # unreadable -> treat as stale
            storage = "local"
        if age <= SESSION_TTL_SECONDS:
            continue
        if storage == "s3":
            # Abort the remote upload so orphaned parts stop accruing cost.
            try:
                _s3().abort_multipart_upload(
                    Bucket=_bucket(),
                    Key=manifest.get("s3_key", ""),
                    UploadId=manifest.get("s3_upload_id", ""),
                )
            except Exception:
                pass
        shutil.rmtree(session, ignore_errors=True)
        removed += 1
    return removed
