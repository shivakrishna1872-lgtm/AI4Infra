# Large-file uploads (3.8–5 GB LAS/LAZ): resumable, proxy-proof

## The failure this fixes

A 3.8–5 GB point cloud uploaded as **one** HTTP request dies at the first
60-second proxy hop ("connection stalled"). The failure is structural:

- the request body must finish *and* be forwarded before any proxy timeout,
- one dropped packet anywhere restarts the whole 5 GB transfer,
- the app server is a bottleneck for bytes that never need to touch it.

## The architecture now deployed

```
browser                          app server (FastAPI)              storage
───────                          ────────────────────              ───────
POST /api/projects/{id}/uploads  ── create session ──────────────▶  manifest
GET  .../uploads/{uid}           ── which parts already arrived ──▶  (resume)
per part (~8 MiB):
  local mode: PUT .../part/{n}   ── one small request each ──────▶  appdata/uploads/…
  s3 mode:    PUT presigned URL  ───────────── direct ─────────────▶  S3 (server never sees bytes)
POST .../uploads/{uid}/complete  ── assemble input.las ──────────▶  project dir
POST /api/projects/{id}/process  ── 202-style job id ───────────▶  background thread
```

Every request is ~8 MiB: it finishes far inside any 60 s body-timeout window,
and a stalled part costs **one retry, not the whole file**. Sessions persist on
disk (`appdata/uploads/<upload_id>/manifest.json`), so a browser refresh or a
server restart resumes from the last acknowledged part. The frontend derives a
stable upload id from `name:size:mtime`, so simply retrying the same file
resumes the same session.

Processing was already decoupled (requirement 4): `POST /process` launches a
background job thread and returns a `job_id` immediately; progress arrives via
`GET /api/jobs/{job_id}`. Upload completion now behaves the same way: the
`complete` call returns as soon as `input.las` is in place.

## Mode 1 — local chunked store (default, zero configuration)

No env vars needed. Chunks land in `appdata/uploads/<upload_id>/parts/` and are
concatenated into `<project>/input.las` on `complete` (with a sha256 of the
assembled file recorded in `project.json` as `input_sha256`).

Tuning:

| env var | default | meaning |
|---|---|---|
| `UPLOAD_CHUNK_BYTES` | `8388608` (8 MiB) | part size (1 MiB – 256 MiB) |
| `STORAGE_BACKEND=local` | – | force local mode even when S3 env exists |

Stale sessions (48 h) are pruned at startup.

## Mode 2 — direct-to-S3 presigned multipart (browser → bucket)

Activate by setting:

| env var | example |
|---|---|
| `AWS_ACCESS_KEY_ID` | `AKIA…` |
| `AWS_SECRET_ACCESS_KEY` | `…` |
| `AWS_REGION` | `us-east-1` |
| `S3_BUCKET` | `ai4infra-lidar-uploads` |
| `S3_ENDPOINT_URL` (optional) | `https://<account>.r2.cloudflarestorage.com` for Cloudflare R2 / MinIO |
| `S3_PRESIGN_TTL` (optional) | `3600` seconds |

When these are present the backend switches automatically: `init_upload` calls
`CreateMultipartUpload`, each part is fetched via `POST .../uploads/{uid}/presign`
(`upload_part` presigned PUT, SigV4), the browser uploads directly to S3, and
`complete` calls `CompleteMultipartUpload` then streams the object down to the
project directory (server→S3 traffic stays in-datacenter). The IAM user needs
`s3:PutObject`, `s3:ListMultipartUploadParts`, `s3:AbortMultipartUpload` on the
bucket. Install with `pip install -e '.[server,s3]'`.

### Bucket CORS (required — the browser needs the ETag back)

```json
[
  {
    "AllowedOrigins": ["https://your-app.example.com", "http://localhost:5173"],
    "AllowedMethods": ["PUT"],
    "AllowedHeaders": ["*"],
    "ExposeHeaders": ["ETag"],
    "MaxAgeSeconds": 3600
  }
]
```

Without `ExposeHeaders: ["ETag"]` the frontend cannot read part ETags and the
upload aborts with a clear error message. For Cloudflare R2 set the same CORS
rule in the bucket settings; for MinIO use `mc cors set`.

### Lifecycle rule (recommended)

Abort incomplete multipart uploads after 7 days and expire `lidar-uploads/*`
after 30 days — `complete` deletes the finished object immediately, but a
crashed session's parts would otherwise linger.

## Proxy / gateway timeout fallback (requirement 3)

See [`deploy/nginx-uploads.conf`](../deploy/nginx-uploads.conf) for the full
NGINX reference: `client_max_body_size 10G;`, `client_body_timeout 3600s;`,
`proxy_read_timeout 3600s;`, `proxy_request_buffering off;`. With the chunked
uploader these are belt-and-suspenders (each part is one small request), but the
legacy `POST /upload` and `POST /api/simulate/data` endpoints still benefit.

`POST /api/simulate/data` now refuses bodies > 128 MiB (HTTP 413) with
instructions to use the chunked path instead — it processes synchronously and
can never survive a proxy timeout for competition-sized files.

## API surface (new)

| endpoint | purpose |
|---|---|
| `POST /api/projects/{id}/uploads` | create/resume session → `{upload_id, storage, chunk_size, chunk_count}` |
| `GET /api/projects/{id}/uploads/{uid}` | acknowledged parts (resume support; authoritative from S3 `ListParts` in s3 mode) |
| `PUT /api/projects/{id}/uploads/{uid}/part/{n}` | send one chunk (local mode) |
| `POST /api/projects/{id}/uploads/{uid}/presign` | presigned part URLs (s3 mode) |
| `POST /api/projects/{id}/uploads/{uid}/complete` | assemble `input.las` (atomic; sha256 recorded) |
| `DELETE /api/projects/{id}/uploads/{uid}` | cancel (drops chunks / aborts the S3 multipart upload) |

### curl walkthrough

```bash
PID=$(curl -s -X POST 'localhost:8766/api/projects?name=demo' | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])')
UID=$(python3 -c 'import uuid;print(uuid.uuid4().hex[:16])')
SIZE=$(stat -c%s big.las)
curl -s -X POST localhost:8766/api/projects/$PID/uploads \
  -H 'Content-Type: application/json' \
  -d "{\"filename\":\"big.las\",\"total_size\":$SIZE,\"upload_id\":\"$UID\"}"
# -> {"chunk_count": N, "chunk_size": 8388608, ...}
split -b 8388608 -d -a 4 big.las part-
for i in $(seq -f "%04g" 0 $(( $(ls part-* | wc -l) - 1 ))); do
  curl -s -X PUT --data-binary @part-$i \
    localhost:8766/api/projects/$PID/uploads/$UID/part/$((10#$i + 1)) > /dev/null
done
curl -s -X POST localhost:8766/api/projects/$PID/uploads/$UID/complete -H 'Content-Type: application/json' -d '{}'
curl -s -X POST localhost:8766/api/projects/$PID/process -H 'Content-Type: application/json' -d '{}'
# -> {"job_id": "..."}  then poll GET /api/jobs/{job_id}
```

## Background workers (scaling beyond one box)

Processing already runs off the request loop via the job thread + persisted job
records. To move heavy PDAL/LAStools-style runs onto dedicated machines, run
the same `process_las` call inside a Celery worker:

```bash
pip install celery[redis]
export CELERY_BROKER_URL=redis://broker:6379/0
```

```python
# infra_inventory/tasks.py
import os
from celery import Celery
from .pipeline import process_las
from .models import ProcessingSettings

celery_app = Celery(__name__, broker=os.environ["CELERY_BROKER_URL"])

@celery_app.task(name="process_pointcloud")
def process_pointcloud(input_path: str, output_dir: str, settings: dict):
    return process_las(input_path, output_dir, ProcessingSettings.from_dict(settings)).summary_dict()
```

Then replace the `threading.Thread(...)` launch in `server.py::_run_job`'s
caller with `process_pointcloud.delay(str(input_path), str(output), asdict(settings))`
and have `GET /api/jobs/{id}` read from Redis. The API contract does not change.

## Frontend

`web/src/api.ts` exports `uploadLasChunked(projectId, file, onProgress)` — used
automatically by `api.uploadLas`, so every upload path in the UI (Landing page
file picker) is chunked + resumable with per-part retry/backoff. Progress
callbacks report `{uploadedBytes, totalBytes, part, parts}` for the status line.
