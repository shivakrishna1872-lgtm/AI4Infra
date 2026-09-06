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

import json
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import __version__
from .models import ProcessingSettings
from .pipeline import process_las
from .simulation import run_quick_simulation, run_data_simulation, run_small_synthetic

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


def _write_job(job: dict) -> None:
    """Atomically persist a job record so it survives process restarts."""
    path = _job_file(job["id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(job, indent=2), encoding="utf-8")
    tmp.replace(path)

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

STAGES = [
    "validating",
    "streaming",
    "pointcept",
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


class ProcessRequest(BaseModel):
    tile_size_m: Optional[float] = None
    viewer_point_limit: Optional[int] = None


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
    (directory / "project.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


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
        info.summary = meta
    info.scene = meta.get("scene")
    return info


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "version": __version__, "jobs": len(_JOBS)}


@app.get("/api/projects")
def list_projects() -> List[ProjectInfo]:
    root = DATA_DIR / "projects"
    if not root.is_dir():
        return []
    result = []
    for directory in sorted(root.iterdir()):
        meta_path = directory / "project.json"
        if meta_path.is_file():
            try:
                result.append(_project_info(json.loads(meta_path.read_text(encoding="utf-8"))))
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
    return _project_info(_read_project(project_id))


@app.post("/api/projects/{project_id}/upload", response_model=ProjectInfo)
async def upload_las(project_id: str, file: UploadFile = File(...)) -> ProjectInfo:
    meta = _read_project(project_id)
    directory = _project_dir(project_id)
    if not file.filename or not file.filename.lower().endswith((".las", ".laz")):
        raise HTTPException(status_code=400, detail="Only .las / .laz files are accepted")
    target = directory / "input.las"
    with target.open("wb") as handle:
        shutil.copyfileobj(file.file, handle)
    meta["input_file"] = file.filename
    meta["input_size_bytes"] = target.stat().st_size
    meta["uploaded_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    meta.pop("processed", None)
    _save_project(project_id, meta)
    return _project_info(meta)


def _run_job(project_id: str, job_id: str, settings: ProcessingSettings) -> None:
    directory = _project_dir(project_id)
    output = directory / "output"
    input_path = directory / "input.las"

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
    except Exception as exc:  # pragma: no cover - defensive
        with _JOBS_LOCK:
            _JOBS[job_id].update({"stage": "error", "message": str(exc)})
            snapshot = dict(_JOBS[job_id])
        _write_job(snapshot)


@app.post("/api/projects/{project_id}/process")
def start_process(project_id: str, request: ProcessRequest) -> dict:
    meta = _read_project(project_id)
    directory = _project_dir(project_id)
    input_path = directory / "input.las"
    if not input_path.is_file():
        raise HTTPException(status_code=400, detail="Upload a LAS file before processing")
    settings = ProcessingSettings()
    if request.tile_size_m:
        settings.tile_size_m = request.tile_size_m
    if request.viewer_point_limit:
        settings.viewer_point_limit = request.viewer_point_limit
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
    project_id = uuid.uuid4().hex[:12]
    directory = _project_dir(project_id)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "input.las"
    with target.open("wb") as handle:
        shutil.copyfileobj(upload.file, handle)
    settings = ProcessingSettings()
    settings.viewer_point_limit = 250000
    try:
        project = run_data_simulation(target, directory, settings=settings)
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"Data simulation failed: {exc}")

    # Stage into the standard project layout so exports/viewer endpoints work.
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
        "input_file": project.get("input_file_name"),
        "input_size_bytes": project.get("input_size_bytes"),
        "point_count": project["point_count"],
        "crs": project["crs"],
        "las_version": project["las_version"],
        "point_format": project["point_format"],
        "asset_count": project["asset_count"],
        "backend": project.get("backend", {}),
        "run_count": project.get("run_count", 0),
        "scanner_ids": project.get("scanner_ids", []),
        "source_kind": project.get("source_kind", "Data Simulation"),
        "processed": True,
        "output_dir": str(output),
        "simulation_source_file": str(target),
        "warnings": project.get("warnings", []),
        "processing_version": project.get("processing_version", "0.3.0"),
        "elapsed_seconds": project.get("elapsed_seconds", 0.0),
        "scene": project.get("scene_summary", {}),
        "simulation_note": "DATA SIMULATION - uploaded LiDAR processed through the pipeline.",
        "simulation_meta": project.get("simulation_meta"),
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
    project_id = uuid.uuid4().hex[:12]
    directory = _project_dir(project_id)
    directory.mkdir(parents=True, exist_ok=True)
    settings = ProcessingSettings()
    settings.viewer_point_limit = 250000
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
    shutil.rmtree(directory)
    return {"deleted": project_id}


# Serve the built frontend last so /api/* routes always win.
if WEB_DIST.is_dir():
    app.mount("/", StaticFiles(directory=str(WEB_DIST), html=True), name="web")


def main(argv: Optional[List[str]] = None) -> int:
    """Run the server with uvicorn (if installed)."""
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
    uvicorn.run(app, host=host, port=port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())