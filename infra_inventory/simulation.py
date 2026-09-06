"""Standalone simulation pipeline.

Generates a synthetic mobile-LiDAR LAS with a realistic infrastructure scene,
optionally processes it through the real extraction pipeline, and returns a
project descriptor suitable for the web app's job/run model.

This is the code path behind "Quick Simulation" on the web UI. It deliberately
does not require any uploaded file: it generates the LAS first, then runs the
same pipeline as real data would go through.
"""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from .models import ProcessingSettings, RunSummary
from .pipeline import process_las
from .synthetic import build_simulated_las, build_synthetic_las
from .validation import validate_las

SIMULATED_SOURCE = "Quick Simulation"


def _summary_to_project(
    summary: RunSummary,
    *,
    name: str,
    simulated: bool,
    source_kind: str,
    input_size_bytes: Optional[int] = None,
) -> Dict[str, Any]:
    return {
        "id": str(uuid.uuid4()),
        "name": name,
        "input_file": summary.input_path,
        "input_size_bytes": input_size_bytes,
        "point_count": summary.point_count,
        "bounds": summary.bounds,
        "crs": summary.crs,
        "las_version": summary.las_version,
        "point_format": summary.point_format,
        "asset_count": len(summary.assets),
        "backend": summary.backend.get("name") if isinstance(summary.backend, dict) else summary.backend,
        "run_count": summary.run_count,
        "scanner_ids": summary.scanner_ids,
        "simulated": simulated,
        "source_kind": source_kind,
        "processed": True,
        "output_dir": str(Path(summary.input_path).resolve().parent),
        "warnings": summary.warnings,
        "processing_version": summary.processing_version,
        "elapsed_seconds": round(summary.elapsed_seconds, 2),
    }


def run_quick_simulation(
    output_root: Path,
    *,
    length_m: float = 400.0,
    seed: Optional[int] = None,
    settings: Optional[ProcessingSettings] = None,
    scene_options: Optional[Dict[str, Any]] = None,
    project_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Generate a Quick Simulation scene and process it through the real pipeline.

    Returns a project descriptor dict ready for ``api.project.open`` style use.
    """
    settings = settings or ProcessingSettings()
    scene_options = scene_options or {}
    out_dir = output_root / "simulations" / f"quick-{uuid.uuid4().hex[:10]}"
    out_dir.mkdir(parents=True, exist_ok=True)
    las_path = out_dir / "quick_simulation.las"

    gen = build_simulated_las(
        las_path,
        length_m=length_m,
        ground_spacing=scene_options.get("ground_spacing", 0.25),
        seed=seed,
    )

    summary = process_las(
        gen["path"],
        out_dir / "pipeline",
        settings=settings,
        progress=False,
    )

    meta = validate_las(gen["path"])
    input_size_bytes = (Path(gen["path"]).stat().st_size
                        if Path(gen["path"]).is_file() else None)

    project = _summary_to_project(
        summary,
        name=f"Quick Simulation · {length_m:.0f} m corridor",
        simulated=True,
        source_kind=SIMULATED_SOURCE,
        input_size_bytes=input_size_bytes,
    )
    if project_overrides:
        project.update(project_overrides)
    project["scene_summary"] = gen["scene"]
    project["simulation_meta"] = {
        "generated_file": gen["path"],
        "bounds": gen["bounds"],
        "scene_parts": gen["scene"],
        "ground_truth": gen.get("ground_truth", []),
        "seed": gen.get("seed"),
        "generated_at": gen["generated_at"],
        "note": "SIMULATED DATA - not competition data; generated for demo/QA only.",
    }
    return project


def run_data_simulation(
    input_path: Path,
    output_root: Path,
    *,
    settings: Optional[ProcessingSettings] = None,
    project_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run the existing data-dependent simulation path on an uploaded LAS/LAZ.

    This preserves the prior behaviour: the user uploads a real file, the
    pipeline validates it, processes it, and returns a project descriptor.
    """
    settings = settings or ProcessingSettings()
    out_dir = output_root / "simulations" / f"data-{uuid.uuid4().hex[:10]}"
    out_dir.mkdir(parents=True, exist_ok=True)
    work_input = out_dir / "uploaded.las"
    shutil.copy2(input_path.resolve(), work_input)

    summary = process_las(
        str(work_input),
        out_dir / "pipeline",
        settings=settings,
        progress=False,
    )
    input_size_bytes = work_input.stat().st_size if work_input.is_file() else None

    project = _summary_to_project(
        summary,
        name=f"Data Simulation · {input_path.name}",
        simulated=True,
        source_kind="Data Simulation",
        input_size_bytes=input_size_bytes,
    )
    if project_overrides:
        project.update(project_overrides)
    project["input_file_name"] = input_path.name
    project["simulation_meta"] = {
        "source_file": str(input_path),
        "processed_file": str(work_input),
        "note": "DATA SIMULATION - uploaded LiDAR processed through the pipeline.",
    }
    return project


def run_small_synthetic(
    output_root: Path,
    *,
    seed: int = 7,
    settings: Optional[ProcessingSettings] = None,
) -> Dict[str, Any]:
    """Small deterministic synthetic used by tests / lightweight smoke paths."""
    settings = settings or ProcessingSettings()
    out_dir = output_root / "simulations" / f"synthetic-{uuid.uuid4().hex[:10]}"
    out_dir.mkdir(parents=True, exist_ok=True)
    las_path = out_dir / "synthetic.las"

    gen = build_synthetic_las(las_path, seed=seed)
    summary = process_las(gen["path"], out_dir / "pipeline", settings=settings, progress=False)

    input_size_bytes = (Path(gen["path"]).stat().st_size
                        if Path(gen["path"]).is_file() else None)
    project = _summary_to_project(
        summary,
        name=f"Synthetic test scene · {seed}",
        simulated=True,
        source_kind="Quick Simulation",
        input_size_bytes=input_size_bytes,
    )
    project["scene_summary"] = gen["scene"]
    project["simulation_meta"] = {
        "generated_file": gen["path"],
        "bounds": gen["bounds"],
        "scene_parts": gen["scene"],
        "seed": gen.get("seed"),
        "generated_at": gen["generated_at"],
        "note": "SMALL SYNTHETIC TEST SCENE - not competition data.",
    }
    return project
