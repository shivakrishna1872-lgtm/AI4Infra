"""Adapter for the external RoadMarkingExtraction C++ subsystem.

`RoadMarkingExtraction <https://github.com/YuePanEdward/RoadMarkingExtraction>`_
is a C++ pipeline (PCL / OpenCV / LibLas / DXFLib) for MLS/ALS road-marking
extraction, classification, and vectorization. It is not vendored here; the
adapter invokes the user's configured run script on the pipeline's exported
tiles and ingests the resulting DXF vector files.

Contract with the external system (documented in docs/PROCEDURE.md):

1. ``roadmarking_command`` points at a run script that consumes LAS input from
   ``<workdir>/input/`` and writes outputs (including DXF) to ``<workdir>/output/``.
2. The adapter stages one tile at a time, invokes the script, then parses every
   ``.dxf`` in the output directory.
3. DXF geometry is parsed for LINE and LWPOLYLINE entities; segments are
   re-grouped into marking instances; each instance becomes an inventory asset
   with ``detection_method: roadmarkingextraction-v1``.

Because this is an external subsystem with a pinned C++ environment, it is
strictly opt-in; the portable native marking detector runs by default.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .errors import RoadMarkingConfigurationError
from .instances import group_line_segments

SEGMENT = Tuple[float, float, float, float, float, float]


def _require(command: Optional[str], config: Optional[str]) -> None:
    if not command:
        raise RoadMarkingConfigurationError(
            "RoadMarkingExtraction requested but no run script was configured.",
            "Pass --roadmarking-command /path/to/run_xxx.sh (and --roadmarking-config for its parameter file).",
        )
    if not Path(command).expanduser().is_file():
        raise RoadMarkingConfigurationError(
            f"RoadMarkingExtraction run script not found: {command}",
            "Build RoadMarkingExtraction (cmake .. && make) and point at the generated run script.",
        )
    if config and not Path(config).expanduser().is_file():
        raise RoadMarkingConfigurationError(
            f"RoadMarkingExtraction config not found: {config}",
            "Point at an existing parameter list (see the upstream ./config/ directory).",
        )


def run_roadmarking(
    command: str,
    config: Optional[str],
    tiles_dir: Path,
    tile_name: str,
    work_dir: Path,
) -> List[Dict[str, object]]:
    """Run the external extractor on one tile and return parsed marking segments."""
    _require(command, config)
    work_dir.mkdir(parents=True, exist_ok=True)
    input_dir = work_dir / "input"
    output_dir = work_dir / "output"
    for directory in (input_dir, output_dir):
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True)

    tile_path = tiles_dir / f"{tile_name}.las"
    if not tile_path.is_file():
        raise RoadMarkingConfigurationError(
            f"Tile not found for RoadMarkingExtraction: {tile_path}",
            "Re-run the pipeline with --save-tiles enabled.",
        )
    shutil.copy2(tile_path, input_dir / f"{tile_name}.las")

    environment = os.environ.copy()
    environment["TERRA_POINT_TILE"] = tile_name
    environment["TERRA_POINT_INPUT_DIR"] = str(input_dir)
    environment["TERRA_POINT_OUTPUT_DIR"] = str(output_dir)
    command_list = [command]
    if config:
        command_list.append(str(Path(config).expanduser().resolve()))
    completed = subprocess.run(command_list, cwd=work_dir, env=environment, capture_output=True, text=True)
    if completed.returncode:
        raise RoadMarkingConfigurationError(
            f"RoadMarkingExtraction failed on tile {tile_name} (exit {completed.returncode}).",
            "Check the C++ environment (PCL/OpenCV/LibLas/DXFLib) and the parameter config. "
            "Tail of output:\n" + (completed.stderr or completed.stdout)[-2000:],
        )
    return parse_dxf_outputs(output_dir)


def parse_dxf_outputs(output_dir: Path) -> List[Dict[str, object]]:
    """Parse LINE / LWPOLYLINE entities from all .dxf files in the output dir.

    Returns one dict per re-grouped marking instance:
      {geometry: [segments...], length_m, width_m, center, source_file}
    """
    segments: List[Tuple[SEGMENT, str]] = []
    if not output_dir.is_dir():
        return []
    for dxf in sorted(output_dir.glob("*.dxf")):
        for segment in _parse_dxf_lines(dxf):
            segments.append((segment, dxf.name))
    if not segments:
        return []
    grouped = group_line_segments([segment for segment, _ in segments], gap_m=1.5)
    instances: List[Dict[str, object]] = []
    for group in grouped:
        xs = [coord for segment in group for coord in (segment[0], segment[3])]
        ys = [coord for segment in group for coord in (segment[1], segment[4])]
        zs = [coord for segment in group for coord in (segment[2], segment[5])]
        instances.append({
            "geometry": group,
            "center": {"x": sum(xs) / len(xs), "y": sum(ys) / len(ys), "z": sum(zs) / len(zs)},
            "length_m": max(max(xs) - min(xs), max(ys) - min(ys)),
            "width_m": min(max(xs) - min(xs), max(ys) - min(ys)),
            "point_count": None,
            "source_files": sorted({name for _, name in segments if _ in group}),
            "detection_method": "roadmarkingextraction-v1",
        })
    return instances


def _parse_dxf_lines(path: Path) -> List[SEGMENT]:
    """Minimal DXF text parser for LINE and LWPOLYLINE entities (ASCII DXF)."""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    segments: List[SEGMENT] = []
    index = 0
    while index < len(lines):
        if lines[index].strip() == "0" and index + 1 < len(lines) and lines[index + 1].strip() == "LINE":
            index += 2
            entity: Dict[int, float] = {}
            while index + 1 < len(lines):
                try:
                    code = int(lines[index].strip())
                except ValueError:
                    break
                value = lines[index + 1].strip()
                index += 2
                if code in (10, 20, 30, 11, 21, 31):
                    try:
                        entity[code] = float(value)
                    except ValueError:
                        pass
                if code == 0:
                    index -= 2
                    break
            if {10, 20, 30, 11, 21, 31}.issubset(entity):
                segments.append((entity[10], entity[20], entity[30], entity[11], entity[21], entity[31]))
            continue
        if lines[index].strip() == "0" and index + 1 < len(lines) and lines[index + 1].strip() == "LWPOLYLINE":
            index += 2
            vertices: List[List[float]] = []
            while index + 1 < len(lines):
                code = lines[index].strip()
                value = lines[index + 1].strip()
                index += 2
                if code == "10":
                    vertices.append([float(value)])
                elif code == "20" and vertices:
                    vertices[-1].append(float(value))
                elif code == "30" and vertices:
                    vertices[-1].append(float(value))
                if code == "0":
                    break
            if len(vertices) >= 2:
                for a, b in zip(vertices, vertices[1:]):
                    if len(a) == 3 and len(b) == 3:
                        segments.append((a[0], a[1], a[2], b[0], b[1], b[2]))
            continue
        index += 1
    return segments