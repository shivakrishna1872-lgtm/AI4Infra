"""AI4Infra command-line interface.

Commands:

    process input.las --output out/     full pipeline (validation -> tiling -> detection -> inventory)
    serve out/                          dependency-free 3D viewer
    validate input.las                  LAS validation report only
    tile input.las --output out/tiles   tiling stage only
    backends                            report available inference backends (GPU/CUDA detection)
    download-models                     fetch documented pretrained weights
    train ...                           optional Pointcept fine-tuning bridge
    demo-data out/                      generate a synthetic test LAS

Exit codes: 0 success, 1 user error, 2 pipeline failure.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Optional

from .errors import InfraError
from .models import ProcessingSettings
from .pipeline import process_las
from .pointcept import describe_backend
from .simulation import run_data_simulation

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]

EXIT_OK, EXIT_USER, EXIT_PIPELINE = 0, 1, 2


def _load_settings_yaml(config: Optional[str]) -> Dict[str, object]:
    if not config:
        return {}
    if yaml is None:
        raise InfraError("YAML support is required for --config.", "Install with: pip install pyyaml")
    path = Path(config).expanduser().resolve()
    if not path.is_file():
        raise InfraError(f"Config file not found: {path}", "Point --config at an existing YAML file.")
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise InfraError(f"Config file must contain a YAML mapping: {path}", "Fix the file format.")
    return data


def _merge_settings(args: argparse.Namespace) -> ProcessingSettings:
    values = _load_settings_yaml(getattr(args, "config", None))
    settings = ProcessingSettings.from_dict(values)
    # CLI flags override YAML
    for name in ("chunk_size", "tile_size", "viewer_points", "strict_las14", "backend",
                 "pointcept_root", "pointcept_config", "pointcept_weight", "pointcept_class_names",
                 "pointcept_num_gpus", "roadmarking_command", "roadmarking_config"):
        if name in ("tile_size", "viewer_points"):
            field = "tile_size_m" if name == "tile_size" else "viewer_point_limit"
            value = getattr(args, name, None)
            if value is not None:
                setattr(settings, field, value)
        elif getattr(args, name, None) is not None:
            setattr(settings, name, getattr(args, name))
    return settings


def _process(args: argparse.Namespace) -> int:
    settings = _merge_settings(args)
    try:
        result = process_las(args.input, args.output, settings)
    except InfraError as exc:
        print(exc.user_message(), file=sys.stderr)
        return EXIT_PIPELINE
    print(json.dumps({
        "assets": len(result.assets),
        "by_class": {key: value for key, value in _class_counts(result.assets).items()},
        "tiles": result.tile_count,
        "point_count": result.point_count,
        "crs": result.crs,
        "elapsed_seconds": round(result.elapsed_seconds, 2),
        "warnings": result.warnings,
        "output": str(Path(args.output).expanduser().resolve()),
    }, indent=2))
    return EXIT_OK


def _class_counts(assets) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for asset in assets:
        counts[asset.asset_class] = counts.get(asset.asset_class, 0) + 1
    return dict(sorted(counts.items()))


def _serve(args: argparse.Namespace) -> int:
    directory = Path(args.output).expanduser().resolve() / "viewer"
    if not (directory / "index.html").is_file():
        print(
            f"Viewer not found at {directory / 'index.html'}. Run `python -m infra_inventory process` first.",
            file=sys.stderr,
        )
        return EXIT_USER
    host = args.host
    port = args.port
    # Freebuff/cloud previews inject a PORT and require binding to 0.0.0.0
    injected = os.environ.get("PORT")
    if injected:
        host = host if args.host != "127.0.0.1" else "0.0.0.0"
        try:
            port = int(injected)
        except ValueError:
            pass
    handler = partial(SimpleHTTPRequestHandler, directory=str(directory))
    server = ThreadingHTTPServer((host, port), handler)
    print(f"AI4Infra viewer: http://{host}:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return EXIT_OK
    finally:
        server.server_close()
    return EXIT_OK


def _validate(args: argparse.Namespace) -> int:
    from .validation import validate_las

    try:
        result = validate_las(args.input, strict_las14=args.strict_las14)
    except InfraError as exc:
        print(exc.user_message(), file=sys.stderr)
        return EXIT_PIPELINE
    for issue in result.issues:
        marker = {"info": "i", "warning": "!", "error": "x"}[issue.severity]
        print(f"[{marker}] {issue.message}" + (f"\n      -> {issue.hint}" if issue.hint else ""))
    return EXIT_OK


def _tile(args: argparse.Namespace) -> int:
    import laspy
    import numpy as np

    from .las_reader import iter_chunks
    from .pipeline import _append_tile, _tile_header  # shared internals

    settings = _merge_settings(args)
    path = Path(args.input).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    try:
        with laspy.open(path) as reader:
            source_header = reader.header
        tile_names: list[str] = []
        for _, _, chunk in iter_chunks(path, settings.chunk_size):
            tx = np.floor(chunk.x / settings.tile_size_m).astype(np.int64)
            ty = np.floor(chunk.y / settings.tile_size_m).astype(np.int64)
            for key in np.unique(np.column_stack((tx, ty)), axis=0):
                tile_mask = (tx == key[0]) & (ty == key[1])
                tile_name = f"tile_{key[0]}_{key[1]}"
                if tile_name not in tile_names:
                    tile_names.append(tile_name)
                _append_tile(output, tile_name, source_header, chunk, tile_mask)
        print(json.dumps({"tiles": len(tile_names), "output": str(output)}, indent=2))
    except InfraError as exc:
        print(exc.user_message(), file=sys.stderr)
        return EXIT_PIPELINE
    return EXIT_OK


def _backends(args: argparse.Namespace) -> int:
    print(json.dumps(describe_backend(args.pointcept_root, args.pointcept_class_names), indent=2))
    return EXIT_OK


def _download_models(args: argparse.Namespace) -> int:
    from .download_models import download_models

    try:
        download_models(args.output, args.model)
    except InfraError as exc:
        print(exc.user_message(), file=sys.stderr)
        return EXIT_PIPELINE
    return EXIT_OK


def _train(args: argparse.Namespace) -> int:
    """Optional Pointcept fine-tuning bridge: run upstream tools/train.py."""
    from .errors import BackendConfigurationError

    root = Path(args.pointcept_root).expanduser().resolve() if args.pointcept_root else None
    script = root / "tools" / "train.py" if root else None
    if root is None or script is None or not script.is_file():
        raise BackendConfigurationError(
            "Fine-tuning requires a Pointcept checkout with tools/train.py.",
            "Clone https://github.com/Pointcept/Pointcept, install its environment, "
            "then pass --pointcept-root. See docs/ENHANCEMENT.md.",
        )
    import os
    import subprocess

    command = [sys.executable, str(script), "--config-file", str(Path(args.pointcept_config).expanduser().resolve())]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root) + os.pathsep + environment.get("PYTHONPATH", "")
    print(f"[train] running: {' '.join(command)}", flush=True)
    completed = subprocess.run(command, cwd=str(root), env=environment)
    return completed.returncode


def _demo_data(args: argparse.Namespace) -> int:
    from .synthetic import build_synthetic_las

    target = Path(args.output) / "mannford_synthetic.las"
    summary = build_synthetic_las(target)
    print(json.dumps(summary, indent=2))
    print("\nThis synthetic file is for TESTING ONLY - never present it as competition data.")
    return EXIT_OK


def _simulate(args: argparse.Namespace) -> int:
    from .simulation import run_quick_simulation
    from .models import ProcessingSettings

    settings = ProcessingSettings()
    settings.viewer_point_limit = args.viewer_points or settings.viewer_point_limit
    project = run_quick_simulation(
        Path(args.output),
        length_m=args.length,
        seed=args.seed,
        settings=settings,
    )
    print(json.dumps({
        "project": project["name"],
        "project_id": project["id"],
        "points": project["point_count"],
        "assets": project["asset_count"],
        "output_dir": project["output_dir"],
        "simulated": True,
        "scene_parts": project.get("scene_summary"),
        "simulation_meta": project.get("simulation_meta"),
    }, indent=2))
    print("\nSIMULATION / DEMO DATA - never present simulated detections as competition results.")
    return EXIT_OK


def _app(args: argparse.Namespace) -> int:
    from .server import main as server_main

    return server_main()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="infra-inventory",
        description="AI4Infra: auditable infrastructure asset inventory from mobile LiDAR LAS files.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    process = subparsers.add_parser("process", help="Full pipeline: LAS -> tiling -> detection -> inventory")
    process.add_argument("input", help="Input LAS/LAZ file")
    process.add_argument("--output", required=True, help="Output directory")
    process.add_argument("--config", help="YAML processing config (configs/processing.yaml)")
    process.add_argument("--chunk-size", type=int, help="Streaming chunk size (points)")
    process.add_argument("--tile-size", type=float, help="Spatial tile edge (metres)")
    process.add_argument("--viewer-points", type=int, help="Points sampled for the 3D viewer")
    process.add_argument("--strict-las14", action="store_true", help="Fail on non-LAS-1.4 input")
    process.add_argument("--backend", choices=("geometry", "pointcept"), help="Inference backend")
    process.add_argument("--pointcept-root", help="Path to the Pointcept checkout")
    process.add_argument("--pointcept-config", help="Adapted Pointcept config consuming output/tiles/")
    process.add_argument("--pointcept-weight", help="PTv3 checkpoint (.pth)")
    process.add_argument("--pointcept-class-names", help="JSON list of the config's class taxonomy")
    process.add_argument("--pointcept-num-gpus", type=int, help="GPUs for Pointcept inference")
    process.add_argument("--roadmarking-command", help="RoadMarkingExtraction run script (opt-in)")
    process.add_argument("--roadmarking-config", help="RoadMarkingExtraction parameter config")
    process.set_defaults(func=_process)

    serve = subparsers.add_parser("serve", help="Serve an output viewer locally")
    serve.add_argument("output", help="Pipeline output directory")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.set_defaults(func=_serve)

    validate = subparsers.add_parser("validate", help="Validate a LAS file and print a report")
    validate.add_argument("input", help="Input LAS/LAZ file")
    validate.add_argument("--strict-las14", action="store_true")
    validate.set_defaults(func=_validate)

    tile = subparsers.add_parser("tile", help="Tiling stage only")
    tile.add_argument("input", help="Input LAS/LAZ file")
    tile.add_argument("--output", required=True, help="Output tiles directory")
    tile.add_argument("--config", help="YAML processing config")
    tile.add_argument("--chunk-size", type=int)
    tile.add_argument("--tile-size", type=float)
    tile.set_defaults(func=_tile)

    backends = subparsers.add_parser("backends", help="Report available inference backends")
    backends.add_argument("--pointcept-root")
    backends.add_argument("--pointcept-class-names")
    backends.set_defaults(func=_backends)

    models = subparsers.add_parser("download-models", help="Download documented pretrained model weights")
    models.add_argument("--output", default="models", help="Output directory (default: ./models)")
    models.add_argument("--model", default="nuscenes-ptv3-semseg", help="Model key (see scripts/download_models.py)")
    models.set_defaults(func=_download_models)

    train = subparsers.add_parser("train", help="Optional Pointcept fine-tuning bridge (requires Pointcept env)")
    train.add_argument("--pointcept-root", required=True, help="Path to the Pointcept checkout")
    train.add_argument("--pointcept-config", required=True, help="Pointcept training config")
    train.set_defaults(func=_train)

    demo = subparsers.add_parser("demo-data", help="Generate a synthetic test LAS (testing only)")
    demo.add_argument("--output", default="data", help="Output directory (default: ./data)")
    demo.set_defaults(func=_demo_data)

    simulate = subparsers.add_parser("simulate", help="Generate a realistic simulated corridor and run the pipeline (demo only)")
    simulate.add_argument("--output", default="data", help="Output directory (default: ./data)")
    simulate.add_argument("--length", type=float, default=400.0, help="Corridor length in metres")
    simulate.add_argument("--seed", type=int, default=None, help="Scene seed (omit for randomized)")
    simulate.add_argument("--viewer-points", type=int, help="Points sampled for the 3D viewer")
    simulate.set_defaults(func=_simulate)

    app_parser = subparsers.add_parser("app", help="Run the FastAPI inspection platform (pip install -e '.[server]')")
    app_parser.set_defaults(func=_app)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except InfraError as exc:
        print(exc.user_message(), file=sys.stderr)
        return EXIT_PIPELINE
    except FileNotFoundError as exc:
        print(f"Input not found: {exc.filename or exc}", file=sys.stderr)
        return EXIT_USER
    except KeyboardInterrupt:
        return EXIT_USER


if __name__ == "__main__":
    raise SystemExit(main())