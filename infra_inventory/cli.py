from __future__ import annotations

import argparse
import json
import os
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .models import ProcessingSettings
from .pipeline import process_las
from .pointcept import describe_backend


def _process(args: argparse.Namespace) -> int:
    settings = ProcessingSettings(
        chunk_size=args.chunk_size, tile_size_m=args.tile_size, viewer_point_limit=args.viewer_points,
        backend=args.backend, pointcept_root=args.pointcept_root, pointcept_config=args.pointcept_config,
        pointcept_weight=args.pointcept_weight, pointcept_num_gpus=args.pointcept_num_gpus, strict_las14=args.strict_las14,
    )
    result = process_las(args.input, args.output, settings)
    print(json.dumps({"assets": len(result.assets), "output": str(Path(args.output).resolve()), "warnings": result.warnings}, indent=2))
    return 0


def _serve(args: argparse.Namespace) -> int:
    directory = Path(args.output).expanduser().resolve() / "viewer"
    if not (directory / "index.html").is_file():
        raise FileNotFoundError(f"Viewer not found: {directory / 'index.html'}")
    handler = partial(SimpleHTTPRequestHandler, directory=str(directory))
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Viewer: http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()


def main() -> int:
    parser = argparse.ArgumentParser(prog="infra-inventory", description="Auditable mobile-LiDAR infrastructure inventory")
    subparsers = parser.add_subparsers(dest="command", required=True)
    process = subparsers.add_parser("process", help="Process a LAS file into inventory artifacts")
    process.add_argument("input", help="Input LAS/LAZ file")
    process.add_argument("--output", required=True, help="Output directory")
    process.add_argument("--chunk-size", type=int, default=500_000)
    process.add_argument("--tile-size", type=float, default=40.0)
    process.add_argument("--viewer-points", type=int, default=120_000)
    process.add_argument("--strict-las14", action="store_true")
    process.add_argument("--backend", choices=("geometry", "pointcept"), default="geometry")
    process.add_argument("--pointcept-root")
    process.add_argument("--pointcept-config")
    process.add_argument("--pointcept-weight")
    process.add_argument("--pointcept-num-gpus", type=int, default=1)
    process.set_defaults(func=_process)
    serve = subparsers.add_parser("serve", help="Serve an output viewer locally")
    serve.add_argument("output", help="Pipeline output directory")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.set_defaults(func=_serve)
    backends = subparsers.add_parser("backends", help="Report available inference backends")
    backends.add_argument("--pointcept-root")
    backends.set_defaults(func=lambda args: (print(json.dumps(describe_backend(args.pointcept_root), indent=2)) or 0))
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
