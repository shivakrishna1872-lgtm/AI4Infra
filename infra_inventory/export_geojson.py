"""Convert assets.json -> GeoJSON FeatureCollection with a proper Point geometry per asset.

Run after processing:
  python -m infra_inventory export-geojson out/ assets.geojson

This is the CLI companion to the existing GeoJSON export that the web app serves
via /api/projects/{id}/exports/assets.geojson.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def _write_geojson(assets_path: Path, out_path: Path) -> dict:
    assets = json.loads(assets_path.read_text(encoding="utf-8"))
    features: list[dict] = []
    for asset in assets:
        geom = asset.get("geometry") or {}
        center = asset.get("center") or {}
        geo = {
            "type": "Feature",
            "id": asset.get("asset_id") or asset.get("id"),
            "properties": {
                "asset_id": asset.get("asset_id") or asset.get("id"),
                "asset_class": asset.get("asset_class"),
                "subclass": asset.get("subclass"),
                "confidence": asset.get("confidence"),
                "point_count": asset.get("point_count"),
                "center": center,
                "dimensions": asset.get("dimensions"),
                "bounding_box": asset.get("bounding_box"),
                "detection_method": asset.get("detection_method"),
                "processing_version": asset.get("processing_version"),
                "coordinate_reference_system": asset.get("coordinate_reference_system"),
                "source_tile": asset.get("source_tile"),
                "source_run": asset.get("source_run"),
                "source_scanner": asset.get("source_scanner"),
            },
            "geometry": {
                "type": "Point",
                "coordinates": [
                    center.get("x"),
                    center.get("y"),
                    center.get("z"),
                ],
            },
        }
        features.append(geo)
    collection = {"type": "FeatureCollection", "features": features}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".geojson.tmp")
    tmp.write_text(json.dumps(collection, indent=2), encoding="utf-8")
    tmp.replace(out_path)
    return collection


def main(argv: list[str] | None = None) -> int:
    args = argv or sys.argv[1:]
    if len(args) < 2:
        print("usage: python -m infra_inventory export-geojson <assets.json> <out.geojson>", file=sys.stderr)
        return 2
    assets_path = Path(args[0]).expanduser().resolve()
    out_path = Path(args[1]).expanduser().resolve()
    if not assets_path.is_file():
        print(f"assets.json not found: {assets_path}", file=sys.stderr)
        return 1
    collection = _write_geojson(assets_path, out_path)
    print(f"wrote {len(collection['features'])} features -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
