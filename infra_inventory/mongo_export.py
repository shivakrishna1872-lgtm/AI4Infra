"""Optional MongoDB mirror of the asset inventory.

Every extracted asset is stored as a single GeoJSON-aware document in the
shape infrastructure owners expect:

    {
      "asset_id": "POL-00017",
      "category": "Utilities",
      "subcategory": "Utility Pole",
      "location": {"type": "Point", "coordinates": [easting, northing, elevation]},
      "attributes": {
        "height_m": 8.5,
        "lean_angle_deg": 1.2,
        "run_source": "Run 1 Laser Left",
        "condition": "Good"
      },
      ...
    }

The export is a *mirror*: MongoDB is never a source of truth for the pipeline.
The canonical inventory is the JSON/CSV/GeoJSON export set; the mirror exists
so agencies can query assets spatially (a ``2dsphere`` index is created on
``location``) without touching the point cloud.

Activation (all optional):

* ``MONGO_URI`` env var set on the server/CLI process -> every pipeline run
  also upserts the inventory (``MONGO_DB`` default ``ai4infra``,
  ``MONGO_COLLECTION`` default ``assets``), or
* ``--mongo-uri`` / ``--mongo-db`` / ``--mongo-collection`` on the CLI, or
* ``scripts/export_mongo.py``-style calls against an existing output dir.

pymongo is an optional dependency (``pip install -e '.[mongo]'``). Without it
(or without a reachable server) the pipeline keeps working and records a
warning - the inventory must never fail because its mirror is down.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

#: Pipeline class -> competition category (the four required asset classes).
CATEGORY_MAP: Dict[str, str] = {
    "pavement": "Pavement",
    "pavement_marking": "Pavement",
    "utility_pole": "Utilities",
    "overhead_conductor": "Utilities",
    "utility_cabinet": "Utilities",
    "traffic_sign": "Signs",
    "guardrail": "Safety",
    "safety_barrier": "Safety",
    "rumble_strip": "Safety",
}

#: Pipeline class -> human subcategory label (attributes schema).
SUBCATEGORY_LABELS: Dict[str, str] = {
    "pavement": "Pavement / Travelled Surface",
    "pavement_marking": "Painted Pavement Marking",
    "utility_pole": "Utility Pole",
    "overhead_conductor": "Overhead Conductor",
    "utility_cabinet": "Utility Cabinet",
    "traffic_sign": "Traffic Sign Panel",
    "guardrail": "Guardrail",
    "safety_barrier": "Safety Barrier",
    "rumble_strip": "Rumble Strip",
}


def _lean_angle_deg(asset: Dict[str, Any]) -> Optional[float]:
    """Lean from eigen-verticality (documented ALP geometric proxy).

    ``geometry.eigen_verticality`` is the strength of the dominant vertical
    axis (1.0 = perfectly vertical); lean is its angular deviation in degrees.
    Missing geometry -> ``None``, never a guessed value.
    """
    geometry = asset.get("geometry")
    if not isinstance(geometry, dict):
        return None
    verticality = geometry.get("eigen_verticality")
    if verticality is None:
        return None
    return round(math.degrees(math.acos(max(0.0, min(1.0, float(verticality))))), 2)


def _run_source(asset: Dict[str, Any]) -> str:
    run = asset.get("source_run")
    scanner = asset.get("source_scanner")
    if run and scanner:
        return f"Run {run} {scanner}"
    if run:
        return f"Run {run}"
    if scanner:
        return scanner
    return "UNKNOWN"


def asset_to_document(asset: Dict[str, Any], run: Dict[str, Any]) -> Dict[str, Any]:
    """Transform one inventory record into the MongoDB document schema."""
    center = asset.get("center") or {}
    x = center.get("x")
    y = center.get("y")
    z = center.get("z")
    if x is None or y is None or z is None:
        raise ValueError(f"Asset {asset.get('asset_id')} has no measured center; refusing to guess a location")

    asset_class = asset.get("class", "")
    dimensions = asset.get("dimensions") or {}
    bounding_box = asset.get("bounding_box")
    run_source = _run_source(asset)
    attributes = {
        "height_m": dimensions.get("height_m"),
        "width_m": dimensions.get("width_m"),
        "length_m": dimensions.get("length_m"),
        "lean_angle_deg": _lean_angle_deg(asset),
        "orientation_deg": asset.get("orientation_deg"),
        "run_source": run_source,
        # Trimble MX9 competition schema aliases (requested document shape):
        "system_source": f"Trimble MX9 - {run_source}" if run_source != "UNKNOWN" else "Trimble MX9",
        "condition": asset.get("condition"),
        "condition_flag": asset.get("condition"),
        "recommended_action": asset.get("recommended_action"),
        "point_count": asset.get("point_count"),
    }
    if asset.get("avg_intensity") is not None:
        attributes["avg_intensity"] = asset.get("avg_intensity")
    elif (asset.get("intensity_stats") or {}).get("mean") is not None:
        attributes["avg_intensity"] = round(float((asset.get("intensity_stats") or {}).get("mean")), 1)
    return {
        "asset_id": asset.get("asset_id"),
        "category": CATEGORY_MAP.get(asset_class, "Other"),
        "subcategory": SUBCATEGORY_LABELS.get(asset_class, asset_class.replace("_", " ").title()),
        "location": {
            "type": "Point",
            "coordinates": [x, y, z],  # [easting, northing, elevation] in the source CRS
        },
        "attributes": attributes,
        "confidence": asset.get("confidence"),
        "review_required": bool(asset.get("review_required")),
        "confidence_factors": asset.get("confidence_factors") or {},
        "detection_method": asset.get("detection_method"),
        "qc_flags": asset.get("qc_flags") or [],
        "geometry": {
            "bbox": list(bounding_box) if bounding_box else None,
            "ground_elevation_m": (asset.get("geometry") or {}).get("ground_elevation_m"),
        },
        "crs": asset.get("coordinate_reference_system"),
        "source_run": asset.get("source_run"),
        "source_scanner": asset.get("source_scanner"),
        "source_tile": asset.get("source_tile"),
        "source_file": run.get("input_path"),
        "processing_version": asset.get("processing_version") or run.get("processing_version"),
        "processed_at": datetime.now(timezone.utc).isoformat(),
    }


def _pymongo():
    """Import pymongo lazily (optional dependency), with a clear install hint."""
    try:
        import pymongo  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "MongoDB export requires pymongo: pip install -e '.[mongo]'"
        ) from exc
    return pymongo


def export_to_mongo(
    inventory: List[Dict[str, Any]],
    run: Dict[str, Any],
    uri: str,
    database: str = "ai4infra",
    collection: str = "assets",
    replace_run: bool = True,
) -> Dict[str, Any]:
    """Upsert the inventory into MongoDB; returns a small summary dict.

    ``replace_run`` removes prior documents of the same ``source_file`` first,
    so the collection mirrors the latest run of each input instead of stacking
    stale copies. GeoJSON ``2dsphere`` and ``asset_id`` indexes are created.
    """
    pymongo = _pymongo()

    client = pymongo.MongoClient(uri, serverSelectionTimeoutMS=5000)
    try:
        client.admin.command("ping")  # fail fast with a real connection error
    except Exception:
        client.close()
        raise
    column = client[database][collection]
    if replace_run:
        column.delete_many({"source_file": run.get("input_path")})
    documents = [asset_to_document(asset, run) for asset in inventory]
    if documents:
        column.insert_many(documents, ordered=False)
    try:
        column.create_index([("location", "2dsphere")])
    except Exception:
        pass  # pre-existing or unsupported index is not fatal for the mirror
    try:
        column.create_index("asset_id", unique=True)
    except Exception:
        pass  # legacy duplicates exist; the replace-run delete keeps it clean going forward
    client.close()
    return {"inserted": len(documents), "database": database, "collection": collection}


def export_output_dir_to_mongo(
    output_dir: Any,
    uri: str,
    database: str = "ai4infra",
    collection: str = "assets",
) -> Dict[str, Any]:
    """Standalone variant: read ``assets.json`` / ``run.json`` from an output dir."""
    import json
    from pathlib import Path

    output = Path(output_dir)
    assets = json.loads((output / "assets.json").read_text(encoding="utf-8"))
    run = json.loads((output / "run.json").read_text(encoding="utf-8"))
    return export_to_mongo(assets, run, uri, database=database, collection=collection)