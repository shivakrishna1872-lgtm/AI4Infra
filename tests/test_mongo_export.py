"""MongoDB mirror export: document schema and export behavior (stubbed client)."""
from __future__ import annotations

import types
from pathlib import Path

import pytest

from infra_inventory.mongo_export import (
    asset_to_document,
    export_to_mongo,
    export_output_dir_to_mongo,
)

RUN = {
    "input_path": "/data/mannford_run1_left.las",
    "point_count": 12_000_000,
    "crs": "EPSG:2248",
    "processing_version": "0.3.0",
}


def _asset(**overrides):
    asset = {
        "asset_id": "POL-00017",
        "class": "utility_pole",
        "subclass": "vertical_support",
        "center": {"x": 613_004.21, "y": 2_447_918.54, "z": 318.42},
        "bounding_box": (613_003.9, 2_447_918.2, 310.0, 613_004.6, 2_447_918.9, 320.2),
        "dimensions": {"length_m": 0.32, "width_m": 0.32, "height_m": 10.2},
        "point_count": 840,
        "source_tile": "tile_15325_61198",
        "coordinate_reference_system": "EPSG:2248",
        "confidence": 0.94,
        "confidence_factors": {"model": None, "geometry": 0.97, "support": 0.91, "spatial_context": 0.95, "class_consistency": 0.88},
        "confidence_explanation": "Vertical columnar component",
        "detection_method": "geometry-v2-heuristic",
        "orientation_deg": 42.0,
        "source_run": "1",
        "source_scanner": "Laser Left",
        "source_point_source_id": 7,
        "model_prior_class": None,
        "model_confidence": None,
        "processing_version": "0.3.0",
        "geometry": {"ground_elevation_m": 310.1, "eigen_verticality": 0.998, "eigen_planarity": 0.03, "eigen_linearity": 0.9},
        "qc_flags": [],
        "flagged": False,
        "condition": "GOOD",
        "recommended_action": "No action — asset within measured tolerances.",
        "review_required": False,
        "assessment_reasoning": "verticality 0.998 -> lean 3.6 deg",
    }
    asset.update(overrides)
    return asset


def test_document_schema_matches_agency_contract() -> None:
    doc = asset_to_document(_asset(), RUN)
    # The exact shape the brief specifies
    assert doc["asset_id"] == "POL-00017"
    assert doc["category"] == "Utilities"
    assert doc["subcategory"] == "Utility Pole"
    assert doc["location"] == {"type": "Point", "coordinates": [613_004.21, 2_447_918.54, 318.42]}
    attributes = doc["attributes"]
    assert attributes["height_m"] == 10.2
    assert attributes["lean_angle_deg"] == pytest.approx(3.62, abs=0.1)
    assert attributes["run_source"] == "Run 1 Laser Left"
    assert attributes["condition"] == "GOOD"
    # audit trail survives into the mirror
    assert doc["confidence"] == 0.94
    assert doc["review_required"] is False
    assert doc["crs"] == "EPSG:2248"
    assert doc["source_file"] == RUN["input_path"]
    assert doc["geometry"]["bbox"] == list(_asset()["bounding_box"])


def test_category_mapping_covers_competition_classes() -> None:
    expected = {
        "pavement": "Pavement", "pavement_marking": "Pavement",
        "utility_pole": "Utilities", "overhead_conductor": "Utilities",
        "utility_cabinet": "Utilities", "traffic_sign": "Signs",
        "guardrail": "Safety", "safety_barrier": "Safety", "rumble_strip": "Safety",
    }
    for asset_class, category in expected.items():
        doc = asset_to_document(_asset(**{"class": asset_class}), RUN)
        assert doc["category"] == category, asset_class


def test_lean_is_none_when_geometry_missing() -> None:
    doc = asset_to_document(_asset(geometry=None), RUN)
    assert doc["attributes"]["lean_angle_deg"] is None


def test_center_is_never_guessed() -> None:
    with pytest.raises(ValueError):
        asset_to_document(_asset(center=None), RUN)


class _FakeCollection:
    def __init__(self) -> None:
        self.deleted: dict = {}
        self.inserted: list = []
        self.indexes: list = []

    def delete_many(self, query: dict) -> None:
        self.deleted = query

    def insert_many(self, documents: list, ordered: bool = True) -> None:
        self.inserted.extend(documents)

    def create_index(self, spec, unique: bool = False) -> None:
        self.indexes.append((spec, unique))


class _FakeDB:
    def __init__(self) -> None:
        self.collections: dict = {}

    def __getitem__(self, name: str) -> _FakeCollection:
        return self.collections.setdefault(name, _FakeCollection())

    def command(self, name: str) -> dict:
        return {"ok": 1}


class _FakeClient:
    def __init__(self) -> None:
        self.databases: dict = {}
        self.admin = _FakeDB()

    def __getitem__(self, name: str) -> _FakeDB:
        return self.databases.setdefault(name, _FakeDB())

    def close(self) -> None:
        pass


@pytest.fixture
def fake_pymongo(monkeypatch) -> _FakeClient:
    client = _FakeClient()
    fake = types.SimpleNamespace(MongoClient=lambda uri, **kwargs: client)
    monkeypatch.setattr("infra_inventory.mongo_export._pymongo", lambda: fake)
    return client


def test_export_inserts_documents_and_indexes(fake_pymongo: _FakeClient) -> None:
    inventory = [_asset(), _asset(asset_id="SGN-00001", condition="FAIR", **{"class": "traffic_sign"})]
    result = export_to_mongo(inventory, RUN, "mongodb://localhost:27017")

    assert result["inserted"] == 2
    column = fake_pymongo["ai4infra"]["assets"]
    assert len(column.inserted) == 2
    assert column.inserted[0]["asset_id"] == "POL-00017"
    assert column.inserted[1]["category"] == "Signs"
    assert [("location", "2dsphere")] in [spec for spec, _ in column.indexes]
    assert any(spec == "asset_id" for spec, _ in column.indexes)
    # prior run of the same source file is replaced, not stacked
    assert column.deleted == {"source_file": RUN["input_path"]}


def test_export_replaces_only_the_same_source_run(fake_pymongo: _FakeClient) -> None:
    export_to_mongo([_asset()], RUN, "mongodb://localhost:27017")
    other = dict(RUN, input_path="/data/mannford_run2_right.las")
    export_to_mongo([_asset(asset_id="POL-00018")], other, "mongodb://localhost:27017")
    column = fake_pymongo["ai4infra"]["assets"]
    assert column.deleted == {"source_file": other["input_path"]}
    assert len(column.inserted) == 2


def test_export_output_dir_roundtrip(synthetic_las: Path, output_dir: Path, fake_pymongo: _FakeClient) -> None:
    from infra_inventory.pipeline import process_las
    from infra_inventory.models import ProcessingSettings

    process_las(synthetic_las, output_dir, ProcessingSettings(), progress=False)
    result = export_output_dir_to_mongo(output_dir, "mongodb://localhost:27017")
    assert result["inserted"] > 0
    docs = fake_pymongo["ai4infra"]["assets"].inserted
    assert docs[0]["source_file"] == str(synthetic_las)
    for doc in docs:
        assert doc["category"] in {"Pavement", "Utilities", "Signs", "Safety"}


def test_pipeline_mirrors_inventory_when_mongo_uri_set(
    synthetic_las: Path, output_dir: Path, monkeypatch
) -> None:
    """MONGO_URI env turns the pipeline run into a mirror update, recorded as a warning."""
    calls: list = []

    def fake_export(inventory, run, uri, database="ai4infra", collection="assets"):
        calls.append((len(inventory), uri))
        return {"inserted": len(inventory), "database": database, "collection": collection}

    monkeypatch.setattr("infra_inventory.pipeline.export_to_mongo", fake_export)
    monkeypatch.setenv("MONGO_URI", "mongodb://localhost:27017")
    from infra_inventory.models import ProcessingSettings
    from infra_inventory.pipeline import process_las

    summary = process_las(synthetic_las, output_dir, ProcessingSettings(), progress=False)
    assert calls and calls[0][1] == "mongodb://localhost:27017"
    assert any("MongoDB mirror updated" in warning for warning in summary.warnings)


def test_pipeline_never_fails_when_mirror_is_down(
    synthetic_las: Path, output_dir: Path, monkeypatch
) -> None:
    def failing_export(*args, **kwargs):
        raise RuntimeError("connection refused")

    monkeypatch.setattr("infra_inventory.pipeline.export_to_mongo", failing_export)
    monkeypatch.setenv("MONGO_URI", "mongodb://localhost:27017")
    from infra_inventory.models import ProcessingSettings
    from infra_inventory.pipeline import process_las

    summary = process_las(synthetic_las, output_dir, ProcessingSettings(), progress=False)
    assert summary.assets
    assert any("MongoDB export skipped" in warning for warning in summary.warnings)