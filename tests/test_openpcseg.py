"""Tests for the OpenPCSeg learned-backend bridge and tensor export."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from infra_inventory.errors import BackendConfigurationError
from infra_inventory.openpcseg import (
    describe_openpcseg,
    load_indexed_predictions,
    run_openpcseg_infer,
    run_openpcseg_train,
)
from infra_inventory.tensor_export import export_tiles_to_tensors, load_tensor_points


# --- describe_openpcseg -------------------------------------------------------


def test_describe_without_root_reports_geometry() -> None:
    detail = describe_openpcseg(None)
    assert detail["name"] == "geometry"
    assert detail["available"] is True
    assert detail["gpu_required"] is False


def test_describe_with_missing_root_reports_unavailable(tmp_path: Path) -> None:
    detail = describe_openpcseg(str(tmp_path / "OpenPCSeg"))
    assert detail["name"] == "openpcseg"
    assert detail["available"] is False
    assert "not found" in str(detail["reason"])


# --- tensor export ------------------------------------------------------------


def _make_tile(tiles_dir: Path, name: str, header_offset: tuple[float, float, float], seed: int) -> None:
    import laspy

    header = laspy.LasHeader(point_format=7, version="1.4")
    header.scales = [0.001, 0.001, 0.001]
    header.offsets = list(header_offset)
    las = laspy.LasData(header)
    rng = np.random.default_rng(seed)
    count = 500
    las.x = header_offset[0] + rng.uniform(0, 40, count)
    las.y = header_offset[1] + rng.uniform(0, 40, count)
    las.z = header_offset[2] + rng.uniform(0, 10, count)
    las.intensity = rng.integers(0, 255, count).astype(np.uint16)
    tiles_dir.mkdir(parents=True, exist_ok=True)
    las.write(tiles_dir / f"{name}.las")


def test_export_tiles_roundtrip_preserves_coordinates(tmp_path: Path) -> None:
    tiles_dir = tmp_path / "tiles"
    _make_tile(tiles_dir, "tile_0_0", (627285.0, 4841948.0, 200.0), seed=1)
    manifest = export_tiles_to_tensors(tiles_dir, tmp_path / "tensors")

    out_dir = tmp_path / "tensors"
    assert (out_dir / "tile_0_0.bin").is_file()
    assert (out_dir / "tensor_manifest.json").is_file()
    assert manifest["tensor_dims"] == 4
    assert manifest["point_counts"]["tile_0_0"] == 500
    assert manifest["input_offset"] == [627285.0, 4841948.0, 200.0]

    rows = load_tensor_points(out_dir / "tile_0_0.bin", manifest["input_offset"])
    assert rows.shape == (500, 4)
    assert rows[:, 0].min() >= 627285.0  # offset re-applied
    assert rows[:, 2].max() <= 210.0
    assert float(rows[:, 3].max()) <= 255.0  # intensity preserved


def test_export_selected_tiles_only(tmp_path: Path) -> None:
    tiles_dir = tmp_path / "tiles"
    _make_tile(tiles_dir, "tile_0_0", (0.0, 0.0, 0.0), seed=2)
    _make_tile(tiles_dir, "tile_1_0", (0.0, 0.0, 0.0), seed=3)
    manifest = export_tiles_to_tensors(tiles_dir, tmp_path / "tensors", tile_names=["tile_0_0"])
    assert set(manifest["point_counts"]) == {"tile_0_0"}


# --- indexed prediction loading ------------------------------------------------


def test_load_indexed_predictions_zips_with_tile_order(tmp_path: Path) -> None:
    pred_dir = tmp_path / "predictions"
    pred_dir.mkdir()
    # Upstream naming: zero-padded index in dataset order.
    np.save(pred_dir / "0000000000.npy", np.array([1, 2, 3], dtype=np.int64))
    np.save(pred_dir / "0000000001.npy", np.array([6, 6], dtype=np.int64))
    # Unrelated file must be ignored by the zip-with-order contract.
    (pred_dir / "notes.txt").write_text("x")
    result = load_indexed_predictions(pred_dir, ["tile_0_0", "tile_1_0", "tile_0_1"])
    assert list(result) == ["tile_0_0", "tile_1_0"]
    assert result["tile_0_0"].tolist() == [1, 2, 3]
    assert result["tile_1_0"].tolist() == [6, 6]
    assert "tile_0_1" not in result


def test_load_indexed_predictions_argmaxes_logits(tmp_path: Path) -> None:
    pred_dir = tmp_path / "predictions"
    pred_dir.mkdir()
    np.save(pred_dir / "0000000000.npy", np.array([[0.1, 0.9], [0.8, 0.2]]))
    result = load_indexed_predictions(pred_dir, ["tile_0_0"])
    assert result["tile_0_0"].tolist() == [1, 0]


def test_load_indexed_predictions_missing_dir(tmp_path: Path) -> None:
    assert load_indexed_predictions(tmp_path / "nope", ["tile_0_0"]) == {}


# --- inference/train guards (no torch / no checkout installed here) ------------


def test_infer_requires_existing_config(tmp_path: Path) -> None:
    (tmp_path / "infer.py").write_text("# stub\n")
    (tmp_path / "train.py").write_text("# stub\n")
    with pytest.raises(BackendConfigurationError):
        run_openpcseg_infer(
            root=str(tmp_path), cfg_file=str(tmp_path / "missing.yaml"),
            checkpoint=str(tmp_path / "model.pth"), num_gpus=1,
            prediction_dir=tmp_path / "preds",
        )


def test_infer_requires_existing_checkpoint(tmp_path: Path) -> None:
    (tmp_path / "infer.py").write_text("# stub\n")
    (tmp_path / "train.py").write_text("# stub\n")
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("DATA:\n  DATASET: semantickitti\n")
    with pytest.raises(Exception):
        run_openpcseg_infer(
            root=str(tmp_path), cfg_file=str(cfg),
            checkpoint=str(tmp_path / "missing.pth"), num_gpus=1,
            prediction_dir=tmp_path / "preds",
        )


def test_train_requires_existing_config(tmp_path: Path) -> None:
    (tmp_path / "infer.py").write_text("# stub\n")
    (tmp_path / "train.py").write_text("# stub\n")
    with pytest.raises(BackendConfigurationError):
        run_openpcseg_train(
            root=str(tmp_path), cfg_file=str(tmp_path / "missing.yaml"),
            init_checkpoint=None, num_gpus=1,
        )


def test_train_missing_init_checkpoint_rejected(tmp_path: Path) -> None:
    (tmp_path / "infer.py").write_text("# stub\n")
    (tmp_path / "train.py").write_text("# stub\n")
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("DATA:\n  DATASET: semantickitti\n")
    with pytest.raises(Exception):
        run_openpcseg_train(
            root=str(tmp_path), cfg_file=str(cfg),
            init_checkpoint=str(tmp_path / "missing.pth"), num_gpus=1,
        )


# --- taxonomy wiring ------------------------------------------------------------


def test_taxonomies_cover_every_mapping_entry() -> None:
    from infra_inventory.pipeline import (
        DEFAULT_OPENPCSEG_CLASS_MAPPING,
        OPENPCSEG_TAXONOMIES,
    )

    for taxonomy, names in OPENPCSEG_TAXONOMIES.items():
        assert names[0] == "unclassified" or names[0] == "car_c"  # id 0 = ignore/context
        # Every mapped upstream class must exist in its taxonomy (shared table
        # is the superset; unknown names simply contribute no evidence).
        known = {"unclassified", "car_c", "bicycle", "motorcycle", "truck",
                 "other-vehicle", "person", "bicyclist", "motorcyclist",
                 "road", "parking", "sidewalk", "other-ground", "building",
                 "fence", "vegetation", "trunk", "terrain", "pole",
                 "traffic-sign", "road_marking", "utility_line", "natural", "car"}
        for name in DEFAULT_OPENPCSEG_CLASS_MAPPING:
            assert name in known, name


def test_mapping_targets_real_asset_classes() -> None:
    from infra_inventory.pipeline import DEFAULT_OPENPCSEG_CLASS_MAPPING

    real_classes = {
        "pavement", "pavement_marking", "utility_pole", "overhead_conductor",
        "utility_cabinet", "traffic_sign", "guardrail", "safety_barrier",
        "rumble_strip",
    }
    for name, entry in DEFAULT_OPENPCSEG_CLASS_MAPPING.items():
        target = entry.get("asset_class")
        if target is not None:
            assert target in real_classes, f"{name} -> {target}"


def test_settings_yaml_roundtrip(tmp_path: Path) -> None:
    from infra_inventory.models import ProcessingSettings

    values = {
        "backend": "openpcseg",
        "openpcseg_root": "/opt/OpenPCSeg",
        "openpcseg_config": "/opt/OpenPCSeg/tools/cfgs/voxel/semantic_kitti/minkunet_mk34_cr10.yaml",
        "openpcseg_weight": "/opt/OpenPCSeg/semkitti_minkunet.pth",
        "openpcseg_taxonomy": "semantickitti",
        "openpcseg_num_gpus": 2,
    }
    settings = ProcessingSettings.from_dict(values)
    assert settings.backend == "openpcseg"
    assert settings.openpcseg_taxonomy == "semantickitti"
    assert settings.openpcseg_num_gpus == 2


def test_processing_yaml_documents_openpcseg() -> None:
    import yaml

    data = yaml.safe_load(Path("configs/processing.yaml").read_text(encoding="utf-8"))
    assert data["openpcseg_taxonomy"] == "toronto3d"
    model = yaml.safe_load(Path("configs/model.yaml").read_text(encoding="utf-8"))
    assert set(model["openpcseg"]["class_mapping"]) >= {"road", "pole", "utility_line"}
    assert model["openpcseg"]["toronto3d"]["class_names"][6] == "pole"
    assert json.loads(json.dumps(model["openpcseg"]["semantickitti"]["model_zoo"]))
