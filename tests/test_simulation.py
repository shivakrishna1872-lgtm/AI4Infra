"""Tests for the simulation subsystem.

Quick Simulation is the no-data-required path: it generates a synthetic mobile
LiDAR scene and runs it through the real extraction pipeline, producing real
assets without any uploaded file.

Data Simulation preserves the old behaviour: upload a LAS/LAZ, process it, get
a project. These tests exercise it on the small synthetic scene (fast,
deterministic) rather than on external data.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from infra_inventory.simulation import run_data_simulation, run_quick_simulation
from infra_inventory.synthetic import build_synthetic_las


def test_quick_simulation_builds_real_project(tmp_path: Path) -> None:
    project = run_quick_simulation(tmp_path / "sim", length_m=150.0, seed=7)

    assert project["simulated"] is True
    assert project["source_kind"] == "Quick Simulation"
    assert project["point_count"] > 0
    assert project["asset_count"] > 0
    assert project["processed"] is True
    assert project["las_version"] == "1.4"
    assert project["point_format"] == 7
    assert project["input_file"].endswith(".las")

    scene = project["scene_summary"]
    assert scene["road"]["points"] > 0
    assert scene["utility_poles"]["count"] == 6
    assert scene["overhead_conductors"]["count"] == 5
    assert scene["traffic_signs"]["count"] == 3
    assert scene["guardrails"]["count"] == 2
    assert scene["markings"]["points"] > 0


def test_quick_simulation_schema_fields(tmp_path: Path) -> None:
    project = run_quick_simulation(tmp_path / "sim", length_m=100.0, seed=11)
    required = {
        "id", "name", "input_file", "point_count", "bounds", "crs",
        "las_version", "point_format", "asset_count", "simulated",
        "source_kind", "processed", "output_dir", "scene_summary", "simulation_meta",
    }
    assert not (required - set(project.keys()))
    assert project["name"].startswith("Quick Simulation")
    assert isinstance(project["point_count"], int) and project["point_count"] > 0
    assert isinstance(project["asset_count"], int) and project["asset_count"] > 0
    assert project["simulation_meta"]["note"].startswith("SIMULATED DATA")
    assert Path(project["simulation_meta"]["generated_file"]).is_file()
    # outputs of the real pipeline exist
    out = Path(project["output_dir"])
    assert (out / "pipeline" / "viewer-data.json").is_file()
    assert (out / "pipeline" / "assets.json").is_file()


def test_quick_simulation_scene_parts_match_las(tmp_path: Path) -> None:
    project = run_quick_simulation(tmp_path / "sim", length_m=120.0, seed=17)
    scene = project["scene_summary"]
    part_sum = sum(part["points"] for part in scene.values())
    assert abs(project["point_count"] - part_sum) < 400


def test_quick_simulation_regenerate_creates_new_run(tmp_path: Path) -> None:
    first = run_quick_simulation(tmp_path / "sim", length_m=120.0, seed=3)
    second = run_quick_simulation(tmp_path / "sim", length_m=120.0, seed=4)
    assert first["id"] != second["id"]
    assert first["simulation_meta"]["generated_file"] != second["simulation_meta"]["generated_file"]


def test_data_simulation_runs_uploaded_scene(tmp_path: Path) -> None:
    input_las = tmp_path / "input.las"
    build_synthetic_las(input_las, seed=5)
    project = run_data_simulation(input_las, tmp_path / "data")

    assert project["simulated"] is True
    assert project["source_kind"] == "Data Simulation"
    assert project["point_count"] > 0
    assert project["asset_count"] > 0
    assert project["processed"] is True
    assert project["input_file_name"] == "input.las"
    assert project["simulation_meta"]["source_file"].endswith("input.las")


def test_evaluate_against_ground_truth_exact_math() -> None:
    """Precision / recall / F1 / positional RMSE on a hand-built example."""
    from infra_inventory.evaluate import evaluate_against_ground_truth

    assets = [
        {"asset_id": "POL-00001", "class": "utility_pole", "center": {"x": 0, "y": 0, "z": 5}},
        {"asset_id": "POL-00002", "class": "utility_pole", "center": {"x": 10, "y": 0, "z": 5}},
        {"asset_id": "POL-00003", "class": "utility_pole", "center": {"x": 50, "y": 0, "z": 5}},  # false positive
    ]
    ground_truth = [
        {"gt_id": "GT-1", "class": "utility_pole", "center": [0.5, 0, 5]},
        {"gt_id": "GT-2", "class": "utility_pole", "center": [9.5, 0, 5]},
    ]
    result = evaluate_against_ground_truth(assets, ground_truth, match_distance_m=3.0)
    assert result["matched_count"] == 2
    assert result["ground_truth_count"] == 2
    assert result["detected_count"] == 3
    assert result["precision"] == round(2 / 3, 4)
    assert result["recall"] == 1.0
    assert result["f1"] == round(2 * (2 / 3) / (2 / 3 + 1), 4)
    assert result["positional_error_rmse_m"] == 0.5
    assert result["false_positive_asset_ids"] == ["POL-00003"]


def test_quick_simulation_scores_against_ground_truth(tmp_path: Path) -> None:
    """Detections on the synthetic scene score real metrics vs its known objects."""
    from infra_inventory.evaluate import evaluate_against_ground_truth

    project = run_quick_simulation(tmp_path / "eval", length_m=400.0, seed=23)
    assets = json.loads(
        (Path(project["output_dir"]) / "pipeline" / "assets.json").read_text(encoding="utf-8")
    )
    ground_truth = project["simulation_meta"]["ground_truth"]
    assert ground_truth, "simulation must export a ground-truth table"

    # Long/area assets (guardrail, barrier, pavement, markings) are extracted per
    # tile component, so they are matched with class-appropriate radii. Conductors
    # are matched at span level (their centroid is mid-span); cabinets at a few
    # meters; rumble strips along their band group.
    result = evaluate_against_ground_truth(
        assets,
        ground_truth,
        match_distance_m={
            "utility_pole": 3.0,
            "traffic_sign": 3.0,
            "utility_cabinet": 3.0,
            "overhead_conductor": 12.0,
            "guardrail": 20.0,
            "safety_barrier": 20.0,
            "rumble_strip": 10.0,
            "pavement": 25.0,
            "pavement_marking": 20.0,
        },
    )
    assert result["matched_count"] >= 15
    assert result["recall"] >= 0.6
    assert result["positional_error_rmse_m"] is not None
    assert result["per_class"]["utility_pole"]["recall"] >= 0.5
    assert result["per_class"]["traffic_sign"]["precision"] >= 0.9
    assert result["per_class"]["overhead_conductor"]["recall"] >= 0.4
    assert result["per_class"]["utility_cabinet"]["recall"] >= 0.5
    assert result["per_class"]["rumble_strip"]["recall"] >= 0.5


def test_quick_simulation_output_assets_are_real(tmp_path: Path) -> None:
    """The generated viewer assets come from the real pipeline, not hardcoding."""
    project = run_quick_simulation(tmp_path / "sim", length_m=100.0, seed=19)
    assets = json.loads(
        (Path(project["output_dir"]) / "pipeline" / "assets.json").read_text(encoding="utf-8")
    )
    classes = {a["class"] for a in assets}
    assert classes & {"utility_pole", "guardrail", "traffic_sign", "pavement_marking"}
    for asset in assets:
        assert asset["asset_id"]
        assert 0.0 <= asset["confidence"] <= 1.0
        assert asset["source_tile"]
        assert asset["detection_method"]
