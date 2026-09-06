"""Shared fixtures: a synthetic LAS scene and scratch output directory.

The synthetic scene contains a road, painted lane line, utility pole, traffic
sign (panel + support), and guardrail - enough to exercise every geometry
detector deterministically. Synthetic data is for testing only.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from infra_inventory.synthetic import build_synthetic_las


@pytest.fixture(scope="session")
def synthetic_las(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("data") / "synthetic.las"
    build_synthetic_las(path)
    return path


@pytest.fixture
def simulated_project(tmp_path: Path) -> dict:
    """A processed simulation project (Quick Simulation) for UI tests."""
    from infra_inventory.simulation import run_quick_simulation

    project = run_quick_simulation(tmp_path, length_m=200.0, seed=11)
    return project


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    return tmp_path / "out"


@pytest.fixture
def synthetic_summary(synthetic_las: Path) -> dict:
    from infra_inventory.las_reader import read_metadata

    return read_metadata(synthetic_las).to_dict()


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """FastAPI TestClient isolated from any real appdata directory."""
    from fastapi.testclient import TestClient

    from infra_inventory import server

    monkeypatch.setattr(server, "DATA_DIR", tmp_path / "appdata")
    with TestClient(server.app) as test_client:
        yield test_client


@pytest.fixture
def clean_appdata():
    """Marker: server tests manage their own projects via the API."""
    yield