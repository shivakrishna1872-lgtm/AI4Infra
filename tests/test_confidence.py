from __future__ import annotations

import numpy as np

from infra_inventory.confidence import (
    ConfidenceFactors,
    class_consistency,
    model_factor_from_prior,
    score_asset,
    spatial_context,
    support_score,
)


def test_support_score_monotonic() -> None:
    previous = -1.0
    for n in (1, 10, 25, 100, 1000, 10000):
        score = support_score(n, min_points=25)
        assert 0.0 <= score <= 1.0
        assert score >= previous
        previous = score


def test_score_asset_renormalizes_weights() -> None:
    factors = ConfidenceFactors(model=None, geometry=0.9, support=0.8, spatial_context=0.7, class_consistency=None)
    confidence, explanation = score_asset("pavement_marking", factors)
    # weights: model 0.25 (absent), geometry .30, support .20, spatial .10, class .15 (absent)
    expected = (0.30 * 0.9 + 0.20 * 0.8 + 0.10 * 0.7) / (0.30 + 0.20 + 0.10)
    assert abs(confidence - expected) < 0.01
    assert "geometry=0.90" in explanation


def test_score_asset_no_factors() -> None:
    confidence, explanation = score_asset("pavement", ConfidenceFactors())
    assert confidence == 0.0
    assert explanation


def test_score_asset_clamps() -> None:
    factors = ConfidenceFactors(model=1.0, geometry=1.0, support=1.0, spatial_context=1.0, class_consistency=1.0)
    confidence, _ = score_asset("utility_pole", factors)
    assert confidence == 1.0


def test_class_consistency_retroreflective() -> None:
    intensity = np.array([20000.0, 22000.0, 21000.0])
    score = class_consistency("pavement_marking", intensity, None)
    assert score == 1.0
    dim = np.array([50.0, 60.0, 55.0])
    assert class_consistency("pavement_marking", dim, None) < 0.7


def test_class_consistency_unknown_class_is_none() -> None:
    assert class_consistency("pavement", None, None) is None


def test_spatial_context_roadside() -> None:
    assert spatial_context("guardrail", 0.0) == 1.0
    assert spatial_context("guardrail", 12.0) == 0.0
    assert spatial_context("guardrail", None) is None


def test_model_factor_from_prior() -> None:
    mapping = {"driveable_surface": {"asset_class": "pavement", "weight": 0.9}}
    prior = np.array([11, 11, 0, 2])  # 11 -> driveable_surface (nuScenes order)
    class_names = [f"c{i}" for i in range(17)]
    class_names[11] = "driveable_surface"
    score = model_factor_from_prior(prior, class_names, mapping, ("pavement",))
    assert score == 0.45  # half the points weighted 0.9
    assert model_factor_from_prior(prior, class_names, {}, ("pavement",)) is None
    assert model_factor_from_prior(None, class_names, mapping, ("pavement",)) is None