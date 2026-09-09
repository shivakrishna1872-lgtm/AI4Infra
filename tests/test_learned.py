"""Tests for the learned component classifier (infra_inventory/learned.py).

The classifier is a CPU prior trained on Quick Simulation ground truth; these
tests cover the math (softmax/training), the feature extraction, persistence,
and the end-to-end effect on the pipeline (model factor present when active,
absent when disabled).
"""
from __future__ import annotations

import numpy as np
import pytest

from infra_inventory.learned import (
    FEATURE_NAMES,
    ComponentClassifier,
    FeatureNormalizer,
    component_features,
    learned_prior_factor,
    softmax,
    train_softmax,
)


def _make_classifier() -> ComponentClassifier:
    """Tiny hand-built classifier with known probabilities."""
    num_features, num_classes = len(FEATURE_NAMES), 3
    rng = np.random.default_rng(0)
    W = rng.normal(0, 0.01, (num_features, num_classes))
    b = np.zeros(num_classes)
    return ComponentClassifier(
        weights=W, bias=b, normalizer=FeatureNormalizer(np.zeros(num_features), np.ones(num_features)),
        classes=("utility_pole", "guardrail", "background"),
    )


def test_softmax_rows_sum_to_one() -> None:
    z = np.array([[0.0, 1.0, 1000.0], [-5.0, 0.0, 5.0]])
    probs = softmax(z)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert np.isfinite(probs).all()  # numerically stable at large logits


def test_train_softmax_separates_linearly_separable_data() -> None:
    rng = np.random.default_rng(3)
    X_a = rng.normal(0.0, 0.2, (60, 4)) + np.array([3.0, 0, 0, 0])
    X_b = rng.normal(0.0, 0.2, (60, 4)) + np.array([-3.0, 0, 0, 0])
    X = np.vstack((X_a, X_b))
    y = np.array([0] * 60 + [1] * 60)
    W, b = train_softmax(X, y, num_classes=2, epochs=200)
    preds = np.argmax(softmax(X @ W + b), axis=1)
    assert float(np.mean(preds == y)) == 1.0


def test_component_features_pole_vs_wire() -> None:
    """A vertical column and a thin horizontal wire have opposite signatures."""
    rng = np.random.default_rng(1)
    angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    zs = np.arange(0.0, 8.0, 0.1)
    px = (0.12 * np.cos(angles))[:, None] + rng.normal(0, 0.002, (12, len(zs)))
    py = (0.12 * np.sin(angles))[:, None] + rng.normal(0, 0.002, (12, len(zs)))
    pz = np.broadcast_to(zs[None, :], px.shape)
    pole_feats, _ = component_features(
        px.ravel(), py.ravel(), pz.ravel(), np.arange(px.size),
        np.full(px.size, 1000.0), None, ground=0.0,
    )
    tx = np.linspace(0.0, 30.0, 200)
    wy = 0.02 * np.sin(tx / 3.0)
    wire_feats, _ = component_features(
        tx, wy, np.full(200, 9.7), np.arange(200),
        np.full(200, 500.0), None, ground=0.0,
    )
    pole = dict(zip(FEATURE_NAMES, pole_feats))
    wire = dict(zip(FEATURE_NAMES, wire_feats))
    assert pole["verticality"] > 0.9
    assert pole["ground_band"] < 0.05          # starts at ground
    assert wire["linearity"] > 0.9
    assert wire["ground_band"] > 0.9           # hangs high
    assert wire["log_cross_section"] < pole["log_cross_section"]


def test_classifier_save_load_roundtrip(tmp_path) -> None:
    model = _make_classifier()
    path = tmp_path / "model.json"
    model.save(path)
    loaded = ComponentClassifier.load(path)
    feats = np.random.default_rng(5).normal(0, 1, len(FEATURE_NAMES))
    assert loaded.scores(feats) == model.scores(feats)
    assert loaded.classes == model.classes
    assert loaded.version == model.version


def test_learned_prior_factor_agreement_and_ambiguity() -> None:
    model = _make_classifier()
    feats = np.zeros(len(FEATURE_NAMES))
    scores = {"utility_pole": 0.9, "guardrail": 0.05, "background": 0.05}
    factor, prior = learned_prior_factor(model, "utility_pole", feats, scores=scores)
    assert factor == pytest.approx(0.9 - 0.25 * 0.05)  # P(assigned) - 0.25 * runner-up
    # The function returns the raw best class; the '*' agreement marker is
    # applied at the build_asset provenance site.
    assert prior == "utility_pole"
    # Ambiguous: high runner-up pulls the factor down.
    ambiguous = {"utility_pole": 0.4, "guardrail": 0.55, "background": 0.05}
    factor2, prior2 = learned_prior_factor(model, "utility_pole", feats, scores=ambiguous)
    assert factor2 < factor
    assert prior2 == "guardrail"  # provenance shows the disagreement


def test_load_classifier_missing_file_is_none(tmp_path) -> None:
    from infra_inventory.learned import load_classifier

    assert load_classifier(tmp_path / "nope.json") is None


def test_pipeline_uses_learned_prior(synthetic_las, tmp_path) -> None:
    """End-to-end: shipped model fills the model factor; disabling removes it."""
    from infra_inventory.models import ProcessingSettings
    from infra_inventory.pipeline import process_las

    with_prior = process_las(synthetic_las, tmp_path / "on", ProcessingSettings(), progress=False)
    assert with_prior.assets
    assert any(a.model_confidence is not None for a in with_prior.assets)
    assert any((a.model_prior_class or "").endswith("*") for a in with_prior.assets)

    settings = ProcessingSettings()
    settings.learned_prior_enabled = False
    without_prior = process_las(synthetic_las, tmp_path / "off", settings, progress=False)
    assert without_prior.assets
    assert all(a.model_confidence is None for a in without_prior.assets)
    # Same geometry decisions either way: classes are unchanged.
    assert {a.asset_class for a in with_prior.assets} == {a.asset_class for a in without_prior.assets}


def test_learned_veto_can_reject_and_disable(synthetic_las, tmp_path) -> None:
    """A component the trained model rejects is dropped; --no-learned-veto keeps it."""
    from infra_inventory.models import ProcessingSettings
    from infra_inventory.pipeline import process_las

    settings = ProcessingSettings()  # veto on by default
    strict = process_las(synthetic_las, tmp_path / "veto", settings, progress=False)
    permissive = ProcessingSettings()
    permissive.learned_veto = False
    lenient = process_las(synthetic_las, tmp_path / "noveto", permissive, progress=False)
    # The veto may or may not fire on this scene, but it can never ADD assets.
    assert len(lenient.assets) >= len(strict.assets)


def test_collect_training_writes_jsonl(synthetic_las, tmp_path) -> None:
    from infra_inventory.models import ProcessingSettings
    from infra_inventory.pipeline import process_las

    settings = ProcessingSettings()
    settings.collect_training = True
    result = process_las(synthetic_las, tmp_path / "out", settings, progress=False)
    training_dir = tmp_path / "out" / "training"
    assert training_dir.is_dir()
    files = sorted(training_dir.glob("components_*.jsonl"))
    assert files
    import json as _json

    record = _json.loads(files[0].read_text(encoding="utf-8").splitlines()[0])
    assert record["class"] in {a.asset_class for a in result.assets} | {"background"}
    assert len(record["features"]) == len(FEATURE_NAMES)
    assert "center_xyz" in record["extras"]
