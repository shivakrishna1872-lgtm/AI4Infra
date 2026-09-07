"""Gemini class-validation booster: parsing, factor mapping, caching, safety.

The booster is optional and must never break a run: no key, network failure,
or malformed response degrades to the geometry-only confidence and a warning.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from infra_inventory.gemini import (
    BOOST_CLASSES,
    GeminiBooster,
    GeminiError,
    apply_verdict,
    build_prompt,
    parse_verdicts,
    verdict_to_model_factor,
)
from infra_inventory.models import Asset, ConfidenceFactors
from infra_inventory.pipeline import process_las
from infra_inventory.synthetic import build_synthetic_las


def _asset(asset_id: str = "POL-00001", asset_class: str = "utility_pole",
           confidence: float = 0.75, confidence_factors: dict | None = None) -> Asset:
    return Asset(
        asset_id=asset_id,
        asset_class=asset_class,
        subclass="vertical_support",
        center={"x": 0.0, "y": 0.0, "z": 5.0},
        bounding_box=(-0.2, -0.2, 0.0, 0.2, 0.2, 10.0),
        dimensions={"length_m": 0.32, "width_m": 0.32, "height_m": 10.0},
        point_count=400,
        source_tile="tile_0_0",
        source_point_indices_sample=[0, 1],
        coordinate_reference_system="EPSG:2248",
        confidence=confidence,
        confidence_factors=confidence_factors if confidence_factors is not None else {
            "model": None, "geometry": 0.8, "support": 0.9,
            "spatial_context": 0.7, "class_consistency": 0.6},
        confidence_explanation="Confidence from geometry=0.80, support=0.90.",
        detection_method="geometry-v1",
        intensity_stats={"mean": 4200.0},
        rgb_stats={"mean": 12000.0},
        orientation_deg=90.0,
    )


def test_parse_verdicts_fenced_json() -> None:
    text = '```json\n[{"asset_id": "POL-00001", "verdict": "AGREE", "plausibility": 0.9, "reason": "tall thin"},\n {"asset_id": "SGN-00002", "verdict": "UNCERTAIN", "plausibility": 0.4, "reason": "small"},\n {"asset_id": "GRD-00003", "verdict": "DISAGREE", "plausibility": 0.1, "reason": "looks like pavement"}]\n```'
    verdicts = parse_verdicts(text, ["POL-00001", "SGN-00002", "GRD-00003"])
    assert verdicts["POL-00001"]["verdict"] == "AGREE"
    assert verdicts["SGN-00002"]["verdict"] == "UNCERTAIN"
    assert verdicts["GRD-00003"]["verdict"] == "DISAGREE"
    assert verdicts["POL-00001"]["plausibility"] == 0.9


def test_parse_verdicts_rejects_unknown_and_bad_rows() -> None:
    text = '[{"asset_id": "POL-00001", "verdict": "AGREE", "plausibility": 0.9},\n {"asset_id": "HACKED", "verdict": "AGREE", "plausibility": 1.0},\n {"asset_id": "POL-00002", "verdict": "MAYBE", "plausibility": 0.5}]'
    verdicts = parse_verdicts(text, ["POL-00001", "POL-00002"])
    assert set(verdicts) == {"POL-00001"}
    assert "HACKED" not in verdicts


def test_parse_verdicts_salvages_truncated_array() -> None:
    text = ('Here is the review:\n'
            '[{"asset_id": "POL-00001", "verdict": "AGREE", "plausibility": 0.95, '
            '"reason": "x"}, {"asset_id": "POL-00002", "verdict": "DISAGREE"')
    verdicts = parse_verdicts(text, ["POL-00001", "POL-00002"])
    assert verdicts.get("POL-00001", {}).get("verdict") == "AGREE"


def test_verdict_to_model_factor_bands() -> None:
    agree = verdict_to_model_factor({"verdict": "AGREE", "plausibility": 1.0})
    uncertain = verdict_to_model_factor({"verdict": "UNCERTAIN", "plausibility": 0.5})
    disagree = verdict_to_model_factor({"verdict": "DISAGREE", "plausibility": 0.9})
    assert agree > uncertain > disagree
    assert 0.0 < disagree < 0.5
    assert agree <= 1.0


def test_apply_verdict_mutates_asset() -> None:
    asset = _asset()
    before = asset.confidence
    apply_verdict(asset, {"verdict": "AGREE", "plausibility": 1.0, "reason": "clear pole"})
    assert asset.model_confidence is not None and asset.model_confidence > 0.9
    assert asset.confidence_factors["model"] == asset.model_confidence
    assert asset.confidence >= before  # agreement never lowers the score
    assert "gemini" in asset.detection_method
    assert "Gemini AGREE" in asset.confidence_explanation


def test_disagreement_lowers_confidence() -> None:
    asset = _asset(confidence=0.9, confidence_factors={"model": None, "geometry": 0.95,
                                                       "support": 0.95, "spatial_context": 0.9,
                                                       "class_consistency": 0.9})
    apply_verdict(asset, {"verdict": "DISAGREE", "plausibility": 1.0, "reason": "not a pole"})
    assert asset.confidence < 0.9  # disagreement drops the score
    assert asset.confidence < 0.8  # ...below the calibrated human-review cut


def test_booster_requires_key() -> None:
    with pytest.raises(GeminiError):
        GeminiBooster(api_key="", model="test-model")


def test_booster_caches_and_applies(monkeypatch, tmp_path: Path) -> None:
    cache_file = tmp_path / "cache.json"
    calls = {"count": 0}

    def fake_call(api_key: str, model: str, prompt: str, timeout: int = 90, retries: int = 2) -> str:
        calls["count"] += 1
        ids = []
        for line in prompt.splitlines():
            if " | label=" in line:
                ids.append(line.split(":", 1)[1].split()[0])
        return json.dumps([{"asset_id": asset_id, "verdict": "AGREE", "plausibility": 0.9,
                            "reason": "mocked"} for asset_id in ids])

    monkeypatch.setattr("infra_inventory.gemini._call_api", fake_call)
    booster = GeminiBooster(api_key="test-key", model="test-model", batch_size=100, cache_path=str(cache_file))
    assets = [_asset(f"POL-{i:05d}") for i in range(5)]
    summary = booster.boost(assets)
    assert summary["api_calls"] == 1
    assert summary["applied"] == 5
    assert all(a.model_confidence is not None for a in assets)
    assert cache_file.is_file()

    # Second run: all cache hits, no API calls, verdicts still applied.
    calls["count"] = 0
    assets2 = [_asset(f"POL-{i:05d}") for i in range(5)]
    summary2 = booster.boost(assets2)
    assert calls["count"] == 0
    assert summary2["cache_hits"] == 5
    assert summary2["applied"] == 5


def test_booster_failure_leaves_assets_untouched(monkeypatch, tmp_path: Path) -> None:
    def failing_call(*args, **kwargs):
        raise GeminiError("network down")

    monkeypatch.setattr("infra_inventory.gemini._call_api", failing_call)
    booster = GeminiBooster(api_key="test-key", model="test-model")
    assets = [_asset()]
    with pytest.raises(GeminiError):
        booster.boost(assets)
    # No partial application: the first asset must not have been half-boosted.
    assert assets[0].model_confidence is None
    assert assets[0].confidence_factors["model"] is None


def test_pipeline_gemini_backend_wiring(monkeypatch, tmp_path: Path) -> None:
    """--backend gemini runs the booster before QC; failures degrade gracefully."""
    input_las = tmp_path / "input.las"
    build_synthetic_las(input_las, seed=7)
    output = tmp_path / "out"

    class FakeBooster:
        def __init__(self, **kwargs):
            pass

        def boost(self, assets):
            for asset in assets:
                apply_verdict(asset, {"verdict": "AGREE", "plausibility": 0.9, "reason": "mocked"})
            return {"eligible": len(assets), "applied": len(assets), "disagreements": 0,
                    "api_calls": 1, "cache_hits": 0, "cache_loaded": 0, "model": "test"}

    monkeypatch.setattr("infra_inventory.gemini.GeminiBooster", FakeBooster)
    from infra_inventory.models import ProcessingSettings
    settings = ProcessingSettings(backend="gemini", gemini_model="test-model")
    summary = process_las(input_las, output, settings, progress=False)
    assert summary.backend["name"] == "gemini"
    assert summary.backend["gpu_required"] is False
    assert any("Gemini validated" in w for w in summary.warnings)
    assert any(a.model_confidence is not None for a in summary.assets)


def test_pipeline_gemini_failure_degrades(monkeypatch, tmp_path: Path) -> None:
    input_las = tmp_path / "input.las"
    build_synthetic_las(input_las, seed=7)
    output = tmp_path / "out2"

    class FailingBooster:
        def __init__(self, **kwargs):
            raise GeminiError("no key")

    monkeypatch.setattr("infra_inventory.gemini.GeminiBooster", FailingBooster)
    from infra_inventory.models import ProcessingSettings
    settings = ProcessingSettings(backend="gemini", gemini_model="test-model")
    summary = process_las(input_las, output, settings, progress=False)
    assert any("Gemini validation skipped" in w for w in summary.warnings)
    assert len(summary.assets) > 0  # run completed with geometry-only confidence


def test_pipeline_gemini_setting_overrides_env(monkeypatch, tmp_path: Path) -> None:
    """The explicit gemini_api_key setting wins over GEMINI_API_KEY (testing and
    per-run use); the booster still applies verdicts through it."""
    input_las = tmp_path / "input.las"
    build_synthetic_las(input_las, seed=7)
    output = tmp_path / "out3"

    seen_keys = []

    class KeyCapturingBooster:
        def __init__(self, api_key=None, **kwargs):
            seen_keys.append(api_key)

        def boost(self, assets):
            for asset in assets:
                apply_verdict(asset, {"verdict": "AGREE", "plausibility": 0.9, "reason": "mocked"})
            return {"eligible": len(assets), "applied": len(assets), "disagreements": 0,
                    "api_calls": 1, "cache_hits": 0, "cache_loaded": 0, "model": "test"}

    monkeypatch.setattr("infra_inventory.gemini.GeminiBooster", KeyCapturingBooster)
    from infra_inventory.models import ProcessingSettings
    settings = ProcessingSettings(backend="gemini", gemini_model="test-model",
                                  gemini_api_key="explicit-key")
    summary = process_las(input_las, output, settings, progress=False)
    assert seen_keys == ["explicit-key"]
    assert any(a.model_confidence is not None for a in summary.assets)


def test_prompt_contains_measured_evidence_only() -> None:
    asset = _asset()
    prompt = build_prompt([asset])
    assert "POL-00001" in prompt
    assert "0.32x0.32x10.0m" in prompt
    assert "pts=400" in prompt
    assert "intensity_mean=4200" in prompt
    assert asset.asset_id in prompt


def test_boost_classes_exclude_area_pavement() -> None:
    assert "pavement" not in BOOST_CLASSES
    assert "utility_pole" in BOOST_CLASSES
    assert "traffic_sign" in BOOST_CLASSES