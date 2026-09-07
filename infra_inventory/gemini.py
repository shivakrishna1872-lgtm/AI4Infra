"""Gemini classification/confidence booster (opt-in ``--backend gemini``).

The geometry detectors always decide *whether* an asset exists. The Gemini
booster adds an independent, learned check: it reads each detected instance's
measured evidence (dimensions, height, point support, intensity/RGB
statistics, orientation) and returns a plausibility verdict for the assigned
class. That verdict becomes the ``model`` factor of the transparent confidence
blend (docs/GEMINI.md) — the same slot a Pointcept prior would fill.

Integrity rules:

* Gemini only ever *reasons over measured facts*; it cannot invent new
  geometry. A ``DISAGREE`` verdict lowers confidence (which routes the asset
  to human review via the calibrated 0.80 threshold) — it never relabels or
  deletes an asset on its own.
* Any API failure is graceful: assets keep their geometry-only confidence and
  the run records a warning. The pipeline must never depend on the network.
* Verdicts are cached by a stable evidence fingerprint (class + geometry +
  radiometry), so re-runs of the same scene do not re-bill.
* Only stdlib HTTP is used — no SDK dependency, no new install step.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from .confidence import score_asset
from .models import Asset, ConfidenceFactors

#: Classes the booster evaluates. Area pavement surfaces get no learned check
#: (their instance identity is per-tile by design); everything else benefits.
BOOST_CLASSES = {
    "pavement_marking", "utility_pole", "overhead_conductor", "utility_cabinet",
    "traffic_sign", "guardrail", "safety_barrier", "rumble_strip",
}

_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


class GeminiError(RuntimeError):
    """Raised when the Gemini API cannot be reached or returns garbage."""


def _fingerprint(asset: Asset) -> str:
    """Stable evidence fingerprint for cache lookups (never the raw points)."""
    dims = asset.dimensions or {}
    intensity = (asset.intensity_stats or {}).get("mean")
    rgb = (asset.rgb_stats or {}).get("mean")
    payload = (
        asset.asset_class, asset.subclass,
        round(float(dims.get("length_m", 0.0)), 2),
        round(float(dims.get("width_m", 0.0)), 2),
        round(float(dims.get("height_m", 0.0)), 2),
        asset.point_count,
        round(float(intensity), 1) if intensity is not None else None,
        round(float(rgb), 1) if rgb is not None else None,
    )
    return json.dumps(payload, sort_keys=True)


def _asset_evidence(asset: Asset) -> str:
    """One compact line of *measured* evidence for the prompt."""
    dims = asset.dimensions or {}
    intensity = asset.intensity_stats or {}
    rgb = asset.rgb_stats or {}
    parts = [
        f"{asset.asset_id}",
        f"label={asset.asset_class}/{asset.subclass or 'unspecified'}",
        f"LxWxH={dims.get('length_m', '?')}x{dims.get('width_m', '?')}x{dims.get('height_m', '?')}m",
        f"pts={asset.point_count}",
    ]
    if intensity.get("mean") is not None:
        parts.append(f"intensity_mean={intensity['mean']:.0f}")
    if rgb.get("mean") is not None:
        parts.append(f"rgb_mean={rgb['mean']:.0f}")
    if asset.orientation_deg is not None:
        parts.append(f"orientation={asset.orientation_deg:.0f}deg")
    return " | ".join(parts)


def build_prompt(assets: List[Asset]) -> str:
    """Build the evidence prompt for one batch of detected assets."""
    lines = [f"Asset evidence: {_asset_evidence(a)}" for a in assets]
    return (
        "You are a LiDAR infrastructure asset QC reviewer. For each detected asset below, "
        "judge whether the assigned label is physically plausible GIVEN ONLY the measured "
        "evidence listed. A 0.3 m-wide, 8 m-tall vertical structure with hundreds of points "
        "is a pole; a thin elevated linear chain is an overhead conductor; a planar panel on "
        "a post is a traffic sign; a long low rail beside the road is a guardrail.\n\n"
        + "\n".join(lines)
        + "\n\nRespond with ONLY a JSON array, one object per asset, in the same order: "
          '[{"asset_id": "...", "verdict": "AGREE|UNCERTAIN|DISAGREE", "plausibility": 0.0-1.0, '
          '"reason": "one short sentence using the measured evidence"}]. '
          "No markdown, no commentary. If evidence is missing or ambiguous use UNCERTAIN."
    )


def _call_api(api_key: str, model: str, prompt: str, timeout: int = 90, retries: int = 4) -> str:
    """POST to the Gemini generateContent endpoint (stdlib urllib only).

    Transient 5xx errors (load-shedding is common on shared API tiers) are
    retried with backoff; anything else fails fast with a GeminiError.
    """
    url = f"{_API_BASE}/{urllib.parse.quote(model)}:generateContent?key={api_key}"
    body = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 8192},
    }).encode("utf-8")
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    last_error: Optional[Exception] = None
    payload: Optional[Dict[str, Any]] = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as exc:
            last_error = exc
            retryable = exc.code >= 500 or exc.code == 429  # load-shedding / quota
            if retryable and attempt < retries:
                # 429s carry a retryDelay in the body ("quotaResetDelay"-style);
                # honor it when present, otherwise linear backoff.
                delay = 5.0 * (attempt + 1)
                try:
                    detail = json.loads(exc.read().decode("utf-8"))
                    delay_str = detail.get("error", {}).get("details", [{}])[0].get("retryDelay")
                    if delay_str and delay_str.endswith("s"):
                        delay = min(float(delay_str[:-1]), 30.0)
                except Exception:
                    pass
                time.sleep(delay)
                continue
            raise GeminiError(f"Gemini API request failed: {exc}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(3 * (attempt + 1))
                continue
            raise GeminiError(f"Gemini API request failed: {exc}") from exc
    try:
        candidates = payload.get("candidates") or []
        text = candidates[0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError) as exc:
        raise GeminiError(f"Unexpected Gemini response shape: {payload}") from exc
    return text


def parse_verdicts(text: str, expected_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """Parse the model's JSON response into ``{asset_id: verdict}``.

    Tolerates markdown fences, leading prose, and per-object extras. Verdicts
    for unknown ids are dropped; missing ids simply get no verdict (the caller
    leaves their confidence untouched — never a guessed factor).
    """
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    array_match = re.search(r"\[.*\]", cleaned, re.DOTALL)
    rows: List[Dict[str, Any]] = []
    if array_match:
        try:
            rows = json.loads(array_match.group(0))
        except json.JSONDecodeError:
            rows = []
    if not rows:
        # Last resort (or the array was truncated mid-stream): salvage the
        # individual object literals that did come back complete.
        for obj in re.finditer(r"\{[^{}]*\}", cleaned):
            try:
                rows.append(json.loads(obj.group(0)))
            except json.JSONDecodeError:
                continue
    expected = set(expected_ids)
    verdicts: Dict[str, Dict[str, Any]] = {}
    for row in rows if isinstance(rows, list) else []:
        asset_id = str(row.get("asset_id", "")).strip()
        if asset_id not in expected:
            continue
        verdict = str(row.get("verdict", "")).upper()
        if verdict not in ("AGREE", "UNCERTAIN", "DISAGREE"):
            continue
        plausibility = row.get("plausibility")
        try:
            plausibility = max(0.0, min(1.0, float(plausibility)))
        except (TypeError, ValueError):
            plausibility = None
        verdicts[asset_id] = {
            "verdict": verdict,
            "plausibility": plausibility,
            "reason": str(row.get("reason", ""))[:200],
        }
    return verdicts


def verdict_to_model_factor(verdict: Dict[str, Any]) -> float:
    """Map a Gemini verdict to the ``model`` confidence factor (0..1).

    AGREE keeps a strong floor and scales with plausibility; UNCERTAIN lands
    mid-band (comparable to weak Pointcept evidence); DISAGREE collapses the
    factor so the weighted blend drops the asset into human review.
    """
    plausibility = verdict.get("plausibility")
    p = plausibility if plausibility is not None else 0.5
    if verdict["verdict"] == "AGREE":
        return round(min(1.0, 0.70 + 0.30 * p), 3)
    if verdict["verdict"] == "UNCERTAIN":
        return round(min(0.70, 0.45 + 0.25 * p), 3)
    return round(max(0.05, 0.25 * (1.0 - p)), 3)


def apply_verdict(asset: Asset, verdict: Dict[str, Any]) -> None:
    """Fold one verdict into the asset: model factor + re-scored confidence.

    Mutates ``asset`` in place: ``model_confidence``, ``model_prior_class``,
    ``confidence_factors.model``, ``confidence``, ``confidence_explanation``
    and ``detection_method`` (records the boost).
    """
    factor = verdict_to_model_factor(verdict)
    factors = ConfidenceFactors(
        model=factor,
        geometry=asset.confidence_factors.get("geometry"),
        support=asset.confidence_factors.get("support"),
        spatial_context=asset.confidence_factors.get("spatial_context"),
        class_consistency=asset.confidence_factors.get("class_consistency"),
    )
    confidence, explanation = score_asset(asset.asset_class, factors)
    asset.confidence_factors = factors.to_dict()
    asset.confidence = confidence
    asset.model_confidence = factor
    asset.model_prior_class = asset.asset_class
    if "gemini" not in asset.detection_method:
        asset.detection_method = f"{asset.detection_method}-gemini"
    reason = verdict.get("reason") or ""
    suffix = f" Gemini {verdict['verdict']}" + (f" ({reason})" if reason else "")
    asset.confidence_explanation = explanation + suffix


class GeminiBooster:
    """Batched, cached, failure-safe Gemini booster."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-3.6-flash",
        timeout: int = 90,
        batch_size: int = 25,
        cache_path: Optional[str] = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not self.api_key:
            raise GeminiError(
                "Gemini backend requires GEMINI_API_KEY (set it in your environment "
                "or Keys/API keys tab, then re-run)."
            )
        self.model = model
        self.timeout = timeout
        self.batch_size = batch_size
        self.cache_path = cache_path
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.loaded = 0
        if cache_path and os.path.isfile(cache_path):
            try:
                with open(cache_path, encoding="utf-8") as handle:
                    self.cache = json.load(handle)
                self.loaded = len(self.cache)
            except (OSError, json.JSONDecodeError):
                self.cache = {}

    def _save_cache(self) -> None:
        if not self.cache_path:
            return
        try:
            with open(self.cache_path, "w", encoding="utf-8") as handle:
                json.dump(self.cache, handle, indent=2)
        except OSError:
            pass  # cache is an optimization; failure is never fatal

    def boost(self, assets: List[Asset]) -> Dict[str, Any]:
        """Run the booster over eligible assets; returns a run summary.

        Assets whose fingerprint is cached skip the API. Any network/parse
        failure raises ``GeminiError`` (callers record a warning and keep the
        geometry-only confidences — never a partial half-boosted scene).
        """
        eligible = [a for a in assets if a.asset_class in BOOST_CLASSES]
        verdicts: Dict[str, Dict[str, Any]] = {}
        uncached: List[Asset] = []
        for asset in eligible:
            key = _fingerprint(asset)
            cached = self.cache.get(key)
            if cached is not None:
                verdicts[asset.asset_id] = cached
            else:
                uncached.append(asset)

        api_calls = 0
        for start in range(0, len(uncached), self.batch_size):
            batch = uncached[start : start + self.batch_size]
            prompt = build_prompt(batch)
            text = _call_api(self.api_key, self.model, prompt, timeout=self.timeout)
            api_calls += 1
            batch_verdicts = parse_verdicts(text, [a.asset_id for a in batch])
            for asset in batch:
                verdict = batch_verdicts.get(asset.asset_id)
                if verdict is None:
                    continue
                verdicts[asset.asset_id] = verdict
                self.cache[_fingerprint(asset)] = verdict

        if api_calls:
            self._save_cache()

        applied = 0
        disagreed = 0
        for asset in eligible:
            verdict = verdicts.get(asset.asset_id)
            if verdict is None:
                continue
            apply_verdict(asset, verdict)
            applied += 1
            if verdict["verdict"] == "DISAGREE":
                disagreed += 1
        return {
            "eligible": len(eligible),
            "applied": applied,
            "disagreements": disagreed,
            "api_calls": api_calls,
            "cache_hits": len(eligible) - len(uncached),
            "cache_loaded": self.loaded,
            "model": self.model,
        }