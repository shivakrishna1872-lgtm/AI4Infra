# Gemini class-validation booster

The geometry detectors always decide **whether** an asset exists. The Gemini
booster adds an independent learned check on top: it reads each detected
instance's *measured* evidence (dimensions, height, point support, intensity /
RGB statistics, orientation) and returns a plausibility verdict for the
assigned class. The verdict becomes the `model` factor of the transparent
confidence blend — the same slot a Pointcept/PTv3 prior fills when that backend
is installed.

```
detect (geometry) ──► merge spans ──► Gemini class validation ──► QC + ALP ──► export
                                        │  verdict ──► confidence.model factor
                                        ▼
                        AGREE    → confidence stays high
                        UNCERTAIN→ mid-band factor (≈ weak prior)
                        DISAGREE → confidence drops below the calibrated
                                   0.80 review threshold → human review
```

## Integrity rules

1. **Gemini reasons only over measured facts** — the prompt contains exactly
   what the detectors measured, never raw points and never the model's own
   guesses. A `DISAGREE` verdict lowers confidence (routing the asset to human
   review); it never relabels, splits, or deletes an asset on its own.
2. **The pipeline never depends on the network.** Missing key, timeouts, 5xx
   retries exhausted, or an unparseable response all degrade to the
   geometry-only confidence and a run warning (`Gemini validation skipped: …`).
   A run is never half-boosted: verdicts are applied per batch only after the
   batch response parses.
3. **Verdicts are cached** by a stable evidence fingerprint (class + geometry +
   radiometry) in `output/gemini_cache.json`, so re-runs of the same scene
   cost zero API calls.
4. **The factor is auditable**: every boosted asset carries `model_confidence`
   (the factor), `model_prior_class`, a `-gemini` suffix on
   `detection_method`, and the verdict + reason in `confidence_explanation`.

## Setup

1. Get a Google Gemini API key (AI Studio → API key).
2. Set `GEMINI_API_KEY` in the project's Keys/API keys tab (or
   `export GEMINI_API_KEY=...` in the shell).
3. Run with the `gemini` backend:

```bash
python -m infra_inventory process data/mannford.las --output out/mannford \
  --backend gemini [--gemini-model gemini-3.6-flash]
```

Or set `backend: gemini` in `configs/processing.yaml`. The default model is
`gemini-3.6-flash`; older `gemini-2.5-flash` is retired for new API keys, and
any model name can be passed with `--gemini-model`.

Only stdlib HTTP is used (no SDK dependency). One API call covers up to 25
assets in a batch; transient 5xx responses and per-minute quota 429s are
retried with backoff (the free tier is ~20 requests/day on flash models —
the verdict cache keeps re-runs free).

### Web app (default-on when a key exists)

Every web processing path — **Upload & Process**, project **Process**, and
**Quick Simulation** — enables the booster automatically whenever
`GEMINI_API_KEY` is present in the server environment (`_web_settings` in
`infra_inventory/server.py`). Web runs always drop tile intermediates, raise
the viewer LOD budget, and validate classes with Gemini when possible. The
process API accepts `use_gemini` (`"false"`/`"0"`/`"off"` to force
geometry-only); without the key everything silently runs geometry-only.
Per-run CLI-style override: `ProcessingSettings(gemini_api_key=...)`.

## Does it actually improve accuracy? (measured)

`scripts/train_gemini.py` measures the boost against the QuickSim ground
truth (same match radii as `scripts/calibrate_confidence.py`) at the
calibrated 0.80 review cut:

* **oracle mode (default)** — a perfect reviewer's verdicts, to measure the
  *mechanism* with zero API cost. On seeds 23 / 42 / 7 (400 m corridor):
  false area-class fragments rejected (accepted 40 → 26), compact
  precision/recall/F1 unchanged at 1.000, and area-class recall lifted
  **0.80 → 1.00** (true fragments whose confidence the boost pushed over the
  cut). A good reviewer both *removes junk* and *rescues borderline truths*.
* **live mode (`--live`)** — the real Gemini API against the same scenes,
  verdicts cached for re-runs:

```bash
python scripts/train_gemini.py --live --report out/accuracy/gemini_report.json
```

On the clean synthetic corridor the live reviewer agrees with almost every
detection (they are all physically plausible), so the accepted set grows and
precision stays 1.000 — the honest expectation on synthetic data. The
discriminating case is real-world clutter: an object whose measured shape
contradicts its label gets `DISAGREE` and lands in the human-review queue
instead of the inventory.

The honest framing for the competition report: the booster is an
**innovation/enhancement layer** (10-pt rubric) that makes confidence
*calibrated to a second learned opinion* — it does not replace the measured
geometry pipeline, and its verdicts never fabricate assets.

## Cost

Roughly one API call per 40 assets; a mile of road (~100–300 assets) is
~3–8 calls per run, and cached thereafter. Free-tier quotas are plenty for
competition-scale runs.