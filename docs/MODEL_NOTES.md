# Model Notes

This document records exactly which learned models are used, where they come from,
their licensing, and the evidence-backed reasoning behind the class adaptation layer.
**No model weights are committed to this repository.**

## Pretrained model: PTv3 nuScenes semantic segmentation

* Model: Point Transformer V3 (PTv3), `pt-v3m1-base`, nuScenes lidar segmentation.
* Source: official Pointcept release on Hugging Face:
  https://huggingface.co/Pointcept/PointTransformerV3
  * Weights: `nuscenes-semseg-pt-v3m1-0-base/model/model_best.pth` (~530 MB)
  * Config: `nuscenes-semseg-pt-v3m1-0-base/config.py`
  * SHA-256 of `model_best.pth`: `e2774d2fa1dd33e640a514afe0bd1e5af94eb08be19d08d1f9402f20dfd6db94`
    (verified by `python -m infra_inventory download-models`)
* License: per the upstream repository the nuScenes-trained release is
  **CC-BY-NC-4.0** (non-commercial). Check the upstream repo before any commercial use.
* Why nuScenes and not Waymo: Waymo-trained weights cannot be released under Waymo
  regulations (documented upstream). nuScenes is the supported outdoor
  semantic-segmentation starting point and includes `driveable_surface` and `barrier`
  — the two classes with direct evidence value for this competition.

## Runtime requirements (upstream-documented)

* Pointcept requires its CUDA point ops to be compiled; inference runs on GPU.
* FlashAttention is **optional**: the upstream docs support
  `enable_flash=False` with smaller patch sizes for non-compatible GPUs.
* The default `geometry` backend in this project never imports torch, spconv, or
  any CUDA extension. `python -m infra_inventory backends` inspects the environment
  without requiring torch.

## Why the pretrained model does NOT know the four competition classes

The nuScenes taxonomy (17 entries with ignore) contains `driveable_surface` and
`barrier`, but **not** utility poles, traffic signs, guardrails, or pavement
markings as first-class labels. Blindly relabeling nuScenes classes to competition
classes would be fabrication. This project therefore separates:

1. **Pretrained capabilities** — PTv3 emits nuScenes-class per-point predictions.
2. **The adaptation layer** (`configs/model.yaml`) — a documented table mapping
   upstream classes to *evidence* for infrastructure classes:
   * `driveable_surface` → strong evidence for `pavement` (weight 0.90)
   * `barrier` → evidence for `safety_barrier` (weight 0.80), geometry must confirm
   * `manmade` → weak evidence for poles/signs/cabinets (weight 0.35)
   * `sidewalk` / `terrain` / `vegetation` → context only, never assets
3. **Geometry detectors** — always gate whether an asset exists.

The `model` confidence factor of each asset is the weighted agreement between the
pretrained prior and the asset's points; the factor list printed with every asset
makes this auditable.

## What improves accuracy most with labeled data

Fine-tuning PTv3 on the competition taxonomy (see `docs/ENHANCEMENT.md`) replaces
the adaptation table with direct supervision, which is the highest-impact
improvement available once labels exist. Until then, geometry + priors is the
honest, documented baseline.

## Integrity

`scripts/download_models.py` / `python -m infra_inventory download-models` streams
downloads to a temp file and verifies the SHA-256 before install, so a corrupted
download never lands in the weights directory. Weights are gitignored (`.gitignore`).