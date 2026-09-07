# Learned models: datasets, pretrained weights, and the adaptation layer

This project's rule for every learned model: **the model contributes evidence;
geometry decides.** No pretrained taxonomy is ever assumed to know the four
competition categories — upstream class ids enter detection only through the
documented adaptation table below, and every asset still passes the geometry
gates (columnarity for poles, linearity for wires, planarity for sign panels,
cross-section shapes for guardrails/barriers). A learned prior can only raise
or lower the `model` confidence factor of an asset the geometry already found.

Three learned backends share this contract:

| backend | entrypoint invoked | taxonomy | GPU |
|---|---|---|---|
| `pointcept` | Pointcept `tools/test.py` | nuScenes (16 ids) | yes |
| `openpcseg` | OpenPCSeg `infer.py` / `train.py` | Toronto-3D (9 ids) or SemanticKITTI (20 ids) | yes |
| `gemini` | Gemini API (per-asset review) | natural language | no |

## 1. OpenPCSeg backend (mobile-LiDAR taxonomies)

[OpenPCSeg](https://github.com/BAI-Yeqi/OpenPCSeg) (Apache-2.0) ships
MinkowskiNet / Cylinder3D / SPVCNN / RPVNet for large outdoor point clouds,
with downloadable SemanticKITTI checkpoints and documented
`infer.py --cfg_file <yaml> --ckp <pth> --set KEY VALUE` invocation. This
application:

1. streams the MX9 LAS into 40 m tiles (pass 1, unchanged),
2. exports the tiles as SemanticKITTI-convention tensors — `float32[~,4]`
   rows (x, y, z, intensity) with the LAS header **offset removed** (the same
   precision fix Toronto-3D documents for UTM coordinates) and recorded in
   `tensors/tensor_manifest.json`,
3. invokes `infer.py` exactly as upstream documents, overriding only
   `DATA.OUTPUT_DIR` (and pointing `DATA.DATA_PATH` at the exported tensors),
4. reads the index-named per-point prediction files
   (`0000000000.npy`, upstream saves in dataset order) back aligned to the
   pipeline's tile order,
5. converts class ids into per-point evidence weights through the mapping in
   [`configs/model.yaml`](../configs/model.yaml).

```bash
python -m infra_inventory download-models --model semkitti-minkunet --output models

python -m infra_inventory process data/mannford.las --output output/mannford \
  --backend openpcseg \
  --openpcseg-root /opt/OpenPCSeg \
  --openpcseg-config tools/cfgs/voxel/semantic_kitti/minkunet_mk34_cr10.yaml \
  --openpcseg-weight models/semkitti-minkunet.pth \
  --openpcseg-taxonomy semantickitti
```

Checkpoint filename mapping: the weight file must match the config's
`MODEL.NAME` + architecture (`minkunet_mk34_cr10.yaml` pairs with the
`semkitti_minkunet_mk34_cr16_checkpoint_epoch_36.pth` release).

### Model zoo (upstream-reported SemanticKITTI mIoU)

| model | mIoU | `--model` key | notes |
|---|---|---|---|
| MinkowskiNet mk34 | 70.04 | `semkitti-minkunet` | ~737 MB checkpoint |
| SPVCNN mk18 | 68.58 | `semkitti-spvcnn` | ~166 MB |
| Cylinder3D cy480 | 66.07 | `semkitti-cylinder3d` | ~56 MB |

Upstream publishes Dropbox links without checksums; the observed SHA-256 is
printed and pinned into `models/manifest.json` at download time.

## 2. Toronto-3D: the fine-tuning dataset

[Toronto-3D](https://github.com/WeikaiTan/Toronto-3D) (CC BY-NC 4.0) is ~1 km
of vehicle-mounted mobile-laser road survey with XYZ / intensity / RGB — the
closest public analogue to the Mannford MX9 capture, and its taxonomy covers
every competition category:

| id | class | competition category |
|---|---|---|
| 1 | road | pavement |
| 2 | road_marking | pavement markings |
| 5 | utility_line | overhead conductors |
| 6 | pole | utility poles (weak evidence for sign posts) |
| 8 | fence | weak evidence for guardrails/barriers |
| 3, 4, 7 | natural / building / car | context only |

The released tiles are binary PLY files (`original_ply/L001..L004.ply` with a
`scalar_Label` field; verified against upstream `data_prepare_toronto3d.py`).
The preparation script converts them to OpenPCSeg training pairs using the
upstream-documented split (train = L001/L003/L004, test = L002) and UTM offset
(`[627285, 4841948, 0]`):

```bash
python scripts/prepare_toronto3d.py --input /data/Toronto_3D --output data_root/toronto3d

python -m infra_inventory train --framework openpcseg \
  --openpcseg-root /opt/OpenPCSeg \
  --openpcseg-config tools/cfgs/voxel/semantic_kitti/minkunet_mk34_cr10.yaml \
  --init-weight models/semkitti-minkunet.pth
```

Fine-tune starting from the SemanticKITTI checkpoint (transfer from a
superset taxonomy), then process with `--openpcseg-taxonomy toronto3d`.
Reference results on Toronto-3D (upstream README): RandLA-Net 77.7 mIoU (XYZ)
/ 81.8 mIoU (XYZ+RGB).

## 3. The adaptation decision table

Verified against the upstream READMEs (2026-09); full table with weights in
[`configs/model.yaml`](../configs/model.yaml) (`openpcseg.class_mapping`):

| upstream class | asset evidence | weight |
|---|---|---|
| road | `pavement` (travelled surface) | 0.90 |
| road_marking | `pavement_marking` | 0.85 |
| utility_line | `overhead_conductor` | 0.85 |
| pole | `utility_pole` | 0.85 |
| traffic-sign (SemanticKITTI only) | `traffic_sign` | 0.85 |
| fence | `safety_barrier`, weak evidence for `guardrail` | 0.55 |
| building | very weak `utility_cabinet` (geometry decides) | 0.10 |
| terrain / trunk | ground/column context only | ≤ 0.2 |
| car, natural, vegetation, sidewalk, parking | context only | 0.0–0.05 |

How it enters confidence: for each detected asset, the fraction of its points
whose upstream class maps onto the asset's class (weighted by the table) is
the `model` factor in the confidence blend
(`model_factor_from_prior` in `infra_inventory/confidence.py`). With no
backend configured the factor is neutral — the inventory never depends on a
model being installed.

## 4. Honest limitations

* Both datasets are outdoor but **not MX9-specific**: resolution, scanner
  pattern, and class balance differ from the competition capture. The
  adaptation table deliberately assigns no upstream class to cabinet or
  rumble-strip detection — those remain geometry-driven.
* Model-zoo checkpoints are trained on automotive LiDAR (SemanticKITTI) at
  ~64-beam density; mobile-mapping data is denser and slower-moving, which
  usually transfers fine but is a distribution shift — measure with the
  QuickSim ground truth (`scripts/train_thresholds.py`) before trusting a
  checkpoint on Mannford data.
* Toronto-3D is CC BY-NC 4.0: fine-tuned weights inherit the non-commercial
  restriction; verify before commercial use.
