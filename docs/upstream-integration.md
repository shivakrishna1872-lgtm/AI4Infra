# Upstream Integration Notes

This project separates an immediately runnable CPU pipeline from optional learned backends. It does not relabel an upstream benchmark taxonomy as a competition class without an evidence-backed adapter.

## Pointcept and Point Transformer V3

[Pointcept](https://github.com/Pointcept/Pointcept) drives evaluation through `tools/test.py`, which constructs its tester from a config and invokes the configured dataset/model stack. The PTv3 nuScenes semantic-segmentation configuration uses `coord` plus `strength` features and declares a 16-class driving taxonomy; that taxonomy includes `driveable_surface` and `barrier`, but not the full AI4Infra asset taxonomy.

The optional `--backend pointcept` bridge therefore invokes the real `tools/test.py` entrypoint only with a supplied Pointcept root, adapted config, and matching checkpoint. The adapted config is responsible for LAS-tile ingestion, declared class mapping, and prediction export. The inventory keeps the geometry-derived result separate until that adapter exists.

[Point Transformer V3](https://github.com/Pointcept/PointTransformerV3) documents two modes: Pointcept-driven training/inference and a detached backbone that accepts a point dictionary. Its recommended environment includes CUDA and FlashAttention, while the repository also documents running without FlashAttention by setting `enable_flash=False` and reducing patch sizes. AI4Infra does not install, import, or require CUDA extensions for its default path.

## RoadMarkingExtraction

[RoadMarkingExtraction](https://github.com/YuePanEdward/RoadMarkingExtraction) is a C++ pipeline for MLS/ALS road-marking extraction, classification, and vectorization. Its documented command path accepts LAS or PCD input and emits vectorizable output after configuring a model pool. Its dependencies include PCL, OpenCV, LibLas, Eigen, and DXFLib.

The portable `native-roadmarking-v1` stage in this repository is not a copy of that project. It measures near-ground, high-reflectance/RGB connected components directly from LAS data and records the exact method in each asset. A future adapter should run the external executable in a pinned environment, preserve its raw DXF/vector output, and create inventory assets with `detection_method: roadmarkingextraction-v1`.

## Competition-specific contribution

The competition layer is the inventory and attribution contract: object identity, source-tile and point provenance, measured attributes, confidence factors, CRS retention, and export artifacts. It is deliberately model-agnostic so a valid Pointcept fine-tuning run or specialist pavement result can improve detection without weakening auditability.
