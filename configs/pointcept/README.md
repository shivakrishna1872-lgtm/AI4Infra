# Pointcept adapter configuration

The `pointcept` backend runs Pointcept's **real** `tools/test.py` entrypoint:

```bash
python -m infra_inventory process data/mannford.las --output out \
  --backend pointcept \
  --pointcept-root /path/to/Pointcept \
  --pointcept-config configs/pointcept/infra_ptv3_nuscenes.py \
  --pointcept-weight models/ptv3_nuscenes_semseg.pth \
  --pointcept-class-names configs/pointcept/nuscenes_class_names.json
```

The pipeline writes one LAS file per spatial tile to `out/tiles/` **before**
Pointcept runs, and expects your adapted config to:

1. Consume those tiles (dataset class + test split pointing at `out/tiles/`).
2. Declare the taxonomy via `data.class_names` **in the same order** as
   `nuscenes_class_names.json`.
3. Export predictions as one `.npy` per tile named `<tile_name>.npy`
   (class-id tensor aligned with the tile's point order) into the config's
   `save_path` (the pipeline passes `save_path=<out>/predictions`).

The template `infra_ptv3_nuscenes.py.example` shows the minimal shape. Class
mapping to infrastructure evidence happens afterwards in `configs/model.yaml` -
never inside the model config.

## Running without FlashAttention

The upstream PTv3 docs allow disabling FlashAttention: set
`model=dict(... enable_flash=False ...)` and use a smaller `patch_size` (e.g.
`patch_size=(512, 512, 256)`) in the model config. CUDA extensions (spconv
point ops) are still required by Pointcept's default PTv3 build.