"""Explicit bridge to OpenPCSeg's real inference entrypoints.

This module invokes OpenPCSeg's documented ``infer.py`` exactly as upstream
documents it (``--cfg_file`` / ``--ckp`` / ``--set``) and its ``train.py`` for
fine-tuning on Toronto-3D. It never pretends a pretrained taxonomy matches the
four competition categories: Toronto-3D class ids are consumed only through the
documented adaptation mapping (``TORONTO3D_CLASS_MAPPING`` in ``pipeline.py``
and ``configs/model.yaml``), and geometry detectors still gate every asset.

Upstream contract (verified against BAI-Yeqi/OpenPCSeg @ master, 2026-09):
* ``infer.py`` accepts ``--cfg_file <yaml> --ckp <checkpoint.pth>`` plus
  ``--set KEY VALUE`` config overrides; per-point predictions are saved as
  zero-padded index-named ``.npy`` files (``0000000000.npy``) in ``DATA.OUTPUT_DIR``.
* ``train.py`` accepts the same ``--cfg_file`` plus ``--ckp`` (init weights)
  and ``--extra_tag``.
* Model zoo weights (SemanticKITTI MinkowskiNet / SPVCNN) are downloadable
  from the links in ``configs/model.yaml``; Waymo weights cannot be released.

Requirements for the ``openpcseg`` backend:
* a checked-out OpenPCSeg repository with its environment installed
* a CUDA-capable GPU (torchreports; torchsparse/MinkowskiEngine ops compiled)
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .errors import BackendConfigurationError, CudaUnavailableError, ModelWeightsUnavailableError

INFER_REL = Path("infer.py")
TRAIN_REL = Path("train.py")


def _torch_cuda() -> tuple[bool, str]:
    """(cuda_available, detail) without forcing a torch dependency."""
    try:
        import torch  # type: ignore[import-not-found]

        if not torch.cuda.is_available():
            return False, "torch installed but CUDA is not available on this machine"
        return True, f"CUDA {torch.version.cuda} available"
    except ImportError:
        return False, "torch is not installed in this Python environment"


def describe_openpcseg(root: Optional[str]) -> Dict[str, object]:
    """Report what the environment can actually run."""
    cuda_ok, cuda_detail = _torch_cuda()
    if not root:
        return {
            "name": "geometry",
            "available": True,
            "gpu_required": False,
            "reason": "CPU-safe geometric baseline; no learned model involved.",
        }
    root_path = Path(root).expanduser()
    infer_script = root_path / INFER_REL
    train_script = root_path / TRAIN_REL
    detail: Dict[str, object] = {
        "name": "openpcseg",
        "gpu_required": True,
        "gpu": cuda_detail,
        "cuda_available": cuda_ok,
        "entrypoint": str(infer_script),
        "entrypoint_present": infer_script.is_file(),
        "train_entrypoint_present": train_script.is_file(),
    }
    if not infer_script.is_file() or not train_script.is_file():
        detail["available"] = False
        detail["reason"] = (
            f"OpenPCSeg infer.py/train.py not found under root: {root_path}"
        )
    elif not cuda_ok:
        detail["available"] = False
        detail["reason"] = "OpenPCSeg requires CUDA; the geometry backend remains available."
    else:
        detail["available"] = True
        detail["reason"] = "OpenPCSeg entrypoints present and CUDA available."
    return detail


def _openpcseg_command(
    root: Path, script: Path, num_gpus: int, launcher_args: List[str],
) -> List[str]:
    """Single-GPU runs invoke upstream directly; multi-GPU mirrors infer.sh."""
    if num_gpus > 1:
        return [
            sys.executable, "-m", "torch.distributed.launch",
            f"--nproc_per_node={num_gpus}", str(script),
            "--launcher", "pytorch", *launcher_args,
        ]
    return [sys.executable, str(script), *launcher_args]


def _run_openpcseg(
    *, root: str, script_rel: Path, num_gpus: int, launcher_args: List[str],
) -> Dict[str, object]:
    root_path = Path(root).expanduser().resolve()
    script = root_path / script_rel
    if not script.is_file():
        raise BackendConfigurationError(
            f"OpenPCSeg entrypoint not found: {script}",
            "Clone https://github.com/BAI-Yeqi/OpenPCSeg, install its environment, "
            "and pass --openpcseg-root pointing at the checkout.",
        )
    cuda_ok, cuda_detail = _torch_cuda()
    if not cuda_ok:
        raise CudaUnavailableError(cuda_detail)

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root_path) + os.pathsep + environment.get("PYTHONPATH", "")
    command = _openpcseg_command(root_path, script, num_gpus, launcher_args)
    print(f"[openpcseg] running: {' '.join(command)}", flush=True)
    completed = subprocess.run(
        command, cwd=str(root_path), env=environment, capture_output=True, text=True,
    )
    if completed.returncode:
        raise RuntimeError(
            "OpenPCSeg failed. Verify its environment, config, checkpoint, and CUDA ops.\n"
            + (completed.stderr or completed.stdout)[-4000:]
        )
    return {
        "name": "openpcseg",
        "command": command,
        "stdout_tail": (completed.stdout or "")[-2000:],
    }


def run_openpcseg_infer(
    *, root: str, cfg_file: str, checkpoint: str, num_gpus: int,
    prediction_dir: Path, extra_options: Optional[Dict[str, str]] = None,
) -> Dict[str, object]:
    """Run OpenPCSeg's real ``infer.py`` on the exported tensors.

    The supplied config consumes the tiles exported by ``tensor_export.py``;
    ``DATA.OUTPUT_DIR`` is overridden to ``prediction_dir`` so the per-point
    ``.npy`` predictions land where ``load_indexed_predictions`` reads them.
    """
    cfg_path = Path(cfg_file).expanduser().resolve()
    if not cfg_path.is_file():
        raise BackendConfigurationError(
            f"OpenPCSeg config not found: {cfg_path}",
            "Point to an OpenPCSeg config (.yaml), e.g. tools/cfgs/voxel/semantic_kitti/"
            "minkunet_mk34_cr10.yaml adapted for the exported tensors.",
        )
    ckpt_path = Path(checkpoint).expanduser().resolve()
    if not ckpt_path.is_file():
        raise ModelWeightsUnavailableError(str(ckpt_path))

    prediction_dir.mkdir(parents=True, exist_ok=True)
    launcher_args = [
        "--cfg_file", str(cfg_path),
        "--ckp", str(ckpt_path),
        "--set", "DATA.OUTPUT_DIR", str(prediction_dir),
    ]
    for key, value in (extra_options or {}).items():
        launcher_args += ["--set", key, str(value)]
    result = _run_openpcseg(
        root=root, script_rel=INFER_REL, num_gpus=num_gpus, launcher_args=launcher_args,
    )
    result["prediction_dir"] = str(prediction_dir)
    return result


def run_openpcseg_train(
    *, root: str, cfg_file: str, init_checkpoint: Optional[str], num_gpus: int,
    extra_options: Optional[Dict[str, str]] = None, extra_tag: str = "toronto3d_finetune",
) -> Dict[str, object]:
    """Run OpenPCSeg's real ``train.py`` for fine-tuning on prepared data."""
    cfg_path = Path(cfg_file).expanduser().resolve()
    if not cfg_path.is_file():
        raise BackendConfigurationError(
            f"OpenPCSeg config not found: {cfg_path}",
            "Point to an OpenPCSeg training config (.yaml); see docs/LEARNED_MODELS.md.",
        )
    launcher_args = ["--cfg_file", str(cfg_path), "--extra_tag", extra_tag]
    if init_checkpoint:
        ckpt_path = Path(init_checkpoint).expanduser().resolve()
        if not ckpt_path.is_file():
            raise ModelWeightsUnavailableError(str(ckpt_path))
        launcher_args += ["--ckp", str(ckpt_path)]
    for key, value in (extra_options or {}).items():
        launcher_args += ["--set", key, str(value)]
    return _run_openpcseg(
        root=root, script_rel=TRAIN_REL, num_gpus=num_gpus, launcher_args=launcher_args,
    )


def load_indexed_predictions(prediction_dir: Path, tile_names: List[str]) -> Dict[str, np.ndarray]:
    """Load OpenPCSeg's index-named per-point predictions, aligned to tiles.

    Upstream saves ``%010d.npy`` (``0000000000.npy``) in dataset order. This
    loader zips that order with the pipeline's streaming tile order, so tile
    ``k`` receives the k-th prediction file, length-checked against the tile's
    point count by the consumer. Missing or truncated files yield a shorter
    dict; tiles without predictions simply run geometry-only.
    """
    if not prediction_dir.is_dir():
        return {}
    files = sorted(prediction_dir.glob("*.npy"))
    result: Dict[str, np.ndarray] = {}
    for tile_name, path in zip(tile_names, files):
        array = np.load(path)
        if array.ndim == 2 and array.shape[1] > 1:
            array = np.argmax(array, axis=1)
        result[tile_name] = array.astype(np.int64).reshape(-1)
    return result
