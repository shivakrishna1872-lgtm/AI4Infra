"""Explicit bridge to Pointcept's real test entrypoint.

This module invokes Pointcept's documented ``tools/test.py`` exactly as the
upstream project documents it (``--config-file`` / ``--num-gpus`` /
``--options``). It never pretends a generic pretrained taxonomy matches the
competition classes: prediction classes are consumed only through the explicit
adaptation mapping in ``configs/model.yaml``, and geometry still decides whether
an asset exists.

Requirements for the ``pointcept`` backend (from the upstream README):
* a checked-out Pointcept repository with its environment installed
* a CUDA-capable GPU with the Pointcept CUDA ops compiled
* FlashAttention optional: configure PTv3 with ``enable_flash=False`` and
  smaller patch sizes to run without it
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .errors import BackendConfigurationError, CudaUnavailableError, ModelWeightsUnavailableError

POINTCEPT_TEST_REL = Path("tools") / "test.py"


def _torch_cuda() -> tuple[bool, str]:
    """(cuda_available, detail) without forcing a torch dependency."""
    try:
        import torch  # type: ignore[import-not-found]

        if not torch.cuda.is_available():
            return False, "torch installed but CUDA is not available on this machine"
        return True, f"CUDA {torch.version.cuda} available"
    except ImportError:
        return False, "torch is not installed in this Python environment"


def describe_backend(root: Optional[str], class_names_file: Optional[str] = None) -> Dict[str, object]:
    """Report what the environment can actually run."""
    cuda_ok, cuda_detail = _torch_cuda()
    if not root:
        return {
            "name": "geometry",
            "available": True,
            "gpu_required": False,
            "reason": "CPU-safe geometric baseline; no learned model involved.",
        }
    test_script = Path(root).expanduser() / POINTCEPT_TEST_REL
    detail: Dict[str, object] = {
        "name": "pointcept",
        "gpu_required": True,
        "gpu": cuda_detail,
        "cuda_available": cuda_ok,
        "entrypoint": str(test_script),
        "entrypoint_present": test_script.is_file(),
    }
    if class_names_file and Path(class_names_file).is_file():
        detail["class_names_file"] = str(Path(class_names_file).resolve())
    if not test_script.is_file():
        detail["available"] = False
        detail["reason"] = f"Pointcept tools/test.py not found under root: {test_script}"
    elif not cuda_ok:
        detail["available"] = False
        detail["reason"] = "Pointcept requires CUDA; the geometry backend remains available."
    else:
        detail["available"] = True
        detail["reason"] = "Pointcept tools/test.py present and CUDA available."
    return detail


def load_class_names(path: Optional[str]) -> List[str]:
    """Load the class taxonomy (JSON list) of the Pointcept config under test.

    Order matters: class id ``i`` in a prediction tensor corresponds to
    ``class_names[i]``. The JSON file must be produced alongside the adapted
    Pointcept dataset config (see docs/PROCEDURE.md).
    """
    if not path:
        return []
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise BackendConfigurationError(
            f"Pointcept class names file not found: {resolved}",
            "Create the JSON taxonomy file for the config you are testing, or omit --pointcept-class-names.",
        )
    names = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(names, list) or not all(isinstance(item, str) for item in names):
        raise BackendConfigurationError(
            f"Class names file must be a JSON list of strings: {resolved}",
            "Fix the file format (e.g. [\"ignore\", \"barrier\", ...]).",
        )
    return names


def run_pointcept_test(
    *, root: str, config: str, weight: str, num_gpus: int, save_path: Path, class_names: Optional[List[str]] = None,
) -> Dict[str, object]:
    """Run Pointcept's real ``tools/test.py`` entrypoint on the exported tiles.

    The supplied config is responsible for consuming the tiles written to
    ``output/tiles/``, declaring its taxonomy, and exporting predictions to the
    config's ``save_path`` (prediction arrays are read back from there).
    """
    root_path = Path(root).expanduser().resolve()
    script = root_path / POINTCEPT_TEST_REL
    if not script.is_file():
        raise BackendConfigurationError(
            f"Pointcept test entrypoint not found: {script}",
            "Clone https://github.com/Pointcept/Pointcept, build its environment, "
            "and pass --pointcept-root pointing at the checkout.",
        )
    config_path = Path(config).expanduser().resolve()
    if not config_path.is_file():
        raise BackendConfigurationError(
            f"Pointcept config not found: {config_path}",
            "Point to a Pointcept config (.py) that consumes the tiled LAS data; "
            "see configs/pointcept/ in this repository for a template.",
        )
    weight_path = Path(weight).expanduser().resolve()
    if not weight_path.is_file():
        raise ModelWeightsUnavailableError(str(weight_path))
    cuda_ok, cuda_detail = _torch_cuda()
    if not cuda_ok:
        raise CudaUnavailableError(cuda_detail)

    save_path.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root_path) + os.pathsep + environment.get("PYTHONPATH", "")
    command = [
        sys.executable,
        str(script),
        "--config-file", str(config_path),
        "--num-gpus", str(num_gpus),
        "--options",
        f"save_path={save_path}",
        f"weight={weight_path}",
    ]
    print(f"[pointcept] running: {' '.join(command)}", flush=True)
    completed = subprocess.run(command, cwd=root_path, env=environment, capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(
            "Pointcept inference failed. Verify the Pointcept environment, config, weight, and CUDA ops.\n"
            + (completed.stderr or completed.stdout)[-4000:]
        )
    return {
        "name": "pointcept",
        "command": command,
        "stdout_tail": (completed.stdout or "")[-2000:],
        "prediction_dir": str(save_path),
        "class_names": class_names or [],
    }


def load_predictions(prediction_dir: Path, tile_names: List[str]) -> Dict[str, np.ndarray]:
    """Load per-tile prediction arrays exported by the Pointcept config.

    The adapted config is expected to export one ``.npy`` file per tile named
    ``<tile_name>.npy`` (see configs/pointcept/ exporter example). Returns a
    dict of tile name -> int64 class-id array aligned with the tile's points.
    """
    if not prediction_dir.is_dir():
        return {}
    result: Dict[str, np.ndarray] = {}
    for tile in tile_names:
        candidates = [
            prediction_dir / f"{tile}.npy",
            prediction_dir / "predictions" / f"{tile}.npy",
            prediction_dir / f"{tile}_pred.npy",
        ]
        for candidate in candidates:
            if candidate.is_file():
                array = np.load(candidate)
                if array.ndim == 2 and array.shape[1] > 1:
                    array = np.argmax(array, axis=1)
                result[tile] = array.astype(np.int64).reshape(-1)
                break
    return result