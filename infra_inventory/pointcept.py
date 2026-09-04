"""Optional, explicit bridge to Pointcept's real test entrypoint.

This module deliberately does not translate a generic pretrained taxonomy into the
competition taxonomy. A deployment must supply a Pointcept config, matching
weights, and an adapted dataset implementation for the exported LAS tiles.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional


class PointceptConfigurationError(RuntimeError):
    pass


def describe_backend(root: Optional[str]) -> Dict[str, object]:
    if not root:
        return {"name": "geometry", "available": True, "reason": "CPU geometric baseline selected"}
    test_script = Path(root) / "tools" / "test.py"
    return {
        "name": "pointcept",
        "available": test_script.is_file(),
        "entrypoint": str(test_script),
        "reason": "Uses Pointcept tools/test.py; requires a dataset adapter and matching weights.",
    }


def run_pointcept_test(
    *, root: str, config: str, weight: str, num_gpus: int, save_path: Path
) -> Dict[str, object]:
    """Run Pointcept's documented `tools/test.py` entrypoint.

    This is intentionally an opt-in execution. Pointcept configuration controls
    the dataset transform, class taxonomy, and output format; this application
    never invents those mappings.
    """
    root_path = Path(root).expanduser().resolve()
    script = root_path / "tools" / "test.py"
    if not script.is_file():
        raise PointceptConfigurationError(f"Pointcept test entrypoint not found: {script}")
    if not Path(config).expanduser().is_file():
        raise PointceptConfigurationError(f"Pointcept config not found: {config}")
    if not Path(weight).expanduser().is_file():
        raise PointceptConfigurationError(f"Pointcept weight not found: {weight}")

    save_path.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root_path) + os.pathsep + environment.get("PYTHONPATH", "")
    command = [
        sys.executable,
        str(script),
        "--config-file",
        str(Path(config).expanduser().resolve()),
        "--num-gpus",
        str(num_gpus),
        "--options",
        f"save_path={save_path}",
        f"weight={Path(weight).expanduser().resolve()}",
    ]
    completed = subprocess.run(command, cwd=root_path, env=environment, capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(
            "Pointcept inference failed. Verify CUDA/extensions, the Pointcept dataset adapter, config, and weight.\n"
            + completed.stderr[-4000:]
        )
    return {"name": "pointcept", "command": command, "stdout": completed.stdout[-2000:]}
