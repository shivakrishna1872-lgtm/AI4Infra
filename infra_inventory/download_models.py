"""Pretrained model downloader.

Sources are documented per model (configs/model.yaml). Downloads stream to a
temporary file and are verified against the published SHA-256 before install, so
a corrupted download never lands in the weights directory. Weights are gitignored
(see .gitignore) - never commit model files to the repository.

Available models:
* ``nuscenes-ptv3-semseg`` - PTv3 nuScenes semantic segmentation (outdoor),
  Pointcept's official release on Hugging Face (model_best.pth + config.py).
* ``semkitti-minkunet`` / ``semkitti-spvcnn`` / ``semkitti-cylinder3d`` -
  OpenPCSeg model-zoo checkpoints trained on SemanticKITTI (see
  configs/model.yaml). Upstream publishes Dropbox links without checksums;
  the SHA-256 observed at download time is printed and recorded in the
  manifest so it can be pinned.
"""
from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from pathlib import Path
from typing import Dict

from .errors import InfraError

MODELS: Dict[str, Dict[str, str]] = {
    "nuscenes-ptv3-semseg": {
        "description": "Point Transformer V3, nuScenes semantic segmentation (16 classes)",
        "license": "CC-BY-NC-4.0 (see upstream repository)",
        "weight_url": (
            "https://huggingface.co/Pointcept/PointTransformerV3/resolve/main/"
            "nuscenes-semseg-pt-v3m1-0-base/model/model_best.pth"
        ),
        "weight_sha256": "e2774d2fa1dd33e640a514afe0bd1e5af94eb08be19d08d1f9402f20dfd6db94",
        "config_url": (
            "https://huggingface.co/Pointcept/PointTransformerV3/resolve/main/"
            "nuscenes-semseg-pt-v3m1-0-base/config.py"
        ),
        "config_sha256": None,  # small file; verified by import at runtime
        "weight_filename": "ptv3_nuscenes_semseg.pth",
        "config_filename": "ptv3_nuscenes_semseg_config.py",
    },
    # --- OpenPCSeg model zoo (SemanticKITTI; upstream README, verified 2026-09) ---
    "semkitti-minkunet": {
        "description": "OpenPCSeg MinkowskiNet mk34, SemanticKITTI (mIoU 70.04)",
        "license": "Apache-2.0 (OpenPCSeg)",
        "weight_url": (
            "https://www.dropbox.com/s/a9gjxeziy6rbiui/"
            "semkitti_minkunet_mk34_cr16_checkpoint_epoch_36.pth?dl=1"
        ),
        "weight_sha256": None,  # upstream publishes no checksum; hash is printed
        "config_url": (
            "https://raw.githubusercontent.com/BAI-Yeqi/OpenPCSeg/master/"
            "tools/cfgs/voxel/semantic_kitti/minkunet_mk34_cr10.yaml"
        ),
        "config_sha256": None,
    },
    "semkitti-spvcnn": {
        "description": "OpenPCSeg SPVCNN mk18, SemanticKITTI (mIoU 68.58)",
        "license": "Apache-2.0 (OpenPCSeg)",
        "weight_url": (
            "https://www.dropbox.com/s/94j8rxkxlo2j924/"
            "semkitti_spvcnn_mk18_cr10_checkpoint_epoch_36.pth?dl=1"
        ),
        "weight_sha256": None,
        "config_url": (
            "https://raw.githubusercontent.com/BAI-Yeqi/OpenPCSeg/master/"
            "tools/cfgs/fusion/semantic_kitti/spvcnn_mk18_cr10.yaml"
        ),
        "config_sha256": None,
    },
    "semkitti-cylinder3d": {
        "description": "OpenPCSeg Cylinder3D cy480, SemanticKITTI (mIoU 66.07)",
        "license": "Apache-2.0 (OpenPCSeg)",
        "weight_url": (
            "https://www.dropbox.com/s/imtcmn9z4qldc2h/"
            "semkitti_cylinder_cy480_cr10_checkpoint_epoch_35.pth?dl=1"
        ),
        "weight_sha256": None,
        "config_url": (
            "https://raw.githubusercontent.com/BAI-Yeqi/OpenPCSeg/master/"
            "tools/cfgs/voxel/semantic_kitti/cylinder_cy480_cr10.yaml"
        ),
        "config_sha256": None,
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(url: str, destination: Path, expected_sha256: str, label: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    print(f"Downloading {label} from {url}", flush=True)
    try:
        with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as handle:
            total = int(response.headers.get("Content-Length") or 0)
            downloaded = 0
            while True:
                block = response.read(1024 * 256)
                if not block:
                    break
                handle.write(block)
                downloaded += len(block)
                if total:
                    percent = int(downloaded * 100 / max(total, 1))
                    print(f"\r  {percent:3d}% ({downloaded / 1e6:.1f} / {total / 1e6:.1f} MB)", end="", flush=True)
        print()
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        raise InfraError(
            f"Download failed: {url} ({exc})",
            "Check the network connection, or download the file manually and place it at "
            f"{destination} with the documented SHA-256.",
        ) from exc
    actual = _sha256(temporary)
    if expected_sha256 and actual != expected_sha256:
        temporary.unlink(missing_ok=True)
        raise InfraError(
            f"Checksum mismatch for {label} (expected {expected_sha256}, got {actual})",
            "The download was corrupted; retry, or fetch the file manually.",
        )
    temporary.rename(destination)
    print(f"  verified SHA-256 {actual[:16]}... -> {destination}")


def download_models(output_dir: str | Path = "models", model_key: str = "nuscenes-ptv3-semseg") -> Path:
    output = Path(output_dir).expanduser().resolve()
    if model_key not in MODELS:
        raise InfraError(
            f"Unknown model '{model_key}'. Available: {', '.join(sorted(MODELS))}",
            "Use --model <name> (see configs/model.yaml for the documented models).",
        )
    entry = MODELS[model_key]
    print(f"Model: {model_key} - {entry['description']}")
    print(f"License: {entry['license']}")
    weight_target = output / entry.get(
        "weight_filename", f"{model_key}.pth"
    )
    config_target = output / entry.get(
        "config_filename", f"{model_key}_config.yaml"
    )
    _download(entry["weight_url"], weight_target, entry["weight_sha256"], "model weights")
    _download(entry["config_url"], config_target, entry.get("config_sha256"), "model config")
    manifest = {
        "model": model_key,
        "description": entry["description"],
        "weight": str(weight_target),
        "config": str(config_target),
        "source": entry["weight_url"].split("/resolve/")[0],
        "license": entry["license"],
        "sha256": entry["weight_sha256"] or _sha256(weight_target),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Done. Manifest: {output / 'manifest.json'}")
    return output


def main() -> int:
    output = sys.argv[1] if len(sys.argv) > 1 else "models"
    model = sys.argv[2] if len(sys.argv) > 2 else "nuscenes-ptv3-semseg"
    try:
        download_models(output, model)
    except InfraError as exc:
        print(exc.user_message(), file=sys.stderr)
        return 1
    return 0