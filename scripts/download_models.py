#!/usr/bin/env python3
"""Download documented pretrained model weights.

Usage:
    python scripts/download_models.py [OUTPUT_DIR] [MODEL_KEY]

Example:
    python scripts/download_models.py models nuscenes-ptv3-semseg

Equivalent to: python -m infra_inventory download-models --output models
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from infra_inventory.download_models import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())