#!/usr/bin/env python3
"""Generate a synthetic mobile-LiDAR test scene (TESTING ONLY).

Usage:
    python scripts/make_synthetic_las.py [OUTPUT.las]

Equivalent to: python -m infra_inventory demo-data --output data
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from infra_inventory.synthetic import build_synthetic_las  # noqa: E402

if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/mannford_synthetic.las")
    summary = build_synthetic_las(target)
    print(summary)
    print("\nSynthetic data is for TESTING ONLY - never present it as competition results.")