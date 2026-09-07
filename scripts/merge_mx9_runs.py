#!/usr/bin/env python3
"""Merge-clean the four Trimble MX9 mobile-mapping files into one LAS/LAZ.

Competition dataset layout (Mannford, Oklahoma — Trimble MX9, dual scanner
heads, two passes):

    Run 1 Laser Left   Run 1 Laser Right
    Run 2 Laser Left   Run 2 Laser Right

Merging first means every physical object is seen once — both heads see the
same pole and both runs re-cover the same pavement, so raw concatenation
would duplicate assets and inflate point support. This script performs the
streaming merge-clean described in ``infra_inventory/merge.py``:

* bounded-memory streaming append (never loads the whole cloud),
* cell-based duplicate removal (LAStools lasmerge -dup analogue; default
  5 cm cell, GPS-time aware),
* per-file ``point_source_id`` stamping (1..4) so the pipeline can report
  which scanner stream each asset came from; run separation survives through
  ``gps_time`` (the pipeline splits runs by GPS k-means),
* LAS 1.4 / point format 7 output (the competition format) with lazrs for
  ``.laz`` outputs.

Usage:
    python scripts/merge_mx9_runs.py \
        run1_left.las run1_right.las run2_left.las run2_right.las \
        --output mannford_merged.laz --cell 0.05
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from infra_inventory.merge import merge_las_files  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", help="LAS/LAZ inputs (order = scanner id order)")
    parser.add_argument("--output", "-o", required=True, help="Merged output (.las or .laz)")
    parser.add_argument("--cell", type=float, default=0.05,
                        help="Dedupe cell size in metres (0 disables duplicate removal)")
    parser.add_argument("--keep-scanner-ids", action="store_true",
                        help="Preserve original point_source_id values instead of stamping 1..N per input")
    parser.add_argument("--chunk-size", type=int, default=500_000, help="Streaming chunk size (points)")
    parser.add_argument("--max-dedupe-keys", type=int, default=25_000_000,
                        help="Cap on distinct dedupe cells before dedupe saturates (large merges)")
    args = parser.parse_args()

    if len(args.inputs) < 2:
        parser.error("Provide at least two input files to merge")
    for path in args.inputs:
        if not Path(path).is_file():
            parser.error(f"Input file not found: {path}")

    summary = merge_las_files(
        args.inputs, args.output,
        dedupe_cell_m=args.cell,
        assign_scanner_ids=not args.keep_scanner_ids,
        chunk_size=args.chunk_size,
        max_dedupe_keys=args.max_dedupe_keys,
    )
    print(json.dumps(summary, indent=2))
    for warning in summary["warnings"]:
        print(f"warning: {warning}", file=sys.stderr)
    if args.cell > 0:
        print(f"Merged {summary['points_read']:,} -> {summary['points_kept']:,} points "
              f"({summary['duplicates_removed']:,} duplicates removed at {args.cell:.2f} m cell).")
    else:
        print(f"Concatenated {summary['points_kept']:,} points (duplicate removal disabled).")
    print(f"Next: python -m infra_inventory process {args.output} --output out/mannford")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())