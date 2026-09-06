"""Asset detectors.

Each module in this package turns measured point geometry into inventory
assets with an explicit detection method and transparent confidence factors.
The learned Pointcept prior (when present) only modulates confidence - geometry
decides whether an asset exists at all.
"""
from __future__ import annotations

from typing import List

from ..models import Asset
from .common import TileContext, assign_ids, build_asset  # noqa: F401
from .pavement import detect_pavement
from .safety import detect_safety
from .signs import detect_signs
from .utilities import detect_utilities


def detect_all(ctx: TileContext) -> List[Asset]:
    """Run every detector over one tile and return all candidate assets."""
    assets: List[Asset] = []
    assets.extend(detect_pavement(ctx))
    assets.extend(detect_utilities(ctx))
    assets.extend(detect_signs(ctx))
    assets.extend(detect_safety(ctx))
    return [asset for asset in assets if asset is not None]