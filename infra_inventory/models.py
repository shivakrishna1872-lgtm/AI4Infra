from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


Bounds = Tuple[float, float, float, float, float, float]


@dataclass
class Asset:
    asset_id: str
    asset_class: str
    subclass: Optional[str]
    center: Dict[str, float]
    bounding_box: Bounds
    dimensions: Dict[str, float]
    point_count: int
    source_tile: str
    source_point_indices_sample: List[int]
    coordinate_reference_system: Optional[str]
    confidence: float
    confidence_factors: Dict[str, Optional[float]]
    confidence_explanation: str
    detection_method: str
    intensity_stats: Optional[Dict[str, float]] = None
    rgb_stats: Optional[Dict[str, float]] = None
    orientation_deg: Optional[float] = None
    source_run: Optional[str] = None
    source_scanner: Optional[str] = None
    processing_version: str = "0.1.0"
    geometry: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["class"] = result.pop("asset_class")
        return result


@dataclass
class ProcessingSettings:
    chunk_size: int = 500_000
    tile_size_m: float = 40.0
    viewer_point_limit: int = 120_000
    point_index_sample_limit: int = 128
    min_pavement_points: int = 150
    backend: str = "geometry"
    pointcept_root: Optional[str] = None
    pointcept_config: Optional[str] = None
    pointcept_weight: Optional[str] = None
    pointcept_num_gpus: int = 1
    roadmarking_command: Optional[str] = None
    strict_las14: bool = False


@dataclass
class RunSummary:
    input_path: str
    point_count: int
    bounds: Bounds
    crs: Optional[str]
    las_version: str
    point_format: int
    assets: List[Asset] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    backend: Dict[str, Any] = field(default_factory=dict)
