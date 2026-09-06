"""Core data model: assets, settings, run summaries, and shared types."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

#: (min_x, min_y, min_z, max_x, max_y, max_z)
Bounds = Tuple[float, float, float, float, float, float]

#: Short identifier prefixes per asset class (used to build asset IDs).
CLASS_PREFIXES: Dict[str, str] = {
    "pavement": "PAV",
    "pavement_marking": "MRK",
    "utility_pole": "POL",
    "overhead_conductor": "CON",
    "utility_cabinet": "CAB",
    "traffic_sign": "SGN",
    "guardrail": "GRD",
    "safety_barrier": "BAR",
    "rumble_strip": "RUM",
}

#: Valid (class, subclass) taxonomy. `None` subclass means "unspecified".
VALID_TAXONOMY: Dict[str, Tuple[Optional[str], ...]] = {
    "pavement": ("travelled_surface",),
    "pavement_marking": ("lane_line", "edge_line", "stop_line", "crosswalk", "symbol", "other"),
    "utility_pole": ("vertical_support",),
    "overhead_conductor": ("conductor",),
    "utility_cabinet": ("cabinet",),
    "traffic_sign": ("panel_with_support", "panel_only", "sign_support"),
    "guardrail": ("roadside_barrier",),
    "safety_barrier": ("concrete_barrier",),
    "rumble_strip": ("rumble_strip",),
}


@dataclass
class ConfidenceFactors:
    """Measured confidence inputs. Missing factors are ``None``, never guessed."""

    model: Optional[float] = None
    geometry: Optional[float] = None
    support: Optional[float] = None
    spatial_context: Optional[float] = None
    class_consistency: Optional[float] = None

    def to_dict(self) -> Dict[str, Optional[float]]:
        return asdict(self)


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
    source_point_source_id: Optional[int] = None
    model_prior_class: Optional[str] = None
    model_confidence: Optional[float] = None
    processing_version: str = "0.3.0"
    geometry: Optional[Dict[str, Any]] = None
    qc_flags: List[str] = field(default_factory=list)
    flagged: bool = False

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["class"] = result.pop("asset_class")
        return result


@dataclass
class ProcessingSettings:
    """Pipeline configuration. Every value has a sensible CPU-safe default.

    CLI flags override YAML config, which overrides these defaults.
    """

    # --- I/O and streaming ---
    chunk_size: int = 500_000
    tile_size_m: float = 40.0
    tile_overlap_m: float = 0.0  # 0 = non-overlapping tiles; instances on boundaries are QC-deduped
    viewer_point_limit: int = 250_000
    point_index_sample_limit: int = 256
    save_tiles: bool = True

    # --- Backends ---
    backend: str = "geometry"  # "geometry" | "pointcept"
    pointcept_root: Optional[str] = None
    pointcept_config: Optional[str] = None
    pointcept_weight: Optional[str] = None
    pointcept_class_names: Optional[str] = None  # JSON list, aligned with the config's taxonomy
    pointcept_num_gpus: int = 1
    roadmarking_command: Optional[str] = None
    roadmarking_config: Optional[str] = None

    # --- Validation ---
    strict_las14: bool = False

    # --- Preprocessing ---
    ground_bin_m: float = 2.0
    ground_quantile: float = 0.15

    # --- Pavement ---
    pavement_height_m: float = 0.3
    min_pavement_points: int = 150
    marking_height_m: float = 0.35
    marking_resolution_m: float = 0.25
    marking_min_points: int = 20
    marking_intensity_quantile: float = 0.86
    marking_brightness_quantile: float = 0.86
    lane_line_min_length_m: float = 2.0
    lane_line_max_width_m: float = 0.6
    stop_line_min_width_m: float = 1.0
    crosswalk_min_width_m: float = 2.5
    symbol_max_side_m: float = 1.5

    # --- Utilities ---
    pole_min_height_m: float = 2.5
    pole_max_footprint_m: float = 1.35
    pole_min_columnarity: float = 0.15
    pole_min_points: int = 40
    pole_resolution_m: float = 0.45
    conductor_min_height_m: float = 4.0
    conductor_min_length_m: float = 8.0
    conductor_max_width_m: float = 0.6
    conductor_max_height_extent_m: float = 3.5  # tolerant of pole-attached wires
    conductor_min_points: int = 60
    conductor_resolution_m: float = 0.5
    cabinet_min_height_m: float = 0.4
    cabinet_max_height_m: float = 3.0
    cabinet_min_side_m: float = 0.3
    cabinet_max_side_m: float = 3.5
    cabinet_min_points: int = 60
    cabinet_resolution_m: float = 0.3

    # --- Signs ---
    sign_min_height_m: float = 1.0
    sign_max_height_m: float = 8.0
    sign_min_points: int = 25
    sign_resolution_m: float = 0.3
    sign_min_planarity: float = 0.55
    sign_min_panel_area_m2: float = 0.04
    sign_max_panel_area_m2: float = 9.0
    sign_max_panel_side_m: float = 4.0
    sign_max_thickness_m: float = 0.35
    sign_support_search_m: float = 1.4

    # --- Safety ---
    guardrail_min_height_m: float = 0.3
    guardrail_max_height_m: float = 1.7
    guardrail_min_length_m: float = 6.0
    guardrail_max_width_m: float = 1.8
    guardrail_max_height_extent_m: float = 1.8
    guardrail_min_points: int = 60
    guardrail_resolution_m: float = 0.5
    barrier_min_length_m: float = 8.0
    barrier_min_width_m: float = 0.35
    barrier_max_width_m: float = 1.4
    barrier_min_height_m: float = 0.5
    barrier_max_height_m: float = 1.5
    rumble_min_length_m: float = 0.4
    rumble_max_length_m: float = 3.0
    rumble_min_width_m: float = 0.15
    rumble_max_width_m: float = 0.8
    rumble_min_band_points: int = 12
    rumble_min_bands: int = 3
    rumble_max_gap_m: float = 1.8

    # --- Confidence / QC ---
    min_asset_points: int = 25
    low_confidence_threshold: float = 0.45
    duplicate_distance_m: float = 1.5
    max_support_points: int = 4000

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "ProcessingSettings":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        extra = set(values) - known
        if extra:
            raise ValueError(f"Unknown processing settings: {sorted(extra)}")
        return cls(**{key: value for key, value in values.items() if key in known})


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
    tile_count: int = 0
    scanner_ids: List[int] = field(default_factory=list)
    run_count: int = 0
    elapsed_seconds: float = 0.0
    processing_version: str = "0.3.0"