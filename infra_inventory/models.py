"""Core data model: assets, settings, run summaries, and shared types."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional, Tuple

from .las_reader import DEFAULT_CRS_FALLBACK

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
    # ALP assessment layer (Observation -> Interpretation -> Recommended Action).
    # Derived only from measured signals + confidence + QC flags; never guesses
    # an unmeasured cause. review_required routes the asset to human review.
    condition: Optional[str] = None          # GOOD | FAIR | POOR | REVIEW
    recommended_action: Optional[str] = None
    review_required: bool = False
    assessment_reasoning: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        # Fast serialization: ``dataclasses.asdict`` deep-copies every nested
        # list/dict recursively, which dominates export time on large runs
        # (35% of total pipeline time on a 1.1M-point profile). One-level
        # copies give the same isolation guarantees serialization consumers
        # need at a fraction of the cost. Field list is derived from the
        # dataclass so new fields are picked up automatically.
        result = {f.name: getattr(self, f.name) for f in fields(self)}
        for name, value in result.items():
            if isinstance(value, list):
                result[name] = list(value)
            elif isinstance(value, dict):
                result[name] = dict(value)
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
    # Decimated overview points for the browser: 120k points is plenty for a
    # digital-twin overview and keeps the JSON payload small enough to load
    # instantly. Full-resolution data is never sent to the browser.
    viewer_point_limit: int = 120_000
    # Hard cap on viewer-data.json bytes. The browser payload is a LOD overview
    # (voxel-downsampled cloud + slim asset records); if it would still exceed
    # this, the background cloud is thinned further so "Loading viewer data"
    # never blocks on multi-gigabyte inputs.
    viewer_payload_max_bytes: int = 9_000_000
    point_index_sample_limit: int = 256
    save_tiles: bool = True
    # Resume a crashed/interrupted run from the per-tile LAS files already on
    # disk (skips the streaming pass). Only ever honored when the tile manifest
    # exists and its input SHA-256 matches the input file - a new upload never
    # reuses stale tiles.
    resume_from_tiles: bool = False
    # Uniform input thinning (LAStools las2las -thin analogue): when > 0 and the
    # input exceeds this many points, every stride-th point is kept per chunk so
    # very large files (e.g. airborne 3DEP tiles) process with bounded memory.
    # 0 = process every point (recommended for mobile competition data).
    max_input_points: int = 0
    # Confidence below this routes an asset to human review instead of trusting
    # it downstream. Default 0.80 was measured by the calibration sweep in
    # scripts/calibrate_confidence.py (compact-object precision 1.00 / recall
    # 0.89 at this cut on the 400 m simulated ground truth), never intuition
    # (docs/ALP.md).
    review_confidence_threshold: float = 0.8

    # --- Backends ---
    # "geometry" (CPU, default) | "pointcept" (PTv3 prior) | "gemini" (LLM
    # class-validation booster; requires GEMINI_API_KEY)
    backend: str = "geometry"
    gemini_model: str = "gemini-3.6-flash"
    # Explicit key override (testing / per-run CLI use). None reads GEMINI_API_KEY.
    gemini_api_key: Optional[str] = None
    pointcept_root: Optional[str] = None
    pointcept_config: Optional[str] = None
    pointcept_weight: Optional[str] = None
    pointcept_class_names: Optional[str] = None  # JSON list, aligned with the config's taxonomy
    pointcept_num_gpus: int = 1
    # OpenPCSeg learned backend (MinkowskiNet/SPVCNN/Cylinder3D; Toronto-3D
    # fine-tuning). Requires a checkout with infer.py/train.py and a GPU.
    openpcseg_root: Optional[str] = None
    openpcseg_config: Optional[str] = None  # OpenPCSeg .yaml (tools/cfgs/...)
    openpcseg_weight: Optional[str] = None  # checkpoint .pth
    # Which documented upstream taxonomy -> asset mapping applies (checkpoint's
    # training set): "toronto3d" (fine-tuned, 9 ids) or "semantickitti"
    # (pretrained model-zoo weights, 20 ids).
    openpcseg_taxonomy: str = "toronto3d"
    openpcseg_num_gpus: int = 1
    roadmarking_command: Optional[str] = None
    roadmarking_config: Optional[str] = None

    # --- Learned component classifier (CPU prior) ---
    # Trained softmax classifier over measured component features, shipped at
    # configs/learned_classifier.json. Fills the `model` confidence factor on
    # runs without a Pointcept/OpenPCSeg prior; a Pointcept/OpenPCSeg prior
    # always wins when present. `learned_veto` additionally drops candidates
    # the classifier rejects with high confidence.
    learned_prior_enabled: bool = True
    learned_classifier_path: Optional[str] = None
    learned_veto: bool = True
    learned_veto_margin: float = 0.60  # P(assigned) below this ...
    learned_veto_runner_margin: float = 0.20  # ... AND runner-up above this
    # Training-data collection (writes <output>/training/components_*.jsonl).
    # Never a schema/API field: set programmatically by --collect-training.
    collect_training: bool = False

    # --- Validation ---
    strict_las14: bool = False
    # CRS fallback forced at load time when the LAS header carries no resolvable
    # CRS (the reader's override_srs analogue): the Trimble MX9 Mannford, OK
    # capture is NAD83(2011) / Oklahoma North, US survey feet = EPSG:6553
    # (docs/PROCEDURE.md §4). None keeps the old CRS_UNRESOLVED behaviour
    # (report unknown, never assume).
    crs_fallback: Optional[str] = DEFAULT_CRS_FALLBACK

    # --- Preprocessing ---
    ground_bin_m: float = 2.0
    ground_quantile: float = 0.15

    # --- Pavement ---
    pavement_height_m: float = 0.3
    min_pavement_points: int = 150
    marking_height_m: float = 0.35
    marking_resolution_m: float = 0.25
    marking_min_points: int = 20
    # Marking brightness cutoffs. The quantile alone is not the effective
    # threshold: the contrast floor (``x * median``) also gates, and a floor at
    # 2.5x median routinely landed the effective cut near the 95th percentile
    # on bright road surfaces, starving markings. 0.90 + a 2.0x floor keeps the
    # effective threshold in the 88th-90th percentile band (docs/ACCURACY_REPORT.md
    # §4a) while the floor still blocks uniformly-bright ground planes.
    marking_intensity_quantile: float = 0.90
    marking_brightness_quantile: float = 0.90
    marking_intensity_floor_multiplier: float = 2.0
    marking_brightness_floor_multiplier: float = 1.6
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
    conductor_min_length_m: float = 5.0
    conductor_max_width_m: float = 0.6
    conductor_max_height_extent_m: float = 3.5  # tolerant of pole-attached wires
    # Wires arrive as short per-tile pieces: 2-D tiles cut a 60 m span at every
    # boundary into slivers (a ~28 m-wide corridor crossing the y=0 band edge
    # leaves 5-12 m slivers), and mobile occlusion drops add more. Precision is
    # held by the cross-section / linearity / height-extent gates, so the
    # per-tile minimums stay low and the merge pass reconstructs full spans.
    conductor_min_points: int = 15
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
    guardrail_linear_aspect_ratio: float = 4.0
    guardrail_linear_overlap_m: float = 3.0
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
    # Overall dataset-confidence report (see infra_inventory/confidence.py);
    # populated at the end of process_las and embedded in run.json /
    # inventory.json / viewer-data.json / the summary report.
    confidence_report: Optional[Dict[str, Any]] = None