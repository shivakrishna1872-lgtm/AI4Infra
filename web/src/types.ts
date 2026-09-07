export interface RunInfo {
  input_path: string;
  point_count: number;
  bounds: number[];
  crs: string | null;
  las_version: string;
  point_format: number;
  scanner_ids: number[];
  run_count: number;
  tile_count: number;
  elapsed_seconds: number;
  warnings: string[];
  backend: Record<string, unknown>;
  processing_version: string;
}

export interface Asset {
  asset_id: string;
  class: string;
  subclass: string | null;
  center: { x: number; y: number; z: number };
  bounding_box: number[];
  dimensions: { length_m: number; width_m: number; height_m: number };
  point_count: number;
  source_tile: string;
  source_point_indices_sample: number[];
  coordinate_reference_system: string | null;
  confidence: number;
  confidence_factors: Record<string, number | null>;
  confidence_explanation: string;
  detection_method: string;
  intensity_stats: Record<string, number> | null;
  rgb_stats: Record<string, number> | null;
  orientation_deg: number | null;
  source_run: string | null;
  source_scanner: string | null;
  source_point_source_id: number | null;
  model_prior_class: string | null;
  model_confidence: number | null;
  processing_version: string;
  geometry: {
    ground_elevation_m?: number;
    eigen_planarity?: number;
    eigen_linearity?: number;
    eigen_verticality?: number;
    highlight_points?: number[][];
  } | null;
  qc_flags: string[];
  flagged: boolean;
  // ALP assessment layer (observation -> interpretation -> recommended action)
  condition?: string | null;
  recommended_action?: string | null;
  review_required?: boolean;
  assessment_reasoning?: string | null;
}

export interface ViewerData {
  run: RunInfo;
  assets: Asset[];
  points: number[][];
  point_colors: number[][];
  point_rgb?: number[][];
  point_intensity?: number[];
  point_class?: number[];
  point_class_names?: string[];
  /** When the viewer loaded from a streaming tile-space package rather than a pipeline run. */
  tile_space?: boolean;
  /** Tile index for lazy tile loading in a future viewer update. */
  manifest?: {
    version: string;
    tile_size_m: number;
    bounds: number[];
    overview: {
      point_count: number;
      bounds: number[];
      file: string;
      color_mode: string;
    };
    tiles: {
      file: string;
      tx: number;
      ty: number;
      point_count: number;
      bounds: number[];
      resolution_m: number;
    }[];
    tile_file_prefix: string;
  };
  overview_point_count?: number;
}

export interface Project {
  id: string;
  name: string;
  created_at: string;
  input_file: string | null;
  input_size_bytes?: number;
  simulated: boolean;
  processed: boolean;
  point_count: number | null;
  crs: string | null;
  las_version: string | null;
  point_format: number | null;
  asset_count: number | null;
  summary: Record<string, unknown> | null;
  /** When the streaming tile-space package has been built for this project. */
  scene?: {
    tile_space?: string;
  } | null;
}

export interface JobStatus {
  id: string;
  project_id: string;
  stage: string;
  message: string;
  points_processed: number;
  point_count: number;
  tiles_done: number;
  tiles_total: number;
  assets: number;
  elapsed_seconds: number;
  updated_at?: number;
}

export type ViewMode = "raw" | "detection" | "inventory";
export type ColorMode = "elevation" | "rgb" | "intensity" | "classification";

export const CLASS_COLORS: Record<string, string> = {
  pavement: "#4d7cff",
  pavement_marking: "#f5cf58",
  utility_pole: "#b082f7",
  overhead_conductor: "#9aa7ff",
  utility_cabinet: "#c98af5",
  traffic_sign: "#ff7e9d",
  guardrail: "#62e8b9",
  safety_barrier: "#3dd6a8",
  rumble_strip: "#ffb46b",
};

export const CLASS_LABELS: Record<string, string> = {
  pavement: "Pavement",
  pavement_marking: "Pavement marking",
  utility_pole: "Utility pole",
  overhead_conductor: "Overhead conductor",
  utility_cabinet: "Utility cabinet",
  traffic_sign: "Traffic sign",
  guardrail: "Guardrail",
  safety_barrier: "Safety barrier",
  rumble_strip: "Rumble strip",
};