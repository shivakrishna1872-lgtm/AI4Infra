import { useState } from "react";
import type { Asset } from "../types";
import { CLASS_LABELS } from "../types";

interface Props {
  asset: Asset;
  onClose: () => void;
  onFocus: () => void;
  onToggleHidden: () => void;
  hidden: boolean;
}

const FACTOR_LABELS: Record<string, string> = {
  model: "Model confidence",
  geometry: "Geometry fit",
  support: "Point support",
  spatial_context: "Spatial context",
  class_consistency: "Class consistency",
};

function pct(value: number | null | undefined): string {
  if (value == null) return "—";
  return `${Math.round(value * 100)}%`;
}

export default function InspectPanel({ asset, onClose, onFocus, onToggleHidden, hidden }: Props) {
  const [showEvidence, setShowEvidence] = useState(true);
  const factors = asset.confidence_factors ?? {};
  const present = Object.entries(factors).filter(
    ([, v]) => v != null
  ) as [string, number][];
  const method = asset.detection_method || "geometry-v2";
  const model = asset.model_prior_class;

  return (
    <aside className="inspect">
      <div className="ip-head">
        <div>
          <div className="aid">{asset.asset_id}</div>
          <div className="cls">
            {CLASS_LABELS[asset.class] ?? asset.class}
            {asset.subclass ? ` · ${asset.subclass.replace(/_/g, " ")}` : ""}
          </div>
        </div>
        <button className="close" onClick={onClose}>✕</button>
      </div>

      <div className="ip-body">
        <div className="evidence">
          <h4>Confidence</h4>
          <div className="ev-final">
            <span className="lbl">Final</span>
            <span className="val">{pct(asset.confidence)}</span>
          </div>
        </div>

        {asset.condition && (
          <div className={`alp alp-${(asset.condition ?? "").toLowerCase()}`}>
            <div className="alp-row">
              <span className={`cond cond-${(asset.condition ?? "").toLowerCase()}`}>
                {asset.condition}
              </span>
              {asset.review_required && <span className="cond cond-review">REVIEW</span>}
            </div>
            {asset.recommended_action && <div className="alp-action">{asset.recommended_action}</div>}
            {asset.assessment_reasoning && (
              <div className="alp-reason">{asset.assessment_reasoning}</div>
            )}
          </div>
        )}

        <dl className="kv">
          <dt>Geometry</dt>
          <dd>
            H {asset.dimensions.height_m.toFixed(2)} m · W {asset.dimensions.width_m.toFixed(2)} m
            {asset.dimensions.length_m > asset.dimensions.width_m + 0.01
              ? ` · L ${asset.dimensions.length_m.toFixed(2)} m`
              : ""}
          </dd>
          <dt>Points</dt>
          <dd>{asset.point_count.toLocaleString()}</dd>
          <dt>Location</dt>
          <dd>
            X {asset.center.x.toFixed(2)}
            <br />
            Y {asset.center.y.toFixed(2)}
            <br />
            Z {asset.center.z.toFixed(2)} m
          </dd>
          <dt>CRS</dt>
          <dd>{asset.coordinate_reference_system ?? "CRS_UNRESOLVED"}</dd>
          <dt>Source</dt>
          <dd>
            {asset.source_run ? `Run ${asset.source_run}` : "Run —"} ·{" "}
            {asset.source_scanner ?? "Scanner —"} · {asset.source_tile}
          </dd>
          <dt>Detection</dt>
          <dd>{method}</dd>
          {model && (
            <>
              <dt>Model prior</dt>
              <dd>
                {model}
                {asset.model_confidence != null ? ` (${pct(asset.model_confidence)})` : ""}
              </dd>
            </>
          )}
          <dt>Version</dt>
          <dd>{asset.processing_version}</dd>
        </dl>

        {present.length > 0 && (
          <div className="evidence">
            <h4>AI evidence — why did AI think this?</h4>
            {present.map(([key, value]) => (
              <div className="ev-row" key={key}>
                <span className="ev-name">{FACTOR_LABELS[key] ?? key.replace(/_/g, " ")}</span>
                <span className="ev-bar">
                  <i style={{ width: `${Math.min(100, value * 100)}%` }} />
                </span>
                <span className="ev-val">{pct(value)}</span>
              </div>
            ))}
            <div className="ev-final">
              <span className="lbl">Final confidence</span>
              <span className="val">{pct(asset.confidence)}</span>
            </div>
          </div>
        )}

        <div className="evidence">
          <h4>Explanation</h4>
          <div style={{ fontSize: 11.5, lineHeight: 1.55, color: "var(--text-dim)" }}>
            {asset.confidence_explanation}
          </div>
        </div>

        {asset.qc_flags.length > 0 && (
          <div className="flags">
            {asset.qc_flags.map((flag) => (
              <span key={flag} className="flag">{flag}</span>
            ))}
          </div>
        )}

        <div className="ip-actions">
          <button className="btn" onClick={onFocus}>Focus</button>
          <button className="btn" onClick={() => setShowEvidence((s) => !s)}>
            {showEvidence ? "Hide AI evidence" : "Inspect AI evidence"}
          </button>
          <button className="btn" onClick={onToggleHidden}>
            {hidden ? "Show asset" : "Hide asset"}
          </button>
        </div>
      </div>
    </aside>
  );
}