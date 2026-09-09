import { useState } from "react";
import type { ConfidenceComponent, ConfidenceReport } from "../types";

interface Props {
  confidence: ConfidenceReport;
}

const COMPONENT_LABELS: Record<string, string> = {
  density_coverage: "Density & coverage",
  crs: "Spatial reference (CRS)",
  intensity_classification: "Intensity & classification",
  geometry_fit: "Geometric fit",
};

function gradeColor(percent: number): string {
  if (percent >= 90) return "#2f9e62";
  if (percent >= 75) return "#c98a1b";
  return "#c14b3e";
}

/**
 * Dashboard card for the overall dataset confidence: score, grade, and the
 * four-way weighted breakdown (density, CRS, intensity, geometry).
 */
export default function ConfidenceCard({ confidence }: Props) {
  const [open, setOpen] = useState(false);
  const percent = confidence.overall_percent;
  const color = gradeColor(percent);
  const components: Record<string, ConfidenceComponent> = confidence.components ?? {};

  return (
    <div className="conf-card" style={{ borderColor: color }}>
      <div className="conf-top" onClick={() => setOpen((v) => !v)} title="Toggle breakdown">
        <div
          className="conf-ring"
          style={{
            background: `conic-gradient(${color} ${percent * 3.6}deg, var(--border-soft) 0deg)`,
          }}
        >
          <div className="conf-ring-inner">
            <b style={{ color }}>{percent}%</b>
          </div>
        </div>
        <div className="conf-meta">
          <div className="conf-eyebrow">OVERALL CONFIDENCE</div>
          <div className="conf-grade" style={{ color }}>
            {confidence.grade}
          </div>
          <div className="conf-hint">{open ? "Hide breakdown" : "Show breakdown"}</div>
        </div>
      </div>

      {open && (
        <div className="conf-breakdown">
          {Object.entries(COMPONENT_LABELS).map(([key, label]) => {
            const component = components[key];
            if (!component) return null;
            return (
              <div className="conf-row" key={key}>
                <div className="conf-row-head">
                  <span className="conf-row-label">{label}</span>
                  <span className="conf-row-value" style={{ color: gradeColor(component.percent) }}>
                    {component.percent}%
                    <span className="conf-row-weight">· w {Math.round(component.weight * 100)}%</span>
                  </span>
                </div>
                <div className="conf-track">
                  <div
                    className="conf-fill"
                    style={{
                      width: `${component.percent}%`,
                      background: gradeColor(component.percent),
                    }}
                  />
                </div>
                <div className="conf-detail">{component.detail}</div>
              </div>
            );
          })}
          {confidence.notes?.map((note, i) => (
            <div className="conf-note" key={i}>
              {note}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}