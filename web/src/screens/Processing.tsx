import { useEffect, useRef, useState } from "react";
import type { JobStatus, Project } from "../types";
import { api } from "../api";

interface Props {
  project: Project;
  jobId: string;
  onDone: (project: Project) => void;
  onBack: () => void;
}

const STAGES: { key: string; label: string; optional?: boolean }[] = [
  { key: "validating", label: "LAS validation & metadata" },
  { key: "streaming", label: "Streaming read + spatial tiling" },
  { key: "pointcept", label: "Pointcept / PTv3 inference", optional: true },
  { key: "roadmarking", label: "Road marking extraction", optional: true },
  { key: "detecting", label: "Instance extraction & classification" },
  { key: "exporting", label: "Attribution, QC & inventory export" },
  { key: "done", label: "Processing complete" },
];

export default function Processing({ project, jobId, onDone, onBack }: Props) {
  const [job, setJob] = useState<JobStatus | null>(null);
  const [failed, setFailed] = useState<string | null>(null);
  const doneRef = useRef(false);

  useEffect(() => {
    let cancelled = false;
    const poll = async () => {
      try {
        const status = await api.job(jobId);
        if (cancelled) return;
        setJob(status);
        if (status.stage === "done" && !doneRef.current) {
          doneRef.current = true;
          window.setTimeout(() => onDone(project), 700);
        }
        if (status.stage === "error") {
          setFailed(status.message || "Processing failed");
        }
      } catch (err) {
        if (!cancelled) setFailed(err instanceof Error ? err.message : String(err));
      }
    };
    void poll();
    const timer = window.setInterval(() => void poll(), 650);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [jobId, project, onDone]);

  const currentIndex = STAGES.findIndex((s) => s.key === (job?.stage ?? "validating"));
  const seen = new Set<string>();
  if (job) {
    for (let i = 0; i <= Math.max(currentIndex, 0); i += 1) seen.add(STAGES[i].key);
  }
  const progress =
    currentIndex < 0 ? 0 : Math.min(1, (currentIndex + 1) / (STAGES.length - 1));

  return (
    <div className="processing">
      <header className="landing-top">
        <div className="brand">
          <strong>TERRA POINT</strong>
          <span>Processing</span>
        </div>
        <button className="btn ghost" onClick={onBack}>← Projects</button>
      </header>

      <div className="processing-body">
        <div className="proc-stages">
          <h1>PROCESSING LiDAR</h1>
          <div className="file">
            {project.simulated && <span className="sim-badge" style={{ marginRight: 10 }}>SIMULATION</span>}
            {project.input_file ?? project.name}
          </div>

          {failed ? (
            <div className="dz-error" style={{ margin: "20px 0" }}>
              Processing failed: {failed}
            </div>
          ) : (
            <div>
              {STAGES.map((stage) => {
                const index = STAGES.findIndex((s) => s.key === stage.key);
                const done = job != null && index < currentIndex;
                const active = job != null && index === currentIndex;
                const skipped = stage.optional && job != null && currentIndex > index && !seen.has(stage.key) && !done;
                return (
                  <div
                    key={stage.key}
                    className={`stage ${done ? "done" : ""} ${active ? "active" : ""} ${skipped ? "skipped" : ""}`}
                  >
                    <span className="ico">{done ? "✓" : skipped ? "—" : active ? "●" : ""}</span>
                    <span>{stage.label}</span>
                    {skipped && <span className="st-msg">skipped · optional backend</span>}
                    {active && <span className="st-msg">{job?.message ?? ""}</span>}
                    {done && <span className="st-msg">done</span>}
                  </div>
                );
              })}
            </div>
          )}
        </div>

        <div className="proc-stats">
          <h2>Pipeline status</h2>
          <div className="stat"><span className="k">Stage</span><span className="v">{job?.stage ?? "queued"}</span></div>
          <div className="stat"><span className="k">Points processed</span><span className="v">{(job?.points_processed ?? 0).toLocaleString()}</span></div>
          <div className="stat"><span className="k">Total points</span><span className="v">{(job?.point_count ?? project.point_count ?? 0).toLocaleString()}</span></div>
          <div className="stat"><span className="k">Tiles</span><span className="v">{job?.tiles_done ?? 0} / {job?.tiles_total ?? "—"}</span></div>
          <div className="stat"><span className="k">Assets detected</span><span className="v">{job?.assets ?? 0}</span></div>
          <div className="stat"><span className="k">Elapsed</span><span className="v">{(job?.elapsed_seconds ?? 0).toFixed(1)} s</span></div>
          <div className="stat"><span className="k">CRS</span><span className="v">{project.crs ?? "—"}</span></div>

          <div className="proc-bar"><i style={{ width: `${progress * 100}%` }} /></div>
          <div style={{ marginTop: 16, color: "var(--text-faint)", fontSize: 11, lineHeight: 1.6, fontFamily: "var(--mono)" }}>
            {job?.message || "Waiting for the worker…"}
          </div>
        </div>
      </div>
    </div>
  );
}