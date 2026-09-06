import { useCallback, useEffect, useRef, useState } from "react";
import type { Project } from "../types";
import { api } from "../api";

interface Props {
  onOpen: (project: Project, jobId: string | null) => void;
}

const FLOW_STEPS = [
  { label: "LAS / LAZ", sub: "Raw point cloud" },
  { label: "VALIDATE", sub: "LAS 1.4 · format · CRS" },
  { label: "TILE", sub: "Spatial tiles" },
  { label: "PTv3", sub: "Optional learned backend" },
  { label: "ROADMARKING", sub: "Marking vectorization" },
  { label: "INVENTORY", sub: "Individual assets" },
  { label: "3D TWIN", sub: "Inspect & export" },
];

function formatBytes(bytes: number): string {
  if (bytes >= 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${bytes} B`;
}

function useReveal(threshold = 0.12) {
  const ref = useRef<HTMLDivElement>(null);
  const [visible, setVisible] = useState(false);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setVisible(true);
          observer.unobserve(el);
        }
      },
      { threshold }
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, [threshold]);
  return { ref, visible };
}

function Reveal({
  children,
  className,
  threshold,
  delay,
}: {
  children: React.ReactNode;
  className?: string;
  threshold?: number;
  delay?: number;
}) {
  const { ref, visible } = useReveal(threshold);
  return (
    <div
      ref={ref}
      className={`reveal ${visible ? "in" : ""} ${className ?? ""}`}
      style={delay ? { transitionDelay: `${delay}ms` } : undefined}
    >
      {children}
    </div>
  );
}

export default function Landing({ onOpen }: Props) {
  const [projects, setProjects] = useState<Project[]>([]);
  const [drag, setDrag] = useState(false);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [envCheck, setEnvCheck] = useState<"checking" | "ok" | "no">("checking");
  const [submitted, setSubmitted] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  const refresh = useCallback(async () => {
    try {
      setProjects(await api.listProjects());
    } catch {
      /* server starting */
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  useEffect(() => {
    (async () => {
      try {
        const health = await api.health();
        setEnvCheck(health?.status === "ok" ? "ok" : "no");
      } catch {
        setEnvCheck("no");
      }
    })();
  }, []);

  const runFile = useCallback(
    (file: File) => {
      const name = file.name.replace(/\.(las|laz)$/i, "");
      setBusy(`Uploading ${file.name}…`);
      setError(null);
      setSubmitted(false);
      (async () => {
        try {
          const created = await api.createProject(name);
          await api.uploadLas(created.id, file);
          setBusy("Starting pipeline…");
          const { job_id } = await api.process(created.id);
          onOpen(created, job_id);
        } catch (err) {
          setError(err instanceof Error ? err.message : String(err));
          setBusy(null);
          setSubmitted(true);
        }
      })();
    },
    [onOpen]
  );

  const runSimulation = useCallback(() => {
    setBusy("Generating simulated LiDAR corridor…");
    setError(null);
    setSubmitted(false);
    (async () => {
      try {
        const project = await api.simulate();
        // /api/simulate already runs the full pipeline and stages the outputs,
        // so only start a background job when the project is not yet processed.
        if (project.processed) {
          onOpen(project, null);
          return;
        }
        setBusy("Starting pipeline…");
        const { job_id } = await api.process(project.id);
        onOpen(project, job_id);
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
        setBusy(null);
        setSubmitted(true);
      }
    })();
  }, [onOpen]);

  const openExisting = useCallback(
    (project: Project) => {
      setError(null);
      if (project.processed) {
        onOpen(project, null);
        return;
      }
      setBusy(`Processing ${project.name}…`);
      (async () => {
        try {
          const { job_id } = await api.process(project.id);
          onOpen(project, job_id);
        } catch (err) {
          setError(err instanceof Error ? err.message : String(err));
          setBusy(null);
          setSubmitted(true);
        }
      })();
    },
    [onOpen]
  );

  return (
    <div className="landing">
      {/* ---------- hero ---------- */}
      <div className="land-hero">
        <div className="land-top">
          <div className="brand">
            <span className="brand-mark" aria-hidden="true" />
            <div>
              <strong>AI4INFRA</strong>
              <span className="brand-sub">LiDAR Infrastructure Intelligence</span>
            </div>
          </div>
          <div className="flow-strip">
            {FLOW_STEPS.map((s, i) => (
              <span key={s.label} className="flow-step">
                <span className="flow-label">{s.label}</span>
                {i < FLOW_STEPS.length - 1 && <span className="flow-arrow" />}
              </span>
            ))}
          </div>
        </div>

        <div className="land-hero-gfx" aria-hidden="true">
          <div className="gfx-cloud" />
          <div className="gfx-grain" />
        </div>

        <div className="land-hero-text">
          <Reveal>
            <div className="eyebrow">
              <span className="badge">ADVANCED TRACK</span>
              <span className="pipe">·</span>
              <span>ASSET MANAGEMENT CAPTURE MODEL</span>
            </div>
            <h1 className="hero-title">
              INFRASTRUCTURE{" "}
              <span className="hero-em">
                <img
                  src="https://images.unsplash.com/photo-1558618666-fcd25c85f82e?auto=format&fit=crop&w=220&q=80"
                  className="hero-img"
                  alt=""
                />
              </span>
            </h1>
            <p className="hero-sub">
              A real{" "}
              <b>
                <span className="hero-em-text">LiDAR asset inventory</span>
              </b>{" "}
              system, not a mockup. Upload a{" "}
              <b className="mono">.las</b>{" "}
              or{" "}
              <b className="mono">.laz</b>{" "}
              point cloud, run the pipeline, and inspect{" "}
              <b>individual</b>{" "}
              infrastructure assets — pavement, utilities, signs and safety — in a
              3D digital twin with per-asset confidence and evidence.
            </p>
          </Reveal>
        </div>

        <div className="land-cards">
          <Reveal threshold={0.25} delay={80}>
            <div className="card">
              <div className="card-icon">
                <span className="ic cloud" />
              </div>
              <div>
                <div>Four required classes</div>
                <div className="card-sub">Pavement · Utilities · Signs · Safety</div>
              </div>
            </div>
          </Reveal>
          <Reveal threshold={0.25} delay={160}>
            <div className="card">
              <div className="card-icon">
                <span className="ic assets" />
              </div>
              <div>
                <div>Individual assets</div>
                <div className="card-sub">Not just segmented points</div>
              </div>
            </div>
          </Reveal>
          <Reveal threshold={0.25} delay={240}>
            <div className="card">
              <div className="card-icon">
                <span className="ic crend" />
              </div>
              <div>
                <div>Attribution &amp; confidence</div>
                <div className="card-sub">Real measurements, not guesses</div>
              </div>
            </div>
          </Reveal>
        </div>

        <Reveal threshold={0.28} delay={120}>
          <div className="land-cta">
            <div
              className={`dropzone ${drag ? "drag" : ""}`}
              onClick={() => fileRef.current?.click()}
              onDragOver={(e) => {
                e.preventDefault();
                setDrag(true);
              }}
              onDragLeave={() => setDrag(false)}
              onDrop={(e) => {
                e.preventDefault();
                setDrag(false);
                const file = e.dataTransfer?.files?.[0];
                if (file) void runFile(file);
              }}
            >
              <div className="cta-glow" aria-hidden="true" />
              <div className="dz-label">DROP LAS DATASET</div>
              <div className="dz-sub">
                {busy
                  ? busy
                  : submitted
                  ? "Upload failed — try again or use the simulation"
                  : "Mannford_Run02_LaserRight.las · LAS 1.4 · Point Format 7"}
              </div>
              <div className="dz-formats">
                <span className="fmt-tag">.LAS</span>
                <span className="fmt-dot" aria-hidden="true" />
                <span className="fmt-tag">.LAZ</span>
                <span className="fmt-dot" aria-hidden="true" />
                <span className="fmt-tag">LAS 1.4</span>
                <span className="fmt-dot" aria-hidden="true" />
                <span className="fmt-tag">PF 7</span>
              </div>
              {error && (
                <div className="dz-error">
                  <span className="err-badge">UPLOAD ERROR</span>
                  {error}
                </div>
              )}
              {submitted && !error && (
                <div className="dz-try-else">
                  <span className="sim-badge">SIMULATION MODE</span>
                  <span>Didn't have the right file? Try the simulation instead.</span>
                </div>
              )}
            </div>
            <input
              ref={fileRef}
              type="file"
              accept=".las,.laz"
              hidden
              onChange={(e) => {
                const file = e.target?.files?.[0];
                if (file) void runFile(file);
                e.target.value = "";
              }}
            />
            <div className="land-cta-row">
              <button className="btn primary" disabled={!!busy} onClick={() => void runSimulation()}>
                <span className="btn-arr" aria-hidden="true" />
                Open Simulation
              </button>
              <div className="sim-note">
                <span className="sim-badge">SIMULATION MODE</span>
                <span>No upload required — opens a synthetic corridor</span>
              </div>
            </div>
          </div>
        </Reveal>
      </div>

      {/* ---------- data section ---------- */}
      <section className="land-data">
        <Reveal>
          <div className="land-data-h">
            <h2 className="section-eyebrow">SUPPORTED DATA</h2>
            <h3>Upload the right file</h3>
          </div>
          <div className="data-grid">
            <div className="data-col">
              <div className="data-col-h">Primary input</div>
              <div className="fmt">
                <span className="fmt-tag">.LAS</span>
                <span className="fmt-desc">/ </span>
                <span className="fmt-tag">.LAZ</span>
              </div>
              <div className="fmt-desc">
                3D point-cloud datasets. The application extracts infrastructure
                assets from the point cloud.
              </div>
            </div>
            <div className="data-col">
              <div className="data-col-h">Competition format</div>
              <div className="fmt">
                <span className="fmt-tag">LAS 1.4</span>
                <span className="fmt-desc"> / </span>
                <span className="fmt-tag">Point Format 7</span>
              </div>
              <div className="fmt-desc">
                Trimble MX9 mobile LiDAR. X, Y, Z, intensity, returns, RGB when
                present, GPS time, scanner ID.
              </div>
            </div>
            <div className="data-col">
              <div className="data-col-h">Development data</div>
              <div className="fmt">
                <span className="fmt-tag">USGS 3DEP</span>
              </div>
              <div className="fmt-desc">
                Public LiDAR from The National Map. Labeled{" "}
                <b>DEVELOPMENT / EXTERNAL DATA</b> — usually airborne, not the
                competition&apos;s vehicle-mounted mobile LiDAR.
              </div>
            </div>
          </div>
        </Reveal>

        <Reveal threshold={0.28}>
          <div className="land-data-h">
            <h2 className="section-eyebrow">WHERE TO GET DATA</h2>
            <h3>Real LiDAR for testing and development</h3>
          </div>
          <div className="data-source">
            <div className="ds-row">
              <div className="ds-title">Recommended source</div>
              <div className="ds-body">
                <div>
                  <b>USGS 3DEP</b> — The National Map. Download a LiDAR point
                  cloud, obtain the LAS/LAZ, upload it here, process it, and
                  inspect the extracted assets.
                </div>
                <div className="ds-link">
                  <a
                    href="https://www.usgs.gov/3d-elevation-program"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    usgs.gov/3d-elevation-program
                    <i className="ext" />
                    <span className="ext-line" />
                  </a>
                </div>
              </div>
            </div>
            <div className="ds-row">
              <div className="ds-title">Important distinction</div>
              <div className="ds-body">
                <div>
                  USGS 3DEP is commonly{" "}
                  <b>airborne</b>{" "}
                  LiDAR. The competition data is{" "}
                  <b>vehicle-collected mobile LiDAR</b>{" "}
                  (Mannford, Oklahoma). Public data should be clearly labeled{" "}
                  <span className="sim-badge">DEVELOPMENT / EXTERNAL DATA</span>{" "}
                  and never presented as competition data.
                </div>
              </div>
            </div>
          </div>
        </Reveal>
      </section>

      {/* ---------- how it looks ---------- */}
      <section className="land-demo">
        <Reveal>
          <div className="land-demo-h">
            <h2 className="section-eyebrow">HOW IT LOOKS</h2>
            <h3>From point cloud to infrastructure inventory</h3>
          </div>
          <div className="demo-strip">
            {FLOW_STEPS.map((s, i) => (
              <Reveal key={s.label} threshold={0.5} delay={i * 60}>
                <div className="demo-step">
                  <span className="step-no">{String(i + 1).padStart(2, "0")}</span>
                  <div>
                    <h3>{s.label}</h3>
                    <p>{s.sub}</p>
                  </div>
                </div>
              </Reveal>
            ))}
          </div>
        </Reveal>
      </section>

      {/* ---------- previous projects ---------- */}
      <section className="projects">
        <Reveal>
          <h2>Previous projects</h2>
          {projects.length === 0 && (
            <div className="empty-projects">
              No projects yet. Upload a dataset or open the simulation.
            </div>
          )}
          {projects.map((p) => (
            <div
              key={p.id}
              className="project-row"
              onClick={() => void openExisting(p)}
            >
              <div className="pname">
                {p.simulated ? (
                  <span
                    className="sim-badge"
                    style={{ marginRight: 8 }}
                  >
                    SIM
                  </span>
                ) : p.point_count && p.point_count > 0 ? (
                  <span className="badge dev">DEV</span>
                ) : (
                  <span className="badge comp">COMP</span>
                )}
                {p.name}
              </div>
              <div className="pmeta">
                {p.input_file ? formatBytes(p.input_size_bytes ?? 0) : ""}
                {p.input_file ? ` · ${p.input_file}` : ""}
                {p.point_count ? ` · ${p.point_count.toLocaleString()} pts` : ""}
                {p.asset_count != null ? ` · ${p.asset_count} assets` : ""}
                {p.processed ? " · processed" : " · pending"}
              </div>
            </div>
          ))}
        </Reveal>
      </section>

      <footer className="land-foot">
        <div>AI4INFRA · Advanced Track Asset Management Capture Model</div>
        <div className="foot-meta">
          Real LiDAR processing · individual asset inventory · 3D inspection
        </div>
      </footer>
    </div>
  );
}
