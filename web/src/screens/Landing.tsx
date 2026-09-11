import { useCallback, useEffect, useRef, useState } from "react";
import type { Project } from "../types";
import { api } from "../api";

interface Props {
  onOpen: (project: Project, jobId: string | null) => void;
}

const TEAM = "ShivaSubscribers";

// The four required competition classes - each card lists exactly what the
// pipeline extracts for that category, with the class color used in the viewer.
const CATEGORY_CARDS = [
  {
    title: "Pavement",
    color: "#7d93b2",
    art: "pavement" as const,
    items: ["Travelled surface", "Painted pavement markings", "Lane lines · stop lines · crosswalks"],
    output: "Surface patches and individual markings, each with area, width, orientation and retro-reflectivity signal.",
  },
  {
    title: "Utilities",
    color: "#8f7fb5",
    art: "utilities" as const,
    items: ["Utility poles", "Overhead conductors", "Cabinets · junction boxes"],
    output: "Every pole, conductor span and cabinet as its own asset with height, lean, scanner and run attribution.",
  },
  {
    title: "Signs",
    color: "#c08a92",
    art: "signs" as const,
    items: ["Sign panels", "Structures carrying signs", "Regulatory & warning signs"],
    output: "Panels and their support structures with panel area, orientation and confidence — not just segmented points.",
  },
  {
    title: "Safety",
    color: "#7c8b6d",
    art: "safety" as const,
    items: ["Guardrails", "Barriers", "Rumble strips"],
    output: "Linear safety assets with measured length, continuity and ALP condition flags.",
  },
];

// Small hand-drawn line illustrations, one per asset class. Inline SVG means
// they always load (no external image host) and always match the topic.
function CatArt({ kind, color }: { kind: string; color: string }) {
  const common = {
    viewBox: "0 0 200 120",
    style: { color },
    fill: "none",
    stroke: "currentColor",
    strokeWidth: 3,
    strokeLinecap: "round" as const,
    strokeLinejoin: "round" as const,
    "aria-hidden": true,
  };
  if (kind === "pavement") {
    // Top-down road: edge lines, dashed centre line, crosswalk stripes.
    return (
      <svg {...common}>
        <line x1="52" y1="8" x2="52" y2="112" strokeOpacity="0.45" />
        <line x1="148" y1="8" x2="148" y2="112" strokeOpacity="0.45" />
        <line x1="100" y1="12" x2="100" y2="26" />
        <line x1="100" y1="36" x2="100" y2="50" />
        <line x1="100" y1="60" x2="100" y2="74" />
        <line x1="100" y1="84" x2="100" y2="98" />
        <line x1="64" y1="100" x2="136" y2="100" strokeOpacity="0.8" />
        <line x1="64" y1="106" x2="136" y2="106" strokeOpacity="0.8" />
      </svg>
    );
  }
  if (kind === "utilities") {
    // Side view: pole with crossarm, insulators and two sagging conductors.
    return (
      <svg {...common}>
        <line x1="62" y1="14" x2="62" y2="114" />
        <line x1="38" y1="26" x2="86" y2="26" />
        <circle cx="40" cy="30" r="2.6" fill="currentColor" stroke="none" />
        <circle cx="84" cy="30" r="2.6" fill="currentColor" stroke="none" />
        <path d="M40 34 C 70 66, 120 30, 194 20" strokeOpacity="0.85" />
        <path d="M84 34 C 110 62, 140 66, 196 58" strokeOpacity="0.85" />
        <line x1="8" y1="114" x2="196" y2="114" strokeOpacity="0.3" />
      </svg>
    );
  }
  if (kind === "signs") {
    // Side view: sign panel on its support, plus a smaller secondary panel.
    return (
      <svg {...common}>
        <rect x="46" y="12" width="84" height="38" rx="4" />
        <line x1="60" y1="24" x2="116" y2="24" strokeOpacity="0.7" />
        <line x1="60" y1="34" x2="96" y2="34" strokeOpacity="0.7" />
        <line x1="88" y1="50" x2="88" y2="116" />
        <line x1="8" y1="116" x2="196" y2="116" strokeOpacity="0.3" />
      </svg>
    );
  }
  // Safety: guardrail with posts and two horizontal rails.
  return (
    <svg {...common}>
      <line x1="8" y1="112" x2="196" y2="112" strokeOpacity="0.3" />
      <line x1="28" y1="66" x2="28" y2="112" />
      <line x1="72" y1="66" x2="72" y2="112" />
      <line x1="116" y1="66" x2="116" y2="112" />
      <line x1="160" y1="66" x2="160" y2="112" />
      <line x1="20" y1="70" x2="188" y2="70" />
      <line x1="20" y1="84" x2="188" y2="84" />
      <path d="M20 88 C 40 94, 60 94, 80 90" strokeOpacity="0.5" />
    </svg>
  );
}

const FLOW_STEPS = [
  { label: "LAS / LAZ", sub: "Raw point cloud" },
  { label: "Validate", sub: "LAS 1.4 · CRS" },
  { label: "Tile", sub: "Spatial tiles" },
  { label: "PTv3", sub: "Learned priors" },
  { label: "Markings", sub: "Vectorized lines" },
  { label: "Inventory", sub: "Individual assets" },
  { label: "3D twin", sub: "Inspect & export" },
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

  // Drop always into a fresh project so the browser does not attempt to start
  // a project name with a space-that-looks-like-a-dash and then fail to parse
  // the resulting URL path.
  const runFile = useCallback(
    (file: File) => {
      const raw = file.name.replace(/\.(las|laz)$/i, "");
      // Collapse any run of non-alphanumeric characters into a single dash,
      // strip leading/trailing separators — produces names like
      // mannford_run02_laserright instead of mangled ones with double spaces.
      const name = raw
        .trim()
        .replace(/[^A-Za-z0-9_-]+/g, "_")
        .replace(/^[-_]+|[-_]+$/g, "")
        .slice(0, 80) || "scan";

      setBusy(`Uploading ${file.name}…`);
      setError(null);
      setSubmitted(false);
      (async () => {
        try {
          const created = await api.createProject(name);
          await api.uploadLas(created.id, file, (p) => {
            const pct = p.totalBytes ? Math.round((p.uploadedBytes / p.totalBytes) * 100) : 0;
            setBusy(`Uploading ${file.name}… ${pct}% (part ${p.part}/${p.parts})`);
          });
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

  return (
    <div className="landing">
      {/* ---------- hero ---------- */}
      <div className="land-hero">
        <div className="land-top">
          <div className="brand">
            <span className="brand-mark" aria-hidden="true">AI</span>
            <div>
              <strong>TERRA POINT</strong>
              <span className="brand-sub">LiDAR Infrastructure Intelligence</span>
            </div>
          </div>
          {envCheck === "ok" && <span className="badge comp">SERVICES OK</span>}
        </div>

        <div className="land-hero-text">
          <Reveal>
            <div className="eyebrow">
              <span className="badge">ADVANCED TRACK</span>
              <span className="pipe">·</span>
              <span>ASSET MANAGEMENT CAPTURE MODEL</span>
            </div>
            <h1 className="hero-title">
              Infrastructure asset inventory,{" "}
              <span className="hero-em">straight from the point cloud.</span>
            </h1>
            <p className="hero-sub">
              A real <b>LiDAR asset inventory</b> system, not a mockup. Drop a{" "}
              <b className="mono">.las</b> or <b className="mono">.laz</b> file,
              run the pipeline, and inspect <b>individual</b> infrastructure
              assets — pavement, utilities, signs and safety — in a 3D digital
              twin with per-asset confidence and evidence.
            </p>
          </Reveal>
        </div>

        <Reveal threshold={0.2} delay={60}>
          <div className="land-cards">
            <div className="card">
              <div>Four required classes</div>
              <div className="card-sub">Pavement · Utilities · Signs · Safety</div>
            </div>
            <div className="card">
              <div>Individual assets</div>
              <div className="card-sub">Not just segmented points</div>
            </div>
            <div className="card">
              <div>Attribution &amp; confidence</div>
              <div className="card-sub">Real measurements, never guesses</div>
            </div>
          </div>
        </Reveal>

        <Reveal threshold={0.25} delay={100}>
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
                  <span>Didn&apos;t have the right file? Try the simulation instead.</span>
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

      {/* ---------- four asset classes ---------- */}
      <section className="land-classes">
        <Reveal>
          <div className="land-data-h">
            <h2 className="section-eyebrow">THE OBJECTIVE</h2>
            <h3>Four asset classes, extracted as individual inventory objects</h3>
            <p className="section-sub">
              Not a segmented point cloud — each category is detected as{" "}
              <b>individual assets</b> with a location, geometry, attributes and
              confidence. Toggle any category in the 3D viewer.
            </p>
          </div>
        </Reveal>
        <div className="cat-grid">
          {CATEGORY_CARDS.map((cat, i) => (
            <Reveal key={cat.title} threshold={0.2} delay={i * 60}>
              <div className="cat-card" style={{ "--cat": cat.color } as React.CSSProperties}>
                <div className="cat-img">
                  <CatArt kind={cat.art} color={cat.color} />
                  <span className="cat-chip">{cat.title}</span>
                </div>
                <ul className="cat-items">
                  {cat.items.map((item) => (
                    <li key={item}>{item}</li>
                  ))}
                </ul>
                <div className="cat-out">{cat.output}</div>
              </div>
            </Reveal>
          ))}
        </div>
      </section>

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
                  </a>
                </div>
              </div>
            </div>
            <div className="ds-row">
              <div className="ds-title">Important distinction</div>
              <div className="ds-body">
                <div>
                  USGS 3DEP is commonly <b>airborne</b> LiDAR. The competition
                  data is <b>vehicle-collected mobile LiDAR</b> (Mannford,
                  Oklahoma). Public data should be clearly labeled{" "}
                  <span className="sim-badge">DEVELOPMENT / EXTERNAL DATA</span>{" "}
                  and never presented as competition data.
                </div>
              </div>
            </div>
          </div>
        </Reveal>
      </section>

      {/* ---------- how it works ---------- */}
      <section className="land-demo">
        <Reveal>
          <div className="land-demo-h">
            <h2 className="section-eyebrow">HOW IT WORKS</h2>
            <h3>From point cloud to infrastructure inventory</h3>
          </div>
          <div className="demo-strip">
            {FLOW_STEPS.map((s, i) => (
              <Reveal key={s.label} threshold={0.5} delay={i * 50}>
                <div className="demo-step">
                  <span className="step-no">{String(i + 1).padStart(2, "0")}</span>
                  <h3>{s.label}</h3>
                  <p>{s.sub}</p>
                </div>
              </Reveal>
            ))}
          </div>
        </Reveal>
      </section>

      {/* ---------- previous projects ---------- */}
      <section className="projects">
        <Reveal>
          <div className="land-data-h">
            <h2 className="section-eyebrow">PROJECTS</h2>
            <h3>Processed scans and simulations</h3>
            <p className="section-sub">
              Each scan is archived here until you finish viewing and close the
              tab — finished scans are released so storage stays free.
            </p>
          </div>
          {projects.length === 0 ? (
            <div className="empty-projects">
              No projects yet. Upload a dataset or open the simulation.
            </div>
          ) : (
            <div className="project-list">
              {projects.map((p) => (
                <button
                  key={p.id}
                  className="project-row"
                  onClick={() => void openExisting(p)}
                >
                  <div className="pname">
                    {p.simulated ? (
                      <span className="sim-badge" style={{ marginRight: 8 }}>
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
                    {p.processed && p.confidence_percent != null
                      ? ` · ${p.confidence_percent}% ${p.confidence_grade?.split(" ")[0] ?? ""}`
                      : ""}
                    {p.processed ? " · processed" : " · pending"}
                  </div>
                </button>
              ))}
            </div>
          )}
        </Reveal>
      </section>

      <footer className="land-foot">
        <div>TERRA POINT · Advanced Track Asset Management Capture Model</div>
        <div className="foot-team">Team {TEAM}</div>
        <div className="foot-meta">
          Real LiDAR processing · individual asset inventory · 3D inspection
        </div>
      </footer>
    </div>
  );
}