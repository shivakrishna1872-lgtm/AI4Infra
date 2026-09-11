import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import type { Asset, ColorMode, Project, ViewMode, ViewerData } from "../types";
import { CLASS_COLORS, CLASS_LABELS } from "../types";
import { api } from "../api";
import Scene3D, { type Fly } from "../scene/Scene3D";
import InspectPanel from "../components/InspectPanel";
import LayersPanel from "../components/LayersPanel";
import ConfidenceCard from "../components/ConfidenceCard";

interface Props {
  project: Project;
  onBack: () => void;
}

const EXPORTS = [
  { kind: "json", label: "JSON" },
  { kind: "csv", label: "CSV" },
  { kind: "geojson", label: "GEOJSON" },
  { kind: "inventory", label: "INVENTORY" },
  { kind: "report", label: "REPORT (.md)" },
];

export default function Viewer({ project, onBack }: Props) {
  const [data, setData] = useState<ViewerData | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  const [mode, setMode] = useState<ViewMode>("raw");
  const [colorMode, setColorMode] = useState<ColorMode>("elevation");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [hiddenIds, setHiddenIds] = useState<Set<string>>(new Set());
  const [visibleClasses, setVisibleClasses] = useState<Set<string> | null>(null);
  const [showLabels, setShowLabels] = useState(false);
  const [showGrid, setShowGrid] = useState(false);
  const [showBBoxes, setShowBBoxes] = useState(false);
  const [compare, setCompare] = useState<{ active: boolean; split: number }>({ active: false, split: 0.5 });
  const [measuring, setMeasuring] = useState(false);
  const [measurePoints, setMeasurePoints] = useState<[number, number, number][]>([]);
  const [pointSize, setPointSize] = useState(1);
  const [search, setSearch] = useState("");
  const [downloading, setDownloading] = useState<string | null>(null);
  const [downloadError, setDownloadError] = useState<string | null>(null);
  const [flySig, setFlySig] = useState(0);
  const [presetKind, setPresetKind] = useState<"reset" | "top" | "side">("reset");
  const [presetSig, setPresetSig] = useState(0);
  const [dragging, setDragging] = useState(false);
  const wrapRef = useRef<HTMLDivElement>(null);
  const [loadKey, setLoadKey] = useState(0);

  useEffect(() => {
    let cancelled = false;
    setLoadError(null);
    api
      .viewerData(project.id)
      .then((d) => {
        if (cancelled) return;
        setData(d);
        setVisibleClasses(new Set(d.assets.map((a) => a.class)));
      })
      .catch((err) => {
        if (!cancelled) setLoadError(err instanceof Error ? err.message : String(err));
      });
    return () => {
      cancelled = true;
    };
  }, [project.id, loadKey]);

  const hasRgb = !!(data && data.point_rgb && data.point_rgb.length > 0);
  const hasClass = !!(data && data.point_class && data.point_class.length > 0);

  useEffect(() => {
    if (mode === "detection" && colorMode === "elevation") setColorMode("classification");
    // Never leave the viewer on a color mode the dataset can't provide.
    if (colorMode === "rgb" && !hasRgb) setColorMode("elevation");
    if (colorMode === "classification" && !hasClass) setColorMode("elevation");
  }, [mode, colorMode, hasRgb, hasClass]);

  const selectedAsset: Asset | null = useMemo(
    () => (data ? data.assets.find((a) => a.asset_id === selectedId) ?? null : null),
    [data, selectedId]
  );

  const visibleAssets = useMemo(() => {
    if (!data) return [];
    return data.assets.filter((a) => visibleClasses?.has(a.class) && !hiddenIds.has(a.asset_id));
  }, [data, visibleClasses, hiddenIds]);

  const fly: Fly | null = useMemo(() => {
    if (!selectedAsset || !data) return null;
    const bb = selectedAsset.bounding_box;
    const b = data.run.bounds;
    const cx = (bb[0] + bb[3]) / 2 - (b[0] + b[3]) / 2;
    const cy = (bb[1] + bb[4]) / 2 - (b[1] + b[4]) / 2;
    const cz = (bb[2] + bb[5]) / 2 - (b[2] + b[5]) / 2;
    return {
      target: new THREE.Vector3(cx, cz, -cy),
      radius: Math.max(bb[3] - bb[0], bb[4] - bb[1], bb[5] - bb[2], 1.2),
    };
  }, [selectedAsset, data]);

  const preset = useMemo(() => ({ kind: presetKind, sig: presetSig }), [presetKind, presetSig]);

  const toggleClass = useCallback((cls: string) => {
    setVisibleClasses((prev) => {
      if (!prev) return prev;
      const next = new Set(prev);
      if (next.has(cls)) next.delete(cls);
      else next.add(cls);
      return next;
    });
  }, []);

  const handleSelect = useCallback((id: string | null) => {
    setSelectedId(id);
    if (id) setFlySig((s) => s + 1);
  }, []);

  const handlePickMeasure = useCallback((p: [number, number, number]) => {
    setMeasurePoints((prev) => (prev.length >= 2 ? [p] : [...prev, p]));
  }, []);

  const applyPreset = useCallback((kind: "reset" | "top" | "side") => {
    setPresetKind(kind);
    setPresetSig((s) => s + 1);
  }, []);

  const classCounts = useMemo(() => {
    if (!data) return [];
    const map = new Map<string, number>();
    for (const a of data.assets) map.set(a.class, (map.get(a.class) ?? 0) + 1);
    return [...map.entries()].sort(([a], [b]) => a.localeCompare(b));
  }, [data]);

  // Dataset diagonal: camera presets / fly-to use it as their framing floor so
  // the scene is always framed from its real bounds instead of a fixed value.
  const autoResetDistance = useMemo(() => {
    if (!data) return 1;
    const b = data.run.bounds;
    return Math.max(
      Math.sqrt((b[3] - b[0]) ** 2 + (b[4] - b[1]) ** 2 + (b[5] - b[2]) ** 2),
      1
    );
  }, [data]);

  const measureReadout = useMemo(() => {
    if (measurePoints.length < 2) return null;
    const [a, b] = measurePoints;
    const d = Math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2 + (b[2] - a[2]) ** 2);
    const dz = Math.abs(b[2] - a[2]);
    const dh = Math.sqrt(d * d - dz * dz);
    return {
      d,
      dh,
      dz,
      a,
      b,
    };
  }, [measurePoints]);

  const backendEngine =
    data && typeof data.run.backend === "object" && data.run.backend
      ? (data.run.backend as Record<string, unknown>).engine ?? "geometry"
      : "geometry";

  const onDividerDown = useCallback((e: React.PointerEvent) => {
    e.preventDefault();
    setDragging(true);
  }, []);

  useEffect(() => {
    if (!dragging) return;
    const move = (e: PointerEvent) => {
      const rect = wrapRef.current?.getBoundingClientRect();
      if (!rect) return;
      const frac = (e.clientX - rect.left) / rect.width;
      setCompare((prev) => ({ ...prev, split: Math.min(0.92, Math.max(0.08, frac)) }));
    };
    const up = () => setDragging(false);
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
    return () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
    };
  }, [dragging]);

  if (loadError) {
    return (
      <div className="err-screen">
        <div>NO DATA PROCESSED</div>
        <div style={{ fontSize: 11, color: "var(--text-faint)", maxWidth: 520, textAlign: "center" }}>{loadError}</div>
        <div style={{ display: "flex", gap: 10 }}>
          <button className="btn" onClick={() => setLoadKey((k) => k + 1)}>↻ Retry load</button>
          <button className="btn ghost" onClick={onBack}>← Back to projects</button>
        </div>
      </div>
    );
  }

  if (!data || !visibleClasses) {
    return <div className="err-screen"><div>Loading viewer data…</div></div>;
  }

  return (
    <div className="viewer-shell">
      {/* ---------------- top bar ---------------- */}
      <header className="topbar">
        <div className="brand">
          <strong>TERRA POINT</strong>
          <span>{project.name}</span>
        </div>
        {project.simulated && <span className="sim-badge">SIM</span>}

        <div className="mode-switch">
          <button className={mode === "raw" ? "on" : ""} onClick={() => setMode("raw")}>
            RAW <span className="tag">LiDAR</span>
          </button>
          <button className={mode === "detection" ? "on" : ""} onClick={() => setMode("detection")}>
            AI <span className="tag">Detection</span>
          </button>
          <button className={mode === "inventory" ? "on" : ""} onClick={() => setMode("inventory")}>
            ASSET <span className="tag">Inventory</span>
          </button>
        </div>

        <select
          className="sel"
          value={mode === "inventory" ? "classification" : colorMode}
          disabled={mode === "inventory"}
          onChange={(e) => setColorMode(e.target.value as ColorMode)}
          title={mode === "inventory" ? "Inventory mode always colors by AI class" : undefined}
        >
          <option value="elevation">Color: Height</option>
          <option value="rgb" disabled={!hasRgb} title={hasRgb ? undefined : "No RGB recorded in this dataset"}>
            Color: RGB{hasRgb ? "" : " (no RGB in dataset)"}
          </option>
          <option value="intensity">Color: Intensity</option>
          <option value="classification" disabled={!hasClass} title={hasClass ? undefined : "No AI classification available"}>
            Color: Asset Class{hasClass ? "" : " (unavailable)"}
          </option>
        </select>

        <button className={`tbtn ${compare.active ? "on" : ""}`} onClick={() => setCompare((p) => ({ active: !p.active, split: p.split }))}>
          Compare
        </button>
        <button className={`tbtn ${measuring ? "on" : ""}`} onClick={() => { setMeasuring((m) => !m); setMeasurePoints([]); }}>
          Measure
        </button>
        <button className={`tbtn ${showLabels ? "on" : ""}`} onClick={() => setShowLabels((s) => !s)}>Labels</button>
        <button className={`tbtn ${showGrid ? "on" : ""}`} onClick={() => setShowGrid((s) => !s)}>Grid</button>
        <button className={`tbtn ${showBBoxes ? "on" : ""}`} onClick={() => setShowBBoxes((s) => !s)}>BBox</button>

        <div className="spacer" />

        <button className="tbtn" onClick={() => applyPreset("reset")}>Reset</button>
        <button className="tbtn" onClick={() => applyPreset("top")}>Top</button>
        <button className="tbtn" onClick={() => applyPreset("side")}>Side</button>

        <div style={{ display: "flex", alignItems: "center", gap: 6, color: "var(--text-faint)", fontSize: 10 }}>
          pts
          <input
            type="range"
            min={0.4}
            max={2.6}
            step={0.1}
            value={pointSize}
            onChange={(e) => setPointSize(parseFloat(e.target.value))}
            style={{ width: 70 }}
          />
        </div>

        {downloadError && <span className="dl-error">{downloadError}</span>}
        {EXPORTS.map((exp) => (
          <button
            key={exp.kind}
            className="tbtn"
            disabled={downloading !== null}
            onClick={() => {
              setDownloadError(null);
              setDownloading(exp.kind);
              api
                .downloadExport(project.id, exp.kind, `terra-point-${exp.kind}.json`)
                .catch((err) =>
                  setDownloadError(err instanceof Error ? err.message : String(err))
                )
                .finally(() => setDownloading(null));
            }}
          >
            {downloading === exp.kind ? "SAVING…" : `↓ ${exp.label}`}
          </button>
        ))}

        <button className="tbtn" onClick={onBack}>Projects</button>
      </header>

      {data.run.warnings.length > 0 && (
        <div className="warn-strip">
          {data.run.warnings.map((w, i) => (
            <span key={i}>{w}</span>
          ))}
        </div>
      )}

      {/* ---------------- main ---------------- */}
      <div className="viewer-main">
        <LayersPanel
          assets={data.assets}
          visibleClasses={visibleClasses}
          onToggleClass={toggleClass}
          selectedId={selectedId}
          onSelect={handleSelect}
        />

        <div className="viewer-canvas-wrap" ref={wrapRef}>
          <Scene3D
            data={data}
            mode={mode}
            colorMode={mode === "inventory" ? "classification" : colorMode}
            selectedId={selectedId}
            onSelect={handleSelect}
            visibleClasses={visibleClasses}
            showLabels={showLabels}
            showGrid={showGrid}
            showBBoxes={showBBoxes}
            compare={compare}
            measuring={measuring}
            measurePoints={measurePoints}
            onPickMeasurePoint={handlePickMeasure}
            pointSize={pointSize}
            fly={fly}
            flySig={flySig}
            preset={preset}
            autoResetDistance={autoResetDistance}
          />

          {project.simulated && (
            <div className="sim-overlay">SIMULATION / DEMO DATA — NOT REAL COMPETITION DATA</div>
          )}

          {compare.active && (
            <div className="compare">
              <div className={`cmp-label left ${compare.active ? "active" : ""}`}>RAW LiDAR</div>
              <div className={`cmp-label right ${compare.active ? "active" : ""}`}>AI ASSET INVENTORY</div>
              <div
                className="divider"
                style={{ left: `${compare.split * 100}%` }}
                onPointerDown={onDividerDown}
              />
            </div>
          )}

          {measuring && (
            <div className="measure-hint">
              {measurePoints.length === 0
                ? "Click in the scene to place the first measurement point"
                : "Click to place the second point"}
            </div>
          )}

          {measureReadout && (
            <div className="measure-readout">
              DIST {measureReadout.d.toFixed(2)} m
              <span style={{ color: "var(--text-dim)", marginLeft: 10 }}>
                ΔH {measureReadout.dh.toFixed(2)} · ΔV {measureReadout.dz.toFixed(2)}
              </span>
              <button className="btn" style={{ marginLeft: 10, padding: "3px 8px", fontSize: 11 }} onClick={() => setMeasurePoints([])}>
                Clear
              </button>
            </div>
          )}
        </div>

        {selectedAsset ? (
          <InspectPanel
            asset={selectedAsset}
            onClose={() => setSelectedId(null)}
            onFocus={() => setFlySig((s) => s + 1)}
            onToggleHidden={() =>
              setHiddenIds((prev) => {
                const next = new Set(prev);
                if (next.has(selectedAsset.asset_id)) next.delete(selectedAsset.asset_id);
                else next.add(selectedAsset.asset_id);
                return next;
              })
            }
            hidden={hiddenIds.has(selectedAsset.asset_id)}
          />
        ) : (
          <aside className="sidebar">
            {data.run.confidence_report && (
              <ConfidenceCard confidence={data.run.confidence_report} />
            )}
            <div className="sb-head">
              <h3>Assets · {visibleAssets.length}</h3>
            </div>
            <div className="sb-body">
              <input
                className="search-box"
                placeholder="Search asset ID…"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
              />
              {visibleAssets
                .filter((a) => a.asset_id.toLowerCase().includes(search.toLowerCase()))
                .map((a) => (
                  <div key={a.asset_id} className="asset-row" onClick={() => handleSelect(a.asset_id)}>
                    <span className="dot" style={{ background: CLASS_COLORS[a.class] ?? "#8899aa" }} />
                    <span className="aid">{a.asset_id}</span>
                    <span className="aname">{CLASS_LABELS[a.class] ?? a.class}</span>
                    <span className="ac">{Math.round(a.confidence * 100)}%</span>
                  </div>
                ))}
              {visibleAssets.length === 0 && <div className="empty">No assets match</div>}
            </div>
          </aside>
        )}
      </div>

      {/* ---------------- status bar ---------------- */}
      <footer className="statsbar">
        <span className="stat">POINTS <b>{data.run.point_count.toLocaleString()}</b></span>
        <span className="stat">TILES <b>{data.run.tile_count}</b></span>
        <span className="stat">ASSETS <b>{data.assets.length}</b></span>
        {data.run.confidence_report && (
          <span className="stat">
            CONFIDENCE{" "}
            <b style={{ color: "var(--accent)" }}>
              {data.run.confidence_report.overall_percent}%{" "}
              {data.run.confidence_report.grade.split(" ")[0]}
            </b>
          </span>
        )}
        {classCounts.map(([cls, count]) => (
          <span className="stat" key={cls}>
            <span className="cls-dot" style={{ background: CLASS_COLORS[cls] ?? "#8899aa" }} />
            {CLASS_LABELS[cls] ?? cls} <b>{count}</b>
          </span>
        ))}
        <span className="stat">CRS <b>{data.run.crs ?? "UNRESOLVED"}</b></span>
        <span className="stat">LAS <b>{data.run.las_version}</b> · fmt {data.run.point_format}</span>
        <span className="stat">BACKEND <b>{String(backendEngine)}</b></span>
        <span className="stat">v<b>{data.run.processing_version}</b></span>
        {data.run.warnings.length > 0 && (
          <span className="stat" style={{ color: "var(--accent)" }}>
            WARNINGS <b>{data.run.warnings.length}</b>
          </span>
        )}
      </footer>
    </div>
  );
}