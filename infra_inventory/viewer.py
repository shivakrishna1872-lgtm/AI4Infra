"""Dependency-free 3D visualization over pipeline artifacts.

The viewer is a single self-contained HTML file with a WebGL point renderer.
It loads ``viewer-data.json`` (written next to it) and offers:

* orbit / zoom navigation
* point cloud colored by elevation
* per-class visibility filters
* asset list with search
* click-to-select an asset -> metadata panel + highlight of its source points
"""
from __future__ import annotations

from pathlib import Path


def write_viewer(viewer_dir: Path) -> None:
    viewer_dir.mkdir(parents=True, exist_ok=True)
    (viewer_dir / "index.html").write_text(_HTML, encoding="utf-8")


_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI4Infra - Infrastructure Asset Inventory</title>
<style>
  :root {
    color-scheme: dark;
    --ink: #e9f2ff; --muted: #8ba3bc; --panel: #101b2a; --line: #21354c;
    --accent: #62e8b9; --danger: #ff7e9d;
  }
  * { box-sizing: border-box; }
  html, body { height: 100%; }
  body {
    margin: 0; overflow: hidden; background: #08111d; color: var(--ink);
    font: 14px/1.45 ui-sans-serif, system-ui, sans-serif;
  }
  #app { display: grid; grid-template-columns: 340px 1fr; height: 100vh; }
  aside {
    background: linear-gradient(180deg, #0e1a29, #0a121e);
    border-right: 1px solid var(--line); display: flex; flex-direction: column; min-height: 0;
  }
  .side-head { padding: 18px 18px 12px; border-bottom: 1px solid var(--line); }
  .eyebrow { color: var(--accent); font-size: 10px; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; }
  h1 { font-size: 18px; margin: 4px 0 2px; }
  .sub { color: var(--muted); font-size: 12px; }
  .metrics { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; padding: 12px 18px; border-bottom: 1px solid var(--line); }
  .metric { background: #0d1928; border: 1px solid var(--line); border-radius: 8px; padding: 8px 10px; }
  .metric b { display: block; font-size: 19px; }
  .metric span { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .06em; }
  .filters { padding: 10px 18px; border-bottom: 1px solid var(--line); }
  .filters h3 { margin: 0 0 6px; font-size: 11px; text-transform: uppercase; letter-spacing: .1em; color: var(--muted); }
  .legend { display: flex; flex-wrap: wrap; gap: 6px; }
  .chip {
    display: inline-flex; align-items: center; gap: 6px; padding: 4px 9px; border-radius: 999px;
    background: #0d1928; border: 1px solid var(--line); cursor: pointer; user-select: none; font-size: 12px;
  }
  .chip.off { opacity: .35; }
  .swatch { width: 9px; height: 9px; border-radius: 50%; }
  .search { margin: 10px 18px 4px; }
  .search input {
    width: 100%; padding: 7px 10px; border-radius: 8px; border: 1px solid var(--line);
    background: #0d1928; color: var(--ink); font-size: 13px; outline: none;
  }
  .search input:focus { border-color: var(--accent); }
  #assetList { flex: 1; overflow-y: auto; padding: 6px 12px 14px; }
  .asset-row {
    display: flex; align-items: center; gap: 8px; padding: 7px 8px; border-radius: 7px;
    cursor: pointer; border: 1px solid transparent;
  }
  .asset-row:hover { background: #0d1928; }
  .asset-row.selected { background: #12263a; border-color: var(--accent); }
  .asset-row .dot { width: 8px; height: 8px; border-radius: 50%; flex: none; }
  .asset-row .id { font-weight: 600; font-size: 12px; }
  .asset-row .cls { color: var(--muted); font-size: 11px; flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .asset-row .conf { font-size: 11px; color: var(--accent); }
  .asset-row.flagged { border-color: rgba(255,126,157,.35); }
  .empty { color: var(--muted); font-size: 12px; padding: 14px 8px; }
  #stage { position: relative; }
  canvas { width: 100%; height: 100%; display: block; cursor: grab; }
  canvas:active { cursor: grabbing; }
  #hint {
    position: absolute; right: 18px; bottom: 16px; color: var(--muted); font-size: 12px;
    background: #0d1928c9; padding: 8px 12px; border: 1px solid var(--line); border-radius: 7px;
  }
  #selection {
    position: absolute; left: 16px; top: 16px; width: min(400px, calc(100% - 32px));
    background: #0d1928f2; border: 1px solid var(--line); border-radius: 10px; padding: 14px 16px;
    display: none; max-height: calc(100% - 60px); overflow-y: auto; backdrop-filter: blur(4px);
  }
  #selection h2 { margin: 0 0 4px; font-size: 15px; display: flex; gap: 8px; align-items: center; }
  #selection .badge { font-size: 10px; padding: 2px 7px; border-radius: 999px; background: #12263a; color: var(--accent); font-weight: 600; }
  #selection .badge.warn { background: #3a1420; color: var(--danger); }
  #selection .conf-bar { height: 6px; border-radius: 3px; background: #1b2c42; margin: 6px 0 10px; overflow: hidden; }
  #selection .conf-bar i { display: block; height: 100%; background: linear-gradient(90deg, #3dd6a8, var(--accent)); }
  #selection table { width: 100%; border-collapse: collapse; font-size: 12px; }
  #selection td { padding: 3px 0; vertical-align: top; }
  #selection td:first-child { color: var(--muted); width: 46%; }
  #selection .factors { margin-top: 8px; color: var(--muted); font-size: 11px; }
  #selection .explain { margin-top: 8px; font-size: 12px; color: #b9cbe0; }
  #closeSel { position: absolute; top: 10px; right: 12px; background: none; border: none; color: var(--muted); font-size: 16px; cursor: pointer; }
  .flag { color: var(--danger); font-size: 11px; margin-top: 6px; }
  @media (max-width: 820px) { #app { grid-template-columns: 1fr; grid-template-rows: auto 1fr; } aside { display: none; } }
</style>
</head>
<body>
<div id="app">
  <aside>
    <div class="side-head">
      <div class="eyebrow">Offline LiDAR inventory</div>
      <h1>AI4Infra</h1>
      <div class="sub" id="runInfo">Loading inventory…</div>
    </div>
    <div class="metrics">
      <div class="metric"><b id="assetCount">—</b><span>assets</span></div>
      <div class="metric"><b id="pointCount">—</b><span>points shown</span></div>
      <div class="metric"><b id="tileCount">—</b><span>tiles</span></div>
      <div class="metric"><b id="crsInfo">—</b><span>CRS</span></div>
    </div>
    <div class="filters">
      <h3>Classes</h3>
      <div class="legend" id="legend"></div>
    </div>
    <div class="search"><input id="search" type="search" placeholder="Filter assets… (e.g. POL, sign)"></div>
    <div id="assetList"></div>
  </aside>
  <main id="stage">
    <canvas id="gl"></canvas>
    <div id="selection"></div>
    <div id="hint">Drag to orbit · scroll to zoom · click an asset for details</div>
  </main>
</div>
<script>
"use strict";
const COLORS = {
  pavement: "#6ba4ff", pavement_marking: "#f5cf58", utility_pole: "#b082f7",
  overhead_conductor: "#9aa7ff", utility_cabinet: "#c98af5", traffic_sign: "#ff7e9d",
  guardrail: "#62e8b9", safety_barrier: "#3dd6a8", rumble_strip: "#ffb46b"
};
const canvas = document.getElementById("gl");
const gl = canvas.getContext("webgl", { antialias: true });
if (!gl) { document.getElementById("runInfo").textContent = "WebGL is not available in this browser."; }

let points = [], colors = [], assets = [], center = [0, 0, 0], radius = 1;
let yaw = -0.75, pitch = 0.6, distance = 2.5, drag = null, selectedId = null;
let visibleClasses = new Set(), markers = []; // projected marker positions

// ---------- shaders ----------
function compile(type, source) {
  const s = gl.createShader(type);
  gl.shaderSource(s, source); gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) console.error(gl.getShaderInfoLog(s));
  return s;
}
const program = gl.createProgram();
gl.attachShader(program, compile(gl.VERTEX_SHADER, `
  attribute vec3 p; attribute vec3 c;
  uniform float yaw; uniform float pitch; uniform float zoom; uniform float size;
  varying vec3 v;
  void main() {
    float cy = cos(yaw), sy = sin(yaw), cp = cos(pitch), sp = sin(pitch);
    vec3 q = vec3(cy*p.x - sy*p.y, sy*p.x + cy*p.y, p.z);
    q = vec3(q.x, cp*q.y - sp*q.z, sp*q.y + cp*q.z);
    float depth = zoom - q.z;
    gl_Position = vec4(q.x/depth, q.y/depth, 0., 1.);
    gl_PointSize = size;
    v = c;
  }
`));
gl.attachShader(program, compile(gl.FRAGMENT_SHADER, `
  precision mediump float; varying vec3 v;
  void main() {
    if (length(gl_PointCoord - .5) > .5) discard;
    gl_FragColor = vec4(v, 1.);
  }
`));
gl.linkProgram(program);

function hex(v) {
  const n = parseInt(v.slice(1), 16);
  return [(n>>16)/255, ((n>>8)&255)/255, (n&255)/255];
}

// ---------- rendering ----------
function resize() {
  const d = devicePixelRatio || 1;
  canvas.width = canvas.clientWidth * d; canvas.height = canvas.clientHeight * d;
  gl.viewport(0, 0, canvas.width, canvas.height);
}
function draw(rows, sizeScale) {
  if (!rows.length) return;
  const pos = new Float32Array(rows.length * 3), col = new Float32Array(rows.length * 3);
  for (let i = 0; i < rows.length; i++) {
    pos[i*3] = (rows[i][0]-center[0])/radius; pos[i*3+1] = (rows[i][1]-center[1])/radius; pos[i*3+2] = (rows[i][2]-center[2])/radius;
    col[i*3] = rows[i][3]; col[i*3+1] = rows[i][4]; col[i*3+2] = rows[i][5];
  }
  const pb = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, pb);
  gl.bufferData(gl.ARRAY_BUFFER, pos, gl.STATIC_DRAW);
  const pa = gl.getAttribLocation(program, "p"); gl.enableVertexAttribArray(pa); gl.vertexAttribPointer(pa, 3, gl.FLOAT, false, 0, 0);
  const cb = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, cb);
  gl.bufferData(gl.ARRAY_BUFFER, col, gl.STATIC_DRAW);
  const ca = gl.getAttribLocation(program, "c"); gl.enableVertexAttribArray(ca); gl.vertexAttribPointer(ca, 3, gl.FLOAT, false, 0, 0);
  gl.uniform1f(gl.getUniformLocation(program, "yaw"), yaw);
  gl.uniform1f(gl.getUniformLocation(program, "pitch"), pitch);
  gl.uniform1f(gl.getUniformLocation(program, "zoom"), distance);
  gl.uniform1f(gl.getUniformLocation(program, "size"), 2.2 * (devicePixelRatio||1) * sizeScale);
  gl.drawArrays(gl.POINTS, 0, rows.length);
}
function project(x, y, z) {
  // Perspective projection matching the vertex shader, used for marker picking.
  const dx = x - center[0], dy = y - center[1], dz = z - center[2];
  const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
  const ry0 = sy*dx + cy*dy;
  const ry = cp*ry0 - sp*dz;
  const rz = sp*ry0 + cp*dz;
  const rx = cy*dx - sy*dy;
  const depth = distance - rz;
  return { x: (rx/depth + 1) * canvas.clientWidth / 2, y: (1 - ry/depth) * canvas.clientHeight / 2 };
}
function render() {
  resize();
  gl.clearColor(.03, .07, .12, 1);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  gl.enable(gl.DEPTH_TEST);
  gl.useProgram(program);

  // point cloud (colored by elevation)
  const pc = [];
  for (let i = 0; i < points.length; i++) {
    pc.push([points[i][0], points[i][1], points[i][2], colors[i][0], colors[i][1], colors[i][2]]);
  }
  draw(pc, 1);

  // highlighted source points of the selected asset (real coordinates from the LAS)
  const sel = selectedId ? assets.find(a => a.asset_id === selectedId) : null;
  const hl = [];
  if (sel && sel.geometry && sel.geometry.highlight_points) {
    const hi = hex("#ffffff");
    for (const p of sel.geometry.highlight_points.slice(0, 4000)) {
      hl.push([p[0], p[1], p[2], hi[0], hi[1], hi[2]]);
    }
  }
  if (hl.length) draw(hl, 2.4);

  // asset markers
  markers = [];
  const rows = [];
  for (const a of assets) {
    if (!visibleClasses.has(a.class)) continue;
    const c = hex(COLORS[a.class] || "#ffffff");
    rows.push([a.center.x, a.center.y, a.center.z, c[0], c[1], c[2]]);
    markers.push({ id: a.asset_id, x: a.center.x, y: a.center.y, z: a.center.z });
  }
  if (rows.length) draw(rows, 4.6);
}

// ---------- UI ----------
function fmt(n) { return n == null ? "null" : (typeof n === "number" ? n.toLocaleString(undefined, {maximumFractionDigits: 3}) : n); }
function row(a) {
  const el = document.createElement("div");
  el.className = "asset-row" + (a.asset_id === selectedId ? " selected" : "") + (a.flagged ? " flagged" : "");
  el.innerHTML = `<span class="dot" style="background:${COLORS[a.class]||"#fff"}"></span>
    <span class="id">${a.asset_id}</span><span class="cls">${a.class}${a.subclass ? " · " + a.subclass.replace(/_/g," ") : ""}</span>
    <span class="conf">${(a.confidence*100).toFixed(0)}%</span>`;
  el.onclick = () => { selectAsset(a.asset_id); };
  return el;
}
function refreshList() {
  const q = document.getElementById("search").value.toLowerCase();
  const list = document.getElementById("assetList");
  list.innerHTML = "";
  const shown = assets.filter(a =>
    (!q || a.asset_id.toLowerCase().includes(q) || a.class.toLowerCase().includes(q) || (a.subclass||"").toLowerCase().includes(q)) &&
    visibleClasses.has(a.class)
  );
  if (!shown.length) { const e = document.createElement("div"); e.className = "empty"; e.textContent = "No matching assets."; list.appendChild(e); }
  for (const a of shown) list.appendChild(row(a));
}
function refreshLegend() {
  const legend = document.getElementById("legend");
  legend.innerHTML = "";
  for (const cls of [...new Set(assets.map(a => a.class))].sort()) {
    const chip = document.createElement("span");
    chip.className = "chip" + (visibleClasses.has(cls) ? "" : " off");
    chip.innerHTML = `<span class="swatch" style="background:${COLORS[cls]||"#fff"}"></span>${cls.replace(/_/g," ")}`;
    chip.onclick = () => {
      visibleClasses.has(cls) ? visibleClasses.delete(cls) : visibleClasses.add(cls);
      chip.classList.toggle("off");
      refreshList(); render();
    };
    legend.appendChild(chip);
  }
}
function selectAsset(id) {
  selectedId = id;
  const a = assets.find(x => x.asset_id === id);
  refreshList(); render();
  const panel = document.getElementById("selection");
  if (!a) { panel.style.display = "none"; return; }
  const f = a.confidence_factors || {};
  const factorRows = Object.entries(f).filter(([,v]) => v != null)
    .map(([k,v]) => `${k.replace(/_/g," ")} ${(v*100).toFixed(0)}%`).join(" · ");
  panel.innerHTML = `<button id="closeSel" title="Close">×</button>
    <h2>${a.asset_id}<span class="badge">${a.class}</span>${a.flagged ? '<span class="badge warn">flagged</span>' : ""}</h2>
    <div style="color:var(--muted);font-size:12px">${a.subclass ? a.subclass.replace(/_/g," ") : "—"}</div>
    <div class="conf-bar"><i style="width:${(a.confidence*100).toFixed(1)}%"></i></div>
    <table>
      <tr><td>Confidence</td><td><b>${(a.confidence*100).toFixed(1)}%</b></td></tr>
      <tr><td>Coordinates</td><td>X ${fmt(a.center.x)}<br>Y ${fmt(a.center.y)}<br>Z ${fmt(a.center.z)}</td></tr>
      <tr><td>Dimensions</td><td>${fmt(a.dimensions.length_m)} × ${fmt(a.dimensions.width_m)} × ${fmt(a.dimensions.height_m)} m</td></tr>
      <tr><td>Point count</td><td>${fmt(a.point_count)}</td></tr>
      <tr><td>Source tile</td><td>${fmt(a.source_tile)}</td></tr>
      <tr><td>Run / scanner</td><td>${fmt(a.source_run)} / ${fmt(a.source_scanner)}</td></tr>
      <tr><td>Detection method</td><td>${fmt(a.detection_method)}</td></tr>
      <tr><td>Orientation</td><td>${a.orientation_deg == null ? "null" : a.orientation_deg.toFixed(1) + "°"}</td></tr>
      <tr><td>Model prior</td><td>${fmt(a.model_prior_class)}${a.model_confidence != null ? " (" + (a.model_confidence*100).toFixed(0) + "%)" : ""}</td></tr>
      <tr><td>CRS</td><td>${fmt(a.coordinate_reference_system)}</td></tr>
    </table>
    ${factorRows ? `<div class="factors">Factors: ${factorRows}</div>` : ""}
    <div class="explain">${a.confidence_explanation}</div>
    ${a.flagged ? `<div class="flag">QC: ${(a.qc_flags||[]).join(", ")}</div>` : ""}`;
  panel.style.display = "block";
  document.getElementById("closeSel").onclick = () => { panel.style.display = "none"; };
}

// ---------- input ----------
canvas.addEventListener("pointerdown", e => drag = [e.clientX, e.clientY]);
canvas.addEventListener("pointerup", () => drag = null);
canvas.addEventListener("pointermove", e => {
  if (!drag) return;
  yaw += (e.clientX - drag[0]) * .009;
  pitch = Math.max(-1.4, Math.min(1.4, pitch + (e.clientY - drag[1]) * .009));
  drag = [e.clientX, e.clientY]; render();
});
canvas.addEventListener("wheel", e => { distance = Math.max(1.1, Math.min(8, distance + e.deltaY * .002)); render(); });
canvas.addEventListener("click", e => {
  if (drag) return;
  const rect = canvas.getBoundingClientRect();
  const mx = e.clientX - rect.left, my = e.clientY - rect.top;
  let best = null, bestDist = 14;
  for (const m of markers) {
    const p = project(m.x, m.y, m.z);
    const d = Math.hypot(p.x - mx, p.y - my);
    if (d < bestDist) { bestDist = d; best = m.id; }
  }
  if (best) selectAsset(best);
  else { document.getElementById("selection").style.display = "none"; selectedId = null; refreshList(); render(); }
});
document.getElementById("search").addEventListener("input", refreshList);
window.addEventListener("resize", render);

// ---------- load ----------
fetch("viewer-data.json").then(r => r.json()).then(data => {
  points = data.points || [];
  colors = data.point_colors || points.map(() => [.35, .52, .72]);
  assets = data.assets || [];
  const b = data.run.bounds;
  center = [(b[0]+b[3])/2, (b[1]+b[4])/2, (b[2]+b[5])/2];
  radius = Math.max(b[3]-b[0], b[4]-b[1], b[5]-b[2], 1);
  visibleClasses = new Set([...new Set(assets.map(a => a.class))]);
  document.getElementById("assetCount").textContent = assets.length.toLocaleString();
  document.getElementById("pointCount").textContent = points.length.toLocaleString();
  document.getElementById("tileCount").textContent = (data.run.tile_count || 0).toLocaleString();
  document.getElementById("crsInfo").textContent = (data.run.crs || "unresolved").replace("EPSG:","");
  document.getElementById("runInfo").textContent =
    `${data.run.point_count.toLocaleString()} source points · LAS ${data.run.las_version} · ${data.run.backend.name || "geometry"}`;
  refreshLegend(); refreshList(); render();
}).catch(() => {
  document.getElementById("runInfo").textContent =
    "Unable to load viewer-data.json. Run: python -m infra_inventory serve <output-dir>";
});
</script>
</body>
</html>
"""