"""Dependency-free 3D visualization over pipeline artifacts.

The viewer is a single self-contained HTML file with a hand-rolled WebGL
engine (no CDN, no libraries). Features:

* orbit camera: drag to rotate, shift-drag / right-drag to pan, wheel to zoom,
  two-finger pinch on touch, double-click to fly to an asset
* elevation-colored point cloud with depth attenuation and fog
* asset markers with glowing halos, always visible in front of the cloud
* projected HTML labels, ground grid, and bounding-box wireframes
* class filter chips with counts, searchable asset list
* click-to-select: camera fly-to, metadata panel, highlighted source points
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
    --bg: #070d16; --panel: #0d1624f2; --panel-solid: #0d1624; --line: #1d3048;
    --ink: #e8f1fb; --muted: #86a0bb; --accent: #5ef0c2; --accent-dim: #2ea581;
    --danger: #ff7e9d; --warn: #ffc46b;
  }
  * { box-sizing: border-box; }
  html, body { height: 100%; }
  body {
    margin: 0; overflow: hidden; background: var(--bg); color: var(--ink);
    font: 13px/1.45 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
  }
  #app { display: grid; grid-template-columns: 330px 1fr; height: 100vh; }

  /* ---------- sidebar ---------- */
  aside {
    background: linear-gradient(180deg, #0c1522, #0a111c);
    border-right: 1px solid var(--line); display: flex; flex-direction: column; min-height: 0;
  }
  .side-head { padding: 16px 16px 10px; border-bottom: 1px solid var(--line); }
  .eyebrow { color: var(--accent); font-size: 10px; font-weight: 700; letter-spacing: .15em; text-transform: uppercase; }
  h1 { font-size: 19px; margin: 3px 0 2px; letter-spacing: .01em; }
  .sub { color: var(--muted); font-size: 12px; }
  .metrics { display: grid; grid-template-columns: 1fr 1fr; gap: 7px; padding: 11px 16px; border-bottom: 1px solid var(--line); }
  .metric { background: #0b1422; border: 1px solid var(--line); border-radius: 8px; padding: 7px 10px; }
  .metric b { display: block; font-size: 18px; font-variant-numeric: tabular-nums; }
  .metric span { color: var(--muted); font-size: 10px; text-transform: uppercase; letter-spacing: .07em; }
  .filters { padding: 9px 16px 6px; border-bottom: 1px solid var(--line); }
  .filters h3 { margin: 0 0 7px; font-size: 10px; text-transform: uppercase; letter-spacing: .12em; color: var(--muted); }
  .legend { display: flex; flex-wrap: wrap; gap: 5px; }
  .chip {
    display: inline-flex; align-items: center; gap: 6px; padding: 3px 9px; border-radius: 999px;
    background: #0b1422; border: 1px solid var(--line); cursor: pointer; user-select: none;
    font-size: 11px; transition: opacity .15s, border-color .15s;
  }
  .chip:hover { border-color: var(--accent-dim); }
  .chip.off { opacity: .32; text-decoration: line-through; }
  .chip .swatch { width: 8px; height: 8px; border-radius: 50%; }
  .chip .n { color: var(--muted); font-variant-numeric: tabular-nums; }
  .search { margin: 9px 16px 3px; }
  .search input {
    width: 100%; padding: 7px 11px; border-radius: 8px; border: 1px solid var(--line);
    background: #0b1422; color: var(--ink); font-size: 12px; outline: none; transition: border-color .15s;
  }
  .search input:focus { border-color: var(--accent-dim); }
  #assetList { flex: 1; overflow-y: auto; padding: 5px 10px 14px; }
  .asset-row {
    display: flex; align-items: center; gap: 8px; padding: 6px 8px; border-radius: 7px;
    cursor: pointer; border: 1px solid transparent; transition: background .12s, border-color .12s;
  }
  .asset-row:hover { background: #0d1a2c; }
  .asset-row.selected { background: #10233a; border-color: var(--accent-dim); }
  .asset-row .dot { width: 8px; height: 8px; border-radius: 50%; flex: none; }
  .asset-row .id { font-weight: 650; font-size: 11.5px; font-family: ui-monospace, "SF Mono", Menlo, monospace; }
  .asset-row .cls { color: var(--muted); font-size: 10.5px; flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .asset-row .confwrap { width: 34px; flex: none; }
  .asset-row .confbar { height: 3px; border-radius: 2px; background: #1c2f48; overflow: hidden; }
  .asset-row .confbar i { display: block; height: 100%; background: linear-gradient(90deg, var(--accent-dim), var(--accent)); }
  .asset-row.flagged .id { color: var(--danger); }
  .empty { color: var(--muted); font-size: 12px; padding: 14px 8px; text-align: center; }

  /* ---------- stage ---------- */
  #stage { position: relative; overflow: hidden; background:
    radial-gradient(1200px 700px at 50% 20%, #0d1b2e 0%, var(--bg) 60%); }
  canvas { width: 100%; height: 100%; display: block; cursor: grab; touch-action: none; }
  canvas.dragging { cursor: grabbing; }
  canvas.panning { cursor: move; }
  #toolbar {
    position: absolute; top: 12px; left: 50%; transform: translateX(-50%);
    display: flex; gap: 6px; padding: 5px 7px; border-radius: 10px;
    background: var(--panel); border: 1px solid var(--line); backdrop-filter: blur(6px);
  }
  .tbtn {
    background: none; border: 1px solid transparent; color: var(--muted); font-size: 11px;
    padding: 4px 10px; border-radius: 7px; cursor: pointer; transition: all .15s; font-weight: 600;
  }
  .tbtn:hover { color: var(--ink); background: #13233a; }
  .tbtn.on { color: var(--accent); border-color: var(--accent-dim); background: #0e2236; }
  #hint {
    position: absolute; right: 14px; bottom: 12px; color: #5d7794; font-size: 11px; pointer-events: none;
    text-shadow: 0 1px 3px #000a; user-select: none;
  }
  #labels { position: absolute; inset: 0; pointer-events: none; overflow: hidden; }
  .alabel {
    position: absolute; transform: translate(-50%, calc(-100% - 8px)); white-space: nowrap;
    font-size: 10.5px; font-family: ui-monospace, "SF Mono", Menlo, monospace; color: #dfe9f5;
    background: #0b1422d9; border: 1px solid #2a4260; border-radius: 5px; padding: 2px 7px;
    display: flex; gap: 6px; align-items: center; box-shadow: 0 2px 8px #0008; pointer-events: none;
  }
  .alabel i { width: 7px; height: 7px; border-radius: 50%; }
  .alabel.hovered { border-color: #fff6; }
  .alabel.selected { border-color: var(--accent); box-shadow: 0 0 12px #5ef0c255; }

  #selection {
    position: absolute; left: 14px; top: 14px; width: min(390px, calc(100% - 28px));
    background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 14px 16px;
    display: none; max-height: calc(100% - 90px); overflow-y: auto; backdrop-filter: blur(8px);
    box-shadow: 0 10px 40px #0009;
  }
  #selection h2 { margin: 0 0 3px; font-size: 15px; display: flex; gap: 8px; align-items: center;
    font-family: ui-monospace, "SF Mono", Menlo, monospace; }
  #selection .badge { font-size: 10px; padding: 2px 8px; border-radius: 999px; background: #12263a;
    color: var(--accent); font-weight: 700; letter-spacing: .03em; }
  #selection .badge.warn { background: #331527; color: var(--danger); }
  #selection .subcls { color: var(--muted); font-size: 12px; margin-bottom: 6px; }
  #selection .conf-bar { height: 7px; border-radius: 4px; background: #16273c; margin: 7px 0 9px; overflow: hidden; }
  #selection .conf-bar i { display: block; height: 100%; background: linear-gradient(90deg, #2ea581, var(--accent)); }
  #selection table { width: 100%; border-collapse: collapse; font-size: 11.5px; }
  #selection td { padding: 2.5px 0; vertical-align: top; }
  #selection td:first-child { color: var(--muted); width: 44%; }
  #selection td:last-child { font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 11px; }
  #selection .factors { margin-top: 9px; display: flex; flex-wrap: wrap; gap: 4px; }
  #selection .factor { font-size: 10px; padding: 2px 7px; border-radius: 999px; background: #0f1e33; color: #b9cbe0; border: 1px solid var(--line); }
  #selection .factor b { color: var(--accent); }
  #selection .explain { margin-top: 9px; font-size: 11px; color: #a9bdd4; background: #0b1422; border: 1px solid var(--line);
    border-radius: 8px; padding: 8px 10px; }
  #selection .flags { margin-top: 8px; }
  #selection .flag { display: inline-block; font-size: 10px; font-weight: 700; padding: 2px 8px; border-radius: 999px;
    background: #331527; color: var(--danger); border: 1px solid #5c2740; margin-right: 4px; }
  #closeSel { position: absolute; top: 9px; right: 11px; background: none; border: none; color: var(--muted);
    font-size: 17px; cursor: pointer; line-height: 1; }
  #closeSel:hover { color: var(--ink); }
  #vignette { position: absolute; inset: 0; pointer-events: none;
    box-shadow: inset 0 0 140px 30px #00000066; }
  @media (max-width: 860px) {
    #app { grid-template-columns: 1fr; grid-template-rows: minmax(0, 42vh) minmax(0, 58vh); }
    aside { border-right: none; border-top: 1px solid var(--line); }
  }
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
    <div class="filters"><h3>Classes</h3><div class="legend" id="legend"></div></div>
    <div class="search"><input id="search" type="search" placeholder="Filter assets… (POL, sign, guardrail)"></div>
    <div id="assetList"></div>
  </aside>
  <main id="stage">
    <canvas id="gl"></canvas>
    <div id="toolbar">
      <button class="tbtn on" id="tLabels" title="Toggle asset labels">Labels</button>
      <button class="tbtn" id="tGrid" title="Toggle ground grid">Grid</button>
      <button class="tbtn" id="tBoxes" title="Toggle bounding boxes">Boxes</button>
      <button class="tbtn" id="tReset" title="Reset camera">Reset</button>
    </div>
    <div id="selection"></div>
    <div id="labels"></div>
    <div id="vignette"></div>
    <div id="hint">drag rotate · shift-drag pan · wheel zoom · click select · double-click fly</div>
  </main>
</div>
<script>
"use strict";
/* ================= data ================= */
const COLORS = {
  pavement: "#5f9df7", pavement_marking: "#f2d066", utility_pole: "#b48bf2",
  overhead_conductor: "#8fa8ff", utility_cabinet: "#d19cf7", traffic_sign: "#ff7e9d",
  guardrail: "#5ef0c2", safety_barrier: "#3fd4ad", rumble_strip: "#ffb46b"
};
let points = [], colors = [], assets = [], run = {};
let center = [0, 0, 0], radius = 1;

/* ================= canvas & gl ================= */
const canvas = document.getElementById("gl");
const gl = canvas.getContext("webgl", { antialias: true, alpha: false });
if (!gl) document.getElementById("runInfo").textContent = "WebGL is not available in this browser.";

function compile(type, source) {
  const s = gl.createShader(type);
  gl.shaderSource(s, source); gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) console.error(gl.getShaderInfoLog(s));
  return s;
}
const VS = `
attribute vec3 p; attribute vec3 c;
uniform vec3 uTarget; uniform float uYaw; uniform float uPitch; uniform float uZoom;
uniform float uAspect; uniform float uFov; uniform float uSize;
uniform float uFogNear; uniform float uFogFar;
varying vec3 vColor; varying float vDepth;
void main() {
  vec3 d = p - uTarget;
  float cy = cos(uYaw), sy = sin(uYaw), cp = cos(uPitch), sp = sin(uPitch);
  vec3 q = vec3(cy*d.x - sy*d.y, sy*d.x + cy*d.y, d.z);
  q = vec3(q.x, cp*q.y - sp*q.z, sp*q.y + cp*q.z);
  vDepth = uZoom - q.z;
  gl_Position = vec4(q.x / (vDepth * uAspect) * uFov, q.y / vDepth * uFov, 0.0, 1.0);
  gl_PointSize = clamp(uSize / vDepth, 1.0, 90.0);
  vColor = c;
}`;
const FS = `
precision mediump float;
varying vec3 vColor; varying float vDepth;
uniform vec3 uFog; uniform float uFogNear; uniform float uFogFar; uniform float uAlpha;
void main() {
  vec2 d = gl_PointCoord - vec2(0.5);
  if (dot(d, d) > 0.25) discard;
  float fog = clamp((vDepth - uFogNear) / (uFogFar - uFogNear), 0.0, 1.0);
  gl_FragColor = vec4(mix(vColor, uFog, fog * fog), uAlpha);
}`;
const program = gl.createProgram();
gl.attachShader(program, compile(gl.VERTEX_SHADER, VS));
gl.attachShader(program, compile(gl.FRAGMENT_SHADER, FS));
gl.linkProgram(program);
const U = {};
for (const name of ["uTarget","uYaw","uPitch","uZoom","uAspect","uFov","uSize","uFogNear","uFogFar","uFog","uAlpha"])
  U[name] = gl.getUniformLocation(program, name);
const A_P = gl.getAttribLocation(program, "p");
const A_C = gl.getAttribLocation(program, "c");

function makeBuffer(data) {
  const b = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, b);
  gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
  return b;
}
/* Separate pos/col buffers (points) */
function bindSeparate(posBuf, colBuf, count) {
  gl.bindBuffer(gl.ARRAY_BUFFER, posBuf);
  gl.enableVertexAttribArray(A_P); gl.vertexAttribPointer(A_P, 3, gl.FLOAT, false, 0, 0);
  gl.bindBuffer(gl.ARRAY_BUFFER, colBuf);
  gl.enableVertexAttribArray(A_C); gl.vertexAttribPointer(A_C, 3, gl.FLOAT, false, 0, 0);
  gl.drawArrays(gl.POINTS, 0, count);
}
/* Interleaved xyz-rgb buffers (markers, highlights, grid, bbox) */
function bindInterleaved(buffer, count, mode) {
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.enableVertexAttribArray(A_P); gl.vertexAttribPointer(A_P, 3, gl.FLOAT, false, 24, 0);
  gl.enableVertexAttribArray(A_C); gl.vertexAttribPointer(A_C, 3, gl.FLOAT, false, 24, 12);
  gl.drawArrays(mode, 0, count);
}

/* cached GPU buffers */
let pointBufs = null, bufGrid = null, bufGridLines = 0;
let bufMarkers = null, markerCount = 0;
let bufHighlights = null, highlightCount = 0;
let bufBBox = null, bboxCount = 0;

/* ================= camera ================= */
const cam = { yaw: -0.8, pitch: 0.55, dist: 2.4, target: [0, 0, 0] };
const camGoal = { ...cam };
let aspect = 1, fov = 1.05;
const FOG = [0.02, 0.05, 0.09];
let showLabels = true, showGrid = false, showBoxes = false;

function setUniforms(size, alpha, fogNear, fogFar) {
  gl.uniform3fv(U.uTarget, cam.target);
  gl.uniform1f(U.uYaw, cam.yaw); gl.uniform1f(U.uPitch, cam.pitch); gl.uniform1f(U.uZoom, cam.dist);
  gl.uniform1f(U.uAspect, aspect); gl.uniform1f(U.uFov, fov);
  gl.uniform1f(U.uSize, size); gl.uniform1f(U.uAlpha, alpha);
  gl.uniform1f(U.uFogNear, fogNear); gl.uniform1f(U.uFogFar, fogFar);
  gl.uniform3fv(U.uFog, FOG);
}
let lastW = 0, lastH = 0;
function resize() {
  const d = Math.min(devicePixelRatio || 1, 2);
  const w = Math.max(1, Math.round(canvas.clientWidth * d));
  const h = Math.max(1, Math.round(canvas.clientHeight * d));
  if (w !== lastW || h !== lastH) {
    canvas.width = w; canvas.height = h;
    gl.viewport(0, 0, w, h);
    lastW = w; lastH = h;
  }
  aspect = canvas.clientWidth / Math.max(1, canvas.clientHeight);
}

function render() {
  resize();
  gl.clearColor(FOG[0], FOG[1], FOG[2], 1);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  gl.useProgram(program);

  // 1. point cloud (depth-tested, fogged)
  if (pointBufs) {
    gl.enable(gl.DEPTH_TEST); gl.disable(gl.BLEND);
    setUniforms(2.6 * (canvas.width / 1100), 1, 0.6, 5.5);
    bindSeparate(pointBufs.pos, pointBufs.col, pointBufs.count);
  }
  // 2. ground grid
  if (bufGrid && showGrid) {
    gl.enable(gl.DEPTH_TEST); gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    setUniforms(1, 0.10, 0.6, 6.0);
    bindInterleaved(bufGrid, bufGridLines, gl.LINES);
  }
  // 3. highlighted source points (additive, always visible)
  if (highlightCount && bufHighlights) {
    gl.disable(gl.DEPTH_TEST); gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE);
    setUniforms(7.0 * (canvas.width / 1100), 0.95, 0.2, 4.0);
    bindInterleaved(bufHighlights, highlightCount, gl.POINTS);
  }
  // 4. marker halos (additive glow)
  if (markerCount && bufMarkers) {
    gl.disable(gl.DEPTH_TEST); gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE);
    setUniforms(34 * (canvas.width / 1100), 0.30, 0.2, 4.0);
    bindInterleaved(bufMarkers, markerCount, gl.POINTS);
  }
  // 5. marker cores (opaque, always visible)
  if (markerCount && bufMarkers) {
    gl.disable(gl.BLEND);
    setUniforms(11 * (canvas.width / 1100), 1, 0.2, 4.0);
    bindInterleaved(bufMarkers, markerCount, gl.POINTS);
  }
  // 6. bounding-box wireframe
  if (bboxCount && bufBBox) {
    gl.disable(gl.DEPTH_TEST); gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE);
    setUniforms(1, 0.9, 0.2, 4.0);
    bindInterleaved(bufBBox, bboxCount, gl.LINES);
  }
  gl.disable(gl.BLEND);
  updateLabels();
}

/* ================= buffers ================= */
function hex(v) { const n = parseInt(v.slice(1), 16); return [(n>>16)/255, ((n>>8)&255)/255, (n&255)/255]; }
function norm(x, y, z) { return [(x - center[0]) / radius, (y - center[1]) / radius, (z - center[2]) / radius]; }

function buildPointBuffers() {
  if (!points.length) return;
  const pos = new Float32Array(points.length * 3);
  const col = new Float32Array(points.length * 3);
  for (let i = 0; i < points.length; i++) {
    const n = norm(points[i][0], points[i][1], points[i][2]);
    pos[i*3] = n[0]; pos[i*3+1] = n[1]; pos[i*3+2] = n[2];
    col[i*3] = colors[i][0]; col[i*3+1] = colors[i][1]; col[i*3+2] = colors[i][2];
  }
  pointBufs = { pos: makeBuffer(pos), col: makeBuffer(col), count: points.length };
}
function buildGrid() {
  const b = run.bounds;
  if (!b) return;
  const z = b[2] + 0.002; // slightly above the lowest points
  const span = Math.max(b[3] - b[0], b[4] - b[1]);
  const step = Math.max(5, Math.round(span / 40 / 5) * 5);
  const x0 = Math.floor(b[0] / step) * step, x1 = Math.ceil(b[3] / step) * step;
  const y0 = Math.floor(b[1] / step) * step, y1 = Math.ceil(b[4] / step) * step;
  const data = [];
  const c = [0.45, 0.72, 1.0];
  for (let gx = x0; gx <= x1; gx += step) {
    data.push(...norm(gx, y0, z), ...c, ...norm(gx, y1, z), ...c);
  }
  for (let gy = y0; gy <= y1; gy += step) {
    data.push(...norm(x0, gy, z), ...c, ...norm(x1, gy, z), ...c);
  }
  if (data.length) { bufGrid = makeBuffer(new Float32Array(data)); bufGridLines = data.length / 6; }
}
function rebuildMarkers() {
  const rows = [];
  for (const a of assets) {
    if (!visibleClasses.has(a.class)) continue;
    const c = hex(COLORS[a.class] || "#ffffff");
    rows.push(...norm(a.center.x, a.center.y, a.center.z), ...c);
  }
  markerCount = rows.length / 6;
  bufMarkers = markerCount ? makeBuffer(new Float32Array(rows)) : null;
}
function rebuildHighlights(asset) {
  const rows = [];
  if (asset && asset.geometry && asset.geometry.highlight_points) {
    for (const p of asset.geometry.highlight_points.slice(0, 4000)) {
      rows.push(...norm(p[0], p[1], p[2]), 1, 1, 1);
    }
  }
  highlightCount = rows.length / 6;
  bufHighlights = highlightCount ? makeBuffer(new Float32Array(rows)) : null;
}
function rebuildBBox(asset) {
  const rows = [];
  if (asset && (showBoxes || asset.asset_id === selectedId || asset.asset_id === hoverId)) {
    const b = asset.bounding_box, c = hex(COLORS[asset.class] || "#ffffff");
    const corners = [
      [b[0], b[1], b[2]], [b[3], b[1], b[2]], [b[3], b[4], b[2]], [b[0], b[4], b[2]],
      [b[0], b[1], b[5]], [b[3], b[1], b[5]], [b[3], b[4], b[5]], [b[0], b[4], b[5]]
    ].map(p => norm(p[0], p[1], p[2]));
    const edges = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
    for (const [i, j] of edges) {
      rows.push(...corners[i], ...c, ...corners[j], ...c);
    }
  }
  bboxCount = rows.length / 6;
  bufBBox = bboxCount ? makeBuffer(new Float32Array(rows)) : null;
}

/* ================= labels ================= */
const labelEl = document.getElementById("labels");
const labelCache = new Map();
function updateLabels() {
  const visible = assets.filter(a => visibleClasses.has(a.class));
  const wanted = new Set();
  for (const a of visible) {
    const isSel = a.asset_id === selectedId, isHov = a.asset_id === hoverId;
    if (showLabels && (isSel || isHov || (cam.dist < 3.2 && visible.length <= 60))) wanted.add(a.asset_id);
  }
  for (const [id, el] of labelCache) if (!wanted.has(id)) el.style.display = "none";
  for (const a of visible) {
    if (!wanted.has(a.asset_id)) continue;
    const p = project(a.center.x, a.center.y, a.center.z);
    if (p.depth <= 0.12 || p.x < -20 || p.x > canvas.clientWidth + 20 || p.y < -20 || p.y > canvas.clientHeight + 20) continue;
    let el = labelCache.get(a.asset_id);
    if (!el) {
      el = document.createElement("div");
      el.className = "alabel";
      el.innerHTML = `<i style="background:${COLORS[a.class] || "#fff"}"></i><span>${a.asset_id}</span>`;
      labelEl.appendChild(el);
      labelCache.set(a.asset_id, el);
    }
    el.style.display = "flex";
    el.style.left = p.x + "px"; el.style.top = p.y + "px";
    el.classList.toggle("selected", a.asset_id === selectedId);
    el.classList.toggle("hovered", a.asset_id === hoverId);
  }
}

/* ================= picking ================= */
function project(x, y, z) {
  const d = norm(x, y, z);
  const dx = d[0] - cam.target[0], dy = d[1] - cam.target[1], dz = d[2] - cam.target[2];
  const cy = Math.cos(cam.yaw), sy = Math.sin(cam.yaw), cp = Math.cos(cam.pitch), sp = Math.sin(cam.pitch);
  const ry0 = sy*dx + cy*dy;
  const ry = cp*ry0 - sp*dz, rz = sp*ry0 + cp*dz, rx = cy*dx - sy*dy;
  const depth = cam.dist - rz;
  return {
    x: (rx / depth / aspect * fov + 1) * canvas.clientWidth / 2,
    y: (1 - ry / depth * fov) * canvas.clientHeight / 2,
    depth
  };
}
function pickMarker(mx, my) {
  let best = null, bestD = 16;
  for (const a of assets) {
    if (!visibleClasses.has(a.class)) continue;
    const p = project(a.center.x, a.center.y, a.center.z);
    if (p.depth <= 0) continue;
    const d = Math.hypot(p.x - mx, p.y - my);
    if (d < bestD) { bestD = d; best = a.asset_id; }
  }
  return best;
}

/* ================= interaction ================= */
let selectedId = null, hoverId = null;
let drag = null, pan = null, pinchLast = null;
const pointers = new Map();

function flyTo(asset) {
  const n = norm(asset.center.x, asset.center.y, asset.center.z);
  const b = asset.bounding_box;
  const diag = Math.hypot((b[3]-b[0])/radius, (b[4]-b[1])/radius, (b[5]-b[2])/radius);
  camGoal.target = n;
  camGoal.dist = Math.max(0.55, diag * 2.4);
  camGoal.pitch = 0.5;
}
function animateTo(dt) {
  const k = 1 - Math.exp(-dt * 7);
  cam.yaw += (camGoal.yaw - cam.yaw) * k;
  cam.pitch += (camGoal.pitch - cam.pitch) * k;
  cam.dist += (camGoal.dist - cam.dist) * k;
  cam.target[0] += (camGoal.target[0] - cam.target[0]) * k;
  cam.target[1] += (camGoal.target[1] - cam.target[1]) * k;
  cam.target[2] += (camGoal.target[2] - cam.target[2]) * k;
  cam.pitch = Math.max(-1.35, Math.min(1.35, cam.pitch));
  cam.dist = Math.max(1.02, Math.min(12, cam.dist));
}
function selectAsset(id) {
  selectedId = id;
  const a = id ? assets.find(x => x.asset_id === id) : null;
  rebuildHighlights(a); rebuildBBox(a);
  if (a) flyTo(a);
  refreshList(); renderPanel(a);
}
function renderPanel(a) {
  const panel = document.getElementById("selection");
  if (!a) { panel.style.display = "none"; return; }
  const f = a.confidence_factors || {};
  const factors = Object.entries(f).filter(([, v]) => v != null)
    .map(([k, v]) => `<span class="factor">${k.replace(/_/g, " ")} <b>${(v*100).toFixed(0)}%</b></span>`).join("");
  const flags = (a.qc_flags || []).map(x => `<span class="flag">${x}</span>`).join("");
  panel.innerHTML = `<button id="closeSel" title="Close">×</button>
    <h2>${a.asset_id}<span class="badge">${a.class}</span>${a.flagged ? '<span class="badge warn">flagged</span>' : ""}</h2>
    <div class="subcls">${a.subclass ? a.subclass.replace(/_/g, " ") : "—"}</div>
    <div class="conf-bar"><i style="width:${Math.round((a.confidence||0)*100)}%"></i></div>
    <table>
      <tr><td>Confidence</td><td>${((a.confidence||0)*100).toFixed(1)}%</td></tr>
      <tr><td>Coordinates</td><td>X ${fmt(a.center.x)}<br>Y ${fmt(a.center.y)}<br>Z ${fmt(a.center.z)}</td></tr>
      <tr><td>Dimensions</td><td>${fmt(a.dimensions.length_m)} × ${fmt(a.dimensions.width_m)} × ${fmt(a.dimensions.height_m)} m</td></tr>
      <tr><td>Point count</td><td>${fmt(a.point_count)}</td></tr>
      <tr><td>Source tile</td><td>${fmt(a.source_tile)}</td></tr>
      <tr><td>Run / scanner</td><td>${fmt(a.source_run)} / ${fmt(a.source_scanner)}</td></tr>
      <tr><td>Detection</td><td>${fmt(a.detection_method)}</td></tr>
      <tr><td>Orientation</td><td>${a.orientation_deg == null ? "null" : a.orientation_deg.toFixed(1) + "°"}</td></tr>
      <tr><td>Model prior</td><td>${fmt(a.model_prior_class)}${a.model_confidence != null ? " (" + ((a.model_confidence||0)*100).toFixed(0) + "%)" : ""}</td></tr>
      <tr><td>CRS</td><td>${fmt(a.coordinate_reference_system)}</td></tr>
    </table>
    ${factors ? `<div class="factors">${factors}</div>` : ""}
    <div class="explain">${a.confidence_explanation || ""}</div>
    ${flags ? `<div class="flags">${flags}</div>` : ""}`;
  panel.style.display = "block";
  document.getElementById("closeSel").onclick = () => selectAsset(null);
}
function fmt(n) { return n == null ? "null" : (typeof n === "number" ? n.toLocaleString(undefined, {maximumFractionDigits: 3}) : n); }

/* ================= sidebar ================= */
let visibleClasses = new Set();
function refreshLegend() {
  const legend = document.getElementById("legend");
  legend.innerHTML = "";
  const counts = {};
  for (const a of assets) counts[a.class] = (counts[a.class] || 0) + 1;
  for (const cls of Object.keys(counts).sort()) {
    const chip = document.createElement("span");
    chip.className = "chip" + (visibleClasses.has(cls) ? "" : " off");
    chip.innerHTML = `<span class="swatch" style="background:${COLORS[cls] || "#fff"}"></span>${cls.replace(/_/g, " ")}<span class="n">${counts[cls]}</span>`;
    chip.onclick = () => {
      visibleClasses.has(cls) ? visibleClasses.delete(cls) : visibleClasses.add(cls);
      chip.classList.toggle("off");
      rebuildMarkers(); refreshList();
    };
    legend.appendChild(chip);
  }
}
function refreshList() {
  const q = document.getElementById("search").value.toLowerCase();
  const list = document.getElementById("assetList");
  list.innerHTML = "";
  const shown = assets.filter(a =>
    (!q || a.asset_id.toLowerCase().includes(q) || a.class.toLowerCase().includes(q) ||
     (a.subclass || "").toLowerCase().includes(q)) && visibleClasses.has(a.class));
  if (!shown.length) {
    const e = document.createElement("div"); e.className = "empty";
    e.textContent = assets.length ? "No matching assets." : "No assets detected.";
    list.appendChild(e); return;
  }
  for (const a of shown) {
    const el = document.createElement("div");
    el.className = "asset-row" + (a.asset_id === selectedId ? " selected" : "") + (a.flagged ? " flagged" : "");
    el.innerHTML = `<span class="dot" style="background:${COLORS[a.class] || "#fff"}"></span>
      <span class="id">${a.asset_id}</span>
      <span class="cls">${a.subclass ? a.subclass.replace(/_/g, " ") : a.class.replace(/_/g, " ")}</span>
      <span class="confwrap"><span class="confbar"><i style="width:${Math.round((a.confidence||0)*100)}%"></i></span></span>`;
    el.onclick = () => selectAsset(a.asset_id);
    list.appendChild(el);
  }
}

/* ================= input ================= */
function handlePointerDown(e) {
  canvas.setPointerCapture(e.pointerId);
  pointers.set(e.pointerId, [e.clientX, e.clientY]);
  pinchLast = null;
  if (pointers.size === 1) {
    if (e.button === 2 || (e.button === 0 && e.shiftKey)) { pan = [e.clientX, e.clientY]; }
    else { drag = [e.clientX, e.clientY]; }
  }
  canvas.classList.toggle("panning", !!pan);
  canvas.classList.toggle("dragging", !!drag);
}
function handlePointerMove(e) {
  if (pointers.has(e.pointerId)) pointers.set(e.pointerId, [e.clientX, e.clientY]);
  if (pointers.size === 2) { handlePinch(); return; }
  if (pan) {
    const dx = e.clientX - pan[0], dy = e.clientY - pan[1];
    panCamera(dx, dy); pan = [e.clientX, e.clientY];
    return;
  }
  if (drag) {
    const dx = e.clientX - drag[0], dy = e.clientY - drag[1];
    camGoal.yaw += dx * 0.008;
    camGoal.pitch = Math.max(-1.35, Math.min(1.35, camGoal.pitch + dy * 0.008));
    drag = [e.clientX, e.clientY];
    return;
  }
  const rect = canvas.getBoundingClientRect();
  const id = pickMarker(e.clientX - rect.left, e.clientY - rect.top);
  if (id !== hoverId) {
    hoverId = id;
    canvas.style.cursor = id ? "pointer" : "grab";
    rebuildBBox(assets.find(a => a.asset_id === id));
  }
}
function handlePointerUp(e) {
  const rect = canvas.getBoundingClientRect();
  if (pointers.size === 1 && drag &&
      Math.hypot(e.clientX - drag[0], e.clientY - drag[1]) < 5) {
    selectAsset(pickMarker(e.clientX - rect.left, e.clientY - rect.top));
  }
  pointers.delete(e.pointerId);
  if (pointers.size < 2) { drag = null; pan = null; pinchLast = null; }
  canvas.classList.remove("dragging", "panning");
}
function handlePinch() {
  const [a, b] = [...pointers.values()];
  const d0 = Math.hypot(a[0] - b[0], a[1] - b[1]);
  if (pinchLast && d0 > 0) {
    camGoal.dist = Math.max(1.02, Math.min(12, camGoal.dist * (pinchLast / d0)));
    camGoal.target = [...cam.target];
  }
  pinchLast = d0;
}
function panCamera(dx, dy) {
  const k = cam.dist * 0.0011;
  const cy = Math.cos(cam.yaw), sy = Math.sin(cam.yaw);
  const cp = Math.cos(cam.pitch), sp = Math.sin(cam.pitch);
  const right = [-sy, cy, 0];
  const up = [-sp * cy, -sp * sy, cp];
  camGoal.target[0] += (-right[0] * dx + up[0] * dy) * k;
  camGoal.target[1] += (-right[1] * dx + up[1] * dy) * k;
  camGoal.target[2] += (-right[2] * dx + up[2] * dy) * k;
  cam.target = [...camGoal.target];
}
canvas.addEventListener("pointerdown", handlePointerDown);
canvas.addEventListener("pointermove", handlePointerMove);
canvas.addEventListener("pointerup", handlePointerUp);
canvas.addEventListener("pointercancel", e => { pointers.delete(e.pointerId); drag = null; pan = null; canvas.classList.remove("dragging", "panning"); });
canvas.addEventListener("contextmenu", e => e.preventDefault());
canvas.addEventListener("wheel", e => {
  e.preventDefault();
  camGoal.dist = Math.max(1.02, Math.min(12, camGoal.dist * (1 + e.deltaY * 0.0012)));
}, { passive: false });
canvas.addEventListener("dblclick", e => {
  const rect = canvas.getBoundingClientRect();
  const id = pickMarker(e.clientX - rect.left, e.clientY - rect.top);
  if (id) selectAsset(id);
});
document.getElementById("search").addEventListener("input", refreshList);
document.getElementById("tLabels").onclick = e => { showLabels = !showLabels; e.target.classList.toggle("on", showLabels); };
document.getElementById("tGrid").onclick = e => { showGrid = !showGrid; e.target.classList.toggle("on", showGrid); };
document.getElementById("tBoxes").onclick = e => {
  showBoxes = !showBoxes; e.target.classList.toggle("on", showBoxes);
  rebuildBBox(assets.find(a => a.asset_id === selectedId || a.asset_id === hoverId));
};
document.getElementById("tReset").onclick = () => {
  camGoal.yaw = -0.8; camGoal.pitch = 0.55; camGoal.dist = 2.4; camGoal.target = [0, 0, 0];
};
window.addEventListener("resize", render);

/* ================= main loop ================= */
let last = performance.now();
function loop(now) {
  animateTo(Math.min(0.05, (now - last) / 1000));
  last = now;
  render();
  requestAnimationFrame(loop);
}

fetch("viewer-data.json").then(r => r.json()).then(data => {
  points = data.points || [];
  colors = data.point_colors || points.map(() => [.33, .5, .7]);
  assets = data.assets || [];
  run = data.run || {};
  const b = run.bounds;
  if (b) {
    center = [(b[0]+b[3])/2, (b[1]+b[4])/2, (b[2]+b[5])/2];
    radius = Math.max(b[3]-b[0], b[4]-b[1], b[5]-b[2], 1);
  }
  visibleClasses = new Set([...new Set(assets.map(a => a.class))]);
  document.getElementById("assetCount").textContent = assets.length.toLocaleString();
  document.getElementById("pointCount").textContent = points.length.toLocaleString();
  document.getElementById("tileCount").textContent = (run.tile_count || 0).toLocaleString();
  document.getElementById("crsInfo").textContent = (run.crs || "unresolved").replace("EPSG:", "");
  document.getElementById("runInfo").textContent =
    `${(run.point_count || 0).toLocaleString()} source points · LAS ${run.las_version || "?"} · ${(run.backend && run.backend.name) || "geometry"}`;
  buildPointBuffers(); buildGrid(); rebuildMarkers();
  refreshLegend(); refreshList(); render();
  requestAnimationFrame(loop);
}).catch(() => {
  document.getElementById("runInfo").textContent =
    "Unable to load viewer-data.json. Run: python -m infra_inventory serve <output-dir>";
});
</script>
</body>
</html>
"""