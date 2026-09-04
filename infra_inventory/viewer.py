from __future__ import annotations

from pathlib import Path


def write_viewer(viewer_dir: Path, title: str = "AI4Infra Asset Inventory") -> None:
    """Write a dependency-free WebGL viewer over pipeline artifacts."""
    viewer_dir.mkdir(parents=True, exist_ok=True)
    template = """<!doctype html>
<html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">
<title>__TITLE__</title><style>
:root {{ color-scheme: dark; --ink:#e9f2ff; --muted:#8ba3bc; --panel:#101b2a; --line:#21354c; --accent:#62e8b9; }}
* {{ box-sizing:border-box }} body {{ margin:0; overflow:hidden; background:#08111d; color:var(--ink); font:14px ui-sans-serif,system-ui,sans-serif }}
#app {{ display:grid; grid-template-columns:310px 1fr; height:100vh }}
aside {{ background:linear-gradient(180deg,#0e1a29,#0a121e); border-right:1px solid var(--line); padding:22px; overflow:auto }}
h1 {{ font-size:18px; margin:0 0 6px }} .eyebrow {{ color:var(--accent); font-size:11px; font-weight:700; letter-spacing:.13em; text-transform:uppercase }}
.meta {{ color:var(--muted); line-height:1.5; margin:18px 0 }} .metric {{ border-top:1px solid var(--line); padding:12px 0 }} .metric b {{ display:block; font-size:24px }}
.legend {{ display:flex; align-items:center; gap:8px; padding:7px 0; color:var(--muted) }} .swatch {{ width:10px; height:10px; border-radius:50% }}
#stage {{ position:relative }} canvas {{ width:100%; height:100%; display:block; cursor:grab }} canvas:active {{ cursor:grabbing }}
#hint {{ position:absolute; right:20px; bottom:18px; color:var(--muted); background:#0d1928c9; padding:9px 12px; border:1px solid var(--line); border-radius:7px }}
#selection {{ position:absolute; left:18px; top:18px; min-width:250px; max-width:420px; background:#0d1928e8; border:1px solid var(--line); border-radius:8px; padding:13px; display:none }}
</style></head><body><div id=\"app\"><aside><div class=\"eyebrow\">Offline LiDAR inventory</div><h1>AI4Infra</h1><div class=\"meta\" id=\"run\">Loading inventory…</div><div class=\"metric\"><b id=\"assetCount\">—</b>detected assets</div><div class=\"metric\"><b id=\"pointCount\">—</b>display points</div><div id=\"legend\"></div><div class=\"meta\">Confidence is derived from measured support and geometry. Click an asset marker for provenance.</div></aside><main id=\"stage\"><canvas id=\"gl\"></canvas><div id=\"selection\"></div><div id=\"hint\">Drag to orbit · wheel to zoom · click marker for details</div></main></div>
<script>
const colors={{pavement:'#6ba4ff',pavement_marking:'#f5cf58',utility_pole:'#b082f7',traffic_sign:'#ff7e9d',guardrail:'#62e8b9',safety_barrier:'#62e8b9'}};
const canvas=document.querySelector('#gl'), gl=canvas.getContext('webgl',{antialias:true});
let points=[],assets=[],center=[0,0,0],radius=1,yaw=-.75,pitch=.65,distance=2.5,drag=null;
function shader(type,source){{const s=gl.createShader(type);gl.shaderSource(s,source);gl.compileShader(s);return s}}
const program=gl.createProgram();gl.attachShader(program,shader(gl.VERTEX_SHADER,`attribute vec3 p;attribute vec3 c;uniform float yaw;uniform float pitch;uniform float zoom;uniform float size;varying vec3 v;void main(){{float cy=cos(yaw),sy=sin(yaw),cp=cos(pitch),sp=sin(pitch);vec3 q=vec3(cy*p.x-sy*p.y,sy*p.x+cy*p.y,p.z);q=vec3(q.x,cp*q.y-sp*q.z,sp*q.y+cp*q.z);float depth=zoom-q.z;gl_Position=vec4(q.x/depth,q.y/depth,0.,1.);gl_PointSize=size;v=c;}}`));gl.attachShader(program,shader(gl.FRAGMENT_SHADER,`precision mediump float;varying vec3 v;void main(){{if(length(gl_PointCoord-.5)>.5)discard;gl_FragColor=vec4(v,1.);}}`));gl.linkProgram(program);
function hex(value){{const n=parseInt(value.slice(1),16);return [(n>>16)/255,((n>>8)&255)/255,(n&255)/255]}}
function resize(){{const d=devicePixelRatio||1;canvas.width=canvas.clientWidth*d;canvas.height=canvas.clientHeight*d;gl.viewport(0,0,canvas.width,canvas.height)}}
function sceneRows(){{const rows=[];for(const p of points)rows.push([p[0],p[1],p[2],.35,.52,.72]);for(const a of assets){{const color=hex(colors[a.class]||'#ffffff');rows.push([a.center.x,a.center.y,a.center.z,...color])}}return rows}}
function render(){{resize();const rows=sceneRows();if(!rows.length)return;const pos=[],col=[];for(const r of rows){{pos.push((r[0]-center[0])/radius,(r[1]-center[1])/radius,(r[2]-center[2])/radius);col.push(r[3],r[4],r[5])}}gl.clearColor(.03,.07,.12,1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);gl.enable(gl.DEPTH_TEST);gl.useProgram(program);const pb=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,pb);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(pos),gl.STATIC_DRAW);const pa=gl.getAttribLocation(program,'p');gl.enableVertexAttribArray(pa);gl.vertexAttribPointer(pa,3,gl.FLOAT,false,0,0);const cb=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,cb);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(col),gl.STATIC_DRAW);const ca=gl.getAttribLocation(program,'c');gl.enableVertexAttribArray(ca);gl.vertexAttribPointer(ca,3,gl.FLOAT,false,0,0);gl.uniform1f(gl.getUniformLocation(program,'yaw'),yaw);gl.uniform1f(gl.getUniformLocation(program,'pitch'),pitch);gl.uniform1f(gl.getUniformLocation(program,'zoom'),distance);gl.uniform1f(gl.getUniformLocation(program,'size'),3*devicePixelRatio);gl.drawArrays(gl.POINTS,0,points.length);gl.uniform1f(gl.getUniformLocation(program,'size'),11*devicePixelRatio);gl.drawArrays(gl.POINTS,points.length,assets.length)}}
function init(data){{points=data.points||[];assets=data.assets||[];const b=data.run.bounds;center=[(b[0]+b[3])/2,(b[1]+b[4])/2,(b[2]+b[5])/2];radius=Math.max(b[3]-b[0],b[4]-b[1],b[5]-b[2],1);document.querySelector('#assetCount').textContent=assets.length;document.querySelector('#pointCount').textContent=points.length.toLocaleString();document.querySelector('#run').textContent=`${{data.run.point_count.toLocaleString()}} source points · ${{data.run.crs||'CRS unavailable'}}`;const kinds=[...new Set(assets.map(a=>a.class))];document.querySelector('#legend').innerHTML=kinds.map(k=>`<div class=\"legend\"><i class=\"swatch\" style=\"background:${{colors[k]||'#fff'}}\"></i>${{k.replace('_',' ')}}</div>`).join('');render()}}
canvas.addEventListener('pointerdown',e=>drag=[e.clientX,e.clientY]);canvas.addEventListener('pointerup',()=>drag=null);canvas.addEventListener('pointermove',e=>{{if(!drag)return;yaw+=(e.clientX-drag[0])*.009;pitch=Math.max(-1.4,Math.min(1.4,pitch+(e.clientY-drag[1])*.009));drag=[e.clientX,e.clientY];render()}});canvas.addEventListener('wheel',e=>{{distance=Math.max(1.1,Math.min(8,distance+e.deltaY*.002));render()}});canvas.addEventListener('click',e=>{{if(drag)return;const a=assets.reduce((best,item)=>!best||item.confidence>best.confidence?item:best,null);if(a){{const el=document.querySelector('#selection');el.style.display='block';el.innerHTML=`<b>${{a.asset_id}}</b><br>${{a.class}} · ${{a.subclass||'unclassified'}}<br>confidence ${{a.confidence.toFixed(2)}}<br><span style=\"color:#8ba3bc\">${{a.confidence_explanation}}</span>`}}}});fetch('../viewer-data.json').then(r=>r.json()).then(init).catch(e=>document.querySelector('#run').textContent='Unable to load viewer-data.json. Start the bundled server.');
</script></main></body></html>"""
    (viewer_dir / "index.html").write_text(
        template.replace("__TITLE__", title).replace("{{", "{").replace("}}", "}"), encoding="utf-8"
    )
