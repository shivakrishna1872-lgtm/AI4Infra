import { useEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import { Canvas, useFrame, type ThreeEvent } from "@react-three/fiber";
import { Grid, Html, OrbitControls } from "@react-three/drei";
import type { OrbitControls as OrbitControlsImpl } from "three-stdlib";
import type { Asset, ColorMode, ViewMode, ViewerData } from "../types";
import { CLASS_COLORS } from "../types";
import { buildAssetObject } from "./assets";

let _roundSprite: THREE.Texture | null = null;
function roundSprite(): THREE.Texture {
  if (_roundSprite) return _roundSprite;
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = 64;
  const ctx = canvas.getContext("2d")!;
  const g = ctx.createRadialGradient(32, 32, 0, 32, 32, 32);
  g.addColorStop(0, "rgba(255,255,255,1)");
  g.addColorStop(0.55, "rgba(255,255,255,0.85)");
  g.addColorStop(1, "rgba(255,255,255,0)");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, 64, 64);
  _roundSprite = new THREE.CanvasTexture(canvas);
  return _roundSprite;
}

// ---------------------------------------------------------------------------
// Color builders
// ---------------------------------------------------------------------------

const ELEVATION_STOPS: [number, [number, number, number]][] = [
  [0.0, [16, 28, 61]],
  [0.22, [29, 78, 137]],
  [0.45, [42, 157, 143]],
  [0.7, [244, 211, 94]],
  [0.88, [240, 140, 62]],
  [1.0, [214, 69, 65]],
];

function ramp(t: number): [number, number, number] {
  const x = Math.min(1, Math.max(0, t));
  for (let i = 1; i < ELEVATION_STOPS.length; i += 1) {
    const [t0, c0] = ELEVATION_STOPS[i - 1];
    const [t1, c1] = ELEVATION_STOPS[i];
    if (x <= t1) {
      const f = (x - t0) / (t1 - t0);
      return [
        Math.round(c0[0] + (c1[0] - c0[0]) * f),
        Math.round(c0[1] + (c1[1] - c0[1]) * f),
        Math.round(c0[2] + (c1[2] - c0[2]) * f),
      ];
    }
  }
  return ELEVATION_STOPS[ELEVATION_STOPS.length - 1][1];
}

function makeGeometry(
  data: ViewerData,
  center: [number, number, number],
  colorMode: ColorMode
): THREE.BufferGeometry {
  const pts = data.points;
  const n = pts.length;
  const positions = new Float32Array(n * 3);
  for (let i = 0; i < n; i += 1) {
    positions[i * 3] = pts[i][0] - center[0];
    positions[i * 3 + 1] = pts[i][1] - center[1];
    positions[i * 3 + 2] = pts[i][2] - center[2];
  }
  const colors = new Float32Array(n * 3);
  const bounds = data.run.bounds;
  const z0 = bounds[2];
  const z1 = Math.max(bounds[5] - z0, 1e-6);

  const rgb = data.point_rgb;
  const intensity = data.point_intensity;
  const pointClass = data.point_class;
  const classNames = data.point_class_names ?? [];

  let iMin = 0;
  let iMax = 1;
  if (intensity && intensity.length === n) {
    iMin = Math.min(...intensity);
    iMax = Math.max(...intensity);
    if (iMax - iMin < 1e-9) iMax = iMin + 1;
  }

  for (let i = 0; i < n; i += 1) {
    let c: [number, number, number];
    if (colorMode === "rgb" && rgb && rgb.length === n) {
      // Backend emits RGB normalized to 0..1 (65535-bit channels / 65535).
      c = [rgb[i][0], rgb[i][1], rgb[i][2]];
    } else if (colorMode === "intensity" && intensity && intensity.length === n) {
      const t = (intensity[i] - iMin) / (iMax - iMin);
      const v = 0.12 + 0.88 * t;
      c = [v, v, v * 1.02];
    } else if (colorMode === "classification" && pointClass && pointClass.length === n) {
      const id = pointClass[i];
      if (id > 0 && id <= classNames.length) {
        const hex = CLASS_COLORS[classNames[id - 1]] ?? "#6b7684";
        c = [
          parseInt(hex.slice(1, 3), 16) / 255,
          parseInt(hex.slice(3, 5), 16) / 255,
          parseInt(hex.slice(5, 7), 16) / 255,
        ];
      } else {
        c = [0.42, 0.46, 0.52];
      }
    } else {
      const t = (pts[i][2] - z0) / z1;
      const [r, g, b] = ramp(t);
      c = [r / 255, g / 255, b / 255];
    }
    colors[i * 3] = c[0];
    colors[i * 3 + 1] = c[1];
    colors[i * 3 + 2] = c[2];
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  return geometry;
}

function makeHighlightGeometry(
  points: number[][],
  center: [number, number, number]
): THREE.BufferGeometry {
  const positions = new Float32Array(points.length * 3);
  for (let i = 0; i < points.length; i += 1) {
    positions[i * 3] = points[i][0] - center[0];
    positions[i * 3 + 1] = points[i][1] - center[1];
    positions[i * 3 + 2] = points[i][2] - center[2];
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  return geometry;
}

// ---------------------------------------------------------------------------
// Cloud
// ---------------------------------------------------------------------------

interface CloudProps {
  geometry: THREE.BufferGeometry;
  visible: boolean;
  opacity: number;
  size: number;
  matRef?: React.MutableRefObject<THREE.PointsMaterial | null>;
}

function Cloud({ geometry, visible, opacity, size, matRef }: CloudProps) {
  return (
    <points geometry={geometry} visible={visible} frustumCulled={false}>
      <pointsMaterial
        ref={matRef}
        map={roundSprite()}
        vertexColors
        transparent
        opacity={opacity}
        size={size}
        sizeAttenuation
        depthWrite={false}
      />
    </points>
  );
}

// ---------------------------------------------------------------------------
// Asset layer
// ---------------------------------------------------------------------------

interface AssetLayerProps {
  data: ViewerData;
  center: [number, number, number];
  mode: ViewMode;
  selectedId: string | null;
  onSelect: (id: string | null) => void;
  visibleClasses: Set<string>;
  showBBoxes: boolean;
  showLabels: boolean;
  measuring: boolean;
  onPickMeasurePoint: (p: [number, number, number]) => void;
  groupRef: React.MutableRefObject<THREE.Group | null>;
}

function AssetLayer({
  data,
  center,
  mode,
  selectedId,
  onSelect,
  visibleClasses,
  showBBoxes,
  showLabels,
  measuring,
  onPickMeasurePoint,
  groupRef,
}: AssetLayerProps) {
  const byId = useMemo(() => new Map(data.assets.map((a) => [a.asset_id, a])), [data]);
  const inventoryMode = mode === "inventory";

  const visible = useMemo(
    () => (mode === "raw" ? [] : data.assets.filter((a) => visibleClasses.has(a.class))),
    [data, visibleClasses, mode]
  );

  const groups = useMemo(
    () =>
      visible.map((a) =>
        buildAssetObject(a, center, {
          inventory: inventoryMode,
          selected: a.asset_id === selectedId,
          showBBox: showBBoxes || a.asset_id === selectedId,
        })
      ),
    [visible, center, inventoryMode, selectedId, showBBoxes]
  );

  const handleClick = (e: ThreeEvent<MouseEvent>) => {
    e.stopPropagation();
    const id = e.object.userData.assetId as string | undefined;
    if (!id) return;
    const asset = byId.get(id);
    if (!asset) return;
    if (measuring) {
      onPickMeasurePoint([asset.center.x, asset.center.y, asset.center.z]);
    } else {
      onSelect(id);
    }
  };

  return (
    <group ref={groupRef}>
      {groups.map((g) => {
        const id = g.userData.assetId as string;
        const asset = byId.get(id);
        return (
          <group key={id}>
            <primitive object={g} onClick={handleClick} />
            {showLabels && inventoryMode && asset && (
              <Html
                position={[asset.center.x - center[0], asset.center.y - center[1], asset.center.z - center[2]]}
                center
                zIndexRange={[50, 0]}
              >
                <div className={`alabel ${selectedId === id ? "selected" : ""}`}>
                  {asset.asset_id}
                </div>
              </Html>
            )}
          </group>
        );
      })}
    </group>
  );
}

// ---------------------------------------------------------------------------
// Measurement layer
// ---------------------------------------------------------------------------

function MeasureLayer({
  points,
  center,
}: {
  points: [number, number, number][];
  center: [number, number, number];
}) {
  const lineObject = useMemo(() => {
    const geometry = new THREE.BufferGeometry();
    const arr = new Float32Array(points.length * 3);
    points.forEach((p, i) => {
      arr[i * 3] = p[0] - center[0];
      arr[i * 3 + 1] = p[1] - center[1];
      arr[i * 3 + 2] = p[2] - center[2];
    });
    geometry.setAttribute("position", new THREE.BufferAttribute(arr, 3));
    const material = new THREE.LineBasicMaterial({ color: "#ffb454", transparent: true, opacity: 0.95 });
    return new THREE.Line(geometry, material);
  }, [points, center]);

  if (points.length === 0) return null;
  return (
    <group>
      <primitive object={lineObject} />
      {points.map((p, i) => (
        <mesh key={i} position={[p[0] - center[0], p[1] - center[1], p[2] - center[2]]}>
          <sphereGeometry args={[0.16, 12, 10]} />
          <meshBasicMaterial color="#ffb454" />
        </mesh>
      ))}
    </group>
  );
}

// ---------------------------------------------------------------------------
// Camera controller: fly-to, view presets, compare-mode clipping
// ---------------------------------------------------------------------------

export interface Fly {
  target: THREE.Vector3;
  radius: number;
}

interface FlyRef {
  target: THREE.Vector3;
  pos: THREE.Vector3;
}

interface CameraControllerProps {
  diag: number;
  bounds: THREE.Box3;
  compare: { active: boolean; split: number };
  leftMatRef: React.MutableRefObject<THREE.PointsMaterial | null>;
  rightMatRef: React.MutableRefObject<THREE.PointsMaterial | null>;
  meshGroupRef: React.MutableRefObject<THREE.Group | null>;
  fly: Fly | null;
  flySig: number;
  preset: { kind: "reset" | "top" | "side"; sig: number };
  autoResetDist: number;
}

function CameraController({
  diag,
  bounds,
  compare,
  leftMatRef,
  rightMatRef,
  meshGroupRef,
  fly,
  flySig,
  preset,
  autoResetDist,
}: CameraControllerProps) {
  const controlsRef = useRef<OrbitControlsImpl | null>(null);
  const flyRef = useRef<FlyRef | null>(null);

  // Framing distance that fits the complete dataset with margin (world units).
  // Derived from the dataset bounds so an 80 m test scene and a 2 km corridor
  // both fill the viewport - never a hardcoded camera position.
  const fitDist = useMemo(() => {
    const size = bounds.getSize(new THREE.Vector3());
    const sceneDiag = Math.max(size.length(), diag, 1);
    return sceneDiag * 1.5;
  }, [bounds, diag]);

  // Hard cut: park the camera at `eye`, looking at `at`.
  function jumpTo(eye: THREE.Vector3, at: THREE.Vector3) {
    const controls = controlsRef.current;
    if (!controls) return;
    controls.object.position.copy(eye);
    controls.target.copy(at);
    controls.update();
  }
  const raycaster = useMemo(() => new THREE.Raycaster(), []);
  const rightVec = useMemo(() => new THREE.Vector3(), []);
  const tmpPoint = useMemo(() => new THREE.Vector3(), []);
  const planeLeft = useMemo(() => new THREE.Plane(), []);
  const planeRight = useMemo(() => new THREE.Plane(), []);

  // Fly-to re-uses the preset-derived distance as a sensible fallback floor
  // so assets on the far side of a large scene still frame nicely.
  const flyFrameDist = useMemo(() => Math.max(autoResetDist * 0.7, 1.2), [autoResetDist]);

  useEffect(() => {
    if (!fly) {
      flyRef.current = null;
      return;
    }
    const controls = controlsRef.current;
    if (!controls) return;
    const dir = new THREE.Vector3()
      .subVectors(controls.object.position, controls.target)
      .normalize();
    const dist = THREE.MathUtils.clamp(fly.radius * 2.6, flyFrameDist * 0.4, diag * 1.4);
    flyRef.current = {
      target: fly.target.clone(),
      pos: fly.target.clone().add(dir.multiplyScalar(dist)),
    };
  }, [fly, flySig, diag, flyFrameDist]);

  // Preset framings derived from the dataset bounds. The point group is rotated
  // -90 deg around X (local z-up becomes world y-up) and re-centered on the
  // origin, so the look-at target is the origin and Top / Side / Reset are eye
  // directions at the fitted distance.
  useEffect(() => {
    const controls = controlsRef.current;
    if (!controls) return;
    const dir =
      preset.kind === "top"
        ? new THREE.Vector3(0, 1, 0.08).normalize()
        : preset.kind === "side"
        ? new THREE.Vector3(0.5, 0.18, 1).normalize()
        : new THREE.Vector3(1, 0.62, 1.3).normalize();
    jumpTo(dir.multiplyScalar(fitDist), new THREE.Vector3(0, 0, 0));
  }, [preset.kind, preset.sig, fitDist]);

  useFrame((state, dt) => {
    const controls = controlsRef.current;
    if (!controls) return;

    // smooth fly-to
    const flyNow = flyRef.current;
    if (flyNow) {
      const k = 1 - Math.exp(-3.2 * dt);
      controls.target.lerp(flyNow.target, k);
      state.camera.position.lerp(flyNow.pos, k);
      if (
        state.camera.position.distanceTo(flyNow.pos) < diag * 0.004 &&
        controls.target.distanceTo(flyNow.target) < diag * 0.004
      ) {
        flyRef.current = null;
      }
      controls.update();
    }

    // compare-mode world-space clipping
    const left = leftMatRef.current;
    const right = rightMatRef.current;
    if (compare.active && left && right) {
      const cam = state.camera;
      raycaster.setFromCamera(new THREE.Vector2(compare.split * 2 - 1, 0), cam);
      const depth = cam.position.distanceTo(controls.target) * 0.6;
      const point = raycaster.ray.at(depth, tmpPoint);
      rightVec.setFromMatrixColumn(cam.matrixWorld, 0);
      planeLeft.setFromNormalAndCoplanarPoint(rightVec.clone(), point);
      planeRight.setFromNormalAndCoplanarPoint(rightVec.clone().negate(), point);
      left.clippingPlanes = [planeLeft];
      left.needsUpdate = true;
      right.clippingPlanes = [planeRight];
      right.needsUpdate = true;
      if (meshGroupRef.current) {
        const mats = new Set<THREE.Material>();
        meshGroupRef.current.traverse((obj) => {
          const sprite = obj as unknown as THREE.Sprite;
          const mesh = obj as THREE.Mesh;
          if (sprite.isSprite || mesh.isMesh) {
            const material = mesh.material;
            if (material) {
              const list = Array.isArray(material) ? material : [material];
              list.forEach((m) => mats.add(m));
            }
          }
        });
        mats.forEach((m) => {
          m.clippingPlanes = [planeRight];
          m.needsUpdate = true;
        });
      }
    } else if (
      left &&
      right &&
      ((left.clippingPlanes?.length ?? 0) > 0 || (right.clippingPlanes?.length ?? 0) > 0)
    ) {
      left.clippingPlanes = [];
      left.needsUpdate = true;
      right.clippingPlanes = [];
      right.needsUpdate = true;
    }
  });

  return (
    <OrbitControls
      ref={controlsRef}
      makeDefault
      target={[0, 0, 0]}
      enableDamping
      dampingFactor={0.09}
      minDistance={diag * 0.02}
      maxDistance={diag * 8}
    />
  );
}

// ---------------------------------------------------------------------------
// Scene
// ---------------------------------------------------------------------------

export interface Scene3DProps {
  data: ViewerData;
  mode: ViewMode;
  colorMode: ColorMode;
  selectedId: string | null;
  onSelect: (id: string | null) => void;
  visibleClasses: Set<string>;
  showLabels: boolean;
  showGrid: boolean;
  showBBoxes: boolean;
  compare: { active: boolean; split: number };
  measuring: boolean;
  measurePoints: [number, number, number][];
  onPickMeasurePoint: (p: [number, number, number]) => void;
  pointSize: number;
  fly: Fly | null;
  flySig: number;    preset: { kind: "reset" | "top" | "side"; sig: number };
  autoResetDistance: number;
}

export default function Scene3D(props: Scene3DProps) {
  const {
    data,
    mode,
    colorMode,
    selectedId,
    onSelect,
    visibleClasses,
    showLabels,
    showGrid,
    showBBoxes,
    compare,
    measuring,
    measurePoints,
    onPickMeasurePoint,
    pointSize,
    fly,
    flySig,
    preset,
    autoResetDistance,
  } = props;

  const bounds = data.run.bounds;
  const center: [number, number, number] = useMemo(
    () => [
      (bounds[0] + bounds[3]) / 2,
      (bounds[1] + bounds[4]) / 2,
      (bounds[2] + bounds[5]) / 2,
    ],
    [bounds]
  );
  const diag = useMemo(
    () => Math.sqrt((bounds[3] - bounds[0]) ** 2 + (bounds[4] - bounds[1]) ** 2 + (bounds[5] - bounds[2]) ** 2),
    [bounds]
  );
  const minZLocal = bounds[2] - center[2];

  // Dataset bounds after the -90 deg X rotation (local x,y,z -> world x,z,-y):
  // used by the camera presets so Reset / Top / Side always frame the full cloud.
  const worldBounds = useMemo(() => {
    const box = new THREE.Box3();
    const p = new THREE.Vector3();
    const corners: [number, number, number][] = [
      [bounds[0], bounds[1], bounds[2]],
      [bounds[0], bounds[1], bounds[5]],
      [bounds[0], bounds[4], bounds[2]],
      [bounds[0], bounds[4], bounds[5]],
      [bounds[3], bounds[1], bounds[2]],
      [bounds[3], bounds[1], bounds[5]],
      [bounds[3], bounds[4], bounds[2]],
      [bounds[3], bounds[4], bounds[5]],
    ];
    for (const [x, y, z] of corners) {
      p.set(x, z, -y);
      box.expandByPoint(p);
    }
    return box;
  }, [bounds]);

  const selectedAsset: Asset | null = useMemo(
    () => data.assets.find((a) => a.asset_id === selectedId) ?? null,
    [data, selectedId]
  );

  // One geometry per color mode (shared positions, per-mode colors).
  const geoms = useMemo(() => {
    const elevation = makeGeometry(data, center, "elevation");
    const rgb = makeGeometry(data, center, "rgb");
    const intensity = makeGeometry(data, center, "intensity");
    const classification = makeGeometry(data, center, "classification");
    return { elevation, rgb, intensity, classification };
  }, [data, center]);

  const mainGeom = useMemo(() => {
    if (mode === "inventory") return geoms.classification;
    if (colorMode === "rgb") return geoms.rgb;
    if (colorMode === "intensity") return geoms.intensity;
    if (colorMode === "classification") return geoms.classification;
    return geoms.elevation;
  }, [mode, colorMode, geoms]);

  const highlightGeom = useMemo(() => {
    const pts = selectedAsset?.geometry?.highlight_points;
    if (!pts || pts.length === 0) return null;
    return makeHighlightGeometry(pts, center);
  }, [selectedAsset, center]);

  const mainMatRef = useRef<THREE.PointsMaterial | null>(null);
  const leftMatRef = useRef<THREE.PointsMaterial | null>(null);
  const rightMatRef = useRef<THREE.PointsMaterial | null>(null);
  const meshGroupRef = useRef<THREE.Group | null>(null);
  const worldRef = useRef<THREE.Group | null>(null);
  const tmpVec = useMemo(() => new THREE.Vector3(), []);

  const pointSizeWorld = diag * 0.0009 * pointSize;
  const mainOpacity = selectedId && !compare.active ? 0.16 : mode === "inventory" ? 0.45 : 1;

  const handleGroundClick = (e: ThreeEvent<MouseEvent>) => {
    e.stopPropagation();
    if (!worldRef.current) return;
    tmpVec.copy(e.point);
    worldRef.current.worldToLocal(tmpVec);
    onPickMeasurePoint([
      tmpVec.x + center[0],
      tmpVec.y + center[1],
      tmpVec.z + center[2],
    ]);
  };

  return (
    <Canvas
      dpr={[1, 2]}
      camera={{
        fov: 45,
        near: Math.max(diag / 2000, 0.01),
        far: diag * 25,
        position: [diag * 0.5, diag * 0.4, diag * 0.85],
      }}
      gl={{ antialias: true, localClippingEnabled: true, powerPreference: "high-performance" }}
      onPointerMissed={() => {
        if (!measuring) onSelect(null);
      }}
    >
      <color attach="background" args={["#211a12"]} />
      <fog attach="fog" args={["#07090d", diag * 0.85, diag * 2.4]} />
      <ambientLight intensity={0.55} />
      <directionalLight position={[diag * 0.4, diag * 0.9, diag * 0.3]} intensity={1.1} />
      <directionalLight position={[-diag * 0.5, diag * 0.3, -diag * 0.4]} intensity={0.35} />

      <group ref={worldRef} rotation={[-Math.PI / 2, 0, 0]}>
        {/* RAW side / main cloud */}
        <Cloud geometry={mainGeom} visible={!compare.active} opacity={mainOpacity} size={pointSizeWorld} matRef={mainMatRef} />
        {/* compare: left = raw elevation, right = classified inventory */}
        <Cloud geometry={geoms.elevation} visible={compare.active} opacity={1} size={pointSizeWorld} matRef={leftMatRef} />
        <Cloud geometry={geoms.classification} visible={compare.active} opacity={0.5} size={pointSizeWorld} matRef={rightMatRef} />

        {/* AI evidence: highlighted source points of the selected asset */}
        {highlightGeom && (
          <points geometry={highlightGeom} frustumCulled={false}>
            <pointsMaterial
              map={roundSprite()}
              color="#ffffff"
              size={pointSizeWorld * 2.6}
              sizeAttenuation
              transparent
              opacity={0.95}
              depthWrite={false}
              blending={THREE.AdditiveBlending}
            />
          </points>
        )}

        <AssetLayer
          data={data}
          center={center}
          mode={mode}
          selectedId={selectedId}
          onSelect={onSelect}
          visibleClasses={visibleClasses}
          showBBoxes={showBBoxes}
          showLabels={showLabels}
          measuring={measuring}
          onPickMeasurePoint={onPickMeasurePoint}
          groupRef={meshGroupRef}
        />

        <MeasureLayer points={measurePoints} center={center} />

        {measuring && (
          <mesh position={[0, minZLocal - 0.2, 0]} onClick={handleGroundClick}>
            <planeGeometry args={[diag * 4, diag * 4]} />
            <meshBasicMaterial transparent opacity={0} depthWrite={false} side={THREE.DoubleSide} />
          </mesh>
        )}

        {showGrid && (
          <Grid
            position={[0, minZLocal, 0]}
            args={[diag, diag]}
            cellSize={diag / 18}
            cellThickness={0.7}
            cellColor="#1f3349"
            sectionSize={diag / 6}
            sectionThickness={1.2}
            sectionColor="#2e4563"
            fadeDistance={diag * 1.6}
            fadeStrength={1.8}
            infiniteGrid
          />
        )}
      </group>

      <CameraController
        diag={diag}
        bounds={worldBounds}
        compare={compare}
        leftMatRef={leftMatRef}
        rightMatRef={rightMatRef}
        meshGroupRef={meshGroupRef}
        fly={fly}
        flySig={flySig}
        preset={preset}
        autoResetDist={autoResetDistance}
      />
    </Canvas>
  );
}