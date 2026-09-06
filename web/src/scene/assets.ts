import * as THREE from "three";
import type { Asset } from "../types";
import { CLASS_COLORS } from "../types";

export type AssetKind = "pole" | "sign" | "elongated" | "box" | "conductor" | "marking" | "pavement";

export function assetKind(asset: Asset): AssetKind {
  switch (asset.class) {
    case "utility_pole":
      return "pole";
    case "traffic_sign":
      return "sign";
    case "guardrail":
    case "safety_barrier":
      return "elongated";
    case "overhead_conductor":
      return "conductor";
    case "pavement_marking":
      return "marking";
    case "pavement":
      return "pavement";
    default:
      return "box";
  }
}

export function assetColor(asset: Asset): THREE.Color {
  return new THREE.Color(CLASS_COLORS[asset.class] ?? "#8899aa");
}

/** Longest horizontal bbox axis direction (yaw), used to orient meshes. */
export function assetYaw(asset: Asset): number {
  const bb = asset.bounding_box;
  const dx = bb[3] - bb[0];
  const dy = bb[4] - bb[1];
  return Math.atan2(dy, dx);
}

export function assetMaxDim(asset: Asset): number {
  const bb = asset.bounding_box;
  return Math.max(bb[3] - bb[0], bb[4] - bb[1], bb[5] - bb[2]);
}

let _haloTexture: THREE.Texture | null = null;
function haloTexture(): THREE.Texture {
  if (_haloTexture) return _haloTexture;
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = 64;
  const ctx = canvas.getContext("2d")!;
  const g = ctx.createRadialGradient(32, 32, 2, 32, 32, 32);
  g.addColorStop(0, "rgba(255,255,255,0.9)");
  g.addColorStop(0.45, "rgba(255,255,255,0.35)");
  g.addColorStop(1, "rgba(255,255,255,0)");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, 64, 64);
  _haloTexture = new THREE.CanvasTexture(canvas);
  return _haloTexture;
}

interface BuildOptions {
  inventory: boolean; // full meshes vs. subtle markers
  selected: boolean;
  showBBox: boolean;
}

export function buildAssetObject(
  asset: Asset,
  center: [number, number, number],
  opts: BuildOptions
): THREE.Group {
  const group = new THREE.Group();
  const px = asset.center.x - center[0];
  const py = asset.center.y - center[1];
  const pz = asset.center.z - center[2];
  group.position.set(px, py, pz);
  group.userData = { assetId: asset.asset_id, assetClass: asset.class };

  const color = assetColor(asset);
  const yaw = assetYaw(asset);
  const dims = asset.dimensions;
  const height = dims.height_m > 0 ? dims.height_m : assetMaxDim(asset);
  const width = dims.width_m > 0 ? dims.width_m : 0.12;
  const length = dims.length_m > 0 ? dims.length_m : width;
  const ground = asset.geometry?.ground_elevation_m;
  const baseZ = ground != null ? ground - pz : -height / 2; // relative to group origin, LAS z-up

  const solid = new THREE.MeshStandardMaterial({
    color,
    emissive: color,
    emissiveIntensity: opts.selected ? 0.7 : opts.inventory ? 0.3 : 0.15,
    roughness: 0.55,
    metalness: 0.15,
  });

  if (opts.inventory) {
    let mesh: THREE.Mesh | null = null;
    switch (assetKind(asset)) {
      case "pole": {
        const radius = Math.max(width / 2, 0.05);
        mesh = new THREE.Mesh(new THREE.CylinderGeometry(radius, radius * 1.15, height, 12), solid);
        mesh.position.y = baseZ + height / 2;
        break;
      }
      case "sign": {
        const panel = new THREE.Mesh(
          new THREE.BoxGeometry(Math.max(length, 0.3), Math.max(height, 0.3), 0.06),
          solid
        );
        panel.position.y = 0; // panel centered on the asset center
        panel.rotation.y = yaw;
        group.add(panel);
        const postH = Math.max(baseZ + height / 2, 0.3);
        const post = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.06, postH, 8), solid);
        post.position.y = baseZ + postH / 2;
        post.rotation.y = yaw;
        group.add(post);
        break;
      }
      case "elongated": {
        mesh = new THREE.Mesh(new THREE.BoxGeometry(length, Math.max(height, 0.15), Math.max(width, 0.1)), solid);
        mesh.position.y = baseZ + Math.max(height, 0.15) / 2;
        mesh.rotation.y = yaw;
        break;
      }
      case "conductor": {
        mesh = new THREE.Mesh(new THREE.BoxGeometry(length, 0.07, 0.07), solid);
        mesh.rotation.y = yaw;
        break;
      }
      case "marking": {
        const marking = new THREE.Mesh(
          new THREE.BoxGeometry(Math.max(length, 0.2), 0.05, Math.max(width, 0.1)),
          new THREE.MeshStandardMaterial({
            color,
            emissive: color,
            emissiveIntensity: 0.8,
            roughness: 0.4,
          })
        );
        marking.position.y = (ground != null ? ground - pz : baseZ) + 0.025;
        marking.rotation.y = yaw;
        group.add(marking);
        break;
      }
      case "pavement": {
        const slab = new THREE.Mesh(
          new THREE.BoxGeometry(Math.max(length, 2), 0.12, Math.max(width, 2)),
          new THREE.MeshStandardMaterial({
            color,
            emissive: color,
            emissiveIntensity: 0.08,
            transparent: true,
            opacity: 0.4,
            depthWrite: false,
          })
        );
        slab.position.y = (ground != null ? ground - pz : baseZ) + 0.06;
        slab.rotation.y = yaw;
        group.add(slab);
        break;
      }
      default: {
        mesh = new THREE.Mesh(new THREE.BoxGeometry(width, height, width), solid);
        mesh.position.y = baseZ + height / 2;
        break;
      }
    }
    if (mesh) {
      mesh.userData = { assetId: asset.asset_id, assetClass: asset.class };
      group.add(mesh);
    }
  }

  // Marker: halo sprite (always visible in front of the cloud) + core sphere.
  const markerScale = THREE.MathUtils.clamp(assetMaxDim(asset) * 1.35, 0.7, 3.2);
  const sprite = new THREE.Sprite(
    new THREE.SpriteMaterial({
      map: haloTexture(),
      color,
      transparent: true,
      opacity: opts.selected ? 0.95 : opts.inventory ? 0.6 : 0.4,
      depthTest: false,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    })
  );
  sprite.scale.setScalar(markerScale);
  sprite.userData = { assetId: asset.asset_id, assetClass: asset.class };
  sprite.raycast = () => {}; // never occlude clicks on the asset itself
  group.add(sprite);

  if (assetKind(asset) !== "pole") {
    const core = new THREE.Mesh(
      new THREE.SphereGeometry(0.22, 12, 10),
      new THREE.MeshStandardMaterial({
        color,
        emissive: color,
        emissiveIntensity: 0.55,
      })
    );
    core.userData = { assetId: asset.asset_id, assetClass: asset.class };
    group.add(core);
  }

  // Generous invisible hit sphere so small assets are easy to click.
  const hitRadius = THREE.MathUtils.clamp(assetMaxDim(asset) * 0.75, 0.9, 3.6);
  const hit = new THREE.Mesh(
    new THREE.SphereGeometry(hitRadius, 12, 8),
    new THREE.MeshBasicMaterial({
      transparent: true,
      opacity: 0,
      depthWrite: false,
    })
  );
  hit.userData = { assetId: asset.asset_id, assetClass: asset.class, hit: true };
  group.add(hit);

  if (opts.showBBox || opts.selected) {
    const bb = asset.bounding_box;
    const box = new THREE.BoxGeometry(bb[3] - bb[0], bb[5] - bb[2], bb[4] - bb[1]);
    const edges = new THREE.LineSegments(
      new THREE.EdgesGeometry(box),
      new THREE.LineBasicMaterial({
        color,
        transparent: true,
        opacity: opts.selected ? 1 : 0.55,
      })
    );
    // bbox spans LAS z from bb[2]..bb[5]; relative to group origin => y offset in Y-up local
    edges.position.y = (bb[5] + bb[2]) / 2 - pz;
    edges.rotation.y = 0;
    edges.userData = { assetId: asset.asset_id, assetClass: asset.class, wire: true };
    edges.raycast = () => {}; // wireframes must not occlude clicks
    group.add(edges);
  }

  return group;
}