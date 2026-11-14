/**
 * Trackside trees, rocks and grass tufts, parented to jungleTrack's per-tile
 * anchors so they scroll and recycle for free.
 *
 * naruto_run_agent_brief.md §P2.5/§F: the previous version placed a single
 * identical cone at a single scale, which read as obviously procedural.
 * This uses **four** distinct tree silhouettes across randomised scale
 * (±25%), rotation, and per-instance trunk/canopy tint, arranged in depth
 * bands (near / mid / far) so the treeline builds up depth instead of
 * sitting as one flat row.
 *
 * All of it runs once at setup -- nothing here allocates per frame.
 * Geometries and materials are created once and shared across every
 * instance; only lightweight Mesh/Group wrappers are per-instance.
 */

import * as THREE from "three";
import {
  ROCK_SIZE_MAX,
  ROCK_SIZE_MIN,
  TRACK_WIDTH,
  TREE_HEIGHT_MAX,
  TREE_HEIGHT_MIN,
} from "../config/gameConfig";
import type { JungleTrack } from "./jungleTrack";

const EDGE_MARGIN = 2.4;

/**
 * Depth bands (brief §F): distance out from the track edge, and how many
 * trees per tile sit in each. Far-band trees are larger so they read as a
 * silhouette treeline rather than individual trees.
 *
 * The near band is deliberately held back from the verge and kept short:
 * a first pass started it at the track edge with full-height trees, which
 * put 10-unit trunks ~4 units from the centre line. They loomed over the
 * lane and swallowed the sky, making the path feel like a tunnel.
 */
const BANDS = [
  { minOffset: 1.5, maxOffset: 7, scale: 0.55, count: 2 },
  { minOffset: 7, maxOffset: 18, scale: 0.9, count: 2 },
  { minOffset: 18, maxOffset: 40, scale: 1.2, count: 3 },
] as const;

/** Deterministic pseudo-random in [0,1), seeded by index -- keeps the layout
 * stable across reloads (easier to eyeball changes) without a dependency. */
function pseudoRandom(seed: number): number {
  const value = Math.sin(seed * 12.9898) * 43758.5453;
  return value - Math.floor(value);
}

interface SharedAssets {
  trunkGeometries: THREE.BufferGeometry[];
  canopyGeometries: THREE.BufferGeometry[][];
  trunkMaterials: THREE.Material[];
  canopyMaterials: THREE.Material[];
  rockGeometry: THREE.BufferGeometry;
  rockMaterial: THREE.Material;
  tuftGeometry: THREE.BufferGeometry;
  tuftMaterial: THREE.Material;
}

function buildSharedAssets(): SharedAssets {
  // Four silhouettes: tall narrow conifer, broad conifer, layered
  // three-tier pine, and a rounded broadleaf.
  const canopyGeometries: THREE.BufferGeometry[][] = [
    [new THREE.ConeGeometry(0.9, 4.2, 7)],
    [new THREE.ConeGeometry(1.7, 3.0, 8)],
    [
      new THREE.ConeGeometry(1.7, 1.7, 8),
      new THREE.ConeGeometry(1.35, 1.6, 8),
      new THREE.ConeGeometry(0.95, 1.5, 8),
    ],
    [new THREE.SphereGeometry(1.5, 9, 7)],
  ];
  const trunkGeometries = [
    new THREE.CylinderGeometry(0.13, 0.2, 2.6, 6),
    new THREE.CylinderGeometry(0.16, 0.26, 2.0, 6),
    new THREE.CylinderGeometry(0.15, 0.24, 2.2, 6),
    new THREE.CylinderGeometry(0.2, 0.3, 2.4, 6),
  ];
  const trunkMaterials = [0x4a3122, 0x5b3a21, 0x3f2b1d].map(
    (color) => new THREE.MeshStandardMaterial({ color, roughness: 0.95 }),
  );
  const canopyMaterials = [0x2f7d3a, 0x27692f, 0x386b2c, 0x1f5a2a, 0x4a8438].map(
    (color) => new THREE.MeshStandardMaterial({ color, roughness: 0.9, flatShading: true }),
  );

  return {
    trunkGeometries,
    canopyGeometries,
    trunkMaterials,
    canopyMaterials,
    rockGeometry: new THREE.DodecahedronGeometry(1),
    rockMaterial: new THREE.MeshStandardMaterial({ color: 0x6f6a63, roughness: 1, flatShading: true }),
    tuftGeometry: new THREE.ConeGeometry(0.22, 0.5, 4),
    tuftMaterial: new THREE.MeshStandardMaterial({ color: 0x547a33, roughness: 1, flatShading: true }),
  };
}

function buildTree(assets: SharedAssets, seed: number, bandScale: number): THREE.Group {
  const group = new THREE.Group();
  const kind = Math.floor(pseudoRandom(seed) * assets.canopyGeometries.length);

  const trunkGeometry = assets.trunkGeometries[kind]!;
  const trunkMaterial = assets.trunkMaterials[Math.floor(pseudoRandom(seed + 3.1) * assets.trunkMaterials.length)]!;
  const trunk = new THREE.Mesh(trunkGeometry, trunkMaterial);
  const trunkHeight = (trunkGeometry as THREE.CylinderGeometry).parameters.height;
  trunk.position.y = trunkHeight / 2;
  trunk.castShadow = true;
  group.add(trunk);

  const canopyMaterial =
    assets.canopyMaterials[Math.floor(pseudoRandom(seed + 7.7) * assets.canopyMaterials.length)]!;
  let stackY = trunkHeight;
  for (const geometry of assets.canopyGeometries[kind]!) {
    const canopy = new THREE.Mesh(geometry, canopyMaterial);
    const height =
      geometry instanceof THREE.ConeGeometry
        ? geometry.parameters.height
        : (geometry as THREE.SphereGeometry).parameters.radius * 2;
    canopy.position.y = stackY + height / 2;
    canopy.castShadow = true;
    group.add(canopy);
    // Overlap each tier slightly so a layered pine reads as one canopy.
    stackY += height * 0.55;
  }

  // Normalise to a real height, then apply the ±25% per-instance variation
  // and the band's own scale multiplier.
  const naturalHeight = new THREE.Box3().setFromObject(group).max.y;
  const targetHeight =
    (TREE_HEIGHT_MIN + pseudoRandom(seed + 11.3) * (TREE_HEIGHT_MAX - TREE_HEIGHT_MIN)) * bandScale;
  const variation = 0.75 + pseudoRandom(seed + 13.9) * 0.5;
  group.scale.setScalar((targetHeight / Math.max(naturalHeight, 0.001)) * variation);
  group.rotation.y = pseudoRandom(seed + 17.1) * Math.PI * 2;

  return group;
}

export function addEnvironmentProps(track: JungleTrack): void {
  const assets = buildSharedAssets();
  const edgeX = TRACK_WIDTH / 2 + EDGE_MARGIN;

  track.anchors.forEach((anchor, tileIndex) => {
    for (const side of [-1, 1] as const) {
      let bandIndex = 0;
      for (const band of BANDS) {
        for (let i = 0; i < band.count; i += 1) {
          const seed = tileIndex * 131 + bandIndex * 29 + i * 7 + (side === -1 ? 1 : 2) * 3;
          const offset = band.minOffset + pseudoRandom(seed) * (band.maxOffset - band.minOffset);
          const tree = buildTree(assets, seed, band.scale);
          tree.position.set(side * (edgeX + offset), 0, (pseudoRandom(seed + 0.4) - 0.5) * 9);
          anchor.add(tree);
        }
        bandIndex += 1;
      }

      // A rock and a couple of grass tufts right at the road edge -- the
      // near-ground detail that makes the path/verge boundary read as
      // organic rather than a hard geometric line.
      const rockSeed = tileIndex * 53 + (side === -1 ? 5 : 9);
      if (pseudoRandom(rockSeed) > 0.45) {
        const rock = new THREE.Mesh(assets.rockGeometry, assets.rockMaterial);
        const size = ROCK_SIZE_MIN + pseudoRandom(rockSeed + 1.7) * (ROCK_SIZE_MAX - ROCK_SIZE_MIN);
        rock.scale.set(size, size * 0.7, size);
        rock.position.set(side * (edgeX + pseudoRandom(rockSeed + 2.3) * 1.5), size * 0.25, (pseudoRandom(rockSeed + 3.1) - 0.5) * 8);
        rock.rotation.set(pseudoRandom(rockSeed + 4) * 0.6, pseudoRandom(rockSeed + 5) * Math.PI, pseudoRandom(rockSeed + 6) * 0.6);
        rock.castShadow = true;
        rock.receiveShadow = true;
        anchor.add(rock);
      }

      for (let i = 0; i < 3; i += 1) {
        const tuftSeed = tileIndex * 71 + i * 13 + (side === -1 ? 2 : 8);
        const tuft = new THREE.Mesh(assets.tuftGeometry, assets.tuftMaterial);
        const scale = 0.7 + pseudoRandom(tuftSeed) * 0.8;
        tuft.scale.setScalar(scale);
        tuft.position.set(
          side * (TRACK_WIDTH / 2 + 0.15 + pseudoRandom(tuftSeed + 1.1) * 1.1),
          0.12 * scale,
          (pseudoRandom(tuftSeed + 2.2) - 0.5) * 9,
        );
        tuft.rotation.y = pseudoRandom(tuftSeed + 3.3) * Math.PI;
        anchor.add(tuft);
      }
    }
  });
}
