/**
 * Tiled, recycled ground plane -- the "infinite runner" illusion built from
 * a small fixed pool of tiles (game_implementation_plan.md Phase C:
 * "recycled 40u tiles, object-pooled"). Positioning math lives in
 * trackRecycling.ts so it's unit-tested independent of Three.js; this file
 * is just the thin scene-graph wrapper around it.
 */

import * as THREE from "three";
import {
  GROUND_RECYCLE_THRESHOLD_Z,
  GROUND_TILE_COUNT,
  GROUND_TILE_LENGTH,
  TERRAIN_WIDTH,
  TRACK_WIDTH,
} from "../config/gameConfig";
import { advanceTilePositions, initialTilePositions } from "./trackRecycling";

const GRASS_COLOR = 0x3f5c2a;

const TEXTURE_BASE = "/textures/forest-road/";
/** How many times the dirt-road texture repeats across one tile's width and
 * length (every tile shares one geometry and one material, so this applies
 * per tile, not across the whole belt -- the pattern doesn't continue
 * seamlessly tile-to-tile, the same way the flat color it replaces didn't
 * need to either). Tuned by eye against the texture's apparent grain size,
 * not derived from any real-world unit the source asset carries. */
const TEXTURE_REPEAT_X = 1.5;
const TEXTURE_REPEAT_Y = 2;

function loadGroundMaterial(): THREE.MeshStandardMaterial {
  const loader = new THREE.TextureLoader();

  const configureRepeat = (texture: THREE.Texture): void => {
    texture.wrapS = THREE.RepeatWrapping;
    texture.wrapT = THREE.RepeatWrapping;
    texture.repeat.set(TEXTURE_REPEAT_X, TEXTURE_REPEAT_Y);
  };

  const diffuse = loader.load(`${TEXTURE_BASE}Dirt_Road_Diffuse.png`);
  diffuse.colorSpace = THREE.SRGBColorSpace;
  configureRepeat(diffuse);

  const normal = loader.load(`${TEXTURE_BASE}Dirt_Road_Normal.png`);
  configureRepeat(normal);

  const roughness = loader.load(`${TEXTURE_BASE}Dirt_Road_Roughness.png`);
  configureRepeat(roughness);

  return new THREE.MeshStandardMaterial({
    map: diffuse,
    normalMap: normal,
    roughnessMap: roughness,
    // DoubleSide: a rotated PlaneGeometry's winding-order-determined front
    // face is easy to get backwards relative to the camera, and there is no
    // real cost to a ground plane being visible "from below" too.
    side: THREE.DoubleSide,
  });
}

export class JungleTrack {
  private readonly tiles: THREE.Mesh[];
  /** One unrotated anchor per tile, positioned at that tile's center, for
   * environmentProps.ts to parent props to -- the ground mesh itself is
   * rotated flat (rotation.x = -PI/2), which would rotate any naive child
   * along with it, so props get their own identity-rotation anchor instead. */
  private readonly segmentAnchors: THREE.Object3D[];
  /** Wide grass planes flanking the path, recycled in lockstep with the
   * dirt tiles. */
  private readonly terrainTiles: THREE.Mesh[];
  private readonly terrainGeometry: THREE.BufferGeometry;
  private readonly terrainMaterial: THREE.Material;
  private positions: number[];

  constructor(scene: THREE.Scene) {
    const geometry = new THREE.PlaneGeometry(TRACK_WIDTH, GROUND_TILE_LENGTH);
    const material = loadGroundMaterial();

    // initialTilePositions() starts the frontmost tile's near edge at 0.
    // Shifted so it instead starts at the recycle threshold -- the maximum
    // extent the belt ever reaches -- so the "no gap in front of the camera"
    // invariant holds from the very first frame, not only after the belt
    // organically scrolls forward long enough to reach that state on its own.
    this.positions = initialTilePositions(GROUND_TILE_COUNT, GROUND_TILE_LENGTH).map(
      (z) => z + GROUND_RECYCLE_THRESHOLD_Z,
    );
    // Grass terrain flanking the path. One wide plane per tile, sitting a
    // hair *below* the path so the dirt road always wins the depth test
    // where they overlap rather than z-fighting along the road edges.
    const terrainGeometry = new THREE.PlaneGeometry(TERRAIN_WIDTH, GROUND_TILE_LENGTH);
    const terrainMaterial = new THREE.MeshStandardMaterial({
      color: GRASS_COLOR,
      roughness: 1,
      side: THREE.DoubleSide,
    });
    this.terrainGeometry = terrainGeometry;
    this.terrainMaterial = terrainMaterial;
    this.terrainTiles = [];

    this.tiles = [];
    this.segmentAnchors = [];
    for (const z of this.positions) {
      const centerZ = z - GROUND_TILE_LENGTH / 2;

      const terrain = new THREE.Mesh(terrainGeometry, terrainMaterial);
      terrain.receiveShadow = true;
      terrain.rotation.x = -Math.PI / 2;
      terrain.position.set(0, -0.02, centerZ);
      scene.add(terrain);
      this.terrainTiles.push(terrain);

      const mesh = new THREE.Mesh(geometry, material);
      mesh.receiveShadow = true; // the path is what the character's shadow lands on
      mesh.rotation.x = -Math.PI / 2;
      // A plane's own origin is its center; offset half a tile so each
      // tile's *far* edge sits at `z`, keeping the strip seamless end to end.
      mesh.position.set(0, 0, centerZ);
      scene.add(mesh);
      this.tiles.push(mesh);

      const anchor = new THREE.Object3D();
      anchor.position.set(0, 0, centerZ);
      scene.add(anchor);
      this.segmentAnchors.push(anchor);
    }
  }

  /** Advance the belt by `deltaZ` world units (positive = toward camera). */
  update(deltaZ: number): void {
    this.positions = advanceTilePositions(this.positions, deltaZ, GROUND_TILE_LENGTH, GROUND_RECYCLE_THRESHOLD_Z);
    for (let i = 0; i < this.tiles.length; i += 1) {
      const centerZ = this.positions[i]! - GROUND_TILE_LENGTH / 2;
      this.tiles[i]!.position.z = centerZ;
      this.terrainTiles[i]!.position.z = centerZ;
      this.segmentAnchors[i]!.position.z = centerZ;
    }
  }

  /** Unrotated per-tile anchors for environmentProps.ts. */
  get anchors(): readonly THREE.Object3D[] {
    return this.segmentAnchors;
  }

  /** Ground mesh Z for a given tile index -- exposed for tests to confirm it
   * stays in sync with the matching anchor's Z, not for scene code to use. */
  tileMeshZ(index: number): number {
    return this.tiles[index]!.position.z;
  }

  dispose(): void {
    // Every tile shares one geometry/material pair, so those are disposed
    // once outside the loop rather than once per tile.
    for (const tile of this.tiles) {
      tile.parent?.remove(tile);
    }
    for (const terrain of this.terrainTiles) {
      terrain.parent?.remove(terrain);
    }
    for (const anchor of this.segmentAnchors) {
      anchor.parent?.remove(anchor);
    }
    const first = this.tiles[0];
    if (first) {
      first.geometry.dispose();
      if (Array.isArray(first.material)) {
        for (const material of first.material) material.dispose();
      } else {
        first.material.dispose();
      }
    }
    this.terrainGeometry.dispose();
    this.terrainMaterial.dispose();
  }
}
