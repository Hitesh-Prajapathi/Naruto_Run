/**
 * Real jungle set-dressing harvested from the Forest_Road asset
 * (Character_3D_Models/Forest_Road, a Wii-adjacent "Dirt Road Forest Scene"
 * pack) -- a fallen log cluster, a mud pile, a broken-rock cluster, and a
 * wooden fence run, placed once alongside the track near the start.
 *
 * Deliberate scope decision: the pack's actual *road* mesh is a winding,
 * curved path (confirmed by rendering it, not guessing) that cannot be
 * sliced into repeating straight tiles for our fixed 3-lane mechanic without
 * a full lane/camera rearchitecture. The user chose (see the conversation)
 * to keep straight recycling lanes and only harvest textures/props from this
 * pack rather than take on that rearchitecture. jungleTrack.ts's ground
 * material now uses this pack's real dirt-road textures; this file supplies
 * the props.
 *
 * Each source object is itself a merged, multi-instance cluster (e.g. the
 * "Wood_Log" object is many logs baked into one mesh spanning ~100 units),
 * not a single reusable prop -- there is no per-instance placement data left
 * once a scene is exported to OBJ. Duplicating a whole cluster many times
 * across the recycled-tile belt would be excessive, so these are placed
 * *once*, as static set-dressing near the start of the track, not part of
 * the recycling pool. They will not reappear on later loops -- an explicit,
 * documented scope choice, not an oversight.
 */

import * as THREE from "three";
import { OBJLoader } from "three/examples/jsm/loaders/OBJLoader.js";
import { TRACK_WIDTH } from "../config/gameConfig";

const MODEL_BASE = "/models/forest-props/";
const TEXTURE_BASE = "/textures/forest-road/";

interface PropDefinition {
  file: string;
  diffuse: string;
  normal?: string;
  roughness?: string;
  /** Desired world-space width (max of scaled X/Z extent) after scaling
   * down from the source cluster's real-world-diorama scale. */
  targetWidth: number;
  position: { x: number; z: number };
  rotationY?: number;
}

const PROPS: PropDefinition[] = [
  {
    file: "wood_log.obj",
    diffuse: "Wood_Log_Diffuse.png",
    targetWidth: 5,
    position: { x: -(TRACK_WIDTH / 2 + 2.2), z: -18 },
    rotationY: 0.3,
  },
  {
    file: "mud_pile.obj",
    diffuse: "Wall_Mud_Diffuse.png",
    normal: "Wall_Mud_Normal.png",
    roughness: "Wall_Mud_Roughness.png",
    targetWidth: 4,
    position: { x: TRACK_WIDTH / 2 + 1.6, z: -26 },
  },
  {
    file: "broken_rocks.obj",
    diffuse: "Broken_Rocks_Diffuse.jpeg",
    normal: "Broken_Rocks_Normal.jpeg",
    roughness: "Broken_Rocks_Roughness.jpeg",
    targetWidth: 6,
    position: { x: -(TRACK_WIDTH / 2 + 2.6), z: -34 },
  },
  {
    file: "wood_fence.obj",
    diffuse: "Wood_Fence_Diffuse.png",
    targetWidth: 7,
    position: { x: TRACK_WIDTH / 2 + 2, z: -42 },
    rotationY: -0.15,
  },
];

async function loadProp(scene: THREE.Scene, definition: PropDefinition): Promise<THREE.Group> {
  const loader = new THREE.TextureLoader();
  const diffuse = loader.load(`${TEXTURE_BASE}${definition.diffuse}`);
  diffuse.colorSpace = THREE.SRGBColorSpace;
  const normal = definition.normal ? loader.load(`${TEXTURE_BASE}${definition.normal}`) : null;
  const roughness = definition.roughness ? loader.load(`${TEXTURE_BASE}${definition.roughness}`) : null;
  const material = new THREE.MeshStandardMaterial({
    map: diffuse,
    normalMap: normal,
    roughnessMap: roughness,
  });

  const object = await new OBJLoader().loadAsync(`${MODEL_BASE}${definition.file}`);
  object.traverse((child) => {
    const mesh = child as THREE.Mesh;
    if (mesh.isMesh) {
      mesh.material = material;
    }
  });

  // Scale from the source diorama's real-world scale down to ours, then
  // place the (now-scaled) bounding box's base at ground level and its
  // horizontal center at the requested track-relative position.
  const rawBox = new THREE.Box3().setFromObject(object);
  const rawSize = rawBox.getSize(new THREE.Vector3());
  const scale = definition.targetWidth / Math.max(rawSize.x, rawSize.z);
  object.scale.setScalar(scale);

  const scaledBox = new THREE.Box3().setFromObject(object);
  const scaledCenter = scaledBox.getCenter(new THREE.Vector3());
  object.position.set(
    definition.position.x - scaledCenter.x,
    -scaledBox.min.y,
    definition.position.z - scaledCenter.z,
  );
  if (definition.rotationY) {
    object.rotation.y = definition.rotationY;
  }

  scene.add(object);
  return object;
}

/**
 * Advances every placed prop by the same `deltaZ` the track/obstacles move
 * by each frame. Without this, these props would sit at a fixed world Z
 * forever while everything else simulates forward motion by shifting
 * position -- looking like they float in place as the ground rushes past
 * them. Unlike ObstacleSpawner, there is no recycling: once a prop scrolls
 * past the camera it is simply left behind, matching the "one-time set
 * dressing near the start" scope documented above.
 */
export function updateForestProps(props: readonly THREE.Group[], deltaZ: number): void {
  for (const prop of props) {
    prop.position.z += deltaZ;
  }
}

/** Loads and places every prop. Failures are logged and skipped individually
 * -- one missing/broken asset should not block the rest of the scene. */
export async function addForestProps(scene: THREE.Scene): Promise<THREE.Group[]> {
  const results = await Promise.allSettled(PROPS.map((definition) => loadProp(scene, definition)));
  const placed: THREE.Group[] = [];
  results.forEach((result, index) => {
    if (result.status === "fulfilled") {
      placed.push(result.value);
    } else {
      console.error(`forestProps: failed to load ${PROPS[index]!.file}`, result.reason);
    }
  });
  return placed;
}
