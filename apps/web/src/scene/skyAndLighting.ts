/**
 * Warm late-afternoon forest lighting, gradient sky, and distance fog --
 * naruto_run_agent_brief.md §G and §F. Replaces the previous flat ambient +
 * flat `#87CEEB` background, which left everything uniformly lit with no
 * tonal separation between path and foliage.
 *
 * Fog is the single cheapest depth cue available and it doubles as the fix
 * for a real problem: it hides the far end of the recycled track belt, so
 * the spawn seam is never visible.
 */

import * as THREE from "three";
import { LANE_X, OBSTACLE_SPAWN_Z, TRACK_WIDTH } from "../config/gameConfig";

const SKY_TOP = new THREE.Color(0x4a83c4);
const SKY_HORIZON = new THREE.Color(0xdcc9a4);
/** Fog is matched to the *horizon* colour, not the zenith -- objects recede
 * toward the horizon, so matching the zenith leaves a visible colour seam. */
const FOG_COLOR = new THREE.Color(0xc8bda0);
const FOG_NEAR = 26;
const FOG_FAR = 115;

const SUN_COLOR = 0xffe8c0;
const HEMI_SKY = 0x9fc4e8;
const HEMI_GROUND = 0x4a5a34;

export interface SkyAndLighting {
  sun: THREE.DirectionalLight;
  hemisphere: THREE.HemisphereLight;
  skyDome: THREE.Mesh;
}

/** Vertical gradient sky, painted onto a canvas texture mapped to the inside
 * of a large sphere. Cheaper and simpler to tune than a six-image skybox,
 * and a runner only ever sees a narrow slice of sky anyway. */
function createSkyDome(): THREE.Mesh {
  const canvas = document.createElement("canvas");
  canvas.width = 4;
  canvas.height = 256;
  const context = canvas.getContext("2d");
  if (context) {
    const gradient = context.createLinearGradient(0, 0, 0, canvas.height);
    gradient.addColorStop(0, `#${SKY_TOP.getHexString()}`);
    gradient.addColorStop(0.62, "#9fc0d8");
    gradient.addColorStop(1, `#${SKY_HORIZON.getHexString()}`);
    context.fillStyle = gradient;
    context.fillRect(0, 0, canvas.width, canvas.height);
  }
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;

  const geometry = new THREE.SphereGeometry(260, 24, 16);
  const material = new THREE.MeshBasicMaterial({
    map: texture,
    side: THREE.BackSide,
    depthWrite: false,
    fog: false, // the dome IS the horizon; fogging it would grey it out
  });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.renderOrder = -1;
  return mesh;
}

export function addSkyAndLighting(scene: THREE.Scene): SkyAndLighting {
  scene.fog = new THREE.Fog(FOG_COLOR, FOG_NEAR, FOG_FAR);

  const skyDome = createSkyDome();
  scene.add(skyDome);

  const hemisphere = new THREE.HemisphereLight(HEMI_SKY, HEMI_GROUND, 1.1);
  scene.add(hemisphere);

  const sun = new THREE.DirectionalLight(SUN_COLOR, 2.2);
  // Low and to the side: a high sun casts a shadow directly under the
  // character where it's hidden by their own body, which defeats the point.
  sun.position.set(-14, 12, 10);
  sun.target.position.set(0, 0, -18);
  sun.castShadow = true;

  // Tight shadow frustum around the play area only (brief §B). The visible
  // track is ~3 lanes wide and the useful shadow range is the near stretch
  // of path -- wasting the shadow map on the full OBSTACLE_SPAWN_Z distance
  // would make the character's own shadow blocky.
  const halfWidth = TRACK_WIDTH / 2 + Math.abs(LANE_X[0]) + 4;
  const shadowDepth = Math.min(46, Math.abs(OBSTACLE_SPAWN_Z));
  sun.shadow.camera.left = -halfWidth;
  sun.shadow.camera.right = halfWidth;
  sun.shadow.camera.top = shadowDepth;
  sun.shadow.camera.bottom = -shadowDepth;
  sun.shadow.camera.near = 1;
  sun.shadow.camera.far = 70;
  sun.shadow.mapSize.set(1024, 1024);
  sun.shadow.bias = -0.0012;
  sun.shadow.normalBias = 0.02;

  scene.add(sun);
  scene.add(sun.target);

  return { sun, hemisphere, skyDome };
}

/**
 * Keep the sun (and therefore its tight shadow frustum) centred on the
 * player. A directional light's shadow camera is a fixed box in world space;
 * without this the character runs out of it and their shadow vanishes.
 */
export function updateSunTarget(lighting: SkyAndLighting, playerX: number): void {
  lighting.sun.position.set(playerX - 14, 12, 10);
  lighting.sun.target.position.set(playerX, 0, -18);
  lighting.sun.target.updateMatrixWorld();
}
