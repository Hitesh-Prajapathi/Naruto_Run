/**
 * Soft contact shadow blob under the character.
 *
 * naruto_run_agent_brief.md §B: "A grounded character reads as grounded
 * largely because of the contact shadow. Even with correct Y placement, a
 * shadowless character floats perceptually." This is the cheap half of the
 * fix (the real shadow-mapped directional light in skyAndLighting.ts is the
 * other half); the blob is what actually sells ground contact frame to
 * frame, because a shadow-mapped shadow from a low sun is cast well off to
 * the side and doesn't read as *contact*.
 *
 * The blob's radial-gradient texture is generated in code rather than
 * shipped as an image file -- it's a handful of canvas calls, and it keeps
 * the asset list smaller.
 */

import * as THREE from "three";
import { JUMP_HEIGHT } from "../config/gameConfig";

const BASE_RADIUS = 0.42;
const BASE_OPACITY = 0.5;
/** Just above the ground plane, to avoid z-fighting with it. */
const GROUND_CLEARANCE = 0.02;

function createBlobTexture(): THREE.Texture {
  const size = 128;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const context = canvas.getContext("2d");
  if (context) {
    const gradient = context.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
    gradient.addColorStop(0, "rgba(0,0,0,0.85)");
    gradient.addColorStop(0.55, "rgba(0,0,0,0.4)");
    gradient.addColorStop(1, "rgba(0,0,0,0)");
    context.fillStyle = gradient;
    context.fillRect(0, 0, size, size);
  }
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

export class CharacterShadow {
  readonly mesh: THREE.Mesh;
  private readonly material: THREE.MeshBasicMaterial;

  constructor(scene: THREE.Scene) {
    const geometry = new THREE.PlaneGeometry(1, 1);
    this.material = new THREE.MeshBasicMaterial({
      map: createBlobTexture(),
      transparent: true,
      depthWrite: false,
      opacity: BASE_OPACITY,
    });
    this.mesh = new THREE.Mesh(geometry, this.material);
    this.mesh.rotation.x = -Math.PI / 2;
    this.mesh.renderOrder = 1;
    scene.add(this.mesh);
  }

  /**
   * Follow the character in X/Z but stay pinned to the ground, shrinking and
   * fading with jump height. That divergence between character and shadow is
   * the entire depth cue: a shadow that scaled with the character, or rose
   * with them, would communicate nothing.
   */
  update(x: number, z: number, feetHeight: number): void {
    const normalisedHeight = Math.min(1, Math.max(0, feetHeight / JUMP_HEIGHT));
    const scale = BASE_RADIUS * 2 * (1 - 0.45 * normalisedHeight);
    this.mesh.position.set(x, GROUND_CLEARANCE, z);
    this.mesh.scale.set(scale, scale, 1);
    this.material.opacity = BASE_OPACITY * (1 - 0.6 * normalisedHeight);
  }

  dispose(): void {
    this.mesh.geometry.dispose();
    this.material.map?.dispose();
    this.material.dispose();
    this.mesh.parent?.remove(this.mesh);
  }
}
