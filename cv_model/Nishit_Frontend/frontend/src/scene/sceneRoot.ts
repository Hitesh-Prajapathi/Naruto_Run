/**
 * Owns the Three.js renderer, scene, camera, and the render loop itself.
 * Everything else in scene/ and game/ receives the `THREE.Scene` from here
 * rather than creating its own.
 *
 * This is a plain requestAnimationFrame loop with delta clamping -- not the
 * fixed-timestep simulation the plan calls for later. Phase F's
 * gameController owns "fixed simulation timestep with render interpolation"
 * for actual gameplay logic; sceneRoot's job in Phase C is just "render
 * smoothly and don't leak," which a variable-timestep rAF loop already does.
 */

import * as THREE from "three";
import { CAMERA_FAR, CAMERA_FOV_DEG, CAMERA_NEAR, CAMERA_POSITION } from "../config/gameConfig";

/**
 * @param deltaSeconds frame delta, clamped to MAX_DELTA_SECONDS. Use this for
 *   anything that integrates -- it is what keeps a long stall from
 *   teleporting the player.
 * @param unclampedDeltaSeconds true wall-clock time since the last frame.
 *   Only for things that must be measured in *real* seconds regardless of
 *   frame rate: below 15fps the clamped delta runs slower than reality, so a
 *   countdown driven from it silently lasts longer than it claims.
 */
export type FrameCallback = (
  deltaSeconds: number,
  elapsedSeconds: number,
  unclampedDeltaSeconds: number,
) => void;

/** Longest delta ever handed to a callback, regardless of a real stall (tab
 * switch, breakpoint, slow GC). Prevents one long frame from teleporting
 * anything that integrates position by delta. */
const MAX_DELTA_SECONDS = 1 / 15;

export class SceneRoot {
  readonly scene: THREE.Scene;
  readonly camera: THREE.PerspectiveCamera;
  readonly renderer: THREE.WebGLRenderer;

  private readonly callbacks = new Set<FrameCallback>();
  private rafHandle: number | null = null;
  private lastTimestampMs: number | null = null;
  private elapsedSeconds = 0;

  constructor(private readonly canvas: HTMLCanvasElement) {
    this.scene = new THREE.Scene();

    this.camera = new THREE.PerspectiveCamera(
      CAMERA_FOV_DEG,
      canvas.clientWidth / Math.max(1, canvas.clientHeight),
      CAMERA_NEAR,
      CAMERA_FAR,
    );
    this.camera.position.set(CAMERA_POSITION.x, CAMERA_POSITION.y, CAMERA_POSITION.z);

    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    // PCFSoftShadowMap is deprecated in this three version (it warns and
    // silently falls back to PCF anyway), so ask for PCF directly.
    this.renderer.shadowMap.type = THREE.PCFShadowMap;
    // Mild filmic tone mapping (brief §G) keeps the warm key light from
    // clipping to white on the character's bright orange jumpsuit.
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.15;
    this.resize();
  }

  /** Register a per-frame callback. Returns an unsubscribe function. */
  onFrame(callback: FrameCallback): () => void {
    this.callbacks.add(callback);
    return () => {
      this.callbacks.delete(callback);
    };
  }

  resize(): void {
    const width = this.canvas.clientWidth || 1;
    const height = this.canvas.clientHeight || 1;
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
  }

  start(): void {
    if (this.rafHandle !== null) {
      return;
    }
    this.lastTimestampMs = null;
    const tick = (timestampMs: number): void => {
      const previous = this.lastTimestampMs;
      this.lastTimestampMs = timestampMs;
      const rawDelta = previous === null ? 0 : (timestampMs - previous) / 1000;
      const delta = Math.min(Math.max(rawDelta, 0), MAX_DELTA_SECONDS);
      this.elapsedSeconds += delta;

      for (const callback of this.callbacks) {
        callback(delta, this.elapsedSeconds, Math.max(rawDelta, 0));
      }
      this.renderer.render(this.scene, this.camera);
      this.rafHandle = requestAnimationFrame(tick);
    };
    this.rafHandle = requestAnimationFrame(tick);
  }

  stop(): void {
    if (this.rafHandle !== null) {
      cancelAnimationFrame(this.rafHandle);
      this.rafHandle = null;
    }
  }

  dispose(): void {
    this.stop();
    this.callbacks.clear();
    this.renderer.dispose();
  }
}
