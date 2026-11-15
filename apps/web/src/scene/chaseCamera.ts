/**
 * Third-person chase camera -- naruto_run_agent_brief.md §C.
 *
 * The previous camera sat at ground level looking slightly *up*, which put
 * the horizon through the middle of the frame and let the character occlude
 * approaching obstacles. That's a gameplay failure, not just an ugly shot:
 * the player couldn't see what they were meant to dodge.
 *
 * Two behaviours here do the heavy lifting:
 * - **Damped follow, not hard parenting.** The lag is what gives lane
 *   changes weight. Damping uses `1 - exp(-lambda*dt)` rather than a raw
 *   `lerp(a, b, k)`, so the smoothing rate is identical at 30, 60 and 144 Hz
 *   instead of silently becoming snappier at higher frame rates.
 * - **Partial lateral follow.** The camera tracks only ~half the player's
 *   sideways offset, so a lane change actually moves the character *within*
 *   the frame. A 1:1 follow would keep them pinned centre-screen and make
 *   the lane change nearly invisible.
 */

import * as THREE from "three";
import {
  CAMERA_FOLLOW_LAMBDA,
  CAMERA_LATERAL_FOLLOW,
  CAMERA_LOOK_AHEAD_Z,
  CAMERA_LOOK_AT_HEIGHT,
  CAMERA_POSITION,
} from "../config/gameConfig";

export class ChaseCamera {
  private readonly desiredPosition = new THREE.Vector3();
  private readonly lookAtTarget = new THREE.Vector3();
  private readonly shakeOffset = new THREE.Vector3();
  private shakeStrength = 0;
  private shakeElapsed = 0;
  private shakeDuration = 0;

  constructor(private readonly camera: THREE.PerspectiveCamera) {
    this.camera.position.set(CAMERA_POSITION.x, CAMERA_POSITION.y, CAMERA_POSITION.z);
  }

  /** Brief §I: a collision must be *felt*. Short, decaying positional noise. */
  shake(strength = 0.35, duration = 0.32): void {
    this.shakeStrength = strength;
    this.shakeDuration = duration;
    this.shakeElapsed = 0;
  }

  /** Snap straight to the target with no damping -- used on the first frame
   * so the camera doesn't visibly fly in from its constructor position. */
  snapTo(playerX: number): void {
    this.computeDesired(playerX);
    this.camera.position.copy(this.desiredPosition);
    this.aim(playerX);
  }

  /**
   * Optional secondary framing blended over the running rig, 0..1.
   *
   * Added for the boss encounter (Feature Brief 02 §5.6), which asks to
   * "reuse the existing smooth-damp follow -- do not write a second camera
   * system, just feed it different target values". At `blend === 0` every
   * computation below is bit-for-bit the running behaviour, so the approved
   * camera is untouched whenever no encounter is active; and because the
   * running values are read from config rather than stored mutably, exiting
   * an encounter restores them exactly by construction.
   */
  private framingBlend = 0;
  private overridePosition = { x: 0, y: 0, z: 0 };
  private overrideLookAheadZ = 0;
  private overrideLookAtHeight = 0;

  setFramingOverride(
    blend: number,
    position: { x: number; y: number; z: number },
    lookAheadZ: number,
    lookAtHeight: number,
  ): void {
    this.framingBlend = Math.min(1, Math.max(0, blend));
    this.overridePosition = position;
    this.overrideLookAheadZ = lookAheadZ;
    this.overrideLookAtHeight = lookAtHeight;
  }

  /** Drop any override and return to the pure running framing. */
  clearFramingOverride(): void {
    this.framingBlend = 0;
  }

  private mix(runningValue: number, overrideValue: number): number {
    return this.framingBlend <= 0
      ? runningValue
      : runningValue + (overrideValue - runningValue) * this.framingBlend;
  }

  private computeDesired(playerX: number): void {
    // Lateral follow eases out as the encounter framing takes over, so the
    // duel stays centred instead of drifting with the player's lane.
    const lateral = playerX * CAMERA_LATERAL_FOLLOW * (1 - this.framingBlend);
    this.desiredPosition.set(
      lateral + this.mix(CAMERA_POSITION.x, this.overridePosition.x),
      this.mix(CAMERA_POSITION.y, this.overridePosition.y),
      this.mix(CAMERA_POSITION.z, this.overridePosition.z),
    );
  }

  private aim(playerX: number): void {
    this.lookAtTarget.set(
      playerX * CAMERA_LATERAL_FOLLOW * (1 - this.framingBlend),
      this.mix(CAMERA_LOOK_AT_HEIGHT, this.overrideLookAtHeight),
      this.mix(CAMERA_LOOK_AHEAD_Z, this.overrideLookAheadZ),
    );
    this.camera.lookAt(this.lookAtTarget);
  }

  update(dt: number, playerX: number): void {
    this.computeDesired(playerX);

    // Frame-rate-independent exponential damping.
    const alpha = 1 - Math.exp(-CAMERA_FOLLOW_LAMBDA * dt);
    this.camera.position.lerp(this.desiredPosition, alpha);

    if (this.shakeElapsed < this.shakeDuration) {
      this.shakeElapsed += dt;
      const remaining = Math.max(0, 1 - this.shakeElapsed / this.shakeDuration);
      const amount = this.shakeStrength * remaining * remaining;
      this.shakeOffset.set(
        (Math.random() * 2 - 1) * amount,
        (Math.random() * 2 - 1) * amount,
        0,
      );
      this.camera.position.add(this.shakeOffset);
    }

    this.aim(playerX);
  }
}
