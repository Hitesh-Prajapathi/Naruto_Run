import { describe, expect, it } from "vitest";
import * as THREE from "three";
import { ChaseCamera } from "../../src/scene/chaseCamera";
import { CAMERA_LATERAL_FOLLOW } from "../../src/config/gameConfig";

function makeCamera(): THREE.PerspectiveCamera {
  return new THREE.PerspectiveCamera(65, 16 / 9, 0.1, 300);
}

describe("ChaseCamera placement", () => {
  it("sits above and behind the origin, per the brief's rig", () => {
    const camera = makeCamera();
    new ChaseCamera(camera);
    expect(camera.position.y).toBeGreaterThan(2);
    expect(camera.position.z).toBeGreaterThan(0); // behind the player at z=0
  });

  it("looks downward and forward down the track", () => {
    const camera = makeCamera();
    const chase = new ChaseCamera(camera);
    chase.snapTo(0);

    const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(camera.quaternion);
    expect(forward.z).toBeLessThan(0); // pointing down-track
    expect(forward.y).toBeLessThan(0); // pitched down, not up

    // Brief §C: pitched down roughly 10-15 degrees.
    const pitchDeg = THREE.MathUtils.radToDeg(Math.asin(-forward.y));
    expect(pitchDeg).toBeGreaterThan(8);
    expect(pitchDeg).toBeLessThan(20);
  });

  it("can actually see at least 25 units of path ahead of the player", () => {
    // Brief §C: the player must be able to see approaching obstacles. Rather
    // than assert an arbitrary distance between the camera and its look
    // target, this projects real ground points into the camera frustum and
    // checks they're genuinely visible -- that's the property that matters.
    const camera = makeCamera();
    const chase = new ChaseCamera(camera);
    chase.snapTo(0);
    camera.updateMatrixWorld(true);
    camera.updateProjectionMatrix();

    const frustum = new THREE.Frustum().setFromProjectionMatrix(
      new THREE.Matrix4().multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse),
    );

    // Ground points straight down the middle of the track, from just ahead
    // of the player out to 25 units, must all be inside the view.
    for (let z = -5; z >= -25; z -= 5) {
      expect(frustum.containsPoint(new THREE.Vector3(0, 0, z)), `ground at z=${z} should be visible`).toBe(true);
    }
  });
});

describe("ChaseCamera follow behaviour", () => {
  it("snapTo places the camera at its target immediately", () => {
    const camera = makeCamera();
    const chase = new ChaseCamera(camera);
    chase.snapTo(2.2);
    expect(camera.position.x).toBeCloseTo(2.2 * CAMERA_LATERAL_FOLLOW, 5);
  });

  it("tracks only part of the player's lateral offset, so lane changes stay visible", () => {
    const camera = makeCamera();
    const chase = new ChaseCamera(camera);
    chase.snapTo(2.2);
    // A 1:1 follow would pin the character centre-screen; partial follow is
    // what lets a lane change actually move them within the frame.
    expect(Math.abs(camera.position.x)).toBeLessThan(2.2);
    expect(CAMERA_LATERAL_FOLLOW).toBeLessThan(1);
  });

  it("lags behind a sudden lane change rather than snapping", () => {
    const camera = makeCamera();
    const chase = new ChaseCamera(camera);
    chase.snapTo(0);

    chase.update(1 / 60, 2.2); // player jumps to the right lane instantly

    const target = 2.2 * CAMERA_LATERAL_FOLLOW;
    expect(camera.position.x).toBeGreaterThan(0); // started moving
    expect(camera.position.x).toBeLessThan(target); // but has not arrived
  });

  it("converges on the target when the player holds a lane", () => {
    const camera = makeCamera();
    const chase = new ChaseCamera(camera);
    chase.snapTo(0);
    for (let i = 0; i < 120; i += 1) {
      chase.update(1 / 60, 2.2);
    }
    expect(camera.position.x).toBeCloseTo(2.2 * CAMERA_LATERAL_FOLLOW, 2);
  });

  it("damping is frame-rate independent", () => {
    // The same elapsed wall time at 30Hz and 144Hz must land the camera in
    // (very nearly) the same place -- a raw lerp(a, b, k) would not.
    const slowCam = makeCamera();
    const fastCam = makeCamera();
    const slowChase = new ChaseCamera(slowCam);
    const fastChase = new ChaseCamera(fastCam);
    slowChase.snapTo(0);
    fastChase.snapTo(0);

    const totalTime = 0.5;
    for (let i = 0; i < 15; i += 1) slowChase.update(totalTime / 15, 2.2);
    for (let i = 0; i < 72; i += 1) fastChase.update(totalTime / 72, 2.2);

    expect(fastCam.position.x).toBeCloseTo(slowCam.position.x, 2);
  });
});
