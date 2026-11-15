/**
 * These exercise real THREE.Scene/Mesh/Geometry/Material objects -- that
 * part of Three.js is plain JS with no WebGL context requirement, so it runs
 * fine under jsdom. Only THREE.WebGLRenderer (sceneRoot.ts) needs a real
 * GPU/canvas and is out of unit-test scope; see the Playwright screenshot
 * check for that piece.
 */

import { describe, expect, it } from "vitest";
import * as THREE from "three";
import { ObstacleSpawner } from "../../src/game/obstacleSpawner";
import { BASE_RUN_SPEED_U_S, LANE_X, OBSTACLE_DESPAWN_Z, OBSTACLE_SPAWN_Z } from "../../src/config/gameConfig";
import type { LevelSegment } from "../../src/config/levelDefinition";

const FAST_SEGMENT: LevelSegment = {
  id: "fast-test",
  durationSeconds: 6,
  obstacles: [
    { atSeconds: 0.1, typeId: "boulder", lanes: [0] },
    { atSeconds: 2.0, typeId: "log", lanes: [1] },
    { atSeconds: 4.0, typeId: "pit", lanes: [2] },
  ],
};

describe("ObstacleSpawner", () => {
  it("spawns nothing before the first obstacle's time", () => {
    const scene = new THREE.Scene();
    const spawner = new ObstacleSpawner(scene, FAST_SEGMENT);
    spawner.update(0.05, 0);
    expect(spawner.activeCount).toBe(0);
  });

  it("spawns an obstacle at its scheduled time, positioned at OBSTACLE_SPAWN_Z in the right lane", () => {
    const scene = new THREE.Scene();
    const spawner = new ObstacleSpawner(scene, FAST_SEGMENT);
    spawner.update(0.2, 0);

    expect(spawner.activeCount).toBe(1);
    const [obstacle] = spawner.collidables;
    expect(obstacle?.typeId).toBe("boulder");
    // Center of the boulder's AABB should sit on lane 0's X and at spawn Z.
    const centerX = (obstacle!.aabb.minX + obstacle!.aabb.maxX) / 2;
    expect(centerX).toBeCloseTo(LANE_X[0], 5);
    expect(obstacle!.aabb.maxZ).toBeGreaterThanOrEqual(OBSTACLE_SPAWN_Z);
  });

  it("moves active obstacles by deltaZ each update", () => {
    const scene = new THREE.Scene();
    const spawner = new ObstacleSpawner(scene, FAST_SEGMENT);
    spawner.update(0.2, 0);
    const before = spawner.collidables[0]!.aabb.minZ;

    spawner.update(0.1, 5);

    const after = spawner.collidables[0]!.aabb.minZ;
    expect(after - before).toBeCloseTo(5, 5);
  });

  it("despawns an obstacle once it scrolls past the despawn threshold", () => {
    const scene = new THREE.Scene();
    const spawner = new ObstacleSpawner(scene, FAST_SEGMENT);
    spawner.update(0.2, 0); // spawns the boulder at OBSTACLE_SPAWN_Z
    expect(spawner.activeCount).toBe(1);

    const totalDistance = OBSTACLE_DESPAWN_Z - OBSTACLE_SPAWN_Z + 1;
    spawner.update(0.001, totalDistance);

    expect(spawner.activeCount).toBe(0);
  });

  it("hides a mesh on release rather than leaving it visible off-screen", () => {
    const scene = new THREE.Scene();
    const spawner = new ObstacleSpawner(scene, FAST_SEGMENT);
    spawner.update(0.2, 0);
    const mesh = scene.children.find((child) => child.visible) as THREE.Mesh | undefined;
    expect(mesh).toBeDefined();

    const totalDistance = OBSTACLE_DESPAWN_Z - OBSTACLE_SPAWN_Z + 1;
    spawner.update(0.001, totalDistance);

    expect(scene.children.some((child) => child.visible)).toBe(false);
  });

  it("marks isScheduleFinished only after every entry has been spawned", () => {
    const scene = new THREE.Scene();
    const spawner = new ObstacleSpawner(scene, FAST_SEGMENT);
    spawner.update(0.2, 0);
    expect(spawner.isScheduleFinished).toBe(false);

    spawner.update(10, 0);
    expect(spawner.isScheduleFinished).toBe(true);
  });

  it("reset() restarts the segment and does not construct any new meshes", () => {
    const scene = new THREE.Scene();
    const spawner = new ObstacleSpawner(scene, FAST_SEGMENT);

    // Drain the whole segment once, moving obstacles all the way through.
    const dt = 0.1;
    const deltaZ = BASE_RUN_SPEED_U_S * dt;
    for (let i = 0; i < 400; i += 1) {
      spawner.update(dt, deltaZ);
    }
    expect(spawner.isScheduleFinished).toBe(true);
    expect(spawner.activeCount).toBe(0);
    const constructedAfterOneLap = spawner.totalConstructed;

    spawner.reset();
    expect(spawner.isScheduleFinished).toBe(false);
    expect(spawner.elapsedSeconds).toBe(0);

    for (let i = 0; i < 400; i += 1) {
      spawner.update(dt, deltaZ);
    }

    expect(spawner.totalConstructed).toBe(constructedAfterOneLap);
  });

  it("stays at a flat totalConstructed across a simulated 3-minute looped soak", () => {
    const scene = new THREE.Scene();
    const spawner = new ObstacleSpawner(scene, FAST_SEGMENT);
    const dt = 1 / 60;
    const deltaZ = BASE_RUN_SPEED_U_S * dt;
    const totalTicks = 60 * 180; // 3 minutes at 60Hz, simulated (not real time)

    let constructedAfterFirstLap: number | null = null;
    for (let tick = 0; tick < totalTicks; tick += 1) {
      spawner.update(dt, deltaZ);
      if (spawner.isScheduleFinished && spawner.activeCount === 0) {
        if (constructedAfterFirstLap === null) {
          constructedAfterFirstLap = spawner.totalConstructed;
        }
        spawner.reset();
      }
    }

    expect(constructedAfterFirstLap).not.toBeNull();
    expect(spawner.totalConstructed).toBe(constructedAfterFirstLap);
  });

  it("dispose() removes every constructed mesh from the scene", () => {
    const scene = new THREE.Scene();
    const spawner = new ObstacleSpawner(scene, FAST_SEGMENT);
    spawner.update(10, 0); // spawn everything
    expect(scene.children.length).toBeGreaterThan(0);

    spawner.dispose();

    expect(scene.children.length).toBe(0);
  });
});
