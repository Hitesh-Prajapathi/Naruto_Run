/**
 * Level-driven, pooled obstacle spawning -- game_implementation_plan.md
 * Phase C: "obstacleSpawner driven by levelDefinition.ts". Combines
 * obstacleSchedule.ts (when) with a per-type ObjectPool (what mesh to reuse)
 * and exposes AABBs for collisionSystem.ts (whether it was hit).
 *
 * Each obstacle type gets a distinct silhouette (naruto_run_agent_brief.md
 * §H: hazards must read as jump-vs-dodge at a distance, which the previous
 * uniform grey boxes did not) plus an accent band that separates hazards
 * from decorative scenery. Geometries and materials are still built once per
 * *type* and shared by every instance, so spawning allocates nothing.
 */

import * as THREE from "three";
import {
  LANE_WIDTH,
  LANE_X,
  OBSTACLE_DESPAWN_Z,
  OBSTACLE_SPAWN_Z,
  OBSTACLE_TYPES,
  type ObstacleTypeId,
} from "../config/gameConfig";
import type { LevelSegment } from "../config/levelDefinition";
import { ObjectPool } from "../scene/objectPool";
import { aabbFromGroundCenter, type CollidableObstacle, type Vector3Like } from "./collisionSystem";
import { ObstacleScheduler, type ScheduledObstacle } from "./obstacleSchedule";

/** Visual/collision gap so two adjacent-lane obstacles don't appear fused. */
const LANE_SPAN_MARGIN = 0.3;

interface TypeAssets {
  body: THREE.BufferGeometry;
  accent: THREE.BufferGeometry | null;
  bodyMaterial: THREE.Material;
  accentMaterial: THREE.Material;
}

/**
 * Unit-sized geometry per shape, authored so that scaling by the type's
 * `size` produces the right proportions. Each is centred on X/Z with its
 * base at y=0, matching aabbFromGroundCenter's convention.
 */
function buildTypeAssets(typeId: ObstacleTypeId): TypeAssets {
  const definition = OBSTACLE_TYPES[typeId];
  const bodyMaterial = new THREE.MeshStandardMaterial({
    color: definition.color,
    roughness: 0.85,
    flatShading: definition.shape === "boulder",
  });
  const accentMaterial = new THREE.MeshStandardMaterial({
    color: definition.accent,
    roughness: 0.6,
    emissive: new THREE.Color(definition.accent).multiplyScalar(0.18),
  });

  let body: THREE.BufferGeometry;
  let accent: THREE.BufferGeometry | null = null;

  switch (definition.shape) {
    case "log": {
      // A lying cylinder: unmistakably low and horizontal, i.e. "jump me".
      body = new THREE.CylinderGeometry(0.5, 0.5, 1, 12);
      body.rotateZ(Math.PI / 2); // lie it along X, spanning the lane(s)
      body.translate(0, 0.5, 0); // base at y=0
      // Cut ends, marked so the log reads as a felled trunk.
      accent = new THREE.CylinderGeometry(0.52, 0.52, 0.06, 12);
      accent.rotateZ(Math.PI / 2);
      accent.translate(0, 0.5, 0);
      break;
    }
    case "boulder": {
      body = new THREE.IcosahedronGeometry(0.5, 0);
      body.scale(1, 0.9, 1);
      body.translate(0, 0.45, 0);
      break;
    }
    case "pit": {
      // A shallow inverted dome reads as a hole; a flat plane would not.
      body = new THREE.SphereGeometry(0.5, 16, 8, 0, Math.PI * 2, 0, Math.PI / 2);
      body.scale(1, 0.9, 1);
      body.rotateX(Math.PI); // open side up
      body.translate(0, 0.02, 0);
      // Raised rim so the hole's edge is visible against the flat path.
      accent = new THREE.TorusGeometry(0.5, 0.05, 6, 20);
      accent.rotateX(Math.PI / 2);
      break;
    }
    case "barrier":
    default: {
      body = new THREE.BoxGeometry(1, 1, 1);
      body.translate(0, 0.5, 0);
      // Bright band across the top edge -- the "don't jump this" read.
      accent = new THREE.BoxGeometry(1.02, 0.12, 1.04);
      accent.translate(0, 0.94, 0);
      break;
    }
  }

  return { body, accent, bodyMaterial, accentMaterial };
}

interface ActiveObstacle {
  id: number;
  typeId: ObstacleTypeId;
  group: THREE.Group;
  size: Vector3Like;
}

export class ObstacleSpawner {
  private readonly pools: Record<ObstacleTypeId, ObjectPool<THREE.Group>>;
  private readonly assets: Record<ObstacleTypeId, TypeAssets>;
  /** Every group ever constructed per type, free or in-use -- ObjectPool only
   * exposes in-use items via `.active`, so dispose() needs its own record to
   * reach pooled-but-currently-free groups too. */
  private readonly allGroupsByType: Record<ObstacleTypeId, THREE.Group[]>;
  private readonly scheduler: ObstacleScheduler;
  private readonly active: ActiveObstacle[] = [];
  private nextInstanceId = 0;

  constructor(
    private readonly scene: THREE.Scene,
    segment: LevelSegment,
    initialPoolSizePerType = 3,
  ) {
    this.scheduler = new ObstacleScheduler(segment);
    this.assets = {} as Record<ObstacleTypeId, TypeAssets>;
    this.pools = {} as Record<ObstacleTypeId, ObjectPool<THREE.Group>>;
    this.allGroupsByType = {} as Record<ObstacleTypeId, THREE.Group[]>;

    for (const typeId of Object.keys(OBSTACLE_TYPES) as ObstacleTypeId[]) {
      const assets = buildTypeAssets(typeId);
      this.assets[typeId] = assets;
      const createdGroups: THREE.Group[] = [];
      this.allGroupsByType[typeId] = createdGroups;
      this.pools[typeId] = new ObjectPool<THREE.Group>(
        () => {
          const group = new THREE.Group();
          const bodyMesh = new THREE.Mesh(assets.body, assets.bodyMaterial);
          bodyMesh.castShadow = true;
          bodyMesh.receiveShadow = true;
          group.add(bodyMesh);
          if (assets.accent) {
            const accentMesh = new THREE.Mesh(assets.accent, assets.accentMaterial);
            accentMesh.castShadow = false;
            group.add(accentMesh);
          }
          group.visible = false;
          scene.add(group);
          createdGroups.push(group);
          return group;
        },
        (group) => {
          group.visible = false;
        },
        initialPoolSizePerType,
      );
    }
  }

  /** Advance the schedule, spawn anything newly due, move active obstacles
   * by `deltaZ`, and recycle any that scrolled past the despawn threshold. */
  update(dt: number, deltaZ: number): void {
    for (const scheduled of this.scheduler.update(dt)) {
      this.spawn(scheduled);
    }

    for (let i = this.active.length - 1; i >= 0; i -= 1) {
      const obstacle = this.active[i]!;
      obstacle.group.position.z += deltaZ;
      if (obstacle.group.position.z > OBSTACLE_DESPAWN_Z) {
        this.pools[obstacle.typeId].release(obstacle.group);
        this.active.splice(i, 1);
      }
    }
  }

  private spawn(scheduled: ScheduledObstacle): void {
    const { entry } = scheduled;
    const definition = OBSTACLE_TYPES[entry.typeId];
    const lanesX = entry.lanes.map((lane) => LANE_X[lane]);
    const minX = Math.min(...lanesX);
    const maxX = Math.max(...lanesX);
    const centerX = (minX + maxX) / 2;
    // Width always follows the actual lane span, not a fixed catalog value --
    // the same "log" type can legitimately be authored as 1 or 2 lanes wide.
    const size: Vector3Like = {
      x: entry.lanes.length * LANE_WIDTH - LANE_SPAN_MARGIN,
      y: definition.size.y,
      z: definition.size.z,
    };

    const group = this.pools[entry.typeId].acquire();
    group.visible = true;
    group.scale.set(size.x, size.y, size.z);
    // Geometry is authored with its base at y=0, so the group sits directly
    // on the ground rather than being centre-offset like the old boxes.
    group.position.set(centerX, 0, OBSTACLE_SPAWN_Z);

    this.active.push({ id: this.nextInstanceId, typeId: entry.typeId, group, size });
    this.nextInstanceId += 1;
  }

  /** Current obstacles as plain AABBs for collisionSystem.ts to query. */
  get collidables(): CollidableObstacle[] {
    return this.active.map((obstacle) => ({
      id: obstacle.id,
      typeId: obstacle.typeId,
      aabb: aabbFromGroundCenter(obstacle.group.position, obstacle.size),
    }));
  }

  get activeCount(): number {
    return this.active.length;
  }

  /** Sum of every type pool's totalConstructed -- flat after warmup proves
   * no per-spawn allocation (the Phase-C "no allocation spikes" gate). */
  get totalConstructed(): number {
    return Object.values(this.pools).reduce((sum, pool) => sum + pool.totalConstructed, 0);
  }

  get isScheduleFinished(): boolean {
    return this.scheduler.isFinished;
  }

  get elapsedSeconds(): number {
    return this.scheduler.elapsedSeconds;
  }

  /** Restarts the same segment's schedule from t=0, reusing every existing
   * pool/material/geometry -- no reconstruction, so looping a segment for a
   * soak test stays a true zero-allocation steady state after the first lap.
   * Any still-active obstacles (there shouldn't be any if the caller waited
   * for activeCount === 0) are released back to their pools first. */
  reset(): void {
    for (const obstacle of this.active) {
      this.pools[obstacle.typeId].release(obstacle.group);
    }
    this.active.length = 0;
    this.scheduler.reset();
  }

  dispose(): void {
    this.active.length = 0;
    for (const typeId of Object.keys(this.pools) as ObstacleTypeId[]) {
      for (const group of this.allGroupsByType[typeId]) {
        this.scene.remove(group);
      }
      const assets = this.assets[typeId];
      assets.body.dispose();
      assets.accent?.dispose();
      assets.bodyMaterial.dispose();
      assets.accentMaterial.dispose();
    }
  }
}
