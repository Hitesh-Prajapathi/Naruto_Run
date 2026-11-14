/**
 * Turns a LevelSegment's time-stamped obstacle list into "what's newly due"
 * per update tick. Pure timeline bookkeeping -- no Three.js, no pooling,
 * no collision. obstacleSpawner.ts (scene layer) consumes this to actually
 * create/move/recycle meshes.
 */

import type { LevelSegment, ObstacleSpawnEntry } from "../config/levelDefinition";

export interface ScheduledObstacle {
  /** Stable identity for this spawn instance, independent of pooling. */
  id: number;
  entry: ObstacleSpawnEntry;
}

export class ObstacleScheduler {
  private elapsed = 0;
  private nextIndex = 0;

  constructor(private readonly segment: LevelSegment) {}

  /** Advances the schedule by `dt` seconds and returns every entry that
   * became due, in level order, each tagged with a stable id. */
  update(dt: number): ScheduledObstacle[] {
    this.elapsed += dt;
    const due: ScheduledObstacle[] = [];
    while (
      this.nextIndex < this.segment.obstacles.length &&
      this.segment.obstacles[this.nextIndex]!.atSeconds <= this.elapsed
    ) {
      due.push({ id: this.nextIndex, entry: this.segment.obstacles[this.nextIndex]! });
      this.nextIndex += 1;
    }
    return due;
  }

  get isFinished(): boolean {
    return this.nextIndex >= this.segment.obstacles.length;
  }

  get elapsedSeconds(): number {
    return this.elapsed;
  }

  reset(): void {
    this.elapsed = 0;
    this.nextIndex = 0;
  }
}
