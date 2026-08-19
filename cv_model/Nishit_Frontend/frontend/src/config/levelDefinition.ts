/**
 * The level script: which obstacles spawn when, in which lane(s).
 *
 * Only Segment 1 is authored here. game_implementation_plan.md's finite
 * level is Segment 1 -> Battle 1 -> Segment 2 -> Battle 2 -> Segment 3 ->
 * Finish; Battles need an enemy/HP model (Phase F) and Segments 2-3 are
 * later content-authoring work (Phase H) once the pattern-authoring
 * conventions below have been proven out on one real segment. The types
 * here (`LevelSegment`, `ObstacleSpawnEntry`) are already general enough for
 * that later content -- nothing about them is Segment-1-specific.
 */

import {
  LANE_COUNT,
  MAX_OBSTACLE_LANE_SPAN,
  type LaneIndex,
  type ObstacleTypeId,
} from "./gameConfig";

export interface ObstacleSpawnEntry {
  /** Seconds from the segment's own start, not wall-clock time. */
  atSeconds: number;
  typeId: ObstacleTypeId;
  /** Which lane(s) this instance occupies. 1 or 2 lanes; MAX_OBSTACLE_LANE_SPAN
   * (never all of LANE_COUNT) is enforced by the level linter, not by this type. */
  lanes: LaneIndex[];
}

export interface LevelSegment {
  id: string;
  durationSeconds: number;
  obstacles: ObstacleSpawnEntry[];
}

export interface LevelDefinition {
  segments: LevelSegment[];
}

function entry(atSeconds: number, typeId: ObstacleTypeId, lanes: LaneIndex[]): ObstacleSpawnEntry {
  return { atSeconds, typeId, lanes };
}

/**
 * Segment 1 -- game_implementation_plan.md: "25s run, obstacle density: low
 * (teaches lanes)". Every gap below is 3.5-4.5s, well past the linter's
 * required minimums (~1.6s general, 1.2s jump/lane-mixed) -- intentionally
 * generous since this segment's job is teaching the two avoidance types, not
 * testing reaction time yet. Pits are withheld here on purpose: the plan
 * introduces those in Segment 2.
 */
export const SEGMENT_1: LevelSegment = {
  id: "segment-1",
  durationSeconds: 25,
  obstacles: [
    entry(3.0, "boulder", [1]), // single-lane blocker: step left or right
    entry(7.0, "log", [0, 1]), // two-lane jump, right lane stays open too
    entry(11.5, "thornWall", [1, 2]), // too tall to jump: must be in the left lane
    entry(15.5, "log", [2]), // single-lane jump, easier variant
    entry(19.0, "boulder", [0]),
    entry(23.0, "log", [1]),
  ],
};

/** Phase-C/dev-harness level: Segment 1 only. Extend `segments` here once
 * Battle 1 (Phase F) and Segment 2 (Phase H) exist. */
export const DEV_LEVEL: LevelDefinition = {
  segments: [SEGMENT_1],
};

// Sanity assertion kept at module scope (not a runtime game rule): if
// LANE_COUNT or MAX_OBSTACLE_LANE_SPAN ever changes, this makes the
// intentional invariant they encode impossible to silently forget.
if (MAX_OBSTACLE_LANE_SPAN >= LANE_COUNT) {
  throw new Error("MAX_OBSTACLE_LANE_SPAN must leave at least one lane open");
}
