/**
 * Validates a LevelSegment's obstacle pattern against the hard CV-driven
 * gameplay constraints from game_implementation_plan.md §1.1 and §4.3:
 *
 *  - no obstacle may span every lane (no valid player input would exist);
 *  - consecutive obstacles must be at least MIN_OBSTACLE_GAP_S apart;
 *  - a jump-avoidance obstacle and a lane-avoidance obstacle specifically
 *    must be at least MIN_MIXED_TYPE_GAP_S apart (checked independently of
 *    the general minimum -- see gameConfig.ts's comment on why both rules
 *    are encoded even though one currently implies the other).
 *
 * Pure and Three.js-free so it can run in a unit test or a CI step, not just
 * by eyeballing the authored level in the browser.
 */

import { LANE_COUNT, MAX_OBSTACLE_LANE_SPAN, MIN_MIXED_TYPE_GAP_S, MIN_OBSTACLE_GAP_S, OBSTACLE_TYPES } from "../config/gameConfig";
import type { LevelSegment, ObstacleSpawnEntry } from "../config/levelDefinition";

export interface LintIssue {
  index: number;
  rule: string;
  message: string;
}

export function lintLevelSegment(segment: LevelSegment): LintIssue[] {
  const issues: LintIssue[] = [];
  const sorted = [...segment.obstacles].sort((a, b) => a.atSeconds - b.atSeconds);

  sorted.forEach((obstacle, index) => {
    issues.push(...lintSingleEntry(obstacle, index));
    if (index > 0) {
      issues.push(...lintGap(sorted[index - 1]!, obstacle, index));
    }
  });

  return issues;
}

function lintSingleEntry(obstacle: ObstacleSpawnEntry, index: number): LintIssue[] {
  const issues: LintIssue[] = [];

  if (obstacle.lanes.length === 0) {
    issues.push({ index, rule: "empty-lanes", message: `obstacle at ${obstacle.atSeconds}s has no lanes` });
  }
  if (obstacle.lanes.length > MAX_OBSTACLE_LANE_SPAN) {
    issues.push({
      index,
      rule: "lane-span",
      message: `obstacle at ${obstacle.atSeconds}s spans ${obstacle.lanes.length} lanes; at most ${MAX_OBSTACLE_LANE_SPAN} of ${LANE_COUNT} may be blocked at once`,
    });
  }
  if (new Set(obstacle.lanes).size !== obstacle.lanes.length) {
    issues.push({ index, rule: "duplicate-lane", message: `obstacle at ${obstacle.atSeconds}s lists a lane more than once` });
  }
  for (const lane of obstacle.lanes) {
    if (lane < 0 || lane >= LANE_COUNT) {
      issues.push({ index, rule: "lane-range", message: `obstacle at ${obstacle.atSeconds}s references out-of-range lane ${lane}` });
    }
  }
  if (obstacle.atSeconds < 0) {
    issues.push({ index, rule: "negative-time", message: `obstacle has a negative spawn time ${obstacle.atSeconds}s` });
  }

  return issues;
}

function lintGap(previous: ObstacleSpawnEntry, current: ObstacleSpawnEntry, index: number): LintIssue[] {
  const issues: LintIssue[] = [];
  const gap = current.atSeconds - previous.atSeconds;

  if (gap < MIN_OBSTACLE_GAP_S) {
    issues.push({
      index,
      rule: "min-gap",
      message: `gap of ${gap.toFixed(2)}s before the obstacle at ${current.atSeconds}s is below the ${MIN_OBSTACLE_GAP_S.toFixed(2)}s minimum`,
    });
  }

  const previousAvoidance = OBSTACLE_TYPES[previous.typeId].avoidance;
  const currentAvoidance = OBSTACLE_TYPES[current.typeId].avoidance;
  if (previousAvoidance !== currentAvoidance && gap < MIN_MIXED_TYPE_GAP_S) {
    issues.push({
      index,
      rule: "mixed-type-gap",
      message: `jump/lane obstacles at ${previous.atSeconds}s and ${current.atSeconds}s are only ${gap.toFixed(2)}s apart (minimum ${MIN_MIXED_TYPE_GAP_S}s)`,
    });
  }

  return issues;
}

/** Throws with every issue listed if the segment fails linting. Intended for
 * a startup assertion (dev harness / tests) rather than user-facing UI. */
export function assertLevelSegmentValid(segment: LevelSegment): void {
  const issues = lintLevelSegment(segment);
  if (issues.length > 0) {
    const details = issues.map((issue) => `  [${issue.rule}] ${issue.message}`).join("\n");
    throw new Error(`level segment "${segment.id}" failed linting:\n${details}`);
  }
}
