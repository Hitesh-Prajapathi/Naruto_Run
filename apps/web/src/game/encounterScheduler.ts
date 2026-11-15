/**
 * Decides *when* an Obito encounter starts -- Feature Brief 02 §5.1.
 *
 * Two fixed distance milestones, never random, so pacing is predictable and
 * tunable. Once a milestone is passed the scheduler waits for a "clean
 * moment" before firing: never mid-jump, never mid-lane-change, and not
 * while an obstacle is close enough to collide during the cutscene.
 *
 * Pure logic (no Three.js, no timers) so the arming/clean-moment rules can
 * be unit-tested directly -- an encounter that never fires, or one that
 * fires into an obstacle, would both be hard to catch by playing.
 */

import {
  ENCOUNTER_DISTANCES_M,
  ENCOUNTER_TRIGGER_PATIENCE_S,
  TOTAL_ENCOUNTERS,
} from "../config/bossConfig";

export interface CleanMomentInput {
  /** Naruto is airborne, or in the anticipation/landing phases. */
  isJumping: boolean;
  isChangingLane: boolean;
  /** Seconds until the nearest active obstacle would reach the player, or
   * null when the path is clear. */
  secondsToNearestObstacle: number | null;
  /** Minimum obstacle clearance required, in seconds. */
  requiredClearanceS: number;
}

export function isCleanMoment(input: CleanMomentInput): boolean {
  if (input.isJumping || input.isChangingLane) {
    return false;
  }
  if (input.secondsToNearestObstacle === null) {
    return true;
  }
  return input.secondsToNearestObstacle >= input.requiredClearanceS;
}

export class EncounterScheduler {
  /** How many encounters have been *started* (not necessarily won). */
  private triggered = 0;
  /** Set once the distance milestone is passed; we then wait for a clean
   * moment before actually firing. */
  private armed = false;
  private armedForS = 0;

  /**
   * @param distances milestone distances in metres; defaults to the config.
   */
  constructor(private readonly distances: readonly number[] = ENCOUNTER_DISTANCES_M) {}

  get encountersTriggered(): number {
    return this.triggered;
  }

  get isArmed(): boolean {
    return this.armed;
  }

  get allEncountersDone(): boolean {
    return this.triggered >= this.distances.length;
  }

  /** Index of the encounter that would fire next (0-based). */
  get nextEncounterIndex(): number {
    return this.triggered;
  }

  /**
   * Advance the scheduler. Returns true on the frame an encounter should
   * begin. `distanceM` is the run's total distance so far.
   */
  update(dt: number, distanceM: number, clean: Omit<CleanMomentInput, "requiredClearanceS"> & { requiredClearanceS: number }): boolean {
    if (this.allEncountersDone) {
      return false;
    }

    if (!this.armed) {
      const milestone = this.distances[this.triggered];
      if (milestone === undefined || distanceM < milestone) {
        return false;
      }
      this.armed = true;
      this.armedForS = 0;
    }

    this.armedForS += dt;

    // Wait for a clean moment -- but not forever. A dense obstacle stream
    // could otherwise postpone an encounter indefinitely and silently drop
    // it from the run.
    const patienceExhausted = this.armedForS >= ENCOUNTER_TRIGGER_PATIENCE_S;
    if (!patienceExhausted && !isCleanMoment(clean)) {
      return false;
    }
    // Even out of patience, never fire mid-air or mid-lane-change: the intro
    // would snap Naruto to the centre lane from an unresolved pose.
    if (clean.isJumping || clean.isChangingLane) {
      return false;
    }

    this.armed = false;
    this.armedForS = 0;
    this.triggered += 1;
    return true;
  }

  /**
   * Arm the next encounter immediately, bypassing its distance milestone.
   * Test/debug only -- lets an end-to-end test reach an encounter without
   * running out hundreds of metres first.
   */
  forceNext(): void {
    if (this.allEncountersDone) return;
    this.armed = true;
    this.armedForS = ENCOUNTER_TRIGGER_PATIENCE_S;
  }

  /** Full reset for Try Again -- both milestones re-armed (brief §4.5). */
  reset(): void {
    this.triggered = 0;
    this.armed = false;
    this.armedForS = 0;
  }
}

export { TOTAL_ENCOUNTERS };
