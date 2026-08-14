/**
 * Lean → lane intent -- Feature Brief 05 §5.2.
 *
 * **Why this reads the metric and not the label.** The pipeline's
 * `classify_metrics` is a priority chain: `jumping > naruto_run >
 * bending_*`. A player jogging on the spot — which this game actively
 * encourages — satisfies `naruto_run`, and their lean is then never
 * reported at all. Reading `lean_x` directly recovers the lean for exactly
 * the players who are most engaged. (Owner's correction; fixture
 * `jog_in_place_with_lean` is the regression test.)
 *
 * **What this deliberately does NOT do.** The pipeline's `ConsensusFilter`
 * already debounces labels and already fires on transition. That machinery
 * is bypassed for lean, so the debounce and return-to-neutral implemented
 * here are the *only* such layer in the lean path, not a second one stacked
 * on top. The jump path is the opposite case — see `jumpDetector`.
 *
 * Input is the calibrated lean value: negative = player's left, having
 * already been mirrored into player space by `poseAdapter`.
 */

import type { BodyTuning } from "./bodyTuning";

export type LeanState = "neutral" | "left" | "right";

export interface LeanDebug {
  /** Smoothed, calibrated lean value the thresholds are compared against. */
  value: number;
  state: LeanState;
  /** Consecutive samples the candidate state has held. */
  heldFrames: number;
  /**
   * False after a lane change until the player comes back inside the exit
   * threshold. §5.2: "enforce this with an explicit flag, not a timeout."
   */
  hasReturnedToNeutral: boolean;
}

export class LeanDetector {
  private smoothed = 0;
  private state: LeanState = "neutral";
  private candidate: LeanState = "neutral";
  private heldFrames = 0;
  private returnedToNeutral = true;
  private primed = false;

  constructor(private tuning: BodyTuning) {}

  /** Tuning is reloadable at runtime, so it can be swapped mid-session. */
  setTuning(tuning: BodyTuning): void {
    this.tuning = tuning;
  }

  get debug(): LeanDebug {
    return {
      value: this.smoothed,
      state: this.state,
      heldFrames: this.heldFrames,
      hasReturnedToNeutral: this.returnedToNeutral,
    };
  }

  /**
   * Feed one calibrated lean reading.
   *
   * @returns -1 for a left lane step, +1 for right, 0 for nothing. The sign
   *   convention matches `LaneController.moveLane`, so the caller does not
   *   translate anything.
   */
  update(calibratedLean: number): -1 | 0 | 1 {
    // Light exponential smoothing. The first reading seeds the filter
    // outright, otherwise every session starts with a spurious ramp from 0.
    if (!this.primed) {
      this.smoothed = calibratedLean;
      this.primed = true;
    } else {
      const alpha = this.tuning.leanSmoothing;
      this.smoothed = this.smoothed + (calibratedLean - this.smoothed) * alpha;
    }

    const { leanEnter, leanExit, leanDebounceFrames } = this.tuning;
    const magnitude = Math.abs(this.smoothed);

    // Hysteresis: entering a lean needs `leanEnter`, but staying in it only
    // needs `leanExit`. The gap is what stops a player resting on the
    // boundary from oscillating (T-03).
    let target: LeanState;
    if (this.state === "neutral") {
      target = magnitude >= leanEnter ? (this.smoothed < 0 ? "left" : "right") : "neutral";
    } else {
      target = magnitude >= leanExit ? this.state : "neutral";
      // A decisive lean straight through neutral to the other side.
      if (magnitude >= leanEnter) {
        target = this.smoothed < 0 ? "left" : "right";
      }
    }

    if (target === this.candidate) {
      this.heldFrames += 1;
    } else {
      this.candidate = target;
      this.heldFrames = 1;
    }

    // Debounce: a one-frame spike never fires (T-04). Returning to neutral is
    // exempt -- releasing a lean should feel immediate, and a slow release
    // would delay the next lane change.
    const settled = this.heldFrames >= leanDebounceFrames || target === "neutral";
    if (!settled || target === this.state) {
      return 0;
    }

    const previous = this.state;
    this.state = target;

    if (target === "neutral") {
      this.returnedToNeutral = true;
      return 0;
    }

    // Fire on the transition *into* a lean, once. A held lean moves exactly
    // one lane (T-05) and the player must pass back through neutral before
    // another registers (T-06).
    if (!this.returnedToNeutral) {
      return 0;
    }
    this.returnedToNeutral = false;
    void previous;
    return target === "left" ? -1 : 1;
  }

  /** Full reset -- new run, or recalibration. */
  reset(): void {
    this.smoothed = 0;
    this.state = "neutral";
    this.candidate = "neutral";
    this.heldFrames = 0;
    this.returnedToNeutral = true;
    this.primed = false;
  }
}
