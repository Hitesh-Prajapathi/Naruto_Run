/**
 * Jump detection -- Feature Brief 05 §5.3.
 *
 * **Driven by the player's actual upward movement, not by the label.** The
 * original design used the pipeline's `jumping` label as the trigger, on the
 * owner's instruction "use labels for jump, metrics for lean". That is no
 * longer possible and the owner has since superseded it: the label is
 * computed from `hip_y`, and in a chest-up webcam framing the hips are
 * outside the frame and merely estimated. The label therefore stopped firing
 * altogether. Jump now fires when the shoulders actually rise, which is both
 * what the owner asked for and the only signal available.
 *
 * `jumpRequireLabel` restores the old behaviour for a full-body camera, where
 * the hips are genuinely visible and the label is worth having.
 *
 * There is deliberately **no debounce and no transition detection here** —
 * the refractory period below is the only rate limit, and anything more would
 * add latency to the one input where latency is most obvious.
 *
 * What this does that the pipeline does not:
 *
 *   - **Normalise by the player's size.** The pipeline's threshold is an
 *     absolute `jump_height > 0.045` in normalised image units, so it means
 *     something different at 1m than at 3m. Ours is a fraction of the
 *     player's own shoulder width (T-11).
 *   - **Require upward velocity.** §5.3: a player bobbing, shifting weight,
 *     or jogging on the spot will cross a height threshold sooner or later.
 *     Demanding that they are actually *moving upward* is what separates a
 *     jump from enthusiasm (T-09).
 *   - **Refractory period.** One jump per airtime, no machine-gunning (T-10).
 *
 * Landing is not detected, per §5.3 — fire on takeoff and let the existing
 * jump arc own the rest.
 *
 * **Why height and velocity are checked over a window, not on one frame.**
 * The pipeline's label edge is not free: `ConsensusFilter` needs 3 `jumping`
 * votes inside a 5-frame window, so at 20fps the edge lands ~150ms after the
 * player actually left the ground. A jump's whole ascent is about 200ms, so
 * by the time the edge arrives the player is at or past the apex — where
 * upward velocity is zero by definition. Sampling velocity on the edge frame
 * therefore tested the one instant guaranteed to fail it, and no jump ever
 * fired in real play. (The synthetic fixtures missed this because they emit
 * the edge on the exact frame the label flips, with no consensus delay.)
 *
 * So both quantities are evaluated as their peak over the last
 * `jumpVelocityWindowMs`, which asks the question §5.3 actually intends —
 * "did this player recently launch upward?" — rather than "are they moving
 * upward at this precise instant?"
 */

import type { BodyTuning } from "./bodyTuning";
import type { BodyBaseline } from "./bodyCalibration";
import { normalisedJumpRise } from "./bodyCalibration";
import type { PoseSample } from "./poseAdapter";

export interface JumpDebug {
  /** Hip rise as a fraction of calibrated torso height. */
  rise: number;
  /** Upward velocity in calibrated heights per second. */
  velocity: number;
  /** Best rise seen inside the recent window -- what the threshold sees. */
  peakRise: number;
  /** Best upward velocity inside the recent window. */
  peakVelocity: number;
  /** Milliseconds left before another jump may fire. */
  refractoryMs: number;
  /** Whether the pipeline's own label is currently `jumping`. */
  labelActive: boolean;
}

/** One frame of history, for the peak-over-window checks. */
interface JumpTrace {
  timestampMs: number;
  rise: number;
  velocity: number;
}

export class JumpDetector {
  private previousRise = 0;
  private previousTimestampMs: number | null = null;
  private refractoryUntilMs = 0;
  private velocity = 0;
  private labelActive = false;
  private trace: JumpTrace[] = [];
  private peakRise = 0;
  private peakVelocity = 0;

  constructor(private tuning: BodyTuning) {}

  setTuning(tuning: BodyTuning): void {
    this.tuning = tuning;
  }

  get debug(): JumpDebug {
    return {
      rise: this.previousRise,
      velocity: this.velocity,
      peakRise: this.peakRise,
      peakVelocity: this.peakVelocity,
      refractoryMs: Math.max(0, this.refractoryUntilMs - (this.previousTimestampMs ?? 0)),
      labelActive: this.labelActive,
    };
  }

  /** @returns true when a jump should fire this sample. */
  update(sample: PoseSample, baseline: BodyBaseline): boolean {
    const rise = sample.personPresent ? normalisedJumpRise(sample, baseline) : 0;

    const previousMs = this.previousTimestampMs;
    const dtSeconds =
      previousMs === null ? 0 : Math.max((sample.timestampMs - previousMs) / 1000, 0);
    this.velocity = dtSeconds > 1e-3 ? (rise - this.previousRise) / dtSeconds : 0;
    this.previousRise = rise;
    this.previousTimestampMs = sample.timestampMs;

    // Keep only the recent past; the window is what absorbs the consensus lag.
    this.trace.push({ timestampMs: sample.timestampMs, rise, velocity: this.velocity });
    const cutoff = sample.timestampMs - this.tuning.jumpVelocityWindowMs;
    while (this.trace.length > 0 && (this.trace[0] as JumpTrace).timestampMs < cutoff) {
      this.trace.shift();
    }
    this.peakRise = Math.max(...this.trace.map((entry) => entry.rise));
    this.peakVelocity = Math.max(...this.trace.map((entry) => entry.velocity));

    // The pipeline's edge, already debounced. `emittedMovement` is non-null
    // only on a change, which is precisely a takeoff.
    const labelEdge = sample.emittedMovement === "jumping";
    this.labelActive = sample.stableLabel === "jumping";

    if (sample.timestampMs < this.refractoryUntilMs) {
      return false;
    }

    const highEnough = this.peakRise >= this.tuning.jumpHeightFraction;
    const fastEnough = this.peakVelocity >= this.tuning.jumpMinVelocity;

    // Rose far enough, and got there fast enough to be a jump rather than a
    // stretch. Height alone would fire on a slow stand-up; velocity alone
    // would fire on a brisk bob.
    const launched = highEnough && fastEnough;

    // Only a full-body camera should also demand the pipeline's label, and
    // even then the label is corroboration, never the measurement.
    const fire = this.tuning.jumpRequireLabel ? labelEdge && launched : launched;

    if (fire) {
      this.refractoryUntilMs = sample.timestampMs + this.tuning.jumpRefractoryMs;
    }
    return fire;
  }

  reset(): void {
    this.previousRise = 0;
    this.previousTimestampMs = null;
    this.refractoryUntilMs = 0;
    this.velocity = 0;
    this.labelActive = false;
    this.trace = [];
    this.peakRise = 0;
    this.peakVelocity = 0;
  }
}
