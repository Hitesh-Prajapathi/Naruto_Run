/**
 * Neutral-pose baseline -- Feature Brief 05 §4.4.
 *
 * "All thresholds are relative to this baseline, never absolute. Absolute
 * thresholds will work for exactly one person at exactly one distance from
 * the camera." The pipeline's own thresholds *are* absolute (`lean_x > 0.06`,
 * `jump_height > 0.045`), which is exactly why the game derives its own
 * values from the metrics instead of trusting the labels for lean.
 *
 * Three things get captured:
 *   - **lean offset** — nobody stands perfectly square to a webcam, so a
 *     relaxed posture reads as a small permanent shoulder tilt. Subtracting
 *     it is what stops the game drifting one lane the moment it starts.
 *   - **shoulder height** — the standing baseline a jump is measured against.
 *   - **shoulder width** — the *scale*. Dividing by it is what makes the same
 *     physical jump register identically at 1m and at 3m (test T-11).
 *
 * All three come from the two shoulder landmarks. Nothing here reads the
 * hips; see the geometry block in `poseAdapter` for why.
 *
 * Uses the median rather than the mean: a player twitching once during the
 * three-second hold should not skew their baseline for the whole session.
 */

import type { PoseSample } from "./poseAdapter";

export interface BodyBaseline {
  /**
   * Resting shoulder tilt, subtracted from every reading. Already a sine, so
   * already free of any dependence on distance from the camera.
   */
  leanOffset: number;
  /** Resting shoulder-midpoint y (top-down normalised image space). */
  riseY: number;
  /** Shoulder width in frame heights; the per-player, per-distance scale. */
  bodyScale: number;
  /** How many samples went into it -- surfaced in the debug overlay. */
  sampleCount: number;
}

/** A baseline that changes nothing, for keyboard-only play and for tests. */
export const IDENTITY_BASELINE: BodyBaseline = {
  leanOffset: 0,
  riseY: 0.35,
  bodyScale: 0.3,
  sampleCount: 0,
};

function median(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  if (sorted.length % 2 === 1) return sorted[middle] as number;
  return (((sorted[middle - 1] as number) + (sorted[middle] as number)) / 2);
}

/**
 * Collects samples during the CALIBRATION gate state.
 *
 * Deliberately a plain accumulator with no timer of its own: the caller
 * decides when the hold is over, exactly like every other system here.
 */
export class BodyCalibrator {
  private readonly leans: number[] = [];
  private readonly rises: number[] = [];
  private readonly scales: number[] = [];

  /** Ignores frames with nobody in them, so a player who steps out mid-hold
   * does not poison the baseline with zeros. */
  add(sample: PoseSample): void {
    if (!sample.personPresent || sample.bodyScale <= 0) return;
    this.leans.push(sample.leanX);
    this.rises.push(sample.riseY);
    this.scales.push(sample.bodyScale);
  }

  get sampleCount(): number {
    return this.leans.length;
  }

  /** True once there is enough signal to trust the baseline. */
  get hasEnough(): boolean {
    return this.leans.length >= 10;
  }

  /** Snapshot the baseline. Falls back to identity when starved, so a
   * failed calibration degrades to "uncalibrated" rather than to garbage. */
  build(): BodyBaseline {
    if (!this.hasEnough) {
      return { ...IDENTITY_BASELINE };
    }
    return {
      leanOffset: median(this.leans),
      riseY: median(this.rises),
      bodyScale: Math.max(median(this.scales), 1e-4),
      sampleCount: this.leans.length,
    };
  }

  reset(): void {
    this.leans.length = 0;
    this.rises.length = 0;
    this.scales.length = 0;
  }
}

/**
 * The calibrated, scaled lean value the thresholds are expressed in:
 * negative = player's left.
 *
 * **No division here any more.** `leanX` is now the sine of the shoulder
 * roll, and an angle is already scale-free: the same physical bend gives the
 * same sine at 1m and at 3m, because both shoulders shrink together. The old
 * hip-based lean was a raw pixel offset that had to be divided by torso
 * height to become distance-independent, and that division was where the
 * unreliable hip estimate did the most damage — a hip landmark being guessed
 * a few percent low inflated the divisor and quietly crushed every reading.
 */
export function calibratedLean(
  sample: PoseSample,
  baseline: BodyBaseline,
  leanScale: number,
): number {
  return (sample.leanX - baseline.leanOffset) * leanScale;
}

/**
 * Shoulder rise above the standing baseline, as a fraction of shoulder width.
 *
 * This is the player's actual upward movement — "if the coordinates move
 * upward, Naruto jumps" — read from the shoulders rather than the hips.
 *
 * Image y grows downward, so rising means a *smaller* y, hence
 * `baseline.riseY - sample.riseY`. Dividing by shoulder width is the distance
 * normalisation (T-11): standing further away shrinks the jump excursion and
 * the shoulders in the same proportion, so the ratio holds. The current
 * frame's width is used where available so a player who drifts closer or
 * further mid-run keeps their sensitivity.
 */
export function normalisedJumpRise(sample: PoseSample, baseline: BodyBaseline): number {
  const scale =
    sample.bodyScale > 1e-4
      ? sample.bodyScale
      : baseline.bodyScale > 1e-4
        ? baseline.bodyScale
        : IDENTITY_BASELINE.bodyScale;
  return (baseline.riseY - sample.riseY) / scale;
}
