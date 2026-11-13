/**
 * Synthetic pose fixtures -- Feature Brief 05 §9.1 and §9.2.
 *
 * §9.1 asks for three input sources behind one interface: live camera,
 * recorded fixture, synthetic generator. This is the third. It matters most,
 * because it is the only one that can run in CI.
 *
 * **These are generated, not recorded — and that is a real limitation.**
 * §9.2 asks for clips of at least two people of different builds, run
 * through the model once and saved. That needs a camera and volunteers,
 * neither of which exist here yet. What these fixtures *can* prove is that
 * the signal-processing logic is correct given a known input: thresholds,
 * hysteresis, debounce, return-to-neutral, normalisation, refractory and
 * gating. What they *cannot* prove is that real bodies produce the signals
 * assumed here. Recorded fixtures remain outstanding and the same tests will
 * run against them unchanged, because both arrive as `PoseSample[]`.
 *
 * Every fixture is emitted in *pipeline* space (pre-mirroring), so the
 * mirroring test T-07 exercises the real adapter rather than assuming it.
 */

import type { BodyOutput } from "../transport/protocol";
import { FRAME_ASPECT, toPoseSample, type PoseSample } from "./poseAdapter";

const FRAME_MS = 1000 / 30;

/**
 * The shoulder tilt a real person produces when they lean to change lane:
 * sin(17 degrees), a clear but unstrenuous tip of the shoulder line.
 *
 * Fixtures here have twice been wrong about what a body does — first 0.12 of
 * hip-relative lean (a 50-degree bend nobody performs), then the same value
 * re-derived in fixed units. Both times the tests asserted against thresholds
 * built on the same mistake, so the suite agreed with itself while the real
 * game ignored real movement. Keeping these magnitudes physically plausible
 * is what makes the suite evidence about the game rather than about itself.
 */
const REALISTIC_LEAN_X = 0.30;

/** Landmark indices we actually populate; the rest are filler. */
const LEFT_SHOULDER = 11;
const RIGHT_SHOULDER = 12;
const LANDMARK_COUNT = 33;

export interface SyntheticFrameOptions {
  /**
   * Shoulder tilt as a sine, in *pipeline* space: positive means the player's
   * anatomical left shoulder is the lower one, which is what leaning to their
   * own left looks like in an unmirrored frame.
   */
  leanX: number;
  /** Shoulder midpoint y, top-down normalised. Smaller = higher up. */
  riseY: number;
  /** Shoulder width in frame heights; the player's apparent size. */
  bodyScale: number;
  /** Label the pipeline's classifier would produce. */
  stableLabel: string;
  /** Non-null only on a change, matching ConsensusFilter's edge behaviour. */
  emittedMovement: string | null;
  personPresent: boolean;
}

function buildBody(options: SyntheticFrameOptions): BodyOutput {
  const landmarks: Array<[number, number]> = [];
  if (options.personPresent) {
    // Place two shoulders a fixed distance apart and roll the line by the
    // requested sine. Hips are left as filler: the adapter no longer reads
    // them, and a fixture that supplied plausible hips would hide the fact.
    const tilt = Math.max(-1, Math.min(1, options.leanX));
    const dy = tilt * options.bodyScale;
    const dxHeights = Math.sqrt(Math.max(options.bodyScale ** 2 - dy ** 2, 0));
    const dxNormalised = dxHeights / FRAME_ASPECT;
    const centre = 0.5;
    for (let i = 0; i < LANDMARK_COUNT; i += 1) {
      landmarks.push([centre, options.riseY]);
    }
    // Anatomical left sits at the higher x in an unmirrored frame, and drops
    // (larger y) when the player leans to their own left.
    landmarks[LEFT_SHOULDER] = [centre + dxNormalised / 2, options.riseY + dy / 2];
    landmarks[RIGHT_SHOULDER] = [centre - dxNormalised / 2, options.riseY - dy / 2];
  }
  return {
    raw_label: options.stableLabel,
    stable_label: options.stableLabel,
    emitted_movement: options.emittedMovement,
    metrics: {
      lean_x: options.leanX,
      jump_height: 0,
      lateral_velocity: 0,
      torso_angle: 0,
      crossed_distance: 0,
    },
    geometry: { landmarks },
  };
}

/**
 * A fixture is just a list of samples. Building them through the real
 * `toPoseSample` is deliberate: it means every test runs through the actual
 * mirroring code rather than a test-only copy of it.
 */
export class FixtureBuilder {
  private readonly samples: PoseSample[] = [];
  private timestampMs: number;
  private lastLabel = "idle";

  constructor(startMs = 1_000_000) {
    this.timestampMs = startMs;
  }

  /** Append `count` frames described by `options`. */
  add(count: number, options: Partial<SyntheticFrameOptions> = {}): this {
    const resolved: SyntheticFrameOptions = {
      leanX: 0,
      riseY: 0.6,
      bodyScale: 0.3,
      stableLabel: "idle",
      emittedMovement: null,
      personPresent: true,
      ...options,
    };
    for (let i = 0; i < count; i += 1) {
      // Emit the label edge only on the frame the label changes, exactly as
      // the pipeline's ConsensusFilter does.
      const changed = resolved.stableLabel !== this.lastLabel;
      const emitted =
        options.emittedMovement !== undefined
          ? resolved.emittedMovement
          : changed && resolved.stableLabel !== "idle"
            ? resolved.stableLabel
            : null;
      this.lastLabel = resolved.stableLabel;
      this.samples.push(
        toPoseSample(buildBody({ ...resolved, emittedMovement: emitted }), this.timestampMs),
      );
      this.timestampMs += FRAME_MS;
    }
    return this;
  }

  /** Skip forward without emitting frames -- for staleness tests. */
  gap(ms: number): this {
    this.timestampMs += ms;
    return this;
  }

  build(): PoseSample[] {
    return this.samples;
  }
}

/** Ten seconds of nothing. */
export function neutralStanding(): PoseSample[] {
  return new FixtureBuilder().add(300).build();
}

/**
 * One clean lean and return.
 *
 * @param side the player's *physical* side. Because fixtures are authored in
 *   unmirrored pipeline space, leaning to the player's left produces a
 *   *positive* `lean_x` -- which is precisely the inversion T-07 checks.
 */
export function leanSingle(side: "left" | "right", magnitude = REALISTIC_LEAN_X): PoseSample[] {
  const leanX = side === "left" ? magnitude : -magnitude;
  const label = side === "left" ? "bending_right" : "bending_left";
  return new FixtureBuilder()
    .add(20)
    .add(30, { leanX, stableLabel: label })
    .add(30)
    .build();
}

export function leanAlternating(): PoseSample[] {
  const builder = new FixtureBuilder().add(15);
  for (const side of ["left", "right", "left", "right"] as const) {
    const leanX = side === "left" ? REALISTIC_LEAN_X : -REALISTIC_LEAN_X;
    const label = side === "left" ? "bending_right" : "bending_left";
    builder.add(20, { leanX, stableLabel: label }).add(20);
  }
  return builder.build();
}

/** Lean and hold for five seconds. Must produce exactly one lane change. */
export function leanHeld(): PoseSample[] {
  return new FixtureBuilder()
    .add(15)
    .add(150, { leanX: REALISTIC_LEAN_X, stableLabel: "bending_right" })
    .add(20)
    .build();
}

/**
 * Hover right at the threshold. The oscillation test (T-03).
 *
 * Sits just under the enter threshold and wobbles across it by a hair, which
 * is what a player standing slightly off-square actually does.
 */
export function leanBoundary(): PoseSample[] {
  const builder = new FixtureBuilder().add(10);
  for (let i = 0; i < 120; i += 1) {
    // The value IS the tilt now, so the boundary sits at leanEnter (0.16)
    // with nothing to convert. Wobble either side of it.
    const leanX = 0.152 + Math.sin(i * 0.7) * 0.012;
    builder.add(1, { leanX });
  }
  return builder.build();
}

/** One clean jump: a smooth rise and fall of the shoulders. */
export function jumpSingle(bodyScale = 0.3, restingRiseY = 0.6): PoseSample[] {
  const builder = new FixtureBuilder().add(30, { bodyScale, riseY: restingRiseY });
  const peak = bodyScale * 0.2; // 20% of shoulder width -- a clear jump
  const frames = 10;
  for (let i = 0; i < frames; i += 1) {
    const rise = Math.sin((i / (frames - 1)) * Math.PI) * peak;
    builder.add(1, {
      bodyScale,
      riseY: restingRiseY - rise,
      stableLabel: rise > peak * 0.4 ? "jumping" : "idle",
    });
  }
  return builder.add(30, { bodyScale, riseY: restingRiseY }).build();
}

/**
 * A jump whose label arrives late, as the real pipeline's always does.
 *
 * `jumpSingle` flips the label on the exact frame the shoulders cross the
 * threshold. The real `ConsensusFilter` needs 3 `jumping` votes inside a
 * 5-frame window first, so the edge lands roughly 150ms after takeoff — at or
 * past the apex, where upward velocity is zero. That gap is the entire reason
 * jumping did nothing in real play while `jumpSingle` passed, so it gets its
 * own fixture rather than being folded into that one.
 *
 * @param delayFrames votes the filter waits for before emitting.
 */
export function jumpWithConsensusLag(delayFrames = 3): PoseSample[] {
  const bodyScale = 0.3;
  const restingRiseY = 0.6;
  const peak = bodyScale * 0.25;
  const frames = 12;
  const builder = new FixtureBuilder().add(30, { bodyScale, riseY: restingRiseY });

  // When the pipeline's own absolute threshold would have been crossed.
  const crossed: boolean[] = [];
  for (let i = 0; i < frames; i += 1) {
    crossed.push(Math.sin((i / (frames - 1)) * Math.PI) * peak > peak * 0.4);
  }

  let emittedAlready = false;
  for (let i = 0; i < frames; i += 1) {
    const rise = Math.sin((i / (frames - 1)) * Math.PI) * peak;
    // The label only turns on once `delayFrames` consecutive frames have been
    // over the threshold -- and it turns on at the END of that run.
    const votesReady =
      i >= delayFrames - 1 && crossed.slice(i - delayFrames + 1, i + 1).every(Boolean);
    const label = votesReady ? "jumping" : "idle";
    const emitted = votesReady && !emittedAlready ? "jumping" : null;
    if (emitted) emittedAlready = true;
    builder.add(1, {
      bodyScale,
      riseY: restingRiseY - rise,
      stableLabel: label,
      emittedMovement: emitted,
    });
  }
  return builder.add(30, { bodyScale, riseY: restingRiseY }).build();
}

export function jumpRepeated(): PoseSample[] {
  const builder = new FixtureBuilder();
  for (let n = 0; n < 3; n += 1) {
    builder.add(25);
    const peak = 0.05;
    for (let i = 0; i < 10; i += 1) {
      const rise = Math.sin((i / 9) * Math.PI) * peak;
      builder.add(1, { riseY: 0.6 - rise, stableLabel: rise > peak * 0.4 ? "jumping" : "idle" });
    }
  }
  return builder.add(25).build();
}

/**
 * Jogging on the spot. Must produce zero jumps (T-09).
 *
 * Small, fast bobs: individually below the height threshold, but frequent,
 * which is exactly the pattern that defeats a height-only check.
 */
export function jogInPlace(): PoseSample[] {
  const builder = new FixtureBuilder();
  for (let i = 0; i < 300; i += 1) {
    const bob = Math.abs(Math.sin(i * 0.5)) * 0.012; // ~4% of shoulder width
    builder.add(1, { riseY: 0.6 - bob });
  }
  return builder.build();
}

/**
 * **Owner-mandated fixture.** Jogging on the spot *while* leaning left and
 * right.
 *
 * The pipeline labels a jogging player `naruto_run`, which outranks
 * `bending_*` in its priority chain, so the lean never reaches the label.
 * Reading `lean_x` directly is what recovers it. Expectation: lane changes
 * fire normally despite every label saying `naruto_run`.
 */
export function jogInPlaceWithLean(): PoseSample[] {
  const builder = new FixtureBuilder();
  const sides: Array<"left" | "right"> = ["left", "right", "left", "right"];
  let frame = 0;
  const bob = (): number => Math.abs(Math.sin(frame * 0.5)) * 0.012;

  for (let i = 0; i < 20; i += 1, frame += 1) {
    builder.add(1, { riseY: 0.6 - bob(), stableLabel: "naruto_run" });
  }
  for (const side of sides) {
    const leanX = side === "left" ? REALISTIC_LEAN_X : -REALISTIC_LEAN_X;
    for (let i = 0; i < 20; i += 1, frame += 1) {
      // Still jogging, so the label stays naruto_run throughout and the
      // lean is only ever visible in the metric.
      builder.add(1, { leanX, riseY: 0.6 - bob(), stableLabel: "naruto_run" });
    }
    for (let i = 0; i < 20; i += 1, frame += 1) {
      builder.add(1, { riseY: 0.6 - bob(), stableLabel: "naruto_run" });
    }
  }
  return builder.build();
}

/**
 * The player drifts off-square after calibrating, then leans both ways.
 *
 * Taken from a real debug capture: the player had been standing still for
 * 947 frames and their lean still read 42% of the enter threshold. A drift
 * that size makes one direction need roughly 2.5x the lean of the other,
 * which is felt as "left works, right doesn't". Both leans here are the same
 * size, so any difference in the result is the bug.
 *
 * @param restingDrift shoulder tilt the player rests at after drifting.
 */
export function driftedRestingPosture(restingDrift = 0.07): PoseSample[] {
  const builder = new FixtureBuilder();
  // Calibrated square, so the baseline offset is ~0 and the drift is real
  // drift rather than something calibration would have absorbed.
  builder.add(60);
  // Then settle into the off-square posture and stay there a while.
  builder.add(150, { leanX: restingDrift });
  for (const side of ["left", "right"] as const) {
    const leanX =
      restingDrift + (side === "left" ? REALISTIC_LEAN_X : -REALISTIC_LEAN_X);
    builder.add(20, { leanX });
    builder.add(40, { leanX: restingDrift });
  }
  return builder.build();
}

/**
 * The player settles lower in their seat, then jumps twice.
 *
 * From a real capture: `jump rise 0.266 peak 0.266` — a rise well past the
 * 0.12 threshold that never changed, because the player had simply moved
 * since calibrating. A standing height captured once is stale as soon as
 * anybody shifts in a chair, and the jump is then measured against a floor
 * that is not where the player is standing.
 *
 * @param settle how far the shoulders move from the calibrated height.
 */
export function settledThenJumps(settle = 0.08): PoseSample[] {
  const bodyScale = 0.3;
  const restingRiseY = 0.4;
  const builder = new FixtureBuilder().add(30, { bodyScale, riseY: restingRiseY });
  // Drift to a new resting height over a couple of seconds, then hold it.
  for (let i = 0; i < 60; i += 1) {
    builder.add(1, { bodyScale, riseY: restingRiseY + (settle * i) / 60 });
  }
  const settled = restingRiseY + settle;
  builder.add(60, { bodyScale, riseY: settled });

  // Two real jumps from the new resting height.
  for (let n = 0; n < 2; n += 1) {
    const peak = bodyScale * 0.25;
    for (let i = 0; i < 10; i += 1) {
      const rise = Math.sin((i / 9) * Math.PI) * peak;
      builder.add(1, { bodyScale, riseY: settled - rise });
    }
    builder.add(40, { bodyScale, riseY: settled });
  }
  return builder.build();
}

/** Player walks out of frame and comes back. */
export function personExits(): PoseSample[] {
  return new FixtureBuilder()
    .add(60)
    .add(90, { personPresent: false, stableLabel: "idle" })
    .add(60)
    .build();
}

/**
 * The same movements at two distances (T-11).
 *
 * At 3m everything shrinks: the torso, the lean offset and the jump
 * excursion all scale together. Normalising by calibrated torso height is
 * what must make the two produce identical decisions.
 */
export function farAndNear(scale: number): PoseSample[] {
  const bodyScale = 0.3 * scale;
  const riseY = 0.6;
  // The tilt itself is NOT scaled: an angle does not shrink with distance,
  // and asserting that the same angle survives a change of apparent size is
  // the whole point of T-11.
  return new FixtureBuilder()
    .add(20, { bodyScale, riseY })
    .add(30, { bodyScale, riseY, leanX: REALISTIC_LEAN_X, stableLabel: "bending_right" })
    .add(20, { bodyScale, riseY })
    .build();
}

/** Events that stop arriving -- for the staleness/disconnect test. */
export function staleAfter(frames: number, gapMs: number): PoseSample[] {
  return new FixtureBuilder().add(frames).gap(gapMs).add(1).build();
}
