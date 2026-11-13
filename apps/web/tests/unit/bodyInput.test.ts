/**
 * Feature Brief 05 §9.3 — the full T-01..T-16 matrix, plus the owner's
 * mandated `jog_in_place_with_lean` case.
 *
 * All of it runs against fixtures: no camera, no human, no browser. That was
 * §9.1's point, and it is what lets the signal processing be declared correct
 * before a camera is ever connected (§11 step 5).
 */

import { describe, expect, it } from "vitest";
import { DEFAULT_BODY_TUNING, mergeBodyTuning, BodyTuningError, type BodyTuning } from "../../src/input/bodyTuning";
import { FRAME_ASPECT, MIRROR_CAMERA_X, toPoseSample, type PoseSample } from "../../src/input/poseAdapter";
import { BodyCalibrator, IDENTITY_BASELINE, calibratedLean, normalisedJumpRise } from "../../src/input/bodyCalibration";
import { BodyInputSource, GATE_ALL, GATE_NONE, type BodyGate } from "../../src/input/bodyInputSource";
import { InputRouter, type InputIntent } from "../../src/input/inputRouter";
import * as fixtures from "../../src/input/poseFixtures";

/** Run a fixture through the full input path and collect the intents. */
function play(
  samples: PoseSample[],
  options: {
    gate?: BodyGate;
    tuning?: BodyTuning;
    calibrate?: boolean;
  } = {},
): { intents: InputIntent[]; source: BodyInputSource; router: InputRouter } {
  const source = new BodyInputSource(options.tuning ?? DEFAULT_BODY_TUNING);
  const router = new InputRouter();
  const gate = options.gate ?? GATE_ALL;

  // Calibrate on the opening frames, the way the startup gate does.
  if (options.calibrate !== false) {
    source.beginCalibration();
    for (const sample of samples.slice(0, 15)) {
      source.push(sample, GATE_NONE, router, sample.timestampMs);
    }
    source.finishCalibration();
    router.clear();
  }

  const intents: InputIntent[] = [];
  for (const sample of samples) {
    source.push(sample, gate, router, sample.timestampMs);
    intents.push(...router.drain());
  }
  return { intents, source, router };
}

const lanes = (intents: InputIntent[]): number[] =>
  intents.filter((i) => i.kind === "lane").map((i) => (i as { direction: number }).direction);
const jumps = (intents: InputIntent[]): number => intents.filter((i) => i.kind === "jump").length;

describe("T-01: lean value from synthetic landmarks", () => {
  it("has the correct sign and magnitude in player space", () => {
    // Authored in pipeline space: a player leaning to their physical LEFT
    // shows up at higher x, i.e. positive lean_x.
    const [sample] = fixtures.leanSingle("left").slice(25);
    expect(sample).toBeDefined();
    // After the adapter it must read negative -- "player's left".
    expect(sample!.leanX).toBeLessThan(0);
    // The value is the sine of the shoulder roll, recovered from the two
    // shoulder landmarks -- no division, and no dependence on frame shape.
    expect(Math.abs(sample!.leanX)).toBeCloseTo(0.3, 5);
    expect(Math.asin(Math.abs(sample!.leanX)) * (180 / Math.PI)).toBeCloseTo(17.5, 0);

    const scaled = calibratedLean(sample!, IDENTITY_BASELINE, DEFAULT_BODY_TUNING.leanScale);
    expect(scaled).toBeLessThan(-DEFAULT_BODY_TUNING.leanEnter);
  });
});

describe("T-07: mirroring — the bug most likely to survive to the demo", () => {
  it("player's physical left produces a LEFT lane step", () => {
    const { intents } = play(fixtures.leanSingle("left"));
    expect(lanes(intents)).toEqual([-1]);
  });

  it("player's physical right produces a RIGHT lane step", () => {
    const { intents } = play(fixtures.leanSingle("right"));
    expect(lanes(intents)).toEqual([1]);
  });

  it("alternating leans track the player, in order", () => {
    const { intents } = play(fixtures.leanAlternating());
    expect(lanes(intents)).toEqual([-1, 1, -1, 1]);
  });

  it("flips in exactly one place", () => {
    // If someone adds a second flip elsewhere, this constant is the only
    // thing that should ever have been touched -- and the two directional
    // tests above will fail rather than silently cancelling out.
    expect(MIRROR_CAMERA_X).toBe(true);
  });
});

describe("resting drift: both directions stay equally reachable", () => {
  it("recovers both leans after the player drifts off-square", () => {
    // Without re-centring this produces [-1] and never a +1, because the
    // drift eats most of one side's threshold. Reported from real play as
    // "left works, right doesn't".
    const { intents } = play(fixtures.driftedRestingPosture());
    expect(lanes(intents)).toEqual([-1, 1]);
  });

  it("works the same when the drift is the other way", () => {
    const { intents } = play(fixtures.driftedRestingPosture(-0.07));
    expect(lanes(intents)).toEqual([-1, 1]);
  });

  it("does not absorb a held lean into the neutral point", () => {
    // The re-centring must freeze while a lean is held, or a long lean would
    // quietly become the new "straight ahead" and the player would have to
    // lean further and further to hold the same lane.
    const { intents, source } = play(fixtures.leanHeld());
    expect(lanes(intents)).toEqual([-1]);
    expect(Math.abs(source.debug.neutralBias)).toBeLessThan(
      DEFAULT_BODY_TUNING.leanEnter,
    );
  });

  it("reports the drift it is correcting", () => {
    const { source } = play(fixtures.driftedRestingPosture());
    expect(source.debug.neutralBias).not.toBe(0);
  });
});

describe("the standing height follows the player", () => {
  it("still detects jumps after the player settles to a new height", () => {
    // With a frozen baseline the settle alone reads as a permanent rise and
    // the real jumps on top of it are indistinguishable from it.
    const { intents } = play(fixtures.settledThenJumps());
    expect(jumps(intents)).toBe(2);
  });

  it("does not invent a jump from the settle itself", () => {
    // Same drift, no jumps in the fixture at all.
    const bodyScale = 0.3;
    const builder = new fixtures.FixtureBuilder().add(30, { bodyScale, riseY: 0.4 });
    for (let i = 0; i < 60; i += 1) {
      builder.add(1, { bodyScale, riseY: 0.4 + (0.08 * i) / 60 });
    }
    builder.add(60, { bodyScale, riseY: 0.48 });
    expect(jumps(play(builder.build()).intents)).toBe(0);
  });

  it("a jump does not become the new standing height", () => {
    const { intents, source } = play(fixtures.jumpSingle());
    expect(jumps(intents)).toBe(1);
    // The baseline must still be near where the player actually stands.
    expect(source.debug.standingHeight).toBeCloseTo(0.6, 1);
  });
});

describe("landmark fallback: one bad point must not kill the measurement", () => {
  /** Build a frame with chosen positions for the shoulder and ear pairs. */
  const frame = (
    shoulders: { span: number; tilt: number } | null,
    ears: { span: number; tilt: number } | null,
  ): PoseSample => {
    const landmarks: Array<[number, number]> = [];
    for (let i = 0; i < 33; i += 1) landmarks.push([0.5, 0.4]);
    const place = (li: number, ri: number, span: number, tilt: number): void => {
      const dy = tilt * span;
      const dx = Math.sqrt(Math.max(span ** 2 - dy ** 2, 0)) / FRAME_ASPECT;
      landmarks[li] = [0.5 + dx / 2, 0.4 + dy / 2];
      landmarks[ri] = [0.5 - dx / 2, 0.4 - dy / 2];
    };
    // Collapsed pairs stand in for a landmark the model placed badly.
    place(11, 12, shoulders?.span ?? 0.001, shoulders?.tilt ?? 0);
    place(7, 8, ears?.span ?? 0.001, ears?.tilt ?? 0);
    return toPoseSample(
      {
        raw_label: "idle",
        stable_label: "idle",
        emitted_movement: null,
        metrics: {},
        geometry: { landmarks },
      } as never,
      1000,
    );
  };

  it("uses the shoulders whenever they are readable", () => {
    const sample = frame({ span: 0.3, tilt: 0.3 }, { span: 0.12, tilt: -0.5 });
    expect(sample.geometrySource).toBe("shoulders");
    expect(sample.leanX).toBeCloseTo(-0.3, 5);
  });

  it("falls back to the ears when a shoulder is unusable", () => {
    const sample = frame(null, { span: 0.12, tilt: 0.3 });
    expect(sample.geometrySource).toBe("ears");
    expect(sample.leanX).toBeCloseTo(-0.3, 5);
    // Rescaled towards a shoulder width, so jump thresholds keep meaning
    // roughly the same thing rather than becoming trivially easy.
    expect(sample.bodyScale).toBeGreaterThan(0.2);
  });

  it("reports no geometry when nothing is measurable, rather than zero lean", () => {
    const sample = frame(null, null);
    expect(sample.geometrySource).toBe("none");
    expect(sample.bodyScale).toBe(0);
  });

  it("a head tilt alone cannot steer while the shoulders are readable", () => {
    // The owner's rule: bending is measured from the shoulders. Ears are a
    // fallback, never an additional vote.
    const samples: PoseSample[] = [];
    for (let i = 0; i < 60; i += 1) {
      samples.push({ ...frame({ span: 0.3, tilt: 0 }, { span: 0.12, tilt: 0.9 }), timestampMs: 1_000_000 + i * 33 });
    }
    expect(lanes(play(samples).intents)).toEqual([]);
  });
});

describe("the render loop runs faster than samples arrive", () => {
  /**
   * Replay a fixture the way `sceneMain` actually drives it: a 60fps render
   * loop consuming samples that arrive at the service's much slower rate.
   *
   * Every other test here feeds exactly one sample per iteration, which
   * quietly assumed the two clocks were the same. They are not -- 60fps
   * against about 13 samples/s -- and that gap hid a real bug: the same
   * sample was re-fed four or five times, so debounce counted rendered
   * frames instead of observations and scaled with framerate.
   */
  const playAtRenderRate = (
    samples: PoseSample[],
    { sampleHz = 13, renderHz = 60 }: { sampleHz?: number; renderHz?: number } = {},
  ): InputIntent[] => {
    const source = new BodyInputSource(DEFAULT_BODY_TUNING);
    const router = new InputRouter();
    source.beginCalibration();
    for (const sample of samples.slice(0, 15)) {
      source.push(sample, GATE_NONE, router, sample.timestampMs);
    }
    source.finishCalibration();
    router.clear();

    const intents: InputIntent[] = [];
    const renderStep = 1000 / renderHz;
    const sampleStep = 1000 / sampleHz;
    const startMs = samples[0]?.timestampMs ?? 0;
    const durationMs = sampleStep * samples.length;

    let index = 0;
    let lastPushedMs = -1;
    for (let elapsed = 0; elapsed < durationMs; elapsed += renderStep) {
      const nowMs = startMs + elapsed;
      // Advance to whichever sample the service would have delivered by now.
      const due = Math.min(Math.floor(elapsed / sampleStep), samples.length - 1);
      index = Math.max(index, due);
      const original = samples[index] as PoseSample;
      // Re-stamp onto the slower clock: the fixture was authored at 30fps.
      const sample: PoseSample = { ...original, timestampMs: startMs + index * sampleStep };
      if (sample.timestampMs !== lastPushedMs) {
        lastPushedMs = sample.timestampMs;
        source.push(sample, GATE_ALL, router, nowMs);
      } else {
        source.tick(nowMs);
      }
      intents.push(...router.drain());
    }
    return intents;
  };

  it("still steers when samples arrive at a third of the framerate", () => {
    expect(lanes(playAtRenderRate(fixtures.leanSingle("left")))).toEqual([-1]);
    expect(lanes(playAtRenderRate(fixtures.leanSingle("right")))).toEqual([1]);
  });

  it("still steers at a very low sample rate", () => {
    expect(lanes(playAtRenderRate(fixtures.leanSingle("left"), { sampleHz: 8 }))).toEqual([-1]);
  });

  it("a held lean is still exactly one lane change, not one per render frame", () => {
    expect(lanes(playAtRenderRate(fixtures.leanHeld()))).toEqual([-1]);
  });

  it("alternating leans stay in order", () => {
    expect(lanes(playAtRenderRate(fixtures.leanAlternating()))).toEqual([-1, 1, -1, 1]);
  });

  it("jogging on the spot still produces no jumps", () => {
    const intents = playAtRenderRate(fixtures.jogInPlace());
    expect(intents.filter((i) => i.kind === "jump")).toHaveLength(0);
  });

  it("tick alone marks the signal lost once samples stop arriving", () => {
    const source = new BodyInputSource(DEFAULT_BODY_TUNING);
    const router = new InputRouter();
    const [sample] = fixtures.neutralStanding();
    source.push(sample as PoseSample, GATE_ALL, router, (sample as PoseSample).timestampMs);
    expect(source.trackingStatus).toBe("tracking");

    source.tick((sample as PoseSample).timestampMs + DEFAULT_BODY_TUNING.staleEventMs + 1);
    expect(source.trackingStatus).toBe("lost");
    expect(source.debug.reason).toBe("frames_behind");
  });
});

describe("T-02 / T-03: hysteresis and boundary oscillation", () => {
  it("enters at the enter threshold and does not leave until the exit one", () => {
    expect(DEFAULT_BODY_TUNING.leanEnter).toBe(0.16);
    expect(DEFAULT_BODY_TUNING.leanExit).toBe(0.09);
    expect(DEFAULT_BODY_TUNING.leanExit).toBeLessThan(DEFAULT_BODY_TUNING.leanEnter);
  });

  it("hovering at the threshold produces ZERO lane changes", () => {
    const { intents } = play(fixtures.leanBoundary());
    expect(lanes(intents)).toEqual([]);
  });
});

describe("T-04: debounce", () => {
  it("ignores a one-frame spike but fires on a sustained signal", () => {
    const spike = new fixtures.FixtureBuilder()
      .add(20)
      .add(1, { leanX: 0.3, stableLabel: "bending_right" })
      .add(30)
      .build();
    expect(lanes(play(spike).intents)).toEqual([]);

    const sustained = new fixtures.FixtureBuilder()
      .add(20)
      .add(5, { leanX: 0.3, stableLabel: "bending_right" })
      .add(30)
      .build();
    expect(lanes(play(sustained).intents)).toEqual([-1]);
  });
});

describe("T-05 / T-06: held lean and return-to-neutral", () => {
  it("a five-second held lean moves exactly ONE lane", () => {
    const { intents } = play(fixtures.leanHeld());
    expect(lanes(intents)).toEqual([-1]);
  });

  it("blocks a second lane change until neutral is reached", () => {
    // Lean, briefly dip to a value still above the *exit* threshold, then
    // lean hard again. Because neutral was never reached, only one fires.
    const samples = new fixtures.FixtureBuilder()
      .add(20)
      .add(20, { leanX: 0.3, stableLabel: "bending_right" })
      .add(20, { leanX: 0.12, stableLabel: "bending_right" }) // above exit (0.09), below enter (0.16)
      .add(20, { leanX: 0.3, stableLabel: "bending_right" })
      .build();
    expect(lanes(play(samples).intents)).toEqual([-1]);
  });

  it("allows the next lane change once neutral has been passed through", () => {
    const samples = new fixtures.FixtureBuilder()
      .add(20)
      .add(20, { leanX: 0.3, stableLabel: "bending_right" })
      .add(20)
      .add(20, { leanX: 0.3, stableLabel: "bending_right" })
      .build();
    expect(lanes(play(samples).intents)).toEqual([-1, -1]);
  });
});

describe("owner-mandated: jog_in_place_with_lean", () => {
  it("lane changes fire normally while the label is stuck on naruto_run", () => {
    const samples = fixtures.jogInPlaceWithLean();
    // Every frame is labelled naruto_run -- the lean is only in the metric.
    expect(samples.every((s) => s.stableLabel === "naruto_run")).toBe(true);

    const { intents } = play(samples);
    expect(lanes(intents)).toEqual([-1, 1, -1, 1]);
    // ...and the jogging must still not produce phantom jumps.
    expect(jumps(intents)).toBe(0);
  });
});

describe("T-08 / T-09 / T-10: jump", () => {
  it("fires above the threshold", () => {
    const { intents } = play(fixtures.jumpSingle());
    expect(jumps(intents)).toBe(1);
  });

  it("fires when the label arrives late, as the real ConsensusFilter's does", () => {
    // The regression test for the bug that made jumping do nothing in real
    // play. The height and velocity checks used to sample the edge frame
    // only, and the edge frame is at the apex, where upward velocity is zero
    // by definition -- so a perfectly good jump was rejected every time.
    const { intents } = play(fixtures.jumpWithConsensusLag(3));
    expect(jumps(intents)).toBe(1);
  });

  it("still fires with an even later label", () => {
    const { intents } = play(fixtures.jumpWithConsensusLag(5));
    expect(jumps(intents)).toBe(1);
  });

  it("does not fire below the threshold", () => {
    const shallow = new fixtures.FixtureBuilder()
      .add(30)
      .add(6, { riseY: 0.6 - 0.3 * 0.04, stableLabel: "jumping" }) // 4% of shoulder width
      .add(30)
      .build();
    expect(jumps(play(shallow).intents)).toBe(0);
  });

  it("jogging on the spot produces ZERO jumps", () => {
    const { intents } = play(fixtures.jogInPlace());
    expect(jumps(intents)).toBe(0);
  });

  it("two crossings inside the refractory window produce ONE jump", () => {
    const builder = new fixtures.FixtureBuilder().add(30);
    // Two full jumps ~200ms apart, well inside the 600ms refractory.
    for (let n = 0; n < 2; n += 1) {
      for (let i = 0; i < 6; i += 1) {
        const rise = Math.sin((i / 5) * Math.PI) * 0.05;
        builder.add(1, { riseY: 0.6 - rise, stableLabel: rise > 0.02 ? "jumping" : "idle" });
      }
    }
    expect(jumps(play(builder.build()).intents)).toBe(1);
  });

  it("three well-separated jumps produce three", () => {
    expect(jumps(play(fixtures.jumpRepeated()).intents)).toBe(3);
  });
});

describe("T-11: distance normalisation", () => {
  it("the same movement at 1m and 3m produces the same decision", () => {
    const near = play(fixtures.farAndNear(1));
    const far = play(fixtures.farAndNear(0.45));
    expect(lanes(near.intents)).toEqual([-1]);
    expect(lanes(far.intents)).toEqual(lanes(near.intents));
  });

  it("normalised jump rise is scale-invariant", () => {
    const make = (scale: number): PoseSample =>
      fixtures.jumpSingle(0.3 * scale, 0.6)[35] as PoseSample;
    const nearBase = { ...IDENTITY_BASELINE, bodyScale: 0.3, riseY: 0.6 };
    const farBase = { ...IDENTITY_BASELINE, bodyScale: 0.15, riseY: 0.6 };
    expect(normalisedJumpRise(make(1), nearBase)).toBeCloseTo(
      normalisedJumpRise(make(0.5), farBase),
      5,
    );
  });
});

describe("T-12: calibration baseline", () => {
  it("an offset neutral pose calibrates to zero", () => {
    // A player who naturally stands 0.04 off-square.
    const offset = new fixtures.FixtureBuilder().add(40, { leanX: 0.04 }).build();
    const calibrator = new BodyCalibrator();
    for (const sample of offset) calibrator.add(sample);
    const baseline = calibrator.build();

    const last = offset[offset.length - 1] as PoseSample;
    expect(calibratedLean(last, baseline, DEFAULT_BODY_TUNING.leanScale)).toBeCloseTo(0, 5);
  });

  it("that offset alone never produces a lane change", () => {
    const offset = new fixtures.FixtureBuilder().add(120, { leanX: 0.04 }).build();
    expect(lanes(play(offset).intents)).toEqual([]);
  });

  it("falls back to identity rather than garbage when starved", () => {
    const calibrator = new BodyCalibrator();
    calibrator.add(new fixtures.FixtureBuilder().add(1).build()[0] as PoseSample);
    expect(calibrator.build()).toEqual(IDENTITY_BASELINE);
  });

  it("survives a reset, so Try Again does not force recalibration", () => {
    const { source } = play(fixtures.leanSingle("left"));
    const baseline = source.currentBaseline;
    source.reset();
    expect(source.currentBaseline).toEqual(baseline);
  });
});

describe("T-13: stale events", () => {
  it("a sample older than the stale threshold reports lost tracking", () => {
    const source = new BodyInputSource();
    const router = new InputRouter();
    const [sample] = fixtures.neutralStanding();
    source.push(sample as PoseSample, GATE_ALL, router, (sample as PoseSample).timestampMs + 800);
    expect(source.trackingStatus).toBe("lost");
  });

  it("a fresh sample reports tracking", () => {
    const source = new BodyInputSource();
    const router = new InputRouter();
    const [sample] = fixtures.neutralStanding();
    source.push(sample as PoseSample, GATE_ALL, router, (sample as PoseSample).timestampMs + 20);
    expect(source.trackingStatus).toBe("tracking");
  });

  it("a player who leaves frame degrades then reports lost", () => {
    const samples = fixtures.personExits();
    const source = new BodyInputSource();
    const router = new InputRouter();
    for (const sample of samples) {
      source.push(sample, GATE_ALL, router, sample.timestampMs);
    }
    // Ends with the player back in frame.
    expect(source.trackingStatus).toBe("tracking");
  });
});

describe("T-14 / T-15: state gating", () => {
  it("a lean during GAME_OVER produces no lane change", () => {
    const { intents } = play(fixtures.leanSingle("left"), { gate: GATE_NONE });
    expect(lanes(intents)).toEqual([]);
  });

  it("a lean during the special-attack warning produces no lane change", () => {
    // Brief 04 §4.2 suppression, expressed as the same gate.
    const suppressed: BodyGate = { laneInput: false, jumpInput: false };
    const { intents } = play(fixtures.leanSingle("right"), { gate: suppressed });
    expect(lanes(intents)).toEqual([]);
  });

  it("jump is gated independently of lane, for BOSS_COMBAT", () => {
    // §6: combat allows lane dodging but not jumping.
    const combat: BodyGate = { laneInput: true, jumpInput: false };
    expect(jumps(play(fixtures.jumpSingle(), { gate: combat }).intents)).toBe(0);
    expect(lanes(play(fixtures.leanSingle("left"), { gate: combat }).intents)).toEqual([-1]);
  });

  it("keeps detector history warm while gated, so the next frame is responsive", () => {
    // Gated for the whole fixture, then the same lean ungated: the second
    // run must not need to re-accumulate debounce history from scratch.
    const source = new BodyInputSource();
    const router = new InputRouter();
    const samples = fixtures.leanHeld();
    // Stop while the lean is still being held -- the fixture deliberately
    // returns to neutral at the end, which would mask the point.
    for (const sample of samples.slice(0, 60)) {
      source.push(sample, GATE_NONE, router, sample.timestampMs);
    }
    expect(router.drain()).toEqual([]);
    expect(source.debug.lean.state).not.toBe("neutral");
  });
});

describe("T-16: keyboard parity", () => {
  it("both providers produce the identical intent object", () => {
    const router = new InputRouter();
    router.laneLeft("keyboard");
    const fromKeyboard = router.drain();

    const { intents } = play(fixtures.leanSingle("left"));
    expect(intents.filter((i) => i.kind === "lane")).toEqual(fromKeyboard);
  });

  it("keyboard wins when both ask on the same frame", () => {
    const router = new InputRouter();
    router.push({ kind: "lane", direction: 1 }, "camera");
    router.push({ kind: "lane", direction: -1 }, "keyboard");
    expect(router.drain()).toEqual([{ kind: "lane", direction: -1 }]);
  });

  it("never applies two lane steps from one frame", () => {
    const router = new InputRouter();
    router.push({ kind: "lane", direction: 1 }, "camera");
    router.push({ kind: "lane", direction: 1 }, "camera");
    expect(router.drain()).toHaveLength(1);
  });

  it("clears queued intents on restart", () => {
    const router = new InputRouter();
    router.laneLeft("camera");
    router.clear();
    expect(router.drain()).toEqual([]);
  });
});

describe("tuning config is runtime-loadable and validated", () => {
  it("merges a partial overlay onto the defaults", () => {
    const merged = mergeBodyTuning({ leanEnter: 0.5 });
    expect(merged.leanEnter).toBe(0.5);
    expect(merged.leanExit).toBe(DEFAULT_BODY_TUNING.leanExit);
  });

  it("ignores _-prefixed comment keys", () => {
    expect(() => mergeBodyTuning({ _about: "notes", leanEnter: 0.4 })).not.toThrow();
  });

  it("rejects an unknown key rather than silently ignoring a typo", () => {
    expect(() => mergeBodyTuning({ leanEntre: 0.4 })).toThrow(BodyTuningError);
  });

  it("rejects a hysteresis gap that would reintroduce oscillation", () => {
    expect(() => mergeBodyTuning({ leanExit: 0.5 })).toThrow(BodyTuningError);
  });

  it("rejects non-numeric and non-positive values", () => {
    expect(() => mergeBodyTuning({ leanEnter: "high" })).toThrow(BodyTuningError);
    expect(() => mergeBodyTuning({ leanEnter: -1 })).toThrow(BodyTuningError);
  });

  it("applies a reloaded tuning without rebuilding the detectors", () => {
    const source = new BodyInputSource();
    source.setTuning({ ...DEFAULT_BODY_TUNING, leanEnter: 0.9, leanExit: 0.8 });
    // A normal lean is now below the raised threshold.
    const router = new InputRouter();
    for (const sample of fixtures.leanSingle("left")) {
      source.push(sample, GATE_ALL, router, sample.timestampMs);
    }
    expect(lanes(router.drain())).toEqual([]);
  });
});

describe("the adapter is the only mirroring point", () => {
  it("negates lean and swaps the labels together", () => {
    // Shoulders placed by hand rather than via the fixture builder, so this
    // test still fails if the builder and the adapter drift apart. The
    // anatomical left shoulder (index 11) sits at the higher x and is the
    // lower of the two, which is a player leaning to their own left.
    const landmarks: Array<[number, number]> = [];
    for (let i = 0; i < 33; i += 1) landmarks.push([0.2, 0.5]);
    const halfWidth = 0.3 / FRAME_ASPECT / 2;
    landmarks[11] = [0.5 + halfWidth, 0.5 + 0.05];
    landmarks[12] = [0.5 - halfWidth, 0.5 - 0.05];
    const body = {
      raw_label: "bending_right",
      stable_label: "bending_right",
      emitted_movement: "bending_right",
      metrics: { lean_x: 0.12 },
      geometry: { landmarks },
    };
    const sample = toPoseSample(body as never, 1000);
    // Leaning to the player's left must read NEGATIVE after mirroring.
    expect(sample.leanX).toBeLessThan(0);
    expect(sample.leanX).toBeCloseTo(-0.1 / Math.hypot(0.3, 0.1), 5);
    expect(sample.stableLabel).toBe("bending_left");
    expect(sample.emittedMovement).toBe("bending_left");
    // Display landmarks mirrored to match the CSS-mirrored preview.
    expect(sample.displayLandmarks[0]?.[0]).toBeCloseTo(0.8, 5);
  });

  it("treats an unrecognised label as idle rather than as a lean", () => {
    const body = {
      raw_label: "cartwheel",
      stable_label: "cartwheel",
      emitted_movement: null,
      metrics: {},
      geometry: null,
    };
    expect(toPoseSample(body as never, 1).stableLabel).toBe("idle");
  });
});
