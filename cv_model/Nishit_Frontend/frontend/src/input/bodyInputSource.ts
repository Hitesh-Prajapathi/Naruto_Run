/**
 * Turns a stream of `PoseSample`s into game intents -- Feature Brief 05
 * §5.4 and §6.
 *
 * Deliberately source-agnostic: it consumes `PoseSample`s and does not care
 * whether they came from a live camera, a recorded fixture, or a synthetic
 * generator (§9.1's "three input sources, one interface"). Everything below
 * is therefore testable with no camera and no human.
 *
 * §6's gating is applied to *consumption*, never to inference: the detectors
 * keep running and keep their history warm even while their output is
 * suppressed, so the frame after a cutscene ends is immediately responsive
 * rather than spending the debounce window catching up.
 */

import type { BodyTuning } from "./bodyTuning";
import { DEFAULT_BODY_TUNING } from "./bodyTuning";
import {
  BodyCalibrator,
  IDENTITY_BASELINE,
  calibratedLean,
  normalisedJumpRise,
  type BodyBaseline,
} from "./bodyCalibration";
import { LeanDetector, type LeanDebug } from "./leanDetector";
import { JumpDetector, type JumpDebug } from "./jumpDetector";
import type { PoseSample } from "./poseAdapter";
import type { InputRouter } from "./inputRouter";

/** What the host permits this frame (§6). Mirrors the boss capability table. */
export interface BodyGate {
  laneInput: boolean;
  jumpInput: boolean;
}

export const GATE_ALL: BodyGate = { laneInput: true, jumpInput: true };
export const GATE_NONE: BodyGate = { laneInput: false, jumpInput: false };

export type TrackingStatus = "no_signal" | "tracking" | "weak" | "lost";

/**
 * Why tracking is not healthy this frame.
 *
 * "Not detected" was covering four unrelated failures -- no frames at all,
 * frames arriving too late to use, frames with no person in them, and frames
 * with a person whose geometry could not be measured. They need completely
 * different fixes (start the service, reduce load, step into frame, improve
 * the lighting or framing), and collapsing them into one red badge meant
 * every report had to be diagnosed from scratch. Naming them costs nothing.
 */
export type TrackingReason =
  | "ok"
  | "no_frames"
  | "frames_behind"
  | "no_person"
  | "no_geometry";

export const TRACKING_REASON_TEXT: Record<TrackingReason, string> = {
  ok: "tracking",
  no_frames: "no frames from the service",
  frames_behind: "frames arriving too late",
  no_person: "nobody in frame",
  no_geometry: "person found, shoulders unreadable",
};

export interface BodyInputDebug {
  status: TrackingStatus;
  lean: LeanDebug;
  jump: JumpDebug;
  baseline: BodyBaseline;
  /**
   * Drift of the player's resting posture away from the calibrated baseline.
   * Large values mean calibration has gone stale -- which is visible as one
   * lean direction being much harder to trigger than the other.
   */
  neutralBias: number;
  /** Why `status` is not "tracking". */
  reason: TrackingReason;
  /** Which landmark pair the geometry came from -- see `poseAdapter`. */
  geometrySource: string;
  /** The adaptive standing shoulder height a jump is measured against. */
  standingHeight: number;
  /** Samples per second, measured over the last second. */
  sampleRate: number;
  /** Age of the newest sample, in ms. */
  sampleAgeMs: number;
  /** Total intents this source has produced -- a cheap sanity counter. */
  laneEvents: number;
  jumpEvents: number;
}

export class BodyInputSource {
  private tuning: BodyTuning;
  private readonly lean: LeanDetector;
  private readonly jump: JumpDetector;
  private readonly calibrator = new BodyCalibrator();

  private baseline: BodyBaseline = { ...IDENTITY_BASELINE };
  /**
   * The baseline the jump actually uses: the calibrated one, with its
   * standing height and apparent size allowed to follow the player.
   * `baseline` stays as captured so recalibration and the debug overlay have
   * something stable to report against.
   */
  private adaptive: BodyBaseline = { ...IDENTITY_BASELINE };
  private calibrating = false;
  private lastSample: PoseSample | null = null;
  private lastPersonMs: number | null = null;
  private status: TrackingStatus = "no_signal";
  private recentTimestamps: number[] = [];
  private laneEvents = 0;
  private jumpEvents = 0;
  private neutralBias = 0;
  private lastRecentreMs: number | null = null;
  private lastRiseY: number | null = null;
  private reason: TrackingReason = "no_frames";

  constructor(tuning: BodyTuning = DEFAULT_BODY_TUNING) {
    this.tuning = tuning;
    this.lean = new LeanDetector(tuning);
    this.jump = new JumpDetector(tuning);
  }

  setTuning(tuning: BodyTuning): void {
    this.tuning = tuning;
    this.lean.setTuning(tuning);
    this.jump.setTuning(tuning);
  }

  get currentBaseline(): BodyBaseline {
    return this.baseline;
  }

  get trackingStatus(): TrackingStatus {
    return this.status;
  }

  /** Restore a baseline captured earlier -- §4.4 requires calibration to
   * survive Try Again rather than being re-run on every retry. */
  restoreBaseline(baseline: BodyBaseline): void {
    this.baseline = { ...baseline };
    this.adaptive = { ...baseline };
  }

  beginCalibration(): void {
    this.calibrator.reset();
    this.calibrating = true;
  }

  /** @returns the captured baseline, or null when there was too little signal. */
  finishCalibration(): BodyBaseline | null {
    this.calibrating = false;
    if (!this.calibrator.hasEnough) {
      return null;
    }
    this.baseline = this.calibrator.build();
    this.adaptive = { ...this.baseline };
    // A new baseline invalidates the smoothing history built against the old
    // one; keeping it would produce a phantom lean on the first frame. The
    // accumulated drift goes with it -- it was drift away from the *previous*
    // baseline and means nothing against this one.
    this.neutralBias = 0;
    this.lastRecentreMs = null;
    this.lastRiseY = null;
    this.lean.reset();
    this.jump.reset();
    return this.baseline;
  }

  get calibrationSampleCount(): number {
    return this.calibrator.sampleCount;
  }

  /**
   * Feed one sample and emit any resulting intents into the router.
   *
   * @param nowMs wall clock, for staleness. Separate from the sample's own
   *   timestamp so a fixture can replay with synthetic times.
   */
  /**
   * Refresh staleness without advancing the detectors.
   *
   * The render loop runs at 60fps and samples arrive at whatever rate the
   * recognition service manages -- about 13/s. Those are different clocks,
   * and conflating them was a real bug: `push` used to be called every
   * rendered frame with whatever the newest sample was, so the same sample
   * was re-fed four or five times. That made "frames" in the debounce mean
   * rendered frames rather than observations (so the lean debounce silently
   * scaled with framerate), and made the event-rate readout report 54/s when
   * the service was emitting 13/s -- a number that sent several debugging
   * sessions in the wrong direction.
   *
   * So the host now calls `push` once per *new* sample, and this every frame
   * in between, which is all that is needed to notice that samples have
   * stopped arriving.
   */
  tick(nowMs: number): void {
    if (this.lastSample === null) return;
    this.status = this.deriveStatus(this.lastSample, nowMs);
  }

  push(sample: PoseSample, gate: BodyGate, router: InputRouter, nowMs: number): void {
    this.lastSample = sample;
    this.trackRate(sample.timestampMs);

    if (this.calibrating) {
      this.calibrator.add(sample);
    }

    if (sample.personPresent) {
      this.lastPersonMs = nowMs;
    }
    this.status = this.deriveStatus(sample, nowMs);

    // Detectors advance regardless of the gate so their history stays warm.
    const rawLean = calibratedLean(sample, this.baseline, this.tuning.leanScale);
    const lane = this.lean.update(rawLean - this.neutralBias);
    // The jump uses the *adaptive* baseline: its standing height follows the
    // player, because a captured-once shoulder height is stale the moment
    // they settle into their chair. See `recentre`.
    const jumped = this.jump.update(sample, this.adaptive);

    this.recentre(rawLean, sample);

    if (!sample.personPresent) {
      return;
    }
    if (lane !== 0 && gate.laneInput) {
      this.laneEvents += 1;
      router.push({ kind: "lane", direction: lane }, "camera");
    }
    if (jumped && gate.jumpInput) {
      this.jumpEvents += 1;
      router.push({ kind: "jump" }, "camera");
    }
  }

  /**
   * Slowly pull the neutral point onto the player's actual resting posture.
   *
   * Only while the detector reports neutral, which is what stops a held lean
   * from being quietly absorbed: entering a lean leaves the neutral state on
   * the same frame, so the bias freezes for as long as the lean lasts and
   * resumes once the player comes back inside the exit threshold.
   *
   * The bias is capped at the enter threshold. Beyond that the reading is not
   * drift, it is a player standing at an angle or a broken hip estimate, and
   * silently following it would make one direction unreachable rather than
   * merely harder -- the failure this exists to prevent.
   */
  private recentre(rawLean: number, sample: PoseSample): void {
    const previousMs = this.lastRecentreMs;
    this.lastRecentreMs = sample.timestampMs;
    if (previousMs === null || !sample.personPresent) return;

    const dtSeconds = (sample.timestampMs - previousMs) / 1000;
    if (dtSeconds <= 0 || dtSeconds > 1) return; // a gap, not a frame

    this.followStandingHeight(sample, dtSeconds);

    if (this.lean.debug.state !== "neutral") return;

    // Exponential approach, framed in seconds so it behaves the same at any
    // capture rate.
    const alpha = 1 - Math.exp(-dtSeconds * this.tuning.leanRecentreRate);
    const cap = this.tuning.leanEnter;
    const target = Math.max(-cap, Math.min(cap, rawLean));
    this.neutralBias += (target - this.neutralBias) * alpha;
  }

  /**
   * Let the standing height a jump is measured against follow the player.
   *
   * What separates a jump from a settle is **speed and direction, not size**.
   * A player sinking into a chair moves as far as a jump does, just slowly
   * and downward, so freezing on magnitude alone would refuse to follow the
   * very drift this exists to absorb — and did, until this test caught it.
   * So the baseline holds still only while the player is *elevated* or moving
   * *fast*, and follows them the rest of the time.
   */
  private followStandingHeight(sample: PoseSample, dtSeconds: number): void {
    if (!sample.personPresent || sample.bodyScale <= 0) return;

    const previousRiseY = this.lastRiseY;
    this.lastRiseY = sample.riseY;

    // Image y grows downward, so a shrinking y is upward movement.
    const upward =
      previousRiseY === null ? 0 : (previousRiseY - sample.riseY) / dtSeconds / sample.bodyScale;
    if (Math.abs(upward) >= this.tuning.jumpMinVelocity) return;

    // Above the floor: could be mid-jump, so do not move the floor.
    const rise = normalisedJumpRise(sample, this.adaptive);
    if (rise >= this.tuning.jumpHeightFraction * 0.5) return;

    const alpha = 1 - Math.exp(-dtSeconds * this.tuning.jumpBaselineRate);
    this.adaptive.riseY += (sample.riseY - this.adaptive.riseY) * alpha;
    this.adaptive.bodyScale += (sample.bodyScale - this.adaptive.bodyScale) * alpha;
  }

  private deriveStatus(sample: PoseSample, nowMs: number): TrackingStatus {
    const age = nowMs - sample.timestampMs;
    if (age > this.tuning.staleEventMs) {
      this.reason = "frames_behind";
      return "lost";
    }
    if (!sample.personPresent) {
      const lostFor = this.lastPersonMs === null ? Infinity : nowMs - this.lastPersonMs;
      this.reason = "no_person";
      return lostFor > this.tuning.lostPlayerPauseMs ? "lost" : "weak";
    }
    if (sample.bodyScale <= 0) {
      // A pose was found but no pair of landmarks in it was measurable.
      this.reason = "no_geometry";
      return "weak";
    }
    this.reason = "ok";
    return "tracking";
  }

  private trackRate(timestampMs: number): void {
    this.recentTimestamps.push(timestampMs);
    const cutoff = timestampMs - 1000;
    while (this.recentTimestamps.length > 0 && (this.recentTimestamps[0] as number) < cutoff) {
      this.recentTimestamps.shift();
    }
  }

  get debug(): BodyInputDebug {
    const newest = this.lastSample?.timestampMs ?? 0;
    return {
      status: this.status,
      lean: this.lean.debug,
      jump: this.jump.debug,
      baseline: this.baseline,
      neutralBias: this.neutralBias,
      reason: this.reason,
      geometrySource: this.lastSample?.geometrySource ?? "none",
      standingHeight: this.adaptive.riseY,
      sampleRate: this.recentTimestamps.length,
      sampleAgeMs: newest === 0 ? 0 : Math.max(0, Date.now() - newest),
      laneEvents: this.laneEvents,
      jumpEvents: this.jumpEvents,
    };
  }

  /** Reset per-run state. The baseline deliberately survives (§4.4). */
  reset(): void {
    this.lean.reset();
    this.jump.reset();
    this.lastSample = null;
    this.lastPersonMs = null;
    this.status = "no_signal";
    this.reason = "no_frames";
    this.recentTimestamps = [];
    this.laneEvents = 0;
    this.jumpEvents = 0;
    // Deliberately kept, like the baseline it corrects (§4.4): the drift the
    // player accumulated is still true of them on the retry.
    this.lastRecentreMs = null;
    this.lastRiseY = null;
  }
}
