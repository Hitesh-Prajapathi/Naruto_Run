/**
 * ============================================================================
 *  THE ONE PLACE THE CAMERA IMAGE IS MIRRORED.  DO NOT FLIP ANYWHERE ELSE.
 * ============================================================================
 *
 * Feature Brief 05 §5.1. Read this before touching anything that has an x
 * coordinate in it.
 *
 * **The contract:** the player's physical left → Naruto moves to screen left.
 *
 * **Why a flip is needed at all.** The recognition pipeline runs on the raw,
 * unmirrored camera frame:
 *   - `framePublisher.ts` captures from the raw video element and documents
 *     that it "never mirrors";
 *   - `combined_pipeline.BodyMovementRecognizer.process()` takes that frame
 *     as-is;
 *   - `run_combined_camera.py` flips only *after* recognition, and
 *     `_mirror_result_for_display` reflects geometry while explicitly leaving
 *     labels alone (asserted in `cv_model/tests/test_display_mirroring.py`).
 *
 * In an unmirrored frame the player's physical left appears at *higher* x.
 * So a player leaning to their own left drops their anatomical left shoulder,
 * which is the one at higher x, producing a positive shoulder tilt — what the
 * pipeline labels **`bending_right`**. Passed through untouched, leaning left
 * would move Naruto right: the exact bug §5.1 warns about.
 *
 * **Upper body only.** Lean and jump are both measured from the two shoulder
 * landmarks and nothing else — no hips. A webcam at desk height frames a
 * player from the chest up, so the hips sit at or past the bottom edge and
 * MediaPipe estimates rather than measures them. Anything derived from them
 * inherits that guess. See the geometry block in `toPoseSample`.
 *
 * **The two separate concerns, both handled here, once each:**
 *
 *   1. *Gameplay*: negate `lean_x` and swap `bending_left`/`bending_right`,
 *      so downstream code is in player space and never thinks about cameras.
 *   2. *Display*: mirror landmark x for the skeleton overlay, because the
 *      preview video is CSS-mirrored. An unmirrored skeleton over a mirrored
 *      video renders backwards and looks broken.
 *
 * These are different fixes for the same underlying fact, which is precisely
 * why they live side by side in one function: it must be impossible to do one
 * and forget the other, and impossible to add a third flip somewhere else
 * without contradicting this comment.
 *
 * **If Naruto ever moves the wrong way, change `MIRROR_CAMERA_X` below — do
 * not add a negation at the call site.** Two flips cancel out and the bug
 * becomes intermittent and unexplainable.
 */

import { CV_CAPTURE_HEIGHT, CV_CAPTURE_WIDTH } from "../config/cvInputConfig";
import type { BodyOutput, PipelineOutputV1 } from "../transport/protocol";

/**
 * Frame aspect ratio, and why anything here cares.
 *
 * MediaPipe normalises landmark x by frame *width* and y by frame *height*.
 * On a 16:9 frame those are different pixel scales: x = 0.1 is 64px, y = 0.1
 * is 36px. So `lean_x / torsoHeight` — an x-normalised numerator over a
 * y-normalised denominator — is not a ratio of two lengths at all, and comes
 * out 1.78x too small.
 *
 * That is not a rounding error. It made the lean threshold demand roughly a
 * 25-degree lean when it was tuned expecting about 14, which is why leaning
 * did nothing in real play while the synthetic fixtures (authored in the same
 * broken units, so self-consistently wrong) passed.
 *
 * Multiplying x by the aspect puts both axes in units of frame height, so
 * `leanX / torsoHeight` becomes the sine of the actual lean angle: a real
 * geometric quantity, independent of capture resolution and of how far away
 * the player is standing.
 */
export const FRAME_ASPECT = CV_CAPTURE_WIDTH / CV_CAPTURE_HEIGHT;

/**
 * The single switch. True because the pipeline is fed unmirrored frames.
 * If the capture path ever starts mirroring before inference, flip this to
 * false and everything downstream stays correct.
 */
export const MIRROR_CAMERA_X = true;

/** Body labels, in *player* space once they have been through here. */
export type BodyLabel =
  | "idle"
  | "bending_left"
  | "bending_right"
  | "jumping"
  | "naruto_run";

/** One frame of pose, in player space. Everything downstream consumes this
 * and nothing downstream knows the camera exists. */
export interface PoseSample {
  /** Wall-clock ms when the frame was captured, for staleness checks. */
  timestampMs: number;
  /** False when no person was detected at all. */
  personPresent: boolean;
  /** Stable (consensus-filtered) label, mirrored into player space. */
  stableLabel: BodyLabel;
  /**
   * Edge-triggered label change from the pipeline's ConsensusFilter, or null.
   * Already debounced and already fire-on-transition -- do not add either
   * again downstream (Brief 05 §2.1: don't duplicate what the model does).
   */
  emittedMovement: BodyLabel | null;
  /**
   * Shoulder-line lean in player space: negative = leaning to the player's
   * left. This is the *sine of the shoulder roll angle* — 0.17 is 10 degrees,
   * 0.34 is 20 — so it is already dimensionless and already free of any
   * dependence on distance from the camera. Not yet calibrated or scaled;
   * `leanDetector` does that.
   */
  leanX: number;
  /** Hip rise from the pipeline's own metric. Unused for gameplay (it is
   * hip-derived); kept for the debug overlay. */
  jumpHeight: number;
  /** Shoulder midpoint lateral speed, normalised units/second. */
  lateralVelocity: number;
  /** Landmarks in *display* space: already mirrored to sit correctly on top
   * of the CSS-mirrored preview. */
  displayLandmarks: ReadonlyArray<readonly [number, number]>;
  /**
   * Shoulder midpoint y in raw image space (top-down). The vertical reference
   * a jump is measured against.
   */
  riseY: number;
  /**
   * Shoulder width, in units of frame height. The player's apparent size, and
   * so the divisor that makes jump thresholds mean the same thing at 1m and
   * at 3m. Measured as the full shoulder-to-shoulder distance rather than its
   * horizontal component, so that leaning — which shortens the horizontal
   * component — does not change the player's apparent size.
   */
  bodyScale: number;
  /**
   * Which landmark pair the geometry came from this frame. Anything other
   * than "shoulders" means the shoulders were unreadable and a fallback was
   * used, which is worth seeing rather than inferring from odd numbers.
   */
  geometrySource: GeometrySource;
}

const LABEL_MIRROR: Record<string, BodyLabel> = {
  bending_left: "bending_right",
  bending_right: "bending_left",
  idle: "idle",
  jumping: "jumping",
  naruto_run: "naruto_run",
};

function mirrorLabel(label: string): BodyLabel {
  const mapped = LABEL_MIRROR[label];
  if (mapped === undefined) {
    // An unrecognised label must not be silently treated as a lean.
    return "idle";
  }
  return MIRROR_CAMERA_X ? mapped : (label as BodyLabel);
}

/**
 * Landmark pairs the geometry can be read from, best first.
 *
 * **Why a chain and not just the shoulders.** MediaPipe returns all 33
 * landmarks or none — presence is its own binary decision — but individual
 * points vary a lot in quality between people: a loose hood, long hair, a
 * high collar or a shoulder clipped by the frame edge can put one shoulder
 * somewhere useless while the rest of the pose is perfectly good. One bad
 * point used to take the whole measurement down with it.
 *
 * Each pair spans the body left-to-right, so each yields the same two
 * quantities: the roll of the line between them, and its length as a measure
 * of apparent size. The first pair that produces a sane reading wins.
 *
 * Shoulders stay first and are used whenever they are usable — the owner's
 * instruction is that bending is measured from the shoulders, and the head
 * pairs are a fallback for when they cannot be read, never an addition to
 * them. A head pair alone would otherwise let a player change lane by tilting
 * their head while their shoulders stayed square.
 */
const LANDMARK_PAIRS: ReadonlyArray<{ name: GeometrySource; left: number; right: number }> = [
  { name: "shoulders", left: 11, right: 12 },
  { name: "ears", left: 7, right: 8 },
  { name: "eyes", left: 3, right: 6 },
];

/** Which pair the frame's geometry actually came from. */
export type GeometrySource = "shoulders" | "ears" | "eyes" | "none";

/**
 * Smallest believable span, in frame heights. Below this the two points are
 * effectively on top of each other and the roll between them is noise
 * amplified by a near-zero divisor.
 */
const MIN_PAIR_SPAN = 0.04;

/**
 * Convert one pipeline body payload into a player-space `PoseSample`.
 *
 * @param timestampMs when the frame was captured (from the transport).
 */
export function toPoseSample(body: BodyOutput, timestampMs: number): PoseSample {
  const landmarks = body.geometry?.landmarks ?? [];
  const personPresent = landmarks.length > 0;

  const metric = (name: string): number => {
    const value = body.metrics[name];
    return typeof value === "number" && Number.isFinite(value) ? value : 0;
  };

  // (2) Display mirroring: reflect x so the skeleton lines up with the
  // CSS-mirrored preview.
  const displayLandmarks: Array<readonly [number, number]> = landmarks.map((point) => {
    const x = point[0] ?? 0;
    const y = point[1] ?? 0;
    return [MIRROR_CAMERA_X ? 1 - x : x, y] as const;
  });

  // ---- Upper-body geometry, from the two shoulders and nothing else ------
  //
  // Everything here used to be measured against the hips, and that is why it
  // did not work. A webcam at desk height sees a player from roughly the
  // chest up, so the hips are at or past the bottom edge of the frame;
  // MediaPipe then stops measuring them and starts *estimating* them. Lean
  // was `shoulder_x - hip_x`, so it inherited the whole of that guess, which
  // is why one direction worked and the other did not, and why it drifted.
  //
  // Shoulders are the right landmarks for this game: they are the last thing
  // to leave frame, the most reliably tracked points on the upper body, and
  // "bend your shoulders" is the instruction players are actually given.
  let riseY = 0;
  let bodyScale = 0;
  let shoulderTilt = 0;
  let geometrySource: GeometrySource = "none";

  for (const pair of LANDMARK_PAIRS) {
    const left = landmarks[pair.left];
    const right = landmarks[pair.right];
    if (!left || !right) continue;

    // Into a common unit (frame heights) so x and y distances are comparable.
    const lx = (left[0] ?? 0) * FRAME_ASPECT;
    const ly = left[1] ?? 0;
    const rx = (right[0] ?? 0) * FRAME_ASPECT;
    const ry = right[1] ?? 0;

    const dx = lx - rx;
    const dy = ly - ry;
    // Full corner-to-corner length, not its horizontal component: the line
    // shortens horizontally as it tilts, and using that would make a leaning
    // player appear to shrink and their thresholds move.
    const span = Math.hypot(dx, dy);
    if (!Number.isFinite(span) || span < MIN_PAIR_SPAN) continue;

    bodyScale = span;
    riseY = (ly + ry) / 2;
    // Rise over the line's own length is the sine of its roll. The left index
    // of every pair is the player's anatomical left, which in an unmirrored
    // frame sits at the higher x; leaning to their left drops that side, so
    // dy > 0 — the same sign the pipeline's hip-based `lean_x` used, which
    // keeps the mirroring rule below unchanged.
    shoulderTilt = dy / span;
    geometrySource = pair.name;
    break;
  }

  // A head pair spans a fraction of the shoulders, so the same physical
  // distance from the camera yields a much smaller `bodyScale`. Jump
  // thresholds are fractions of it, and would silently become far easier to
  // trip. Scale head spans up to the shoulder width they imply.
  if (geometrySource === "ears" || geometrySource === "eyes") {
    const factor = geometrySource === "ears" ? 2.4 : 3.6;
    bodyScale *= factor;
  }

  // (1) Gameplay mirroring: negate lean so negative means "player's left".
  const leanX = MIRROR_CAMERA_X ? -shoulderTilt : shoulderTilt;

  return {
    timestampMs,
    personPresent,
    stableLabel: mirrorLabel(body.stable_label),
    emittedMovement: body.emitted_movement === null ? null : mirrorLabel(body.emitted_movement),
    leanX,
    jumpHeight: metric("jump_height"),
    lateralVelocity: metric("lateral_velocity"),
    displayLandmarks,
    riseY,
    bodyScale,
    geometrySource,
  };
}

/** Convenience for the whole snapshot. */
export function poseSampleFromOutput(
  output: PipelineOutputV1,
  timestampMs: number,
): PoseSample {
  return toPoseSample(output.body, timestampMs);
}
