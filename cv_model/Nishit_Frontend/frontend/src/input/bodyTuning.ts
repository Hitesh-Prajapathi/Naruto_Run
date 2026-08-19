/**
 * Every tunable constant for body-movement input -- Feature Brief 05.
 *
 * **One file, and it is loadable at runtime.** The owner's instruction: since
 * there is no trained body model, every threshold is just a constant and they
 * will be re-tuned a lot, so changing one must not require a rebuild. The
 * defaults below are the source of truth for types and for tests; at startup
 * the game fetches `/config/bodyTuning.json` and overlays whatever it finds,
 * so tuning in practice is: edit the JSON, refresh the page.
 *
 * The loader is deliberately forgiving about *missing* keys and strict about
 * *bad* ones: a half-written file during tuning should fall back to defaults
 * for the keys it omits, but a value of the wrong type or a nonsensical
 * threshold should say so loudly rather than silently producing a control
 * scheme nobody can explain.
 */

export interface BodyTuning {
  // --- Lean (Brief 05 §5.2) --------------------------------------------
  /**
   * Hysteresis on the calibrated lean value. Enter is deliberately well
   * above exit: without the gap a player resting near the boundary
   * oscillates lanes continuously (§5.2, test T-03).
   */
  leanEnter: number;
  leanExit: number;
  /** Consecutive samples the lean state must hold before firing (§5.2). */
  leanDebounceFrames: number;
  /**
   * Exponential smoothing on the raw lean metric, 0..1 -- higher is more
   * responsive. Every millisecond of smoothing is input lag, so this is
   * light by default.
   */
  leanSmoothing: number;
  /**
   * Pure sensitivity multiplier on the torso-normalised lean. At 1 the value
   * the thresholds see is the sine of the player's lean angle; raise it to
   * make the game respond to smaller leans without touching the geometry.
   * Applied after distance normalisation, so this is about feel only.
   */
  leanScale: number;
  /**
   * How fast the neutral point re-centres on the player's actual resting
   * posture, in units of "fraction of the remaining error per second".
   *
   * Calibration captures the resting lean once, at the gate. Players then
   * shift: they turn slightly, put their weight on one foot, or drift across
   * the camera. The captured offset goes stale and every reading inherits the
   * error — a drift of 0.093 against a 0.22 threshold makes one side 2.5x
   * harder to trigger than the other, which reads as "left works, right
   * doesn't". Re-centring only happens while the detector is in its neutral
   * state, so a held lean is never absorbed.
   *
   * Set to a very small number to effectively pin the baseline to
   * calibration; 0 is rejected, since a static baseline is what caused this.
   */
  leanRecentreRate: number;
  /**
   * How fast the *standing height* a jump is measured against follows the
   * player, per second. The lean equivalent, and needed for the same reason.
   *
   * A seated or standing player settles, shifts their weight, or rolls their
   * chair, and their shoulders end up at a different height than during
   * calibration. The captured baseline then reports a large permanent "rise"
   * that never changes -- 0.266 against a 0.12 threshold in one real capture
   * -- which is meaningless on its own and, worse, means a genuine jump adds
   * to an already-saturated number. Frozen automatically whenever the player
   * is actually off the ground, so a jump cannot re-baseline itself.
   */
  jumpBaselineRate: number;

  // --- Jump (Brief 05 §5.3) --------------------------------------------
  /**
   * Rise of the shoulder midpoint above the standing baseline, as a fraction
   * of the player's own shoulder width. Being relative to their size is what
   * makes it work at 1m and at 3m.
   */
  jumpHeightFraction: number;
  /** Required upward velocity, in shoulder widths per second. Guards against
   * a slow stand-up or weight shift reading as a jump. */
  jumpMinVelocity: number;
  /**
   * How far back the height and velocity checks look, in ms.
   *
   * The pipeline's `jumping` label arrives ~150ms late (ConsensusFilter needs
   * 3 of 5 frames), by which point the player is at their apex and their
   * upward velocity has already returned to zero. This window is what lets
   * the late label still be matched against the launch that produced it.
   * Shorter than a jump's airtime, or the checks stop discriminating.
   */
  jumpVelocityWindowMs: number;
  /** Refractory period after a jump fires, roughly the jump's airtime. */
  jumpRefractoryMs: number;
  /**
   * Also require the pipeline's own `jumping` label before firing.
   *
   * **Off by default, and it must stay off for a chest-up webcam.** That
   * label is computed from `hip_y`, and when the hips are below the bottom of
   * the frame MediaPipe estimates them rather than measuring them -- the
   * label then never fires and jumping stops working entirely, which is
   * exactly what happened. Turn it on only for a full-body camera where the
   * hips are genuinely visible and the extra corroboration is free.
   */
  jumpRequireLabel: boolean;

  // --- Detection and calibration (Brief 05 §4.3, §4.4) ------------------
  /** Sustained presence required before the startup gate advances. */
  detectionHoldMs: number;
  /** How long the player holds neutral to capture a baseline. */
  calibrationMs: number;
  /** Samples older than this are treated as a disconnect (§8). */
  staleEventMs: number;
  /** Lost detection for longer than this pauses the run (§8). */
  lostPlayerPauseMs: number;
}

/**
 * Defaults. These are the values the unit tests pin, so changing one here is
 * a deliberate act with test consequences; changing the JSON is not.
 */
export const DEFAULT_BODY_TUNING: BodyTuning = {
  // With `leanScale` at 1, the calibrated lean value *is* the sine of the
  // angle the player's shoulder line makes with horizontal, so these read
  // directly as geometry: tip your shoulders about 9 degrees to change lane,
  // and come back inside about 5 to release it.
  //
  // Deliberately lower than the 0.22 used when lean was measured from the
  // hips. A shoulder tilt is a smaller number than a whole-torso lean for the
  // same intent -- a player who bends sideways to steer tips their shoulders
  // well before their spine reaches the same angle -- so keeping the old
  // threshold would have made the new measurement feel stiffer than the one
  // it replaced.
  leanEnter: 0.16,
  leanExit: 0.09,
  leanDebounceFrames: 3,
  leanSmoothing: 0.5,
  leanScale: 1.0,
  leanRecentreRate: 0.4,
  jumpBaselineRate: 0.5,

  // Fractions of shoulder width now, not torso height. Shoulder width is
  // roughly 40cm on an adult, so 0.12 is a rise of about 5cm -- a real hop,
  // but not a demand for athleticism.
  jumpHeightFraction: 0.12,
  jumpMinVelocity: 0.3,
  jumpVelocityWindowMs: 350,
  jumpRefractoryMs: 600,
  jumpRequireLabel: false,

  detectionHoldMs: 2000,
  calibrationMs: 3000,
  staleEventMs: 500,
  lostPlayerPauseMs: 1000,
};

/** Where the runtime overlay is fetched from. */
export const BODY_TUNING_URL = "/config/bodyTuning.json";

const NUMERIC_KEYS = [
  "leanEnter",
  "leanExit",
  "leanDebounceFrames",
  "leanSmoothing",
  "leanScale",
  "leanRecentreRate",
  "jumpBaselineRate",
  "jumpHeightFraction",
  "jumpMinVelocity",
  "jumpVelocityWindowMs",
  "jumpRefractoryMs",
  "detectionHoldMs",
  "calibrationMs",
  "staleEventMs",
  "lostPlayerPauseMs",
] as const satisfies ReadonlyArray<keyof BodyTuning>;

export class BodyTuningError extends Error {}

/**
 * Merge a parsed JSON overlay onto the defaults.
 *
 * Exported separately from the fetch so it can be tested without a network,
 * and so a future in-game tuning panel can reuse exactly this validation.
 */
export function mergeBodyTuning(overlay: unknown): BodyTuning {
  if (overlay === null || overlay === undefined) {
    return { ...DEFAULT_BODY_TUNING };
  }
  if (typeof overlay !== "object" || Array.isArray(overlay)) {
    throw new BodyTuningError("body tuning must be a JSON object");
  }
  const source = overlay as Record<string, unknown>;
  const merged: BodyTuning = { ...DEFAULT_BODY_TUNING };

  for (const key of Object.keys(source)) {
    // `_`-prefixed keys are comments. JSON has none of its own, and this
    // file is meant to be hand-edited during tuning sessions.
    if (key.startsWith("_")) continue;
    if (!(key in DEFAULT_BODY_TUNING)) {
      throw new BodyTuningError(`unknown body tuning key: ${key}`);
    }
  }

  for (const key of NUMERIC_KEYS) {
    const value = source[key];
    if (value === undefined) continue;
    if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
      throw new BodyTuningError(`${key} must be a finite positive number`);
    }
    merged[key] = value;
  }

  if (source["jumpRequireLabel"] !== undefined) {
    const value = source["jumpRequireLabel"];
    if (typeof value !== "boolean") {
      throw new BodyTuningError("jumpRequireLabel must be a boolean");
    }
    merged.jumpRequireLabel = value;
  }

  // The one cross-field rule that matters: without a real gap between them,
  // hysteresis does nothing and T-03 (boundary oscillation) will fail.
  if (merged.leanExit >= merged.leanEnter) {
    throw new BodyTuningError("leanExit must be below leanEnter (hysteresis gap)");
  }
  if (merged.leanSmoothing > 1) {
    throw new BodyTuningError("leanSmoothing must be at most 1");
  }
  if (merged.leanDebounceFrames < 1) {
    throw new BodyTuningError("leanDebounceFrames must be at least 1");
  }
  return merged;
}

/**
 * Fetch the runtime overlay. Never throws: tuning must not be able to break
 * startup, so a missing or malformed file logs and falls back to defaults.
 */
export async function loadBodyTuning(
  fetchImpl: typeof fetch = fetch,
  url: string = BODY_TUNING_URL,
): Promise<BodyTuning> {
  try {
    const response = await fetchImpl(url, { cache: "no-store" });
    if (!response.ok) {
      return { ...DEFAULT_BODY_TUNING };
    }
    return mergeBodyTuning(await response.json());
  } catch (error: unknown) {
    console.warn(`body tuning: falling back to defaults (${String(error)})`);
    return { ...DEFAULT_BODY_TUNING };
  }
}
