/**
 * Obito's once-per-encounter special attack and the Hare counter --
 * Feature Brief 04.
 *
 * Its own file, alongside bossConfig.ts rather than inside it, for the same
 * reason bossConfig.ts is separate from gameConfig.ts: brief §0 requires this
 * work to be strictly additive, so the approved boss encounter's tunables are
 * not touched at all.
 *
 * Everything here is a named constant. Brief §2.4 is explicit that the window
 * duration in particular must be tunable rather than a literal, because it
 * has to be re-tested against a first-time player.
 */

/** Master switch. Off => encounters behave exactly as the approved build. */
export const ENABLE_SPECIAL_ATTACK = true;

// --- The counter input (brief §2.1, §4.6) ------------------------------

/**
 * The one sign that saves Naruto. Deliberately its own type rather than a
 * member of `SEAL_POOL`: adding it to the pool would change the approved
 * jutsu sequences, which §0 forbids. The name matches the recognition
 * pipeline's vocabulary (`cv_model/inference/hand_config.py`) so wiring CV
 * later is a mapping, not a redesign.
 */
export const COUNTER_SIGN = "hare" as const;
export type CounterSign = typeof COUNTER_SIGN;

/**
 * Keyboard fallback. Brief §4.6 suggests `H` for hare and requires the key be
 * shown on screen during the window, not left as a developer secret.
 *
 * `H` already toggled the dev panel. Rather than take that away, the dev
 * panel gained `` ` `` as an always-available alias and `H` is only consumed
 * here while an encounter is on screen -- see sceneMain.ts.
 */
export const COUNTER_KEY_CODE = "KeyH";
export const COUNTER_KEY_LABEL = "H";

/**
 * Zodiac glyph for hare, matching the SEAL_GLYPHS style in bossConfig.
 *
 * Deliberately *not* used by the warning UI. On the canvas-drawn marker the
 * CJK fallback is whatever the OS supplies, and it rendered as a broken
 * glyph next to the word HARE -- flavour is not worth risking legibility on
 * the one prompt the player cannot afford to misread. Kept here so a future
 * seal catalog (Brief 03) has it in the same place as the others.
 */
export const HARE_GLYPH = "卯";

/**
 * Optional drop-in reference photo for the sign (brief §4.1 wants the `hare`
 * reference shown large). No such asset exists in the repo yet -- the CV
 * training set is not checked in -- so the overlay draws a schematic and
 * swaps in this image automatically if someone adds it.
 */
export const HARE_REFERENCE_IMAGE_URL = "/textures/seals/hare.png";

// --- Trigger (brief §3) ------------------------------------------------

/**
 * Fires when Obito drops to this HP or below, i.e. after the player's first
 * successful jutsu (he starts at 120 and the first hit deals 60). Reads as
 * retaliation, guarantees the attack always happens, and guarantees the
 * player is already engaged rather than being ambushed on arrival.
 */
export const SPECIAL_TRIGGER_OBITO_HP = 60;

/**
 * Fallback trigger: fire this many seconds into combat even if Obito is
 * still at full health.
 *
 * §3 claims the HP threshold "guarantees the attack always happens". It does
 * not. Landing a jutsu means completing a three-seal sequence *and* Obito
 * still being in the lane it was aimed at -- he deliberately moves out of it
 * most of the time -- while his own attacks kill Naruto in three hits, at
 * roughly 4.1s, 8.2s and 12.3s. A real playthrough that never lands a hit
 * therefore never sees the special attack at all. Observed: an end-to-end
 * run of the real fight died at 12s with Obito still on 120 HP.
 *
 * Five seconds is after his first attack has resolved but comfortably before
 * his third lands, so the mechanic is genuinely guaranteed once per
 * encounter. The HP threshold usually fires first and keeps the intended
 * "retaliation" read; this is only the floor.
 */
export const SPECIAL_COMBAT_FALLBACK_S = 5;

/** Beat between the qualifying hit landing and the warning opening, so the
 * two events don't collide on one frame. */
export const SPECIAL_TRIGGER_DELAY_S = 1;

/**
 * How long to hold the attack back while asking the player to try `hare`
 * (§2.2), before firing anyway.
 *
 * This **must** be bounded. The first implementation waited indefinitely,
 * which quietly turned §2.2's safety guard into an off switch: a player who
 * never pressed the key simply never saw the mechanic, and the feature
 * looked unimplemented. §2.2 asks to "prompt them to try it first, *or*
 * delay the special attack" -- the intent is that nobody meets it cold, not
 * that it can be skipped forever. So the prompt goes up, and after this long
 * the attack comes regardless; by then the player has had the practice
 * prompt on screen the whole time, and the warning window itself shows the
 * sign and the key at full size for another 3.2 seconds.
 *
 * Short, because it is a grace period *after* the trigger, not the whole
 * teaching time: the practice banner goes up the moment combat starts. Any
 * longer and Obito's ordinary attacks kill Naruto before the special can
 * fire, which is the very outcome this feature must not have.
 */
export const SPECIAL_PRACTICE_PATIENCE_S = 3;

/** Brief §3: never within this long of Obito's death. Combined with the
 * "at most once per encounter" flag this makes a death-frame trigger
 * impossible. */
export const SPECIAL_MIN_OBITO_HP_MARGIN = 1;

// --- The warning window (brief §2.4, §4.1) -----------------------------

/**
 * Window length per encounter index, in **real** seconds. The first is longer
 * so the mechanic teaches itself (brief §2.4's last mitigation); the second
 * is the brief's nominal 2.5s.
 *
 * Budget from brief §2.4: notice ~0.4s + form ~0.5s + hold ~0.25s +
 * classifier ~0.2s = ~1.35s, so 2.5s leaves roughly a second of slack and
 * 3.2s leaves nearly two on the first, teaching attempt.
 */
export const SPECIAL_WARNING_S = [3.2, 2.5] as const;

/**
 * Shorter sustained hold for the counter than the general jutsu recogniser
 * uses (brief §2.4): the window already tells the game a `hare` is expected,
 * so the prior is strong enough to justify a looser threshold. Consumed by
 * the CV input layer when it is wired; the keyboard path is instantaneous.
 */
export const COUNTER_HOLD_S = 0.25;

/** World time scale during the window (brief §4.1's optional slowdown).
 * The countdown itself always runs on unscaled real seconds. */
export const SPECIAL_TIME_SCALE = 0.7;

// --- Resolution beats (brief §4.3, §4.4) -------------------------------

/** Slow-motion success beat before combat resumes. */
export const SPECIAL_SUCCESS_BEAT_S = 1.1;
/** Impact pause before the defeat animation takes over. */
export const SPECIAL_FAILURE_BEAT_S = 0.55;

export const SPECIAL_HIT_STOP_S = 0.24;
export const SPECIAL_CAMERA_SHAKE = 0.6;
export const SPECIAL_COUNTER_SHAKE = 0.18;

// --- VFX (brief §4.5) --------------------------------------------------

/** Multiplier on the ordinary projectile size. "Scaled up substantially." */
export const SPECIAL_VFX_SCALE = 3.6;
/** Slower than a normal projectile so the impact is readable. */
export const SPECIAL_VFX_TRAVEL_S = 0.55;

// --- Palette (brief §4.1: categorically not the HUD orange) ------------

/** Obito's dark red/purple. Shared by the world-space warning, the overlay
 * card and the screen vignette so they read as one alarm. */
export const DANGER_COLOR = "#ff1f4b";
export const DANGER_COLOR_DEEP = "#4a0d24";
export const DANGER_COLOR_MID = "#8f1236";
