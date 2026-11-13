/**
 * Every tunable and feature flag for Feature Brief 02 (Obito boss + Game
 * Over). Kept in its own file rather than added to gameConfig.ts so the
 * approved running game's configuration is not touched at all -- brief §0
 * requires this work to be strictly additive.
 *
 * Brief §0 rule 4: both features sit behind a flag so the original game can
 * be switched back on in isolation to confirm it is unchanged.
 */

/** Turn the Obito encounter off to verify the approved running game. */
export const ENABLE_BOSS = true;
/** Turn the Game Over screen off to verify the approved running game. */
export const ENABLE_GAME_OVER = true;

// --- Encounter scheduling (brief §5.1) --------------------------------

/**
 * Distance in metres at which each encounter triggers. Fixed, not random,
 * so pacing is predictable and tunable. Roughly 1/3 and 2/3 of an intended
 * ~900m run.
 */
export const ENCOUNTER_DISTANCES_M = [300, 600] as const;
export const TOTAL_ENCOUNTERS = ENCOUNTER_DISTANCES_M.length;

/**
 * An encounter waits for a clean moment: never while mid-jump or
 * mid-lane-change, and never within this long of an active obstacle.
 */
export const ENCOUNTER_OBSTACLE_CLEARANCE_S = 2;
/** Give up waiting for a clean moment after this long and start anyway, so
 * a pathological obstacle stream can't skip an encounter entirely. */
export const ENCOUNTER_TRIGGER_PATIENCE_S = 6;

// --- Intro / outro transitions (brief §5.2, §5.8) ----------------------

export const BOSS_INTRO_DECEL_S = 1.5;
export const BOSS_INTRO_HOLD_S = 1.2;
export const BOSS_OUTRO_ACCEL_S = 1.4;
/** Grace period after victory before obstacles may spawn again (§5.8.6). */
export const BOSS_RESUME_GRACE_S = 2;

/**
 * Where Obito stands during the duel, and where he first appears. Tuned
 * against a real screenshot: at -11 he was far enough away to read as a
 * small distant figure rather than an opponent, while still leaving both
 * health bars comfortably in frame at -8.
 */
export const OBITO_COMBAT_Z = -8;
export const OBITO_SPAWN_Z = -70;

// --- Combat camera (brief §5.6) ---------------------------------------

/** Wider/higher/more centred than the running rig so both fighters and both
 * health bars fit. Fed to the *existing* camera as different target values;
 * the running values are stored and restored exactly (§0, §5.6). */
export const COMBAT_CAMERA_POSITION = { x: 0, y: 4.2, z: 8.5 } as const;
export const COMBAT_CAMERA_LOOK_AHEAD_Z = -6;
export const COMBAT_CAMERA_LOOK_AT_HEIGHT = 1.5;

// --- Health and damage (brief §5.4, resolved per §2.1) ----------------

/**
 * Naruto keeps 100 HP and takes 60 then 20 per hit; Obito has 120 HP and
 * takes the escalating 60/40/20. Both therefore fall on the third hit.
 *
 * This is the brief's *alternative* resolution to the §2.1 contradiction
 * (60/40/20 against 100 HP would have killed Obito on the second hit),
 * chosen by the project owner to preserve the escalating-damage feel.
 */
export const NARUTO_MAX_HP = 100;
export const OBITO_MAX_HP = 120;
export const NARUTO_DAMAGE_SEQUENCE = [60, 40, 20] as const;
export const OBITO_DAMAGE_SEQUENCE = [60, 20, 20] as const;

// --- Obito behaviour (brief §5.3) -------------------------------------

/** Seconds between Obito's attacks, per encounter index. Tightens in the
 * second encounter. */
export const OBITO_ATTACK_INTERVAL_S = [2.5, 1.8] as const;
/**
 * How long the wind-up + lane marker is held before the attack fires.
 * Brief §5.3: "The telegraph is the difference between a fair fight and a
 * frustrating one" -- must be >= 600ms.
 */
export const OBITO_TELEGRAPH_S = 0.75;
/** How long Obito's projectile takes to travel to Naruto's lane. */
export const OBITO_PROJECTILE_TRAVEL_S = 0.32;
/** Obito repositions between attacks; this is the pause before he slides. */
export const OBITO_DODGE_TELEGRAPH_S = 0.22;
export const OBITO_LANE_SLIDE_S = 0.3;

// --- Naruto's seal sequence (brief §5.3) -------------------------------

export const SEAL_SEQUENCE_LENGTH = 3;
/** Whole sequence must be completed within this window. */
export const SEAL_SEQUENCE_WINDOW_S = 4;
/** Cooldown after a wrong seal or a timeout. */
export const SEAL_FAIL_COOLDOWN_S = 1.5;
/** Cooldown after a successful attack, so it can't be spammed. */
export const SEAL_SUCCESS_COOLDOWN_S = 1.2;
/** How long Naruto's projectile takes to reach Obito. */
export const NARUTO_PROJECTILE_TRAVEL_S = 0.28;

/**
 * The seal pool. Deliberately small and keyboard-mapped (1/2/3) so combat
 * is fully testable without a camera -- brief §6 requires this fallback.
 * These names match the recognition pipeline's own hand-seal vocabulary, so
 * wiring real CV later is a mapping change rather than a redesign.
 */
export const SEAL_POOL = ["tiger", "ram", "snake"] as const;
export type SealName = (typeof SEAL_POOL)[number];

/** Display glyph per seal, for the combat HUD prompts. */
export const SEAL_GLYPHS: Record<SealName, string> = {
  tiger: "寅",
  ram: "未",
  snake: "巳",
};

// --- Feel / polish -----------------------------------------------------

export const COMBAT_HIT_STOP_S = 0.08;
export const COMBAT_CAMERA_SHAKE = 0.22;
/** Pause between the killing blow and the Game Over panel (§4.2). */
export const GAME_OVER_DELAY_S = 0.7;
export const OBITO_DEFEAT_S = 1.1;
export const NARUTO_VICTORY_S = 1;
