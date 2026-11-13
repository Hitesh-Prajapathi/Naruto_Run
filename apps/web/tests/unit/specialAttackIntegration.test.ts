/**
 * Feature Brief 04, at the encounter level: how the special attack composes
 * with the approved combat loop.
 *
 * The controller's own tests cover its state machine in isolation. These
 * cover the joins -- the places a strictly-additive feature can still break
 * the system it was bolted onto: damage, the jutsu prompt, the loss cause,
 * and the guarantee that combat really is frozen rather than merely ignored.
 */

import { describe, expect, it, vi } from "vitest";
import { EncounterDirector, type EncounterHooks } from "../../src/game/encounterDirector";
import { NARUTO_MAX_HP, OBITO_MAX_HP, SEAL_POOL } from "../../src/config/bossConfig";
import * as specialConfig from "../../src/config/specialAttackConfig";
import {
  SPECIAL_FAILURE_BEAT_S,
  SPECIAL_SUCCESS_BEAT_S,
  SPECIAL_PRACTICE_PATIENCE_S,
  SPECIAL_TRIGGER_OBITO_HP,
  SPECIAL_WARNING_S,
} from "../../src/config/specialAttackConfig";

function makeHooks(): EncounterHooks & Record<string, ReturnType<typeof vi.fn>> {
  return {
    onObitoMoveLane: vi.fn(),
    onObitoAttackPose: vi.fn(),
    onObitoHit: vi.fn(),
    onObitoDefeated: vi.fn(),
    onTelegraph: vi.fn(),
    onFireVfx: vi.fn(),
    onNarutoHit: vi.fn(),
    onDamageNumber: vi.fn(),
    onHitStop: vi.fn(),
    onBanner: vi.fn(),
    onSpecialPracticeRequired: vi.fn(),
    onSpecialWarning: vi.fn(),
    onSpecialReleased: vi.fn(),
    onSpecialCountered: vi.fn(),
    onSpecialStruck: vi.fn(),
    onSpecialTeardown: vi.fn(),
  } as unknown as EncounterHooks & Record<string, ReturnType<typeof vi.fn>>;
}

function step(director: EncounterDirector, seconds: number, hare = true, dt = 1 / 60): void {
  for (let t = 0; t < seconds; t += dt) director.update(dt, 1, hare);
}

/** Reach combat, chip Obito to the trigger HP, and open the window. */
function armed(hare = true): {
  director: EncounterDirector;
  hooks: ReturnType<typeof makeHooks>;
} {
  const hooks = makeHooks();
  const director = new EncounterDirector(0, hooks);
  director.skipIntro();
  step(director, 0.1, hare);
  expect(director.currentPhase).toBe("combat");
  director.combat.applyDebugObitoDamage(OBITO_MAX_HP - SPECIAL_TRIGGER_OBITO_HP);
  for (let i = 0; i < 600 && director.special?.state !== "warning"; i += 1) {
    director.update(1 / 60, 1, hare);
  }
  return { director, hooks };
}

describe("brief §0: the additive contract", () => {
  it("exposes its own feature flag, separate from the boss flags", () => {
    expect(specialConfig).toHaveProperty("ENABLE_SPECIAL_ATTACK");
    expect(typeof specialConfig.ENABLE_SPECIAL_ATTACK).toBe("boolean");
  });

  it("keeps the counter sign out of the approved jutsu pool", () => {
    // Adding `hare` to SEAL_POOL would change the approved sequences, which
    // §0 forbids. It is its own input with its own type.
    const pool: readonly string[] = SEAL_POOL;
    expect(pool).not.toContain(specialConfig.COUNTER_SIGN);
  });

  it("makes the warning duration a tunable constant, not a literal (§4.1)", () => {
    expect(Array.isArray(SPECIAL_WARNING_S)).toBe(true);
    for (const duration of SPECIAL_WARNING_S) expect(duration).toBeGreaterThan(0);
  });
});

describe("brief §4.1: the window lasts its duration in *real* seconds", () => {
  it("ignores the host's clamped game delta and uses wall-clock time", () => {
    // SceneRoot clamps the frame delta to 1/15s so a stall cannot teleport
    // anything that integrates. Below 15fps that makes game time run slower
    // than the clock, so a countdown fed the clamped delta silently lasts
    // longer than it claims -- measured at ~5fps in a software renderer, a
    // 3.2s window took over 8 real seconds. The director therefore takes a
    // separate unclamped delta for the special attack alone.
    const CLAMP = 1 / 15;
    const REAL_FRAME = 0.2; // 5fps

    const hooks = makeHooks();
    const director = new EncounterDirector(0, hooks);
    director.skipIntro();
    director.update(CLAMP, 1, true, REAL_FRAME);
    director.combat.applyDebugObitoDamage(OBITO_MAX_HP - SPECIAL_TRIGGER_OBITO_HP);

    let realSeconds = 0;
    for (let i = 0; i < 400 && director.special?.state !== "struck"; i += 1) {
      director.update(CLAMP, 1, true, REAL_FRAME);
      realSeconds += REAL_FRAME;
    }

    // Trigger delay + window, in real seconds, within one slow frame.
    const expected = 1 + SPECIAL_WARNING_S[0]!;
    expect(realSeconds).toBeGreaterThan(expected - 0.3);
    expect(realSeconds).toBeLessThan(expected + 0.3);
  });
});

describe("brief §4.2: the window really suspends combat", () => {
  it("cancels an in-progress jutsu sequence when the window opens", () => {
    const { director } = armed();
    expect(director.seals.currentPhase).toBe("idle");
    expect(director.seals.sequence).toHaveLength(0);
  });

  it("advances no combat state at all while the window is open", () => {
    const { director } = armed();
    const before = {
      narutoHp: director.combat.naruto.hp,
      obitoHp: director.combat.obito.hp,
      obitoLane: director.combat.obitoCurrentLane,
    };
    step(director, SPECIAL_WARNING_S[0]! - 0.2);
    expect(director.combat.naruto.hp).toBe(before.narutoHp);
    expect(director.combat.obito.hp).toBe(before.obitoHp);
    expect(director.combat.obitoCurrentLane).toBe(before.obitoLane);
  });

  it("suppresses lane input for the window and restores it afterwards", () => {
    const { director } = armed();
    expect(director.suppressLaneInput).toBe(true);
    director.submitCounter();
    step(director, SPECIAL_SUCCESS_BEAT_S + 0.2);
    expect(director.suppressLaneInput).toBe(false);
    expect(director.worldTimeScale).toBe(1);
  });

  it("restores the jutsu prompt after a successful counter", () => {
    const { director } = armed();
    director.submitCounter();
    step(director, SPECIAL_SUCCESS_BEAT_S + 0.2);
    expect(director.seals.currentPhase).toBe("active");
    expect(director.seals.sequence.length).toBeGreaterThan(0);
  });
});

describe("brief §2.6 / §4.3: a successful counter", () => {
  it("damages nobody -- not Naruto, not Obito", () => {
    const { director } = armed();
    const obitoBefore = director.combat.obito.hp;
    director.submitCounter();
    step(director, SPECIAL_SUCCESS_BEAT_S + 0.5);
    expect(director.combat.naruto.hp).toBe(NARUTO_MAX_HP);
    expect(director.combat.obito.hp).toBe(obitoBefore);
  });

  it("does not end the encounter", () => {
    const { director } = armed();
    director.submitCounter();
    step(director, SPECIAL_SUCCESS_BEAT_S + 0.5);
    expect(director.currentPhase).toBe("combat");
    expect(director.result).toBeNull();
  });
});

describe("brief §2.5 / §4.4: a missed counter", () => {
  it("kills from full health, bypassing the HP system", () => {
    const { director } = armed();
    expect(director.combat.naruto.hp).toBe(NARUTO_MAX_HP);
    step(director, SPECIAL_WARNING_S[0]! + 0.2);
    expect(director.combat.naruto.hp).toBe(0);
    expect(director.result).toBe("lost");
  });

  it("is attributed to the special attack, not to an ordinary defeat", () => {
    const { director } = armed();
    step(director, SPECIAL_WARNING_S[0]! + 0.2);
    expect(director.lostToSpecialAttack).toBe(true);
  });

  it("leaves an ordinary defeat attributed normally", () => {
    const hooks = makeHooks();
    const director = new EncounterDirector(0, hooks);
    expect(director.lostToSpecialAttack).toBe(false);
  });
});

describe("brief §5: one teardown, on every exit path", () => {
  it("tears down exactly once after a counter", () => {
    const { director, hooks } = armed();
    director.submitCounter();
    step(director, SPECIAL_SUCCESS_BEAT_S + 2);
    expect(hooks["onSpecialTeardown"]).toHaveBeenCalledTimes(1);
  });

  it("tears down exactly once after being struck -- even though the encounter left combat on the same frame", () => {
    // This is the regression that matters: `struck` transitions the
    // encounter to `defeat`, so a teardown driven only from the `combat`
    // branch would never run and the overlay would survive into Game Over.
    const { director, hooks } = armed();
    step(director, SPECIAL_WARNING_S[0]! + SPECIAL_FAILURE_BEAT_S + 1);
    expect(director.currentPhase).not.toBe("combat");
    expect(hooks["onSpecialTeardown"]).toHaveBeenCalledTimes(1);
  });

  it("asks for the practice prompt once, then fires anyway when it is ignored", () => {
    const hooks = makeHooks();
    const director = new EncounterDirector(0, hooks);
    director.skipIntro();
    step(director, 0.1, false);
    director.combat.applyDebugObitoDamage(OBITO_MAX_HP - SPECIAL_TRIGGER_OBITO_HP);

    // Inside the patience window: prompted, held back, nobody hurt.
    step(director, SPECIAL_PRACTICE_PATIENCE_S - 1.5, false);
    expect(director.special?.state).toBe("awaiting_practice");
    expect(hooks["onSpecialPracticeRequired"]).toHaveBeenCalledTimes(1);
    expect(hooks["onSpecialWarning"]).not.toHaveBeenCalled();

    // Past it: the attack comes regardless. Waiting forever here is what
    // made the feature look unimplemented in real play.
    step(director, 5, false);
    expect(hooks["onSpecialWarning"]).toHaveBeenCalledTimes(1);
  });
});
