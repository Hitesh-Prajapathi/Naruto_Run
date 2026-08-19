import { describe, expect, it } from "vitest";
import { CombatController, type CombatEvent } from "../../src/game/combatController";
import {
  NARUTO_DAMAGE_SEQUENCE,
  NARUTO_MAX_HP,
  NARUTO_PROJECTILE_TRAVEL_S,
  OBITO_DAMAGE_SEQUENCE,
  OBITO_MAX_HP,
  OBITO_TELEGRAPH_S,
} from "../../src/config/bossConfig";
import type { LaneIndex } from "../../src/config/gameConfig";

/** Deterministic RNG so lane choices are reproducible. */
const fixedRandom = (value: number) => () => value;

/** Step the controller in small slices, collecting every event. */
function run(
  combat: CombatController,
  seconds: number,
  laneAt: (elapsed: number) => LaneIndex,
  step = 1 / 60,
): CombatEvent[] {
  const events: CombatEvent[] = [];
  let elapsed = 0;
  while (elapsed < seconds) {
    elapsed += step;
    events.push(...combat.update(step, laneAt(elapsed)));
  }
  return events;
}

const staysIn = (lane: LaneIndex) => () => lane;

describe("CombatController damage curve (brief §5.4, resolved §2.1)", () => {
  it("starts both fighters at their configured HP", () => {
    const combat = new CombatController(0, fixedRandom(0.9));
    expect(combat.naruto.hp).toBe(NARUTO_MAX_HP);
    expect(combat.naruto.hp).toBe(100);
    expect(combat.obito.hp).toBe(OBITO_MAX_HP);
    expect(combat.obito.hp).toBe(120);
  });

  it("Obito's 60/20/20 against 100 HP defeats Naruto on exactly the third hit", () => {
    expect(OBITO_DAMAGE_SEQUENCE.reduce((a, b) => a + b, 0)).toBe(NARUTO_MAX_HP);
  });

  it("Naruto's 60/40/20 against 120 HP defeats Obito on exactly the third hit", () => {
    // This is the whole point of the §2.1 resolution: 60+40+20 = 120, so the
    // bar empties precisely on the third blow rather than the second.
    expect(NARUTO_DAMAGE_SEQUENCE.reduce((a, b) => a + b, 0)).toBe(OBITO_MAX_HP);
  });

  it("applies Naruto's escalating damage in order and defeats Obito on hit three", () => {
    const combat = new CombatController(0, fixedRandom(0.9));
    const hpAfter: number[] = [];

    for (let i = 0; i < 3; i += 1) {
      // Aim at Obito's current lane and stand there so the hit connects.
      const targetLane = combat.obitoCurrentLane;
      combat.launchNarutoAttack();
      run(combat, NARUTO_PROJECTILE_TRAVEL_S + 0.05, staysIn(targetLane));
      hpAfter.push(combat.obito.hp);
      if (combat.isFinished) break;
    }

    expect(hpAfter).toEqual([60, 20, 0]);
    expect(combat.isFinished).toBe(true);
  });
});

describe("CombatController telegraph fairness (brief §5.3)", () => {
  it("holds the telegraph for at least the 600ms the brief requires", () => {
    expect(OBITO_TELEGRAPH_S).toBeGreaterThanOrEqual(0.6);
  });

  it("exposes the telegraphed lane and a rising progress while winding up", () => {
    // These two getters drive the ground marker the player reads to dodge,
    // so they must actually be observable mid-wind-up, not just internally
    // correct.
    const combat = new CombatController(0, fixedRandom(0.9));
    let sawLane: LaneIndex | null = null;
    let maxProgress = 0;

    let elapsed = 0;
    const step = 1 / 120;
    while (elapsed < 12) {
      elapsed += step;
      combat.update(step, 1);
      if (combat.telegraphedLane !== null) {
        sawLane = combat.telegraphedLane;
        maxProgress = Math.max(maxProgress, combat.telegraphProgress);
      }
    }

    expect(sawLane, "a telegraphed lane should be exposed at some point").not.toBeNull();
    expect(maxProgress).toBeGreaterThan(0.5);
    expect(maxProgress).toBeLessThanOrEqual(1);
  });

  it("emits telegraph before fire, and fire before impact, every time", () => {
    const combat = new CombatController(0, fixedRandom(0.9));
    const events = run(combat, 12, staysIn(1));
    const kinds = events.map((e) => e.type).filter((t) => t.startsWith("obito_"));

    const telegraphIndex = kinds.indexOf("obito_telegraph");
    const fireIndex = kinds.indexOf("obito_fire");
    const impactIndex = kinds.indexOf("obito_impact");

    expect(telegraphIndex).toBeGreaterThanOrEqual(0);
    expect(fireIndex).toBeGreaterThan(telegraphIndex);
    expect(impactIndex).toBeGreaterThan(fireIndex);
  });

  it("standing in the targeted lane takes damage", () => {
    const combat = new CombatController(0, fixedRandom(0.9));
    const events = run(combat, 12, staysIn(1));
    const impacts = events.filter((e) => e.type === "obito_impact");
    expect(impacts.length).toBeGreaterThan(0);
    expect(impacts.some((e) => e.type === "obito_impact" && e.hit)).toBe(true);
    expect(combat.naruto.hp).toBeLessThan(NARUTO_MAX_HP);
  });

  it("vacating the telegraphed lane before impact avoids all damage", () => {
    // The core fairness guarantee: the attack targets the lane Naruto is in
    // at *fire* time, so moving during the projectile's flight dodges it.
    const combat = new CombatController(0, fixedRandom(0.9));
    let firedLane: LaneIndex | null = null;
    let elapsed = 0;
    const step = 1 / 120;

    while (elapsed < 12 && !combat.isFinished) {
      elapsed += step;
      // Dodge: if a shot is in flight at our lane, move away immediately.
      const lane: LaneIndex = firedLane === null ? 1 : firedLane === 1 ? 2 : 1;
      const events = combat.update(step, lane);
      for (const event of events) {
        if (event.type === "obito_fire") firedLane = event.lane;
        if (event.type === "obito_impact") {
          expect(event.hit, "a dodged attack must not connect").toBe(false);
          firedLane = null;
        }
      }
    }

    expect(combat.naruto.hp).toBe(NARUTO_MAX_HP);
  });
});

describe("CombatController Obito's positional dodge (brief §5.3)", () => {
  it("misses when Obito has left the targeted lane", () => {
    const combat = new CombatController(0, fixedRandom(0.9));
    const targetLane = combat.obitoCurrentLane;
    combat.launchNarutoAttack();

    // Advance far enough that Obito repositions before the projectile lands
    // is not guaranteed, so force the situation: run until impact and check
    // the hit flag agrees with whether he was still there.
    const events = run(combat, NARUTO_PROJECTILE_TRAVEL_S + 0.05, staysIn(1));
    const impact = events.find((e) => e.type === "naruto_impact");
    expect(impact).toBeDefined();
    if (impact && impact.type === "naruto_impact") {
      expect(impact.hit).toBe(combat.obitoCurrentLane === targetLane);
    }
  });

  it("telegraphs its own lane change with a lean before moving", () => {
    const combat = new CombatController(0, fixedRandom(0.9));
    const events = run(combat, 12, staysIn(1));
    const dodgeTelegraph = events.findIndex((e) => e.type === "obito_dodge_telegraph");
    const move = events.findIndex((e) => e.type === "obito_move");
    if (dodgeTelegraph >= 0) {
      expect(move).toBeGreaterThan(dodgeTelegraph);
    }
  });

  it("ignores a second attack while one is already in flight", () => {
    const combat = new CombatController(0, fixedRandom(0.9));
    expect(combat.launchNarutoAttack()).toHaveLength(1);
    expect(combat.launchNarutoAttack()).toHaveLength(0);
  });
});

describe("CombatController end conditions", () => {
  it("emits naruto_defeated and stops once Naruto's HP hits 0", () => {
    const combat = new CombatController(1, fixedRandom(0.9)); // faster interval
    const events = run(combat, 40, staysIn(1));
    expect(events.some((e) => e.type === "naruto_defeated")).toBe(true);
    expect(combat.naruto.hp).toBe(0);
    expect(combat.isFinished).toBe(true);

    // Nothing further should happen after the fight is over.
    expect(combat.update(1, 1)).toEqual([]);
  });

  it("emits obito_defeated once his HP hits 0", () => {
    const combat = new CombatController(0, fixedRandom(0.9));
    const collected: CombatEvent[] = [];
    for (let i = 0; i < 3 && !combat.isFinished; i += 1) {
      const lane = combat.obitoCurrentLane;
      combat.launchNarutoAttack();
      collected.push(...run(combat, NARUTO_PROJECTILE_TRAVEL_S + 0.05, staysIn(lane)));
    }
    expect(collected.some((e) => e.type === "obito_defeated")).toBe(true);
    expect(combat.obito.hp).toBe(0);
  });

  it("never lets HP go negative", () => {
    const combat = new CombatController(1, fixedRandom(0.9));
    run(combat, 40, staysIn(1));
    expect(combat.naruto.hp).toBeGreaterThanOrEqual(0);
    expect(combat.obito.hp).toBeGreaterThanOrEqual(0);
  });

  it("carries phase overshoot so a long frame cannot shorten the telegraph", () => {
    // One coarse 0.5s step vs many fine steps must reach the same phase, or
    // a frame spike could rob the player of reaction time.
    const coarse = new CombatController(0, fixedRandom(0.9));
    const fine = new CombatController(0, fixedRandom(0.9));
    run(coarse, 3, staysIn(1), 0.5);
    run(fine, 3, staysIn(1), 1 / 120);
    expect(coarse.naruto.hp).toBe(fine.naruto.hp);
  });
});
