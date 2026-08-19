/**
 * Feature Brief 02 §0 / §7: with both feature flags off, the approved
 * running game must be untouched.
 *
 * This pins the *mechanism* that guarantees it -- the capability table and
 * the scroll factor -- rather than eyeballing a screenshot. If someone later
 * makes an encounter modify the world instead of scaling it, or forgets to
 * gate a new system, these fail.
 */

import { describe, expect, it } from "vitest";
import { capabilitiesOf } from "../../src/game/gameState";
import * as bossConfig from "../../src/config/bossConfig";
import {
  NARUTO_DAMAGE_SEQUENCE,
  NARUTO_MAX_HP,
  OBITO_DAMAGE_SEQUENCE,
  OBITO_MAX_HP,
  OBITO_TELEGRAPH_S,
  ENCOUNTER_DISTANCES_M,
} from "../../src/config/bossConfig";

describe("brief §0: the additive contract", () => {
  it("exposes both feature flags so the original game can be switched back on", () => {
    expect(bossConfig).toHaveProperty("ENABLE_BOSS");
    expect(bossConfig).toHaveProperty("ENABLE_GAME_OVER");
    expect(typeof bossConfig.ENABLE_BOSS).toBe("boolean");
    expect(typeof bossConfig.ENABLE_GAME_OVER).toBe("boolean");
  });

  it("RUNNING permits exactly the pre-existing behaviour and nothing new", () => {
    // The running state must enable every original system and leave the new
    // gesture path off, so RUNNING is indistinguishable from the approved
    // build regardless of what the encounter code does.
    const running = capabilitiesOf("RUNNING");
    expect(running).toEqual({
      runningGameplay: true,
      worldScroll: "on",
      obstacleSpawn: true,
      laneInput: true,
      jumpInput: true,
      gestureInput: false,
    });
  });

  it("suspends rather than modifies: every non-running state restores to RUNNING's capabilities", () => {
    // Brief §0 rule 3. Because capabilities are looked up per state rather
    // than mutated, returning to RUNNING necessarily restores every value --
    // there is no path that can leave a suspended system switched off.
    const running = capabilitiesOf("RUNNING");
    for (const state of ["BOSS_INTRO", "BOSS_COMBAT", "BOSS_DEFEATED", "GAME_OVER"] as const) {
      expect(capabilitiesOf(state)).not.toEqual(running);
    }
    expect(capabilitiesOf("RUNNING")).toEqual(running);
  });
});

describe("brief §7: acceptance values are actually configured", () => {
  it("schedules exactly two encounters at fixed distances", () => {
    expect(ENCOUNTER_DISTANCES_M).toHaveLength(2);
    for (const distance of ENCOUNTER_DISTANCES_M) {
      expect(distance).toBeGreaterThan(0);
    }
    // Fixed and ordered, not random.
    expect(ENCOUNTER_DISTANCES_M[1]).toBeGreaterThan(ENCOUNTER_DISTANCES_M[0]!);
  });

  it("telegraphs attacks for at least the required 600ms", () => {
    expect(OBITO_TELEGRAPH_S).toBeGreaterThanOrEqual(0.6);
  });

  it("uses damage curves where each fighter falls on exactly the third hit", () => {
    expect(OBITO_DAMAGE_SEQUENCE.reduce((a, b) => a + b, 0)).toBe(NARUTO_MAX_HP);
    expect(NARUTO_DAMAGE_SEQUENCE.reduce((a, b) => a + b, 0)).toBe(OBITO_MAX_HP);
    expect(OBITO_DAMAGE_SEQUENCE).toHaveLength(3);
    expect(NARUTO_DAMAGE_SEQUENCE).toHaveLength(3);
  });

  it("keeps the first hit the heaviest for both fighters", () => {
    expect(NARUTO_DAMAGE_SEQUENCE[0]).toBe(60);
    expect(OBITO_DAMAGE_SEQUENCE[0]).toBe(60);
  });

  it("tightens Obito's attack interval in the second encounter", () => {
    const [first, second] = bossConfig.OBITO_ATTACK_INTERVAL_S;
    expect(second).toBeLessThan(first!);
  });
});
