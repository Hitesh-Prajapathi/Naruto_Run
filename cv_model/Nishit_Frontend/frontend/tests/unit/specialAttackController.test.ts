/**
 * Feature Brief 04 §3-§5: the special attack's trigger guards, window
 * timing, counter handling, and -- most importantly -- the reset contract.
 *
 * The last group is the reason this file is long. §5 names two failures as
 * "the two most dangerous leaks": a run that starts with lean input still
 * suppressed, or with the world stuck at 70% time scale. Both are silent.
 * These tests walk every exit path and assert both are released.
 */

import { describe, expect, it } from "vitest";
import { SpecialAttackController } from "../../src/game/specialAttackController";
import {
  SPECIAL_FAILURE_BEAT_S,
  SPECIAL_COMBAT_FALLBACK_S,
  SPECIAL_PRACTICE_PATIENCE_S,
  SPECIAL_SUCCESS_BEAT_S,
  SPECIAL_TIME_SCALE,
  SPECIAL_TRIGGER_DELAY_S,
  SPECIAL_TRIGGER_OBITO_HP,
  SPECIAL_VFX_TRAVEL_S,
  SPECIAL_WARNING_S,
} from "../../src/config/specialAttackConfig";

/** A context in which the attack is allowed to arm. */
function ready(overrides: Partial<Parameters<SpecialAttackController["update"]>[1]> = {}) {
  return {
    obitoHp: SPECIAL_TRIGGER_OBITO_HP,
    combatBusy: false,
    hareEverPerformed: true,
    inCombat: true,
    // 0 so these cases exercise the HP threshold specifically; the elapsed
    // fallback has its own tests below.
    combatElapsedS: 0,
    ...overrides,
  };
}

/** Advance in small steps so nothing depends on a single huge frame. */
function run(
  special: SpecialAttackController,
  seconds: number,
  context = ready(),
  step = 1 / 60,
): string[] {
  const types: string[] = [];
  for (let t = 0; t < seconds; t += step) {
    for (const event of special.update(step, context)) types.push(event.type);
  }
  return types;
}

/**
 * Fast-forward to the *instant* the warning opens. Deliberately stops on
 * the transition frame rather than overshooting, so the timing assertions
 * below measure the window itself and not the window plus slop.
 */
function openWindow(encounterIndex = 0): SpecialAttackController {
  const special = new SpecialAttackController(encounterIndex);
  for (let i = 0; i < 600 && special.state !== "warning"; i += 1) {
    special.update(1 / 60, ready());
  }
  expect(special.state).toBe("warning");
  return special;
}

describe("brief §3: trigger conditions", () => {
  it("does not fire while Obito is above the HP threshold", () => {
    const special = new SpecialAttackController(0);
    run(special, 10, ready({ obitoHp: SPECIAL_TRIGGER_OBITO_HP + 1 }));
    expect(special.state).toBe("idle");
    expect(special.hasFired).toBe(false);
  });

  it("fires once Obito drops to the threshold", () => {
    const special = openWindow();
    expect(special.hasFired).toBe(true);
  });

  it("never fires outside combat -- not during the intro or the defeat beat", () => {
    const special = new SpecialAttackController(0);
    run(special, 10, ready({ inCombat: false }));
    expect(special.state).toBe("idle");
  });

  it("never fires on top of an unresolved normal attack", () => {
    const special = new SpecialAttackController(0);
    run(special, 10, ready({ combatBusy: true }));
    expect(special.state).toBe("idle");
    // ...but it arms as soon as that attack resolves.
    run(special, SPECIAL_TRIGGER_DELAY_S + 0.2, ready());
    expect(special.state).toBe("warning");
  });

  it("never fires on the frame Obito dies", () => {
    const special = new SpecialAttackController(0);
    run(special, 10, ready({ obitoHp: 0 }));
    expect(special.state).toBe("idle");
  });

  it("fires even if the player never damages Obito at all", () => {
    // The reachability bug that made this look unimplemented in play: the HP
    // threshold alone is not a guarantee. Landing a jutsu needs a completed
    // seal sequence *and* Obito still standing in the targeted lane, while
    // his own attacks kill Naruto in three. A real end-to-end run died at
    // 12s with Obito untouched on 120 HP, having never seen the mechanic.
    const special = new SpecialAttackController(0);
    let elapsed = 0;
    while (special.state !== "warning" && elapsed < 30) {
      special.update(1 / 60, ready({ obitoHp: 120, combatElapsedS: elapsed }));
      elapsed += 1 / 60;
    }
    expect(special.state).toBe("warning");
    expect(elapsed).toBeLessThan(SPECIAL_COMBAT_FALLBACK_S + SPECIAL_TRIGGER_DELAY_S + 1);
  });

  it("still prefers the HP threshold, so the retaliation read survives", () => {
    // At full health with no elapsed time it must not fire; drop him to the
    // threshold and it does, well before the elapsed fallback would.
    const special = new SpecialAttackController(0);
    run(special, 1, ready({ obitoHp: 120, combatElapsedS: 0 }));
    expect(special.state).toBe("idle");
    run(special, SPECIAL_TRIGGER_DELAY_S + 0.3, ready({ combatElapsedS: 1 }));
    expect(special.state).toBe("warning");
  });

  it("fires at most once per encounter, whatever the outcome", () => {
    const special = openWindow();
    special.submitCounter();
    run(special, SPECIAL_SUCCESS_BEAT_S + 0.2);
    expect(special.state).toBe("spent");
    // Conditions are still met, but it must never come back.
    run(special, 20);
    expect(special.state).toBe("spent");
  });
});

describe("brief §2.2: the practice guard", () => {
  it("waits, rather than firing, when the player has never performed hare", () => {
    const special = new SpecialAttackController(0);
    const events = run(special, SPECIAL_PRACTICE_PATIENCE_S - 1, ready({ hareEverPerformed: false }));
    expect(special.state).toBe("awaiting_practice");
    expect(special.hasFired).toBe(false);
    // The prompt is asked for exactly once, not every frame.
    expect(events.filter((type) => type === "practice_required")).toHaveLength(1);
  });

  it("fires anyway once the practice patience runs out", () => {
    // The regression that made the whole feature look unimplemented: this
    // used to wait indefinitely, so a player who never pressed the key never
    // saw the special attack at all. §2.2 wants them warned first, not
    // exempted forever.
    const special = new SpecialAttackController(0);
    run(special, SPECIAL_PRACTICE_PATIENCE_S + SPECIAL_TRIGGER_DELAY_S + 0.5, ready({ hareEverPerformed: false }));
    expect(special.state).toBe("warning");
  });

  it("fires sooner when the player does practise than when they never do", () => {
    const practised = new SpecialAttackController(0);
    let practisedS = 0;
    while (practised.state !== "warning" && practisedS < 60) {
      practised.update(1 / 60, ready());
      practisedS += 1 / 60;
    }

    const ignored = new SpecialAttackController(0);
    let ignoredS = 0;
    while (ignored.state !== "warning" && ignoredS < 60) {
      ignored.update(1 / 60, ready({ hareEverPerformed: false }));
      ignoredS += 1 / 60;
    }

    expect(practisedS).toBeLessThan(ignoredS);
    expect(ignoredS - practisedS).toBeCloseTo(SPECIAL_PRACTICE_PATIENCE_S, 0);
  });

  it("proceeds as soon as the player has tried it", () => {
    const special = new SpecialAttackController(0);
    run(special, 3, ready({ hareEverPerformed: false }));
    expect(special.state).toBe("awaiting_practice");
    run(special, SPECIAL_TRIGGER_DELAY_S + 0.2, ready());
    expect(special.state).toBe("warning");
  });

  it("suspends nothing while it waits -- the fight carries on normally", () => {
    const special = new SpecialAttackController(0);
    run(special, SPECIAL_PRACTICE_PATIENCE_S - 1, ready({ hareEverPerformed: false }));
    expect(special.suppressLaneInput).toBe(false);
    expect(special.freezeCombat).toBe(false);
    expect(special.worldTimeScale).toBe(1);
  });
});

describe("brief §4.1: the warning window", () => {
  it("lasts the configured duration in real seconds", () => {
    const special = openWindow();
    run(special, SPECIAL_WARNING_S[0]! - 0.1);
    expect(special.state).toBe("warning");
    run(special, 0.2);
    expect(special.state).toBe("struck");
  });

  it("gives the first encounter a longer window than the second, to teach it", () => {
    expect(new SpecialAttackController(0).windowDurationS).toBeGreaterThan(
      new SpecialAttackController(1).windowDurationS,
    );
  });

  it("is long enough for the §2.4 reaction budget with margin to spare", () => {
    // notice 0.4 + form 0.5 + hold 0.25 + classifier 0.2 = ~1.35s.
    for (const duration of SPECIAL_WARNING_S) {
      expect(duration).toBeGreaterThan(1.35 + 0.75);
    }
  });

  it("drains its countdown monotonically from full to empty", () => {
    const special = openWindow();
    expect(special.windowProgress).toBeCloseTo(0, 1);
    let previous = -1;
    for (let t = 0; t < SPECIAL_WARNING_S[0]!; t += 0.1) {
      special.update(0.1, ready());
      if (special.state !== "warning") break;
      expect(special.windowProgress).toBeGreaterThanOrEqual(previous);
      previous = special.windowProgress;
    }
    expect(previous).toBeGreaterThan(0.9);
  });

  it("releases the projectile in time to arrive as the window closes", () => {
    const special = openWindow();
    const early = run(special, SPECIAL_WARNING_S[0]! - SPECIAL_VFX_TRAVEL_S - 0.15);
    expect(early).not.toContain("attack_released");
    const rest = run(special, SPECIAL_VFX_TRAVEL_S + 0.3);
    expect(rest.filter((type) => type === "attack_released")).toHaveLength(1);
    expect(rest).toContain("struck");
  });
});

describe("brief §2.1 / §4.2: input during the window", () => {
  it("accepts the counter at any point in the window, including immediately", () => {
    for (const delay of [0, 0.5, SPECIAL_WARNING_S[0]! - 0.2]) {
      const special = openWindow();
      run(special, delay);
      expect(special.submitCounter()).toBe(true);
      expect(special.state).toBe("countered");
    }
  });

  it("accepts a counter after the projectile is already in flight", () => {
    const special = openWindow();
    run(special, SPECIAL_WARNING_S[0]! - SPECIAL_VFX_TRAVEL_S + 0.05);
    expect(special.state).toBe("warning");
    expect(special.submitCounter()).toBe(true);
  });

  it("rejects a hold that began before the window opened", () => {
    // §2.1: "a `hare` already in progress when the window opens does not
    // carry over" -- otherwise a player mid-ikazuchi gets a free save they
    // did not earn, and one who thinks a stale input protects them is wrong.
    const special = openWindow();
    run(special, 0.2);
    expect(special.submitCounter(/* holdStartedSecondsAgo */ 1.5)).toBe(false);
    expect(special.state).toBe("warning");
    // A fresh hold within the window is accepted.
    expect(special.submitCounter(0.1)).toBe(true);
  });

  it("ignores the counter outside the window, so it can mean ikazuchi there", () => {
    const special = new SpecialAttackController(0);
    expect(special.submitCounter()).toBe(false);
    run(special, 10, ready({ obitoHp: 120 }));
    expect(special.submitCounter()).toBe(false);
  });

  it("consumes the input entirely, so nothing else can also fire", () => {
    const special = openWindow();
    expect(special.submitCounter()).toBe(true);
    // A second press during the success beat is not a second counter.
    expect(special.submitCounter()).toBe(false);
  });

  it("suppresses lane input and freezes combat only while the window is open", () => {
    const special = new SpecialAttackController(0);
    expect(special.suppressLaneInput).toBe(false);
    run(special, SPECIAL_TRIGGER_DELAY_S + 0.2);
    expect(special.suppressLaneInput).toBe(true);
    expect(special.freezeCombat).toBe(true);
    expect(special.worldTimeScale).toBeCloseTo(SPECIAL_TIME_SCALE);
  });
});

describe("brief §4.3 / §4.4: outcomes", () => {
  it("resolves a successful counter after the success beat", () => {
    const special = openWindow();
    special.submitCounter();
    const events = run(special, SPECIAL_SUCCESS_BEAT_S + 0.1);
    expect(events).toContain("resolved");
    expect(special.state).toBe("spent");
  });

  it("strikes when the window elapses, then resolves", () => {
    const special = openWindow();
    const events = run(special, SPECIAL_WARNING_S[0]! + SPECIAL_FAILURE_BEAT_S + 0.2);
    expect(events).toContain("struck");
    expect(events).toContain("resolved");
    expect(special.state).toBe("spent");
  });

  it("emits exactly one resolved event on each path", () => {
    const countered = openWindow();
    countered.submitCounter();
    const a = run(countered, 20);
    expect(a.filter((type) => type === "resolved")).toHaveLength(1);

    const struck = openWindow();
    const b = run(struck, 20);
    expect(b.filter((type) => type === "resolved")).toHaveLength(1);
    expect(b.filter((type) => type === "struck")).toHaveLength(1);
  });
});

describe("brief §5: reset and the two dangerous leaks", () => {
  // Every exit path, walked to its end, must leave the world unsuspended.
  const paths: Array<[string, () => SpecialAttackController]> = [
    ["never triggered", () => {
      const special = new SpecialAttackController(0);
      run(special, 5, ready({ obitoHp: 120 }));
      return special;
    }],
    ["waiting for practice", () => {
      const special = new SpecialAttackController(0);
      run(special, SPECIAL_PRACTICE_PATIENCE_S - 1, ready({ hareEverPerformed: false }));
      return special;
    }],
    ["countered", () => {
      const special = openWindow();
      special.submitCounter();
      run(special, 20);
      return special;
    }],
    ["struck", () => {
      const special = openWindow();
      run(special, 20);
      return special;
    }],
    ["struck, then the encounter left combat mid-beat", () => {
      const special = openWindow();
      run(special, SPECIAL_WARNING_S[0]! + 0.05);
      // The encounter transitions to `defeat` on the frame it strikes, so
      // the remaining updates arrive with inCombat false. The teardown must
      // still be reached -- this is the leak §5 warns about.
      run(special, 20, ready({ inCombat: false }));
      return special;
    }],
  ];

  for (const [name, build] of paths) {
    it(`releases lean suppression and the time scale after: ${name}`, () => {
      const special = build();
      expect(special.suppressLaneInput).toBe(false);
      expect(special.worldTimeScale).toBe(1);
      expect(special.freezeCombat).toBe(false);
    });
  }

  it("still reaches resolved when combat ends on the striking frame", () => {
    const special = openWindow();
    run(special, SPECIAL_WARNING_S[0]! + 0.05);
    expect(special.state).toBe("struck");
    const events = run(special, SPECIAL_FAILURE_BEAT_S + 0.2, ready({ inCombat: false }));
    expect(events).toContain("resolved");
  });

  it("holds no state that can survive into a new encounter", () => {
    // A fresh controller is constructed per encounter, so "reset" is really
    // "is a new instance clean?". Ten in a row, to catch shared state.
    for (let i = 0; i < 10; i += 1) {
      const special = new SpecialAttackController(i % 2);
      expect(special.state).toBe("idle");
      expect(special.hasFired).toBe(false);
      expect(special.suppressLaneInput).toBe(false);
      expect(special.worldTimeScale).toBe(1);
      expect(special.remainingS).toBe(0);
    }
  });
});
