import { describe, expect, it } from "vitest";
import { EncounterScheduler, isCleanMoment } from "../../src/game/encounterScheduler";
import {
  ENCOUNTER_DISTANCES_M,
  ENCOUNTER_OBSTACLE_CLEARANCE_S,
  ENCOUNTER_TRIGGER_PATIENCE_S,
} from "../../src/config/bossConfig";

const CLEAR = {
  isJumping: false,
  isChangingLane: false,
  secondsToNearestObstacle: null,
  requiredClearanceS: ENCOUNTER_OBSTACLE_CLEARANCE_S,
};

describe("isCleanMoment (brief §5.1)", () => {
  it("is clean on an empty path while grounded and settled", () => {
    expect(isCleanMoment(CLEAR)).toBe(true);
  });

  it("is never clean mid-jump", () => {
    expect(isCleanMoment({ ...CLEAR, isJumping: true })).toBe(false);
  });

  it("is never clean mid-lane-change", () => {
    expect(isCleanMoment({ ...CLEAR, isChangingLane: true })).toBe(false);
  });

  it("is not clean while an obstacle is inside the clearance window", () => {
    expect(isCleanMoment({ ...CLEAR, secondsToNearestObstacle: 0.5 })).toBe(false);
  });

  it("is clean once the nearest obstacle is beyond the clearance window", () => {
    expect(
      isCleanMoment({ ...CLEAR, secondsToNearestObstacle: ENCOUNTER_OBSTACLE_CLEARANCE_S + 0.1 }),
    ).toBe(true);
  });
});

describe("EncounterScheduler triggering", () => {
  it("does not fire before the first milestone", () => {
    const scheduler = new EncounterScheduler([100, 200]);
    expect(scheduler.update(1 / 60, 99, CLEAR)).toBe(false);
    expect(scheduler.encountersTriggered).toBe(0);
  });

  it("fires once the milestone is reached on a clean path", () => {
    const scheduler = new EncounterScheduler([100, 200]);
    expect(scheduler.update(1 / 60, 100, CLEAR)).toBe(true);
    expect(scheduler.encountersTriggered).toBe(1);
  });

  it("fires exactly once per milestone, not repeatedly", () => {
    const scheduler = new EncounterScheduler([100, 200]);
    expect(scheduler.update(1 / 60, 120, CLEAR)).toBe(true);
    // Still past the first milestone, but the second is far away.
    for (let i = 0; i < 100; i += 1) {
      expect(scheduler.update(1 / 60, 150, CLEAR)).toBe(false);
    }
    expect(scheduler.encountersTriggered).toBe(1);
  });

  it("fires the second encounter at its own milestone", () => {
    const scheduler = new EncounterScheduler([100, 200]);
    scheduler.update(1 / 60, 100, CLEAR);
    expect(scheduler.update(1 / 60, 200, CLEAR)).toBe(true);
    expect(scheduler.encountersTriggered).toBe(2);
    expect(scheduler.allEncountersDone).toBe(true);
  });

  it("never fires more than the configured number of encounters", () => {
    // Brief §5.1: "Obito appears exactly twice per run."
    const scheduler = new EncounterScheduler([100, 200]);
    let fired = 0;
    for (let i = 0; i < 5000; i += 1) {
      if (scheduler.update(1 / 60, 10_000, CLEAR)) fired += 1;
    }
    expect(fired).toBe(2);
  });

  it("the shipped config schedules exactly two encounters", () => {
    expect(ENCOUNTER_DISTANCES_M).toHaveLength(2);
  });
});

describe("EncounterScheduler waiting for a clean moment", () => {
  it("arms at the milestone but waits while an obstacle is close", () => {
    const scheduler = new EncounterScheduler([100]);
    const blocked = { ...CLEAR, secondsToNearestObstacle: 0.3 };

    expect(scheduler.update(1 / 60, 100, blocked)).toBe(false);
    expect(scheduler.isArmed).toBe(true);
    expect(scheduler.encountersTriggered).toBe(0);
  });

  it("fires as soon as the path clears", () => {
    const scheduler = new EncounterScheduler([100]);
    scheduler.update(1 / 60, 100, { ...CLEAR, secondsToNearestObstacle: 0.3 });
    expect(scheduler.update(1 / 60, 101, CLEAR)).toBe(true);
  });

  it("gives up waiting after the patience window so an encounter is never skipped", () => {
    const scheduler = new EncounterScheduler([100]);
    const blocked = { ...CLEAR, secondsToNearestObstacle: 0.1 };
    let fired = false;
    // A pathological, permanently-blocked obstacle stream.
    for (let t = 0; t < ENCOUNTER_TRIGGER_PATIENCE_S + 1; t += 1 / 60) {
      if (scheduler.update(1 / 60, 100, blocked)) fired = true;
    }
    expect(fired).toBe(true);
  });

  it("still refuses to fire mid-jump even once patience is exhausted", () => {
    // Firing here would snap Naruto to the centre lane from an airborne
    // pose, which is exactly the "not sliding, not T-posing" failure the
    // brief's acceptance criteria call out.
    const scheduler = new EncounterScheduler([100]);
    const airborne = { ...CLEAR, isJumping: true };
    let fired = false;
    for (let t = 0; t < ENCOUNTER_TRIGGER_PATIENCE_S + 2; t += 1 / 60) {
      if (scheduler.update(1 / 60, 100, airborne)) fired = true;
    }
    expect(fired).toBe(false);
  });
});

describe("EncounterScheduler reset (brief §4.5)", () => {
  it("re-arms both encounters for a fresh run", () => {
    const scheduler = new EncounterScheduler([100, 200]);
    scheduler.update(1 / 60, 100, CLEAR);
    scheduler.update(1 / 60, 200, CLEAR);
    expect(scheduler.allEncountersDone).toBe(true);

    scheduler.reset();

    expect(scheduler.encountersTriggered).toBe(0);
    expect(scheduler.allEncountersDone).toBe(false);
    expect(scheduler.isArmed).toBe(false);
    expect(scheduler.update(1 / 60, 100, CLEAR)).toBe(true);
  });
});
