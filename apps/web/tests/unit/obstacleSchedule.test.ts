import { describe, expect, it } from "vitest";
import { ObstacleScheduler } from "../../src/game/obstacleSchedule";
import type { LevelSegment } from "../../src/config/levelDefinition";

const SEGMENT: LevelSegment = {
  id: "test",
  durationSeconds: 10,
  obstacles: [
    { atSeconds: 1, typeId: "boulder", lanes: [0] },
    { atSeconds: 3, typeId: "log", lanes: [1] },
    { atSeconds: 3, typeId: "pit", lanes: [2] }, // same instant as the previous one
    { atSeconds: 8, typeId: "thornWall", lanes: [1, 2] },
  ],
};

describe("ObstacleScheduler", () => {
  it("returns nothing before the first obstacle is due", () => {
    const scheduler = new ObstacleScheduler(SEGMENT);
    expect(scheduler.update(0.5)).toEqual([]);
  });

  it("returns an obstacle exactly when its time is reached", () => {
    const scheduler = new ObstacleScheduler(SEGMENT);
    scheduler.update(0.9);
    const due = scheduler.update(0.1); // elapsed now 1.0
    expect(due).toHaveLength(1);
    expect(due[0]?.entry.typeId).toBe("boulder");
  });

  it("returns every entry due within one large tick, in level order", () => {
    const scheduler = new ObstacleScheduler(SEGMENT);
    const due = scheduler.update(3.5); // covers t=1 and both t=3 entries
    expect(due.map((d) => d.entry.typeId)).toEqual(["boulder", "log", "pit"]);
  });

  it("assigns each due entry a stable, distinct id", () => {
    const scheduler = new ObstacleScheduler(SEGMENT);
    const due = scheduler.update(3.5);
    const ids = due.map((d) => d.id);
    expect(new Set(ids).size).toBe(ids.length);
  });

  it("never returns the same entry twice", () => {
    const scheduler = new ObstacleScheduler(SEGMENT);
    const firstBatch = scheduler.update(1.5);
    const secondBatch = scheduler.update(2.0); // now elapsed 3.5, crosses the t=3 entries
    expect(firstBatch.map((d) => d.entry.typeId)).toEqual(["boulder"]);
    expect(secondBatch.map((d) => d.entry.typeId)).toEqual(["log", "pit"]);
  });

  it("reports isFinished once every entry has been returned", () => {
    const scheduler = new ObstacleScheduler(SEGMENT);
    expect(scheduler.isFinished).toBe(false);
    scheduler.update(100);
    expect(scheduler.isFinished).toBe(true);
  });

  it("reset() restarts the timeline and re-emits from the beginning", () => {
    const scheduler = new ObstacleScheduler(SEGMENT);
    scheduler.update(100);
    expect(scheduler.isFinished).toBe(true);

    scheduler.reset();

    expect(scheduler.isFinished).toBe(false);
    expect(scheduler.elapsedSeconds).toBe(0);
    const due = scheduler.update(1.5);
    expect(due.map((d) => d.entry.typeId)).toEqual(["boulder"]);
  });

  it("handles an empty segment without ever having anything due", () => {
    const scheduler = new ObstacleScheduler({ id: "empty", durationSeconds: 5, obstacles: [] });
    expect(scheduler.update(100)).toEqual([]);
    expect(scheduler.isFinished).toBe(true);
  });
});
