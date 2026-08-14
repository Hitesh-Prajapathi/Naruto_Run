import { describe, expect, it } from "vitest";
import { HIT_STOP_S, HitFeedback, INVULNERABLE_S, MAX_LIVES } from "../../src/game/hitFeedback";

describe("HitFeedback lives", () => {
  it("starts at full lives and undefeated", () => {
    const hits = new HitFeedback();
    expect(hits.remainingLives).toBe(MAX_LIVES);
    expect(hits.isDefeated).toBe(false);
  });

  it("a landed hit costs exactly one life", () => {
    const hits = new HitFeedback();
    expect(hits.registerHit()).toBe(true);
    expect(hits.remainingLives).toBe(MAX_LIVES - 1);
  });

  it("becomes defeated once lives run out", () => {
    const hits = new HitFeedback();
    for (let i = 0; i < MAX_LIVES; i += 1) {
      hits.registerHit();
      hits.update(INVULNERABLE_S + 0.01); // clear i-frames between hits
    }
    expect(hits.remainingLives).toBe(0);
    expect(hits.isDefeated).toBe(true);
  });

  it("ignores further hits once defeated", () => {
    const hits = new HitFeedback();
    for (let i = 0; i < MAX_LIVES; i += 1) {
      hits.registerHit();
      hits.update(INVULNERABLE_S + 0.01);
    }
    expect(hits.registerHit()).toBe(false);
    expect(hits.remainingLives).toBe(0);
  });
});

describe("HitFeedback invulnerability", () => {
  it("absorbs a second hit inside the grace window", () => {
    const hits = new HitFeedback();
    hits.registerHit();
    hits.update(INVULNERABLE_S / 2);

    expect(hits.isInvulnerable).toBe(true);
    expect(hits.registerHit()).toBe(false);
    expect(hits.remainingLives).toBe(MAX_LIVES - 1); // only the first one counted
  });

  it("accepts a hit again once the grace window expires", () => {
    const hits = new HitFeedback();
    hits.registerHit();
    hits.update(INVULNERABLE_S + 0.01);

    expect(hits.isInvulnerable).toBe(false);
    expect(hits.registerHit()).toBe(true);
    expect(hits.remainingLives).toBe(MAX_LIVES - 2);
  });

  it("one sustained overlap cannot drain every life across frames", () => {
    const hits = new HitFeedback();
    // Simulate 2 seconds of continuous contact at 60Hz.
    for (let i = 0; i < 120; i += 1) {
      hits.registerHit();
      hits.update(1 / 60);
    }
    // 2s of contact spans only ~1 full invulnerability window, so at most a
    // couple of hits should have landed -- certainly not 120.
    expect(hits.remainingLives).toBeGreaterThan(0);
  });
});

describe("HitFeedback hit-stop", () => {
  it("freezes simulation time immediately after a hit", () => {
    const hits = new HitFeedback();
    hits.registerHit();
    expect(hits.isHitStopped).toBe(true);
    expect(hits.timeScale).toBe(0);
  });

  it("resumes normal time after the hit-stop window", () => {
    const hits = new HitFeedback();
    hits.registerHit();
    hits.update(HIT_STOP_S + 0.001);

    expect(hits.isHitStopped).toBe(false);
    expect(hits.timeScale).toBe(1);
  });

  it("runs at normal speed when nothing has happened", () => {
    const hits = new HitFeedback();
    hits.update(1 / 60);
    expect(hits.timeScale).toBe(1);
  });
});

describe("HitFeedback wasJustHit", () => {
  it("is true only on the frame the hit landed", () => {
    const hits = new HitFeedback();
    hits.registerHit();
    expect(hits.wasJustHit).toBe(true);

    hits.update(1 / 60);
    expect(hits.wasJustHit).toBe(false);
  });

  it("is not set by an absorbed hit", () => {
    const hits = new HitFeedback();
    hits.registerHit();
    hits.update(1 / 60);
    hits.registerHit(); // absorbed by i-frames
    expect(hits.wasJustHit).toBe(false);
  });
});

describe("HitFeedback reset", () => {
  it("restores full lives and clears every timer", () => {
    const hits = new HitFeedback();
    hits.registerHit();

    hits.reset();

    expect(hits.remainingLives).toBe(MAX_LIVES);
    expect(hits.isInvulnerable).toBe(false);
    expect(hits.isHitStopped).toBe(false);
    expect(hits.timeScale).toBe(1);
  });
});
