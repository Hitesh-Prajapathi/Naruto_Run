import { describe, expect, it } from "vitest";
import { SealSequence } from "../../src/game/sealSequence";
import {
  SEAL_FAIL_COOLDOWN_S,
  SEAL_POOL,
  SEAL_SEQUENCE_LENGTH,
  SEAL_SEQUENCE_WINDOW_S,
  type SealName,
} from "../../src/config/bossConfig";

/** Always picks SEAL_POOL[0], so the expected sequence is deterministic. */
const firstSeal = () => 0;
const otherSeal = (): SealName => SEAL_POOL[1]!;

describe("SealSequence start", () => {
  it("generates a sequence of the configured length", () => {
    const seals = new SealSequence(firstSeal);
    const events = seals.start();
    expect(seals.sequence).toHaveLength(SEAL_SEQUENCE_LENGTH);
    expect(events[0]?.type).toBe("sequence_started");
  });

  it("only draws seals from the pool", () => {
    const seals = new SealSequence(Math.random);
    seals.start();
    for (const seal of seals.sequence) {
      expect(SEAL_POOL).toContain(seal);
    }
  });

  it("exposes the seal the player should perform next", () => {
    const seals = new SealSequence(firstSeal);
    seals.start();
    expect(seals.expectedSeal).toBe(seals.sequence[0]);
  });

  it("is a no-op if already active", () => {
    const seals = new SealSequence(firstSeal);
    seals.start();
    expect(seals.start()).toEqual([]);
  });
});

describe("SealSequence correct input", () => {
  it("advances one step per correct seal", () => {
    const seals = new SealSequence(firstSeal);
    seals.start();

    const events = seals.submit(seals.sequence[0]!);

    expect(events[0]).toEqual({ type: "seal_accepted", index: 0 });
    expect(seals.matchedCount).toBe(1);
    expect(seals.expectedSeal).toBe(seals.sequence[1]);
  });

  it("completes the sequence on the final correct seal", () => {
    const seals = new SealSequence(firstSeal);
    seals.start();
    const target = [...seals.sequence];

    let completed = false;
    for (const seal of target) {
      const events = seals.submit(seal);
      if (events.some((e) => e.type === "sequence_completed")) completed = true;
    }

    expect(completed).toBe(true);
    expect(seals.currentPhase).toBe("cooldown");
  });
});

describe("SealSequence failure handling (brief §5.3)", () => {
  it("fails immediately on a wrong seal", () => {
    const seals = new SealSequence(firstSeal);
    seals.start();

    const events = seals.submit(otherSeal());

    expect(events[0]).toEqual({ type: "sequence_failed", reason: "wrong_seal" });
    expect(seals.currentPhase).toBe("cooldown");
    expect(seals.matchedCount).toBe(0);
  });

  it("fails on timeout when the window elapses", () => {
    const seals = new SealSequence(firstSeal);
    seals.start();

    const events = seals.update(SEAL_SEQUENCE_WINDOW_S + 0.01);

    expect(events.some((e) => e.type === "sequence_failed" && e.reason === "timeout")).toBe(true);
  });

  it("ignores input during the cooldown so stray presses cannot queue up", () => {
    const seals = new SealSequence(firstSeal);
    seals.start();
    seals.submit(otherSeal()); // fail into cooldown
    expect(seals.currentPhase).toBe("cooldown");

    expect(seals.submit(seals.sequence[0] ?? SEAL_POOL[0]!)).toEqual([]);
  });

  it("auto-starts a fresh attempt after the cooldown expires", () => {
    // The player should always have something to work on without pressing
    // anything to re-arm.
    const seals = new SealSequence(firstSeal);
    seals.start();
    seals.submit(otherSeal());

    const events = seals.update(SEAL_FAIL_COOLDOWN_S + 0.01);

    expect(events.some((e) => e.type === "sequence_started")).toBe(true);
    expect(seals.currentPhase).toBe("active");
    expect(seals.matchedCount).toBe(0);
  });
});

describe("SealSequence window progress", () => {
  it("reports rising progress through the window", () => {
    const seals = new SealSequence(firstSeal);
    seals.start();
    expect(seals.windowProgress).toBeCloseTo(0, 5);

    seals.update(SEAL_SEQUENCE_WINDOW_S / 2);

    expect(seals.windowProgress).toBeCloseTo(0.5, 2);
  });

  it("reports zero progress when not active", () => {
    const seals = new SealSequence(firstSeal);
    expect(seals.windowProgress).toBe(0);
  });
});

describe("SealSequence reset (brief §4.5)", () => {
  it("clears the sequence and returns to idle", () => {
    const seals = new SealSequence(firstSeal);
    seals.start();
    seals.submit(seals.sequence[0]!);

    seals.reset();

    expect(seals.currentPhase).toBe("idle");
    expect(seals.sequence).toEqual([]);
    expect(seals.matchedCount).toBe(0);
    expect(seals.expectedSeal).toBeNull();
  });
});
