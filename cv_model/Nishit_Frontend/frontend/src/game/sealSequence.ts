/**
 * Naruto's attack input: a randomised hand-seal sequence -- Feature Brief
 * 02 §5.3.
 *
 * Pure logic, deliberately decoupled from *how* a seal arrives. It exposes
 * a single `submit(seal)` entry point, so the keyboard fallback (1/2/3) and
 * real CV hand-seal recognition are the same code path with different
 * producers. Brief §6 requires the keyboard fallback to exist permanently so
 * combat stays testable without a camera.
 *
 * The sequence is re-randomised from a small pool on every attempt so it
 * can't be muscle-memoried blindly.
 */

import {
  SEAL_FAIL_COOLDOWN_S,
  SEAL_POOL,
  SEAL_SEQUENCE_LENGTH,
  SEAL_SEQUENCE_WINDOW_S,
  SEAL_SUCCESS_COOLDOWN_S,
  type SealName,
} from "../config/bossConfig";

export type SealPhase = "idle" | "active" | "cooldown";

export type SealEvent =
  | { type: "sequence_started"; sequence: readonly SealName[] }
  | { type: "seal_accepted"; index: number }
  | { type: "sequence_failed"; reason: "wrong_seal" | "timeout" }
  | { type: "sequence_completed" };

export class SealSequence {
  private phase: SealPhase = "idle";
  private target: SealName[] = [];
  private progress = 0;
  private timer = 0;

  constructor(private readonly random: () => number = Math.random) {}

  get currentPhase(): SealPhase {
    return this.phase;
  }

  /** The seals the player must perform, in order. Empty when idle. */
  get sequence(): readonly SealName[] {
    return this.target;
  }

  /** How many seals have been matched so far. */
  get matchedCount(): number {
    return this.progress;
  }

  /** The seal the player should be performing right now, or null. */
  get expectedSeal(): SealName | null {
    return this.phase === "active" ? (this.target[this.progress] ?? null) : null;
  }

  /** Seconds left in the sequence window (or in the cooldown). */
  get remainingS(): number {
    return Math.max(0, this.timer);
  }

  /** 0..1, for a countdown ring in the HUD. */
  get windowProgress(): number {
    if (this.phase !== "active") return 0;
    return 1 - this.timer / SEAL_SEQUENCE_WINDOW_S;
  }

  /** Begin a new randomised attempt. No-op unless idle. */
  start(): SealEvent[] {
    if (this.phase !== "idle") {
      return [];
    }
    // No two consecutive seals the same. With a 3-seal pool, plain uniform
    // draws produce an all-identical sequence about 11% of the time
    // (measured: 20/200), which is trivially easy and reads as a bug rather
    // than a roll of the dice.
    this.target = [];
    for (let i = 0; i < SEAL_SEQUENCE_LENGTH; i += 1) {
      const previous = this.target[i - 1];
      const choices = SEAL_POOL.filter((seal) => seal !== previous);
      const index = Math.floor(this.random() * choices.length);
      this.target.push(choices[Math.min(index, choices.length - 1)] as SealName);
    }
    this.progress = 0;
    this.timer = SEAL_SEQUENCE_WINDOW_S;
    this.phase = "active";
    return [{ type: "sequence_started", sequence: [...this.target] }];
  }

  /**
   * Offer a seal. Correct advances the sequence (and completes it on the
   * last one); wrong fails immediately into a cooldown. Ignored outside the
   * active phase, so stray input during cooldown can't queue up.
   */
  submit(seal: SealName): SealEvent[] {
    if (this.phase !== "active") {
      return [];
    }
    if (seal !== this.target[this.progress]) {
      return this.fail("wrong_seal");
    }

    this.progress += 1;
    const events: SealEvent[] = [{ type: "seal_accepted", index: this.progress - 1 }];
    if (this.progress >= this.target.length) {
      this.phase = "cooldown";
      this.timer = SEAL_SUCCESS_COOLDOWN_S;
      events.push({ type: "sequence_completed" });
    }
    return events;
  }

  private fail(reason: "wrong_seal" | "timeout"): SealEvent[] {
    this.phase = "cooldown";
    this.timer = SEAL_FAIL_COOLDOWN_S;
    this.progress = 0;
    return [{ type: "sequence_failed", reason }];
  }

  /** Advance timers. Auto-restarts a fresh attempt after a cooldown so the
   * player always has a sequence to work on without pressing anything. */
  update(dt: number): SealEvent[] {
    if (this.phase === "idle") {
      return this.start();
    }

    this.timer -= dt;
    if (this.timer > 0) {
      return [];
    }

    if (this.phase === "active") {
      const events = this.fail("timeout");
      return events;
    }
    // Cooldown elapsed.
    this.phase = "idle";
    this.target = [];
    this.progress = 0;
    return this.start();
  }

  /** Full reset between encounters and on Try Again. */
  reset(): void {
    this.phase = "idle";
    this.target = [];
    this.progress = 0;
    this.timer = 0;
  }
}
