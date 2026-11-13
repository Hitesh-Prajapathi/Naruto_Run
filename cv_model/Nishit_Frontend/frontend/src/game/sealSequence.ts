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
  ATTACK_CATALOG,
  SEAL_FAIL_COOLDOWN_S,
  SEAL_POOL,
  SEAL_SEQUENCE_LENGTH,
  SEAL_SEQUENCE_WINDOW_S,
  SEAL_SUCCESS_COOLDOWN_S,
  type AttackDefinition,
  type SealName,
} from "../config/bossConfig";

export type SealPhase = "idle" | "active" | "cooldown";

export type SealEvent =
  | { type: "sequence_started"; sequence: readonly SealName[] }
  | { type: "seal_accepted"; index: number; seal?: SealName }
  | { type: "attack_matched"; attack: AttackDefinition }
  | { type: "sequence_failed"; reason: "wrong_seal" | "timeout" }
  | { type: "sequence_completed"; attack?: AttackDefinition };

export class SealSequence {
  private phase: SealPhase = "idle";
  private target: SealName[] = [];
  private progress = 0;
  private timer = 0;
  private lastSeal: SealName | null = null;
  private lastAttack: AttackDefinition | null = null;
  private recentSeals: Array<{ seal: SealName; time: number }> = [];

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

  /** The full attack catalog for UI display. */
  get catalog(): readonly AttackDefinition[] {
    return ATTACK_CATALOG;
  }

  /** Last recognized seal name. */
  get lastRecognizedSeal(): SealName | null {
    return this.lastSeal;
  }

  /** Last triggered attack definition. */
  get lastTriggeredAttack(): AttackDefinition | null {
    return this.lastAttack;
  }

  /**
   * Progress towards each attack in the catalog: attack.id -> count of matched seals.
   */
  get attackProgress(): Record<string, number> {
    const result: Record<string, number> = {};
    const recent = this.recentSeals.map((r) => r.seal);
    for (const attack of ATTACK_CATALOG) {
      let matchCount = 0;
      for (let len = attack.seals.length; len > 0; len -= 1) {
        const prefix = attack.seals.slice(0, len);
        if (recent.length >= len && this.arrayEndsWith(recent, prefix)) {
          matchCount = len;
          break;
        }
      }
      result[attack.id] = matchCount;
    }
    return result;
  }

  private arrayEndsWith(arr: readonly SealName[], suffix: readonly SealName[]): boolean {
    if (arr.length < suffix.length) return false;
    for (let i = 0; i < suffix.length; i += 1) {
      if (arr[arr.length - suffix.length + i] !== suffix[i]) return false;
    }
    return true;
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
    this.target = [];
    for (let i = 0; i < SEAL_SEQUENCE_LENGTH; i += 1) {
      const previous = this.target[i - 1];
      const choices = SEAL_POOL.filter((seal) => seal !== previous);
      const index = Math.floor(this.random() * choices.length);
      this.target.push(choices[Math.min(index, choices.length - 1)] as SealName);
    }
    this.progress = 0;
    this.timer = SEAL_SEQUENCE_WINDOW_S;
    this.recentSeals = [];
    this.phase = "active";
    return [{ type: "sequence_started", sequence: [...this.target] }];
  }

  /**
   * Offer a seal. Checks for matches in the Jutsu Catalog and against the target.
   */
  submit(seal: SealName, now = performance.now() / 1000): SealEvent[] {
    if (this.phase !== "active") {
      return [];
    }
    this.lastSeal = seal;
    this.recentSeals.push({ seal, time: now });
    // Keep at most last 4 seals
    if (this.recentSeals.length > 4) {
      this.recentSeals.shift();
    }

    const recent = this.recentSeals.map((r) => r.seal);
    const events: SealEvent[] = [];

    // Check if any jutsu in the catalog is completed!
    for (const attack of ATTACK_CATALOG) {
      if (this.arrayEndsWith(recent, attack.seals)) {
        this.lastAttack = attack;
        this.phase = "cooldown";
        this.timer = SEAL_SUCCESS_COOLDOWN_S;
        this.recentSeals = [];
        this.progress = this.target.length;
        events.push({ type: "seal_accepted", index: this.progress - 1 });
        events.push({ type: "attack_matched", attack });
        events.push({ type: "sequence_completed", attack });
        return events;
      }
    }

    // Also support matching the target sequence directly
    if (this.target.length > 0 && seal === this.target[this.progress]) {
      this.progress += 1;
      events.push({ type: "seal_accepted", index: this.progress - 1 });
      if (this.progress >= this.target.length) {
        this.phase = "cooldown";
        this.timer = SEAL_SUCCESS_COOLDOWN_S;
        this.recentSeals = [];
        events.push({ type: "sequence_completed" });
      }
      return events;
    }

    // If seal doesn't match any partial attack prefix and doesn't match target:
    const matchesAnyPrefix = ATTACK_CATALOG.some((attack) =>
      this.arrayEndsWith(recent, attack.seals.slice(0, recent.length)),
    );
    if (!matchesAnyPrefix && (this.target.length === 0 || seal !== this.target[this.progress])) {
      return this.fail("wrong_seal");
    }

    events.push({ type: "seal_accepted", index: Math.max(0, this.progress - 1) });
    return events;
  }

  /**
   * Direct trigger for a specific attack (e.g. from keyboard keys 1-5).
   */
  triggerAttack(attackId: string): SealEvent[] {
    const attack = ATTACK_CATALOG.find((a) => a.id === attackId);
    if (!attack) return [];
    this.lastAttack = attack;
    this.phase = "cooldown";
    this.timer = SEAL_SUCCESS_COOLDOWN_S;
    this.recentSeals = [];
    this.progress = this.target.length;
    return [
      { type: "attack_matched", attack },
      { type: "sequence_completed", attack },
    ];
  }

  private fail(reason: "wrong_seal" | "timeout"): SealEvent[] {
    this.phase = "cooldown";
    this.timer = SEAL_FAIL_COOLDOWN_S;
    this.progress = 0;
    this.recentSeals = [];
    return [{ type: "sequence_failed", reason }];
  }

  /** Advance timers. Auto-restarts a fresh attempt after a cooldown. */
  update(dt: number, now = performance.now() / 1000): SealEvent[] {
    if (this.phase === "idle") {
      return this.start();
    }

    // Prune stale seals older than 3.0s
    this.recentSeals = this.recentSeals.filter((s) => now - s.time <= 3.0);

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
    this.recentSeals = [];
    return this.start();
  }

  /** Full reset between encounters and on Try Again. */
  reset(): void {
    this.phase = "idle";
    this.target = [];
    this.progress = 0;
    this.timer = 0;
    this.lastSeal = null;
    this.lastAttack = null;
    this.recentSeals = [];
  }
}
