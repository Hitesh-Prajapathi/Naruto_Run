/**
 * Obito's special attack and the Hare counter -- Feature Brief 04 §3-§5.
 *
 * Pure logic: no Three.js, no DOM, and -- as with every other system here --
 * **no timers**. It advances on the frame delta and emits typed events the
 * scene layer turns into VFX, UI and camera work.
 *
 * Two structural decisions do most of the safety work the brief asks for.
 *
 * 1. **The dangerous suspensions are derived, not stored.** Brief §5 calls
 *    out lean suppression and the 70% time scale as "the two most dangerous
 *    leaks", because both are silent when they persist into the next run.
 *    Neither is a flag that has to be cleared: `suppressLaneInput` and
 *    `worldTimeScale` are getters computed from the current state, so any
 *    exit -- success, failure, encounter end, restart, or a state this file
 *    doesn't even know about yet -- restores them by construction. There is
 *    nothing to forget to undo. `teardown()` still exists for the imperative
 *    leftovers (VFX, UI, the jutsu prompt), and every exit path routes
 *    through it exactly as the brief requires.
 *
 * 2. **The countdown runs on unscaled real seconds.** `update()` is fed the
 *    raw frame delta, never the slowed one, so brief §4.1's "2.5s must still
 *    mean 2.5s under a time-scale change" holds by construction too.
 */

import {
  SPECIAL_FAILURE_BEAT_S,
  SPECIAL_COMBAT_FALLBACK_S,
  SPECIAL_PRACTICE_PATIENCE_S,
  SPECIAL_MIN_OBITO_HP_MARGIN,
  SPECIAL_SUCCESS_BEAT_S,
  SPECIAL_TIME_SCALE,
  SPECIAL_TRIGGER_DELAY_S,
  SPECIAL_TRIGGER_OBITO_HP,
  SPECIAL_VFX_TRAVEL_S,
  SPECIAL_WARNING_S,
} from "../config/specialAttackConfig";

export type SpecialAttackState =
  /** Trigger conditions not met yet. */
  | "idle"
  /** Trigger met, but the player has never performed `hare` (§2.2 guard). */
  | "awaiting_practice"
  /** Trigger met and armed; short beat before the warning opens. */
  | "pending"
  /** The window is open. This is the only state that suspends anything. */
  | "warning"
  /** Countered successfully; playing the success beat. */
  | "countered"
  /** Counter missed; playing the impact beat before Game Over. */
  | "struck"
  /** Used for this encounter; can never fire again (§2.3). */
  | "spent";

export type SpecialAttackEvent =
  /** Show the "practise HARE first" prompt (§2.2). */
  | { type: "practice_required" }
  /** Open the warning UI. `durationS` is in real seconds. */
  | { type: "warning_opened"; durationS: number }
  /** Release the projectile visual; still mid-window. */
  | { type: "attack_released" }
  /** The player got it. Nobody takes damage (§2.6). */
  | { type: "countered" }
  /** The player missed. Lethal regardless of HP (§2.5). */
  | { type: "struck" }
  /** Every exit path ends here: restore the world (§5). */
  | { type: "resolved"; outcome: "countered" | "struck" };

/** What the encounter tells the controller about this frame. */
export interface SpecialAttackContext {
  /** Obito's current HP -- the trigger condition (§3). */
  obitoHp: number;
  /** True while a normal Obito attack or Naruto projectile is resolving.
   * §3: never fire on top of one. */
  combatBusy: boolean;
  /** §2.2: the player must have performed `hare` at least once first. */
  hareEverPerformed: boolean;
  /** False during intro/victory/defeat. §3: combat only. */
  inCombat: boolean;
  /** Seconds spent in combat, for the fallback trigger. */
  combatElapsedS: number;
}

export class SpecialAttackController {
  private phase: SpecialAttackState = "idle";
  /** Seconds spent in the current phase, always unscaled real time. */
  private elapsed = 0;
  /** The projectile has left Obito's hand; it is still counterable. */
  private released = false;
  private readonly windowS: number;

  constructor(encounterIndex: number) {
    this.windowS =
      SPECIAL_WARNING_S[Math.min(encounterIndex, SPECIAL_WARNING_S.length - 1)] ??
      SPECIAL_WARNING_S[0];
  }

  get state(): SpecialAttackState {
    return this.phase;
  }

  /** Configured window length for this encounter, in real seconds. */
  get windowDurationS(): number {
    return this.windowS;
  }

  /** True once it has fired, whatever the outcome (§3's last guard rail). */
  get hasFired(): boolean {
    return this.phase === "warning" || this.phase === "countered" || this.phase === "struck" || this.phase === "spent";
  }

  /** The window is open and accepting the counter. */
  get isWarning(): boolean {
    return this.phase === "warning";
  }

  /** Waiting for the player to try `hare` once before it is safe to fire. */
  get isAwaitingPractice(): boolean {
    return this.phase === "awaiting_practice";
  }

  /** Real seconds left in the window; 0 outside it. */
  get remainingS(): number {
    return this.phase === "warning" ? Math.max(0, this.windowS - this.elapsed) : 0;
  }

  /** 0..1 through the window, for the countdown ring. */
  get windowProgress(): number {
    return this.phase === "warning" ? Math.min(1, this.elapsed / this.windowS) : 0;
  }

  // --- Derived suspensions (see the header note) -------------------------

  /**
   * §4.2: lane changes are suppressed for the window. The player raises
   * their hands to form the sign, which risks a phantom lean-driven lane
   * change, and lane position is irrelevant here anyway -- only `hare`
   * saves him.
   */
  get suppressLaneInput(): boolean {
    return this.phase === "warning";
  }

  /**
   * §4.2: the rest of combat is frozen for the window and the resolution
   * beat. "No other input does anything. No jutsu, no jump. One escape
   * route." Freezing rather than merely ignoring input also disposes of
   * brief §3's "never while a normal attack is resolving" hazard for the
   * duration -- nothing can land mid-window.
   */
  get freezeCombat(): boolean {
    return this.phase === "warning" || this.phase === "countered" || this.phase === "struck";
  }

  /** §4.1's optional slowdown. Applies to the world, never the countdown. */
  get worldTimeScale(): number {
    return this.phase === "warning" ? SPECIAL_TIME_SCALE : 1;
  }

  /**
   * Offer the counter.
   *
   * @param holdStartedSecondsAgo how long the recognised hold has been in
   *   progress. §2.1: "a `hare` already in progress when the window opens
   *   does not carry over" -- a hold that began before the window opened is
   *   rejected and a fresh one is required. Keyboard presses are
   *   instantaneous, so they pass trivially; this exists for the CV path.
   * @returns true when the input was consumed as the counter. §2.1's
   *   context-gated priority: inside the window `hare` means the counter and
   *   nothing else, so a `true` return tells the caller to swallow the input
   *   rather than also firing `ikazuchi`.
   */
  submitCounter(holdStartedSecondsAgo = 0): boolean {
    if (this.phase !== "warning") {
      return false;
    }
    if (holdStartedSecondsAgo > this.elapsed) {
      // Stale: the hold predates the window. Discard and require a fresh one.
      return false;
    }
    this.enter("countered");
    return true;
  }

  /**
   * Advance. `rawDt` must be the **unscaled** frame delta so the window
   * lasts its configured number of real seconds (§4.1).
   */
  update(rawDt: number, context: SpecialAttackContext): SpecialAttackEvent[] {
    const events: SpecialAttackEvent[] = [];
    this.elapsed += rawDt;

    switch (this.phase) {
      case "idle":
      case "awaiting_practice": {
        if (!this.canArm(context)) {
          // Conditions lapsed (e.g. a normal attack started); fall back to
          // idle so the practice prompt doesn't linger on screen.
          if (this.phase === "awaiting_practice" && !this.triggerMet(context)) {
            this.enter("idle");
          }
          break;
        }
        if (!context.hareEverPerformed) {
          // §2.2: nobody should meet this mechanic for the first time in the
          // moment it can kill them. Put the prompt up and hold the attack
          // back -- but only for SPECIAL_PRACTICE_PATIENCE_S. Waiting forever
          // (the first version) meant a player who never pressed the key
          // never saw the feature at all.
          if (this.phase !== "awaiting_practice") {
            this.enter("awaiting_practice");
            events.push({ type: "practice_required" });
            break;
          }
          if (this.elapsed < SPECIAL_PRACTICE_PATIENCE_S) {
            break;
          }
          // Patience exhausted: fire anyway. The prompt has been up the whole
          // time and the warning window teaches the sign again at full size.
        }
        this.enter("pending");
        break;
      }

      case "pending": {
        if (this.elapsed >= SPECIAL_TRIGGER_DELAY_S) {
          this.enter("warning");
          events.push({ type: "warning_opened", durationS: this.windowS });
        }
        break;
      }

      case "warning": {
        // The projectile leaves in time to *arrive* as the window closes, so
        // the last stretch of the window is spent watching it come in. §2.4:
        // "accept the counter from the instant the warning appears through to
        // impact" -- a counter during the flight still saves him, and the
        // shot visibly veers off instead.
        if (!this.released && this.elapsed >= this.windowS - SPECIAL_VFX_TRAVEL_S) {
          this.released = true;
          events.push({ type: "attack_released" });
        }
        if (this.elapsed >= this.windowS) {
          this.enter("struck");
          events.push({ type: "struck" });
        }
        break;
      }

      case "countered": {
        if (this.elapsed >= SPECIAL_SUCCESS_BEAT_S) {
          this.enter("spent");
          events.push({ type: "resolved", outcome: "countered" });
        }
        break;
      }

      case "struck": {
        if (this.elapsed >= SPECIAL_FAILURE_BEAT_S) {
          this.enter("spent");
          events.push({ type: "resolved", outcome: "struck" });
        }
        break;
      }

      case "spent":
      default:
        break;
    }

    return events;
  }

  /** §3's guard rails, all of them, in one place. */
  private canArm(context: SpecialAttackContext): boolean {
    return (
      context.inCombat &&
      this.triggerMet(context) &&
      // Never on the frame he dies, and never once he is down.
      context.obitoHp >= SPECIAL_MIN_OBITO_HP_MARGIN &&
      // Never on top of a normal attack -- resolve that first.
      !context.combatBusy
    );
  }

  /**
   * §3's HP threshold, or the elapsed-combat floor that actually makes the
   * attack guaranteed. See SPECIAL_COMBAT_FALLBACK_S for why the threshold
   * alone is not enough in a real fight.
   */
  private triggerMet(context: SpecialAttackContext): boolean {
    return (
      context.obitoHp <= SPECIAL_TRIGGER_OBITO_HP ||
      context.combatElapsedS >= SPECIAL_COMBAT_FALLBACK_S
    );
  }

  private enter(phase: SpecialAttackState): void {
    this.phase = phase;
    this.elapsed = 0;
  }
}
