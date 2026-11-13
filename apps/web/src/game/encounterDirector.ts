/**
 * Orchestrates a whole Obito encounter: intro -> combat -> victory/defeat
 * -- Feature Brief 02 §5.2, §5.8, §5.9.
 *
 * Design constraint from brief §0: this must **suspend and restore**, never
 * modify. It owns a `scrollFactor` (1 = normal running speed, 0 = stopped)
 * that the host multiplies into its existing scroll speed, and it captures
 * the camera's running values on entry and hands them back untouched on
 * exit. Nothing in the approved running game is rewritten; it is scaled by
 * a factor that is exactly 1 again the moment the encounter ends.
 *
 * Owning all encounter timing here (rather than scattering setTimeouts) is
 * also what makes the brief's "no orphaned timer fires during the next run"
 * requirement tractable: there are no timers, only this object's elapsed
 * counters, and dropping the object drops all of them.
 */

import {
  BOSS_INTRO_DECEL_S,
  BOSS_INTRO_HOLD_S,
  BOSS_OUTRO_ACCEL_S,
  COMBAT_HIT_STOP_S,
  NARUTO_PROJECTILE_TRAVEL_S,
  NARUTO_VICTORY_S,
  OBITO_COMBAT_Z,
  OBITO_DEFEAT_S,
  OBITO_PROJECTILE_TRAVEL_S,
  OBITO_SPAWN_Z,
} from "../config/bossConfig";
import { ENABLE_SPECIAL_ATTACK } from "../config/specialAttackConfig";
import { type LaneIndex } from "../config/gameConfig";
import { CombatController, type CombatEvent } from "./combatController";
import { SealSequence, type SealEvent } from "./sealSequence";
import { SpecialAttackController, type SpecialAttackEvent } from "./specialAttackController";

export type EncounterPhase = "intro" | "combat" | "victory" | "defeat" | "done";

/** What the director asks the host scene to do; the host owns the meshes. */
export interface EncounterHooks {
  onObitoMoveLane(lane: LaneIndex): void;
  onObitoAttackPose(): void;
  onObitoHit(): void;
  onObitoDefeated(durationS: number): void;
  onTelegraph(lane: LaneIndex | null): void;
  onFireVfx(owner: "naruto" | "obito", fromLane: LaneIndex, toLane: LaneIndex, travelS: number): void;
  onNarutoHit(damage: number): void;
  onDamageNumber(target: "naruto" | "obito", damage: number, lane: LaneIndex): void;
  onHitStop(seconds: number): void;
  onBanner(text: string): void;

  // --- Feature Brief 04 (special attack) ------------------------------
  // Optional so nothing that already builds an EncounterHooks has to change
  // -- brief §0 limits edits to existing code to "adding a call, a state
  // check, or a listener".
  /** §2.2: the player must try `hare` once before it can kill them. */
  onSpecialPracticeRequired?(): void;
  /** §4.1: open the warning. `durationS` is in real seconds. */
  onSpecialWarning?(durationS: number): void;
  /** §4.1/§4.5: the projectile leaves Obito's hand, still counterable. */
  onSpecialReleased?(): void;
  /** §4.3: countered -- the attack visibly misses, nobody takes damage. */
  onSpecialCountered?(): void;
  /** §4.4: it connected. Lethal regardless of HP. */
  onSpecialStruck?(): void;
  /** §5: the single teardown, run on *every* exit path. */
  onSpecialTeardown?(): void;
}

export type EncounterOutcome = "won" | "lost";

function easeInOutCubic(t: number): number {
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
}

export class EncounterDirector {
  readonly combat: CombatController;
  readonly seals = new SealSequence();
  /**
   * Feature Brief 04. Null when ENABLE_SPECIAL_ATTACK is off, which is what
   * makes the flag-off build byte-for-byte the approved encounter: every
   * call site below is guarded on this being non-null.
   */
  readonly special: SpecialAttackController | null;

  private phase: EncounterPhase = "intro";
  private phaseElapsed = 0;
  private outcome: EncounterOutcome | null = null;
  private lastFailure: "wrong_seal" | "timeout" | null = null;
  private failureFlashS = 0;
  /** True when the loss came from the special attack, so the Game Over
   * screen can name the right cause (Brief 04 §2.5). */
  private lostToSpecial = false;

  constructor(
    readonly encounterIndex: number,
    private readonly hooks: EncounterHooks,
  ) {
    this.combat = new CombatController(encounterIndex);
    this.special = ENABLE_SPECIAL_ATTACK ? new SpecialAttackController(encounterIndex) : null;
  }

  /** Brief 04 §2.5: distinguishes the two combat losses on the Game Over
   * screen. False for an ordinary defeat. */
  get lostToSpecialAttack(): boolean {
    return this.lostToSpecial;
  }

  /** Brief 04 §4.2: lane input is suppressed for the warning window only. */
  get suppressLaneInput(): boolean {
    return this.special?.suppressLaneInput ?? false;
  }

  /** Brief 04 §4.1: world time scale during the window. Never applied to
   * the warning countdown, which runs in real seconds. */
  get worldTimeScale(): number {
    return this.special?.worldTimeScale ?? 1;
  }

  /**
   * Offer the Hare counter -- Brief 04 §2.1's context-gated priority.
   *
   * Returns true when it was consumed as the counter, which tells the caller
   * to swallow the input entirely rather than also firing `ikazuchi`. This
   * is the whole resolution to the triple-booked `hare`: the decision lives
   * in one place, and it is made by asking whether the window is open.
   *
   * Outside the window it returns false and the input is free to mean
   * whatever the jutsu catalog says -- but it still counts as practice
   * (§2.2), which is tracked by the caller since it persists across
   * encounters.
   */
  submitCounter(holdStartedSecondsAgo = 0): boolean {
    if (this.phase !== "combat") return false;
    const consumed = this.special?.submitCounter(holdStartedSecondsAgo) ?? false;
    if (consumed) {
      // §4.3: nobody takes damage. Purely defensive -- surviving is the
      // reward, and letting it chip Obito would break the 3-hit kill maths.
      this.hooks.onSpecialCountered?.();
      this.hooks.onBanner("COUNTERED!");
    }
    return consumed;
  }

  get currentPhase(): EncounterPhase {
    return this.phase;
  }

  get result(): EncounterOutcome | null {
    return this.outcome;
  }

  get failureReason(): "wrong_seal" | "timeout" | null {
    return this.failureFlashS > 0 ? this.lastFailure : null;
  }

  /**
   * World-scroll multiplier for this frame: 1 during normal running, easing
   * to 0 through the intro, 0 in combat, easing back to 1 on the way out.
   * The host multiplies its existing speed by this; it never sets speed.
   */
  get scrollFactor(): number {
    switch (this.phase) {
      case "intro": {
        const t = Math.min(1, this.phaseElapsed / BOSS_INTRO_DECEL_S);
        return 1 - easeInOutCubic(t);
      }
      case "victory": {
        // Hold still through the defeat beat, then accelerate back.
        const accelStart = OBITO_DEFEAT_S + NARUTO_VICTORY_S;
        if (this.phaseElapsed < accelStart) return 0;
        const t = Math.min(1, (this.phaseElapsed - accelStart) / BOSS_OUTRO_ACCEL_S);
        return easeInOutCubic(t);
      }
      case "combat":
      case "defeat":
      case "done":
      default:
        return 0;
    }
  }

  /** Obito's Z as he walks in during the intro, then holds for the duel. */
  get obitoZ(): number {
    if (this.phase !== "intro") return OBITO_COMBAT_Z;
    const t = Math.min(1, this.phaseElapsed / BOSS_INTRO_DECEL_S);
    return OBITO_SPAWN_Z + (OBITO_COMBAT_Z - OBITO_SPAWN_Z) * easeInOutCubic(t);
  }

  /** 0..1 blend toward the combat camera framing. */
  get cameraBlend(): number {
    switch (this.phase) {
      case "intro":
        return Math.min(1, this.phaseElapsed / (BOSS_INTRO_DECEL_S + BOSS_INTRO_HOLD_S));
      case "combat":
      case "defeat":
        return 1;
      case "victory": {
        const accelStart = OBITO_DEFEAT_S + NARUTO_VICTORY_S;
        if (this.phaseElapsed < accelStart) return 1;
        return Math.max(0, 1 - (this.phaseElapsed - accelStart) / BOSS_OUTRO_ACCEL_S);
      }
      default:
        return 0;
    }
  }

  /** True once the encounter is completely finished and the host may return
   * to RUNNING (victory) or GAME_OVER (defeat). */
  get isComplete(): boolean {
    return this.phase === "done";
  }

  /** Skip the intro (brief §5.2 suggests this -- you watch it a lot). */
  skipIntro(): void {
    if (this.phase === "intro") {
      this.phaseElapsed = BOSS_INTRO_DECEL_S + BOSS_INTRO_HOLD_S;
    }
  }

  /** Route a seal from either the keyboard fallback or CV recognition. */
  submitSeal(seal: Parameters<SealSequence["submit"]>[0]): void {
    if (this.phase !== "combat") return;
    this.handleSealEvents(this.seals.submit(seal));
  }

  /**
   * @param dt raw, unscaled frame delta -- the special attack's countdown
   *   must run in real seconds (Brief 04 §4.1).
   * @param hareEverPerformed Brief 04 §2.2's guard. Tracked by the caller
   *   because it persists across encounters within a run.
   */
  update(
    dt: number,
    narutoLane: LaneIndex,
    hareEverPerformed = false,
    realDt: number = dt,
  ): void {
    this.phaseElapsed += dt;
    this.failureFlashS = Math.max(0, this.failureFlashS - dt);

    // Brief 04: advanced on every phase, not only `combat`, and on the raw
    // delta. It only *arms* in combat (`inCombat` below), but once it has
    // fired it must be allowed to reach its `resolved` event wherever the
    // encounter has got to -- the failure path transitions the encounter to
    // `defeat` on the same frame it strikes, and a teardown that only ran in
    // `combat` would never fire there. That is precisely the silent leak
    // §5 warns about.
    if (this.special) {
      this.handleSpecialEvents(
        // `realDt`, not `dt`: the host clamps the frame delta to 1/15s so a
        // stall cannot teleport anything that integrates. That clamp makes
        // game time run slower than the clock below 15fps, which would
        // silently stretch the warning window -- and §4.1 requires it to
        // last its configured duration in *real* seconds.
        this.special.update(realDt, {
          obitoHp: this.combat.obito.hp,
          combatBusy: this.combat.isResolvingAttack,
          hareEverPerformed,
          inCombat: this.phase === "combat",
          combatElapsedS: this.phase === "combat" ? this.phaseElapsed : 0,
        }),
        narutoLane,
      );
    }

    switch (this.phase) {
      case "intro":
        if (this.phaseElapsed >= BOSS_INTRO_DECEL_S + BOSS_INTRO_HOLD_S) {
          this.enterPhase("combat");
          this.hooks.onBanner("VS OBITO");
          this.handleSealEvents(this.seals.start());
        }
        break;

      case "combat": {
        // Brief 04 §4.2: while the window is open the rest of combat does
        // not advance at all -- "no other input does anything... one escape
        // route." Freezing rather than filtering also disposes of §3's
        // "never while a normal attack is resolving" hazard for the duration:
        // nothing can land mid-window.
        if (this.special?.freezeCombat) break;
        this.handleSealEvents(this.seals.update(dt));
        const events = this.combat.update(dt, narutoLane);
        for (const event of events) this.handleCombatEvent(event, narutoLane);
        this.hooks.onTelegraph(this.combat.telegraphedLane);
        break;
      }

      case "victory": {
        const total = OBITO_DEFEAT_S + NARUTO_VICTORY_S + BOSS_OUTRO_ACCEL_S;
        if (this.phaseElapsed >= total) this.enterPhase("done");
        break;
      }

      case "defeat":
        // Brief §5.9: short pause on the killing blow before Game Over.
        if (this.phaseElapsed >= 0.8) this.enterPhase("done");
        break;

      case "done":
      default:
        break;
    }
  }

  private enterPhase(phase: EncounterPhase): void {
    this.phase = phase;
    this.phaseElapsed = 0;
  }

  private handleCombatEvent(event: CombatEvent, narutoLane: LaneIndex): void {
    switch (event.type) {
      case "obito_dodge_telegraph":
        // The lean happens before the slide; the slide itself is triggered
        // by obito_move so the visual and logical lanes stay in step.
        break;
      case "obito_move":
        this.hooks.onObitoMoveLane(event.to);
        break;
      case "obito_telegraph":
        this.hooks.onTelegraph(event.lane);
        break;
      case "obito_fire":
        this.hooks.onObitoAttackPose();
        this.hooks.onTelegraph(null);
        this.hooks.onFireVfx("obito", this.combat.obitoCurrentLane, event.lane, OBITO_PROJECTILE_TRAVEL_S);
        break;
      case "obito_impact":
        if (event.hit) {
          this.hooks.onNarutoHit(event.damage);
          this.hooks.onDamageNumber("naruto", event.damage, event.lane);
          this.hooks.onHitStop(COMBAT_HIT_STOP_S);
        }
        break;
      case "naruto_fire":
        this.hooks.onFireVfx("naruto", narutoLane, event.lane, NARUTO_PROJECTILE_TRAVEL_S);
        break;
      case "naruto_impact":
        if (event.hit) {
          this.hooks.onObitoHit();
          this.hooks.onDamageNumber("obito", event.damage, event.lane);
          this.hooks.onHitStop(COMBAT_HIT_STOP_S);
        }
        break;
      case "obito_defeated":
        this.outcome = "won";
        this.hooks.onObitoDefeated(OBITO_DEFEAT_S);
        this.hooks.onTelegraph(null);
        this.hooks.onBanner("ENCOUNTER WON");
        this.seals.reset();
        this.enterPhase("victory");
        break;
      case "naruto_defeated":
        this.outcome = "lost";
        this.hooks.onTelegraph(null);
        this.seals.reset();
        this.enterPhase("defeat");
        break;
      default:
        break;
    }
  }

  /**
   * Feature Brief 04 §4.3-§4.5.
   *
   * Note that both resolutions converge on the same `resolved` event, and
   * that `resolved` is the *only* place the teardown hook is called. §5 is
   * explicit: "do not scatter restores across the success and failure
   * branches." The suspensions themselves (lean lock, time scale) need no
   * restoring at all -- they are getters off the controller's state.
   */
  private handleSpecialEvents(events: SpecialAttackEvent[], narutoLane: LaneIndex): void {
    for (const event of events) {
      switch (event.type) {
        case "practice_required":
          this.hooks.onSpecialPracticeRequired?.();
          break;
        case "warning_opened":
          // §4.2: cancel any jutsu sequence in progress rather than asking
          // the player to track two prompts at once. It restarts on
          // resolution, so this is a suspension, not a loss.
          this.seals.reset();
          this.hooks.onTelegraph(null);
          this.hooks.onSpecialWarning?.(event.durationS);
          break;
        case "attack_released":
          this.hooks.onSpecialReleased?.();
          break;
        case "struck": {
          // §2.5: lethal regardless of remaining HP, and it does not consume
          // an obstacle life -- combat loss ends the run outright.
          this.lostToSpecial = true;
          this.hooks.onSpecialStruck?.();
          for (const combatEvent of this.combat.applySpecialAttackKill()) {
            this.handleCombatEvent(combatEvent, narutoLane);
          }
          break;
        }
        case "resolved":
          this.hooks.onSpecialTeardown?.();
          if (event.outcome === "countered") {
            // Hand combat back exactly as it was, including a fresh sequence
            // to work on so the player is never left with no prompt.
            this.handleSealEvents(this.seals.start());
          }
          break;
        default:
          break;
      }
    }
  }

  private handleSealEvents(events: SealEvent[]): void {
    for (const event of events) {
      if (event.type === "sequence_completed") {
        for (const combatEvent of this.combat.launchNarutoAttack()) {
          this.handleCombatEvent(combatEvent, this.combat.obitoCurrentLane);
        }
      } else if (event.type === "sequence_failed") {
        this.lastFailure = event.reason;
        this.failureFlashS = 1.2;
      }
    }
  }
}
