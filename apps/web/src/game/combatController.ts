/**
 * The Obito duel's rules and timing -- Feature Brief 02 §5.3-§5.4.
 *
 * Pure logic: no Three.js, no DOM, no timers of its own. It advances on a
 * delta and emits typed events the scene layer turns into meshes, VFX and
 * camera shake. Keeping it pure is what makes the fairness rules -- the
 * telegraph window, "moving out of the lane avoids damage", the damage
 * curve, and the positional dodge -- testable without rendering a frame.
 *
 * Damage (resolved with the project owner, brief §2.1): Naruto has 100 HP
 * and takes 60/20/20; Obito has 120 HP and takes 60/40/20. Both fall on the
 * third hit. HP resets each encounter (§2.3), and none of this touches the
 * running game's obstacle lives, which stay a separate system (§2.2).
 */

import {
  NARUTO_DAMAGE_SEQUENCE,
  NARUTO_MAX_HP,
  NARUTO_PROJECTILE_TRAVEL_S,
  OBITO_ATTACK_INTERVAL_S,
  OBITO_DAMAGE_SEQUENCE,
  OBITO_DODGE_TELEGRAPH_S,
  OBITO_LANE_SLIDE_S,
  OBITO_MAX_HP,
  OBITO_PROJECTILE_TRAVEL_S,
  OBITO_TELEGRAPH_S,
} from "../config/bossConfig";
import { LANE_COUNT, type LaneIndex } from "../config/gameConfig";

export type CombatEvent =
  /** Obito has begun winding up; `lane` is where the marker shows. */
  | { type: "obito_telegraph"; lane: LaneIndex }
  /** The wind-up finished and the projectile is away, aimed at `lane`. */
  | { type: "obito_fire"; lane: LaneIndex }
  /** The projectile arrived. `hit` false means Naruto vacated in time. */
  | { type: "obito_impact"; lane: LaneIndex; hit: boolean; damage: number }
  /** Obito is about to slide lanes (short lean before he moves). */
  | { type: "obito_dodge_telegraph"; from: LaneIndex; to: LaneIndex }
  | { type: "obito_move"; from: LaneIndex; to: LaneIndex }
  /** Naruto's attack is away, aimed at the lane Obito occupied on release. */
  | { type: "naruto_fire"; lane: LaneIndex }
  /** `hit` false means Obito had already moved out of the targeted lane. */
  | { type: "naruto_impact"; lane: LaneIndex; hit: boolean; damage: number }
  | { type: "naruto_defeated" }
  | { type: "obito_defeated" };

type Phase =
  | { kind: "waiting"; remaining: number }
  | { kind: "repositioning"; remaining: number; to: LaneIndex; telegraphing: boolean }
  | { kind: "telegraphing"; remaining: number; lane: LaneIndex }
  | { kind: "projectile"; remaining: number; lane: LaneIndex };

interface NarutoProjectile {
  remaining: number;
  targetLane: LaneIndex;
}

export class CombatController {
  readonly narutoMaxHp = NARUTO_MAX_HP;
  readonly obitoMaxHp = OBITO_MAX_HP;

  private narutoHp = NARUTO_MAX_HP;
  private obitoHp = OBITO_MAX_HP;
  private narutoHitsTaken = 0;
  private obitoHitsTaken = 0;

  private obitoLane: LaneIndex = 1;
  private phase: Phase;
  private narutoProjectile: NarutoProjectile | null = null;
  private finished = false;

  private readonly attackInterval: number;

  /**
   * @param encounterIndex 0 for the first encounter, 1 for the second (which
   *   attacks faster).
   * @param random injectable for deterministic tests.
   */
  constructor(
    encounterIndex: number,
    private readonly random: () => number = Math.random,
  ) {
    this.attackInterval =
      OBITO_ATTACK_INTERVAL_S[Math.min(encounterIndex, OBITO_ATTACK_INTERVAL_S.length - 1)] ??
      OBITO_ATTACK_INTERVAL_S[0];
    // Open with a full interval so the player gets a beat before the first
    // wind-up rather than being attacked the instant control is handed over.
    this.phase = { kind: "waiting", remaining: this.attackInterval };
  }

  get naruto(): { hp: number; max: number } {
    return { hp: this.narutoHp, max: this.narutoMaxHp };
  }

  get obito(): { hp: number; max: number } {
    return { hp: this.obitoHp, max: this.obitoMaxHp };
  }

  get obitoCurrentLane(): LaneIndex {
    return this.obitoLane;
  }

  get isFinished(): boolean {
    return this.finished;
  }

  /** The lane Obito is telegraphing, or null when he isn't winding up.
   * Drives the ground marker the player reads to dodge. */
  get telegraphedLane(): LaneIndex | null {
    return this.phase.kind === "telegraphing" ? this.phase.lane : null;
  }

  /**
   * True while an attack from either side is mid-flight or mid-wind-up.
   *
   * Added for Feature Brief 04 §3: Obito's special attack must never fire on
   * top of an unresolved normal attack. Read-only -- it changes nothing about
   * the approved combat loop.
   */
  get isResolvingAttack(): boolean {
    return (
      this.phase.kind === "telegraphing" ||
      this.phase.kind === "projectile" ||
      this.narutoProjectile !== null
    );
  }

  /**
   * The special attack connecting -- Feature Brief 04 §2.5.
   *
   * It bypasses the HP system entirely: a player at full health dies. The
   * bar is set to 0 rather than decremented so it *drains* to empty (§4.4),
   * which is what visually explains a death at 100/100. Any in-flight Naruto
   * projectile is dropped, since the run is over.
   */
  applySpecialAttackKill(): CombatEvent[] {
    if (this.finished) {
      return [];
    }
    this.narutoProjectile = null;
    this.narutoHp = 0;
    this.finished = true;
    return [{ type: "naruto_defeated" }];
  }

  /**
   * Test-only: chip Obito's HP without going through a real attack, so an
   * end-to-end test can reach the special attack's HP trigger in one frame
   * instead of fighting a whole duel. Nothing in the game calls this.
   */
  applyDebugObitoDamage(amount: number): void {
    this.obitoHp = Math.max(0, this.obitoHp - amount);
  }

  /**
   * Test-only: park Obito's ordinary attack loop indefinitely.
   *
   * The special-attack tests need to observe a 3.2s warning window and its
   * resolution in isolation, but an unattended Naruto takes 60/20/20 from
   * the normal loop and dies partway through -- which made those tests
   * intermittently fail for a reason that had nothing to do with what they
   * were checking. Parking resets the phase rather than freezing it in
   * place, so no wind-up is left half-finished (which would read as an
   * unresolved attack and block the special from arming at all).
   */
  setDebugOrdinaryAttacksPaused(paused: boolean): void {
    this.narutoProjectile = null;
    this.phase = paused
      ? { kind: "waiting", remaining: Number.POSITIVE_INFINITY }
      : { kind: "waiting", remaining: this.attackInterval };
  }

  /** 0..1 progress through the current telegraph, for the marker's fill. */
  get telegraphProgress(): number {
    if (this.phase.kind !== "telegraphing") return 0;
    return 1 - this.phase.remaining / OBITO_TELEGRAPH_S;
  }

  /**
   * Fire Naruto's attack at whichever lane Obito occupies right now. Whether
   * it lands is decided on *impact*, not here -- that is what gives Obito's
   * dodge meaning and makes the player's positioning matter (§5.3).
   */
  launchNarutoAttack(): CombatEvent[] {
    if (this.finished || this.narutoProjectile) {
      return [];
    }
    const lane = this.obitoLane;
    this.narutoProjectile = { remaining: NARUTO_PROJECTILE_TRAVEL_S, targetLane: lane };
    return [{ type: "naruto_fire", lane }];
  }

  /**
   * Advance combat. `narutoLane` is read live so that vacating a
   * telegraphed lane before impact avoids the hit.
   */
  update(dt: number, narutoLane: LaneIndex): CombatEvent[] {
    if (this.finished) {
      return [];
    }
    const events: CombatEvent[] = [];
    this.advanceObito(dt, narutoLane, events);
    this.advanceNarutoProjectile(dt, events);
    return events;
  }

  private advanceObito(dt: number, narutoLane: LaneIndex, events: CombatEvent[]): void {
    const phase = this.phase;
    phase.remaining -= dt;
    if (phase.remaining > 0) {
      return;
    }
    // Carry the overshoot into the next phase so a long frame doesn't lose
    // time (and so the telegraph can never be shortened by a frame spike).
    const overshoot = -phase.remaining;

    switch (phase.kind) {
      case "waiting": {
        // Between attacks Obito repositions, which is also his dodge: being
        // in a different lane from Naruto's next attack is what saves him.
        const to = this.pickLane(narutoLane);
        if (to === this.obitoLane) {
          this.phase = { kind: "telegraphing", remaining: OBITO_TELEGRAPH_S - overshoot, lane: narutoLane };
          events.push({ type: "obito_telegraph", lane: narutoLane });
        } else {
          this.phase = {
            kind: "repositioning",
            remaining: OBITO_DODGE_TELEGRAPH_S - overshoot,
            to,
            telegraphing: true,
          };
          events.push({ type: "obito_dodge_telegraph", from: this.obitoLane, to });
        }
        break;
      }
      case "repositioning": {
        if (phase.telegraphing) {
          // Lean finished; now actually slide.
          events.push({ type: "obito_move", from: this.obitoLane, to: phase.to });
          this.obitoLane = phase.to;
          this.phase = {
            kind: "repositioning",
            remaining: OBITO_LANE_SLIDE_S - overshoot,
            to: phase.to,
            telegraphing: false,
          };
        } else {
          // Slide finished; wind up at Naruto's *current* lane.
          this.phase = { kind: "telegraphing", remaining: OBITO_TELEGRAPH_S - overshoot, lane: narutoLane };
          events.push({ type: "obito_telegraph", lane: narutoLane });
        }
        break;
      }
      case "telegraphing": {
        // Brief §5.3: the attack targets Naruto's lane at the moment of
        // *firing*, so moving after the telegraph is what dodges it.
        const lane = narutoLane;
        this.phase = { kind: "projectile", remaining: OBITO_PROJECTILE_TRAVEL_S - overshoot, lane };
        events.push({ type: "obito_fire", lane });
        break;
      }
      case "projectile": {
        const hit = narutoLane === phase.lane;
        const damage = hit ? this.nextDamage(OBITO_DAMAGE_SEQUENCE, this.narutoHitsTaken) : 0;
        if (hit) {
          this.narutoHitsTaken += 1;
          this.narutoHp = Math.max(0, this.narutoHp - damage);
        }
        events.push({ type: "obito_impact", lane: phase.lane, hit, damage });
        if (this.narutoHp <= 0) {
          this.finished = true;
          events.push({ type: "naruto_defeated" });
          return;
        }
        this.phase = { kind: "waiting", remaining: this.attackInterval - overshoot };
        break;
      }
    }
  }

  private advanceNarutoProjectile(dt: number, events: CombatEvent[]): void {
    const projectile = this.narutoProjectile;
    if (!projectile) {
      return;
    }
    projectile.remaining -= dt;
    if (projectile.remaining > 0) {
      return;
    }
    this.narutoProjectile = null;

    // Positional dodge (§5.3): Obito is hit only if he's still in the lane
    // the attack was aimed at. Flat random chance was the alternative; this
    // makes his lane movement, and the player's timing, actually matter.
    const hit = this.obitoLane === projectile.targetLane;
    const damage = hit ? this.nextDamage(NARUTO_DAMAGE_SEQUENCE, this.obitoHitsTaken) : 0;
    if (hit) {
      this.obitoHitsTaken += 1;
      this.obitoHp = Math.max(0, this.obitoHp - damage);
    }
    events.push({ type: "naruto_impact", lane: projectile.targetLane, hit, damage });
    if (this.obitoHp <= 0) {
      this.finished = true;
      events.push({ type: "obito_defeated" });
    }
  }

  /** Damage curves are fixed sequences; past the end, repeat the last value
   * so an unexpected extra hit can never deal zero. */
  private nextDamage(sequence: readonly number[], hitsSoFar: number): number {
    return sequence[Math.min(hitsSoFar, sequence.length - 1)] ?? 0;
  }

  /** Choose where Obito goes next: usually away from Naruto (so Naruto must
   * reposition to land an attack), occasionally into Naruto's lane to keep
   * him from being trivially predictable. */
  private pickLane(narutoLane: LaneIndex): LaneIndex {
    const stayAggressive = this.random() < 0.25;
    const candidates: LaneIndex[] = [];
    for (let lane = 0; lane < LANE_COUNT; lane += 1) {
      const index = lane as LaneIndex;
      if (stayAggressive ? index === narutoLane : index !== narutoLane) {
        candidates.push(index);
      }
    }
    if (candidates.length === 0) {
      return this.obitoLane;
    }
    return candidates[Math.floor(this.random() * candidates.length)] ?? this.obitoLane;
  }
}
