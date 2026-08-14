/**
 * Collision consequence + feedback timing -- naruto_run_agent_brief.md §I:
 * "Currently a collision is invisible." A hit now costs a life and drives
 * hit-stop, camera shake, a screen flash, and a brief invulnerability window.
 *
 * Pure timing/state logic, no Three.js and no DOM, so the rules (i-frames,
 * hit-stop duration, life loss) are unit-testable without a renderer. The
 * scene layer reads `timeScale` and the `onHit` signal to actually shake the
 * camera and flash the HUD.
 */

/** Simulation freeze on impact -- brief §I asks for 60-100ms. */
export const HIT_STOP_S = 0.09;
/** Grace period after a hit during which further contact is ignored, so one
 * obstacle can't drain every life across consecutive frames of overlap. */
export const INVULNERABLE_S = 1.1;
export const MAX_LIVES = 3;

export class HitFeedback {
  private lives = MAX_LIVES;
  private hitStopRemaining = 0;
  private invulnerableRemaining = 0;
  private justHit = false;

  /**
   * Report a collision this frame. Returns true if it actually landed (i.e.
   * wasn't absorbed by invulnerability), so the caller only shakes/flashes
   * on a real hit.
   */
  registerHit(): boolean {
    if (this.invulnerableRemaining > 0 || this.lives <= 0) {
      return false;
    }
    this.lives -= 1;
    this.hitStopRemaining = HIT_STOP_S;
    this.invulnerableRemaining = INVULNERABLE_S;
    this.justHit = true;
    return true;
  }

  update(dt: number): void {
    this.justHit = false;
    this.hitStopRemaining = Math.max(0, this.hitStopRemaining - dt);
    this.invulnerableRemaining = Math.max(0, this.invulnerableRemaining - dt);
  }

  /**
   * Multiplier the scene should apply to its own delta time. Zero during
   * hit-stop: the world freezes for a beat so the impact registers, which is
   * what makes a hit *feel* like a hit rather than a number changing.
   */
  get timeScale(): number {
    return this.hitStopRemaining > 0 ? 0 : 1;
  }

  get isHitStopped(): boolean {
    return this.hitStopRemaining > 0;
  }

  get isInvulnerable(): boolean {
    return this.invulnerableRemaining > 0;
  }

  /** True only on the frame a hit landed. */
  get wasJustHit(): boolean {
    return this.justHit;
  }

  get remainingLives(): number {
    return this.lives;
  }

  get isDefeated(): boolean {
    return this.lives <= 0;
  }

  reset(): void {
    this.lives = MAX_LIVES;
    this.hitStopRemaining = 0;
    this.invulnerableRemaining = 0;
    this.justHit = false;
  }
}
