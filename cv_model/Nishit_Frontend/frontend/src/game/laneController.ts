/**
 * Real player movement controller -- game_implementation_plan.md Phase D:
 * "laneController (relative one-step movement, jump arc, coyote/forgiveness
 * window)". Framework-agnostic (plain numbers in/out, no Three.js) so its
 * timing math is unit-testable and provably frame-rate independent, exactly
 * like DebugPlayerRig was in Phase C -- this supersedes that file.
 *
 * scene/playerCharacter.ts reads this controller's output (lane X, feet
 * height, airborne flag) to position the visual model and choose an
 * animation pose. This module never touches a mesh.
 */

import {
  JUMP_ANTICIPATION_S,
  JUMP_DURATION_S,
  JUMP_HEIGHT,
  JUMP_INPUT_BUFFER_S,
  JUMP_LANDING_S,
  LANE_CHANGE_DURATION_S,
  LANE_COUNT,
  LANE_LEAN_MAX_RAD,
  LANE_X,
  type LaneIndex,
} from "../config/gameConfig";

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function easeOutCubic(t: number): number {
  const inverse = 1 - t;
  return 1 - inverse * inverse * inverse;
}

function clampLane(lane: number): LaneIndex {
  return Math.min(LANE_COUNT - 1, Math.max(0, lane)) as LaneIndex;
}

/**
 * Jump height over normalised airtime. Deliberately *not* a symmetric
 * physics parabola (brief §D: "fast rise / slower hang / fast fall -- games
 * are not physics simulators"): raising sin() to a power below 1 widens the
 * peak and steepens both ends, which gives the arc a readable hang time
 * without lengthening the jump.
 */
function jumpArc(t: number): number {
  return Math.pow(Math.sin(Math.PI * t), 0.72) * JUMP_HEIGHT;
}

/**
 * A jump runs anticipation -> airborne -> landing (brief §D). Anticipation
 * and landing are short, ground-contact phases: the feet stay planted and
 * the *body* compresses (see `crouchAmount`), so they read as a crouch and
 * an impact absorption rather than moving the character.
 */
export type JumpPhase = "grounded" | "anticipation" | "airborne" | "landing";

export class LaneController {
  private lane: LaneIndex = 1;
  private laneStartX: number;
  private laneTargetX: number;
  private laneElapsed = LANE_CHANGE_DURATION_S; // starts already settled
  private laneDirection: -1 | 0 | 1 = 0;

  private phase: JumpPhase = "grounded";
  private phaseElapsed = 0;
  private groundOffsetY = 0;
  private crouch = 0;

  /** Seconds since a jump was requested while ineligible (mid-jump); null
   * when there is no pending buffered request. */
  private bufferedJumpAge: number | null = null;

  constructor() {
    this.laneStartX = LANE_X[this.lane];
    this.laneTargetX = LANE_X[this.lane];
  }

  /** Move one lane step, clamped at the track edges. A press while already
   * mid-transition retargets smoothly from the character's current position
   * rather than snapping or queuing. */
  moveLane(direction: -1 | 1): void {
    const next = clampLane(this.lane + direction);
    if (next === this.lane) {
      return;
    }
    this.lane = next;
    this.laneStartX = this.currentX;
    this.laneTargetX = LANE_X[next];
    this.laneElapsed = 0;
    this.laneDirection = direction;
  }

  /** Request a jump. If grounded, the anticipation crouch starts
   * immediately. Otherwise the request is buffered for JUMP_INPUT_BUFFER_S
   * and consumed the instant the character is eligible again, rather than
   * being silently dropped -- which matters doubly here, since Phase E's CV
   * input adds its own ~150-200ms of latency on top of human reaction time. */
  requestJump(): void {
    if (this.phase === "grounded") {
      this.startJump();
    } else {
      this.bufferedJumpAge = 0;
    }
  }

  private startJump(): void {
    this.phase = "anticipation";
    this.phaseElapsed = 0;
    this.bufferedJumpAge = null;
  }

  update(dt: number): void {
    if (this.laneElapsed < LANE_CHANGE_DURATION_S) {
      this.laneElapsed = Math.min(LANE_CHANGE_DURATION_S, this.laneElapsed + dt);
      if (this.laneElapsed >= LANE_CHANGE_DURATION_S) {
        this.laneDirection = 0;
      }
    }

    // Age the buffered request from the moment it was made, independent of
    // phase -- a request buffered early in a long jump arc must still be able
    // to expire before landing, not just be checked once on impact with
    // whatever stale age it happened to have.
    if (this.bufferedJumpAge !== null) {
      this.bufferedJumpAge += dt;
      if (this.bufferedJumpAge > JUMP_INPUT_BUFFER_S) {
        this.bufferedJumpAge = null; // expired unconsumed
      }
    }

    this.advanceJump(dt);
  }

  private advanceJump(dt: number): void {
    if (this.phase === "grounded") {
      this.groundOffsetY = 0;
      this.crouch = 0;
      return;
    }

    this.phaseElapsed += dt;

    if (this.phase === "anticipation") {
      if (this.phaseElapsed >= JUMP_ANTICIPATION_S) {
        this.phase = "airborne";
        this.phaseElapsed -= JUMP_ANTICIPATION_S; // carry the remainder, don't drop it
      } else {
        this.groundOffsetY = 0;
        this.crouch = this.phaseElapsed / JUMP_ANTICIPATION_S; // deepening crouch
        return;
      }
    }

    if (this.phase === "airborne") {
      if (this.phaseElapsed >= JUMP_DURATION_S) {
        this.phase = "landing";
        this.phaseElapsed -= JUMP_DURATION_S;
      } else {
        this.groundOffsetY = jumpArc(this.phaseElapsed / JUMP_DURATION_S);
        this.crouch = 0;
        return;
      }
    }

    if (this.phase === "landing") {
      if (this.phaseElapsed >= JUMP_LANDING_S) {
        this.phase = "grounded";
        this.phaseElapsed = 0;
        this.groundOffsetY = 0;
        this.crouch = 0;
        if (this.bufferedJumpAge !== null) {
          this.startJump(); // still within the window -- consume it on landing
        }
      } else {
        this.groundOffsetY = 0;
        // Compress hard on impact, then spring back out.
        this.crouch = 1 - this.phaseElapsed / JUMP_LANDING_S;
      }
    }
  }

  get currentX(): number {
    if (this.laneElapsed >= LANE_CHANGE_DURATION_S) {
      return this.laneTargetX;
    }
    const t = easeOutCubic(this.laneElapsed / LANE_CHANGE_DURATION_S);
    return lerp(this.laneStartX, this.laneTargetX, t);
  }

  /** Height of the feet above ground; 0 whenever in ground contact. */
  get feetHeight(): number {
    return this.groundOffsetY;
  }

  /**
   * Body compression, 0 (upright) to 1 (fully crouched), during the
   * anticipation and landing phases. The renderer squashes the model on Y
   * by this amount; the feet do not move, so the character stays grounded.
   */
  get crouchAmount(): number {
    return this.crouch;
  }

  /**
   * Bank angle in radians, signed: **positive means leaning toward +X
   * (right)**. Peaks mid-transition and eases back to zero, so the character
   * banks into a lane change like a real runner instead of sliding flat.
   * The renderer is responsible for the sign convention of whichever axis it
   * applies this to (a roll about +Z tilts the body toward -X, so it negates).
   */
  get leanRad(): number {
    if (this.laneDirection === 0 || this.laneElapsed >= LANE_CHANGE_DURATION_S) {
      return 0;
    }
    const t = this.laneElapsed / LANE_CHANGE_DURATION_S;
    return this.laneDirection * LANE_LEAN_MAX_RAD * Math.sin(Math.PI * t);
  }

  /** True only while genuinely off the ground -- the anticipation crouch and
   * landing absorption are ground-contact phases, so an obstacle is not
   * cleared during them. */
  get isAirborne(): boolean {
    return this.phase === "airborne";
  }

  get jumpPhase(): JumpPhase {
    return this.phase;
  }

  get isTransitioningLane(): boolean {
    return this.laneElapsed < LANE_CHANGE_DURATION_S;
  }

  get currentLane(): LaneIndex {
    return this.lane;
  }

  get hasBufferedJump(): boolean {
    return this.bufferedJumpAge !== null;
  }

  /**
   * Return to a clean centre-lane, grounded, settled state.
   *
   * Added for Feature Brief 02 §4.5 (Try Again must produce a completely
   * clean run: "Naruto -> centre lane, grounded"). Purely additive -- no
   * existing behaviour changed, and nothing calls this during normal play.
   */
  reset(): void {
    this.lane = 1;
    this.laneStartX = LANE_X[1];
    this.laneTargetX = LANE_X[1];
    this.laneElapsed = LANE_CHANGE_DURATION_S;
    this.laneDirection = 0;
    this.phase = "grounded";
    this.phaseElapsed = 0;
    this.groundOffsetY = 0;
    this.crouch = 0;
    this.bufferedJumpAge = null;
  }
}
