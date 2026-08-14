/**
 * Explicit game state machine -- Feature Brief 02 §3.
 *
 * This exists to keep boss logic from leaking into running logic. Only
 * `RUNNING` may execute the existing, approved gameplay path; every other
 * state gates it. The capability table below is the single source of truth
 * for "what is allowed right now", so callers ask questions like
 * `caps.worldScroll` rather than re-deriving rules from the state name.
 *
 * Pure and framework-free: no Three.js, no DOM, no timers. That keeps the
 * transition rules unit-testable, which matters because a stuck or skipped
 * transition here would strand the player in a frozen world.
 */

export type GameState =
  | "RUNNING"
  | "BOSS_INTRO"
  | "BOSS_COMBAT"
  | "BOSS_DEFEATED"
  | "GAME_OVER";

/** What each state permits. Mirrors the table in brief §3 exactly. */
export interface StateCapabilities {
  /** Existing running gameplay (obstacle collision, distance, run cycle). */
  runningGameplay: boolean;
  /** World scroll: full speed, easing, or stopped. */
  worldScroll: "on" | "decelerating" | "stopped" | "accelerating" | "frozen";
  obstacleSpawn: boolean;
  laneInput: boolean;
  jumpInput: boolean;
  /** Hand-seal gestures -- only ever active in combat (brief §6: no CPU
   * cost and no misfires outside it). */
  gestureInput: boolean;
}

const CAPABILITIES: Record<GameState, StateCapabilities> = {
  RUNNING: {
    runningGameplay: true,
    worldScroll: "on",
    obstacleSpawn: true,
    laneInput: true,
    jumpInput: true,
    gestureInput: false,
  },
  BOSS_INTRO: {
    runningGameplay: false,
    worldScroll: "decelerating",
    obstacleSpawn: false,
    laneInput: false,
    jumpInput: false,
    gestureInput: false,
  },
  BOSS_COMBAT: {
    runningGameplay: false,
    worldScroll: "stopped",
    obstacleSpawn: false,
    // Lane changes stay fully enabled -- dodging Obito reuses the exact
    // same input and timing as the running game (brief §5.3).
    laneInput: true,
    jumpInput: false,
    gestureInput: true,
  },
  BOSS_DEFEATED: {
    runningGameplay: false,
    worldScroll: "accelerating",
    obstacleSpawn: false,
    laneInput: false,
    jumpInput: false,
    gestureInput: false,
  },
  GAME_OVER: {
    runningGameplay: false,
    worldScroll: "frozen",
    obstacleSpawn: false,
    laneInput: false,
    jumpInput: false,
    gestureInput: false,
  },
};

/** Legal transitions, straight from the brief's diagram. */
const TRANSITIONS: Record<GameState, readonly GameState[]> = {
  RUNNING: ["BOSS_INTRO", "GAME_OVER"],
  BOSS_INTRO: ["BOSS_COMBAT", "GAME_OVER"],
  BOSS_COMBAT: ["BOSS_DEFEATED", "GAME_OVER"],
  BOSS_DEFEATED: ["RUNNING", "GAME_OVER"],
  // Only the Try Again button leaves GAME_OVER.
  GAME_OVER: ["RUNNING"],
};

export type GameStateListener = (next: GameState, previous: GameState) => void;

export class GameStateMachine {
  private current: GameState = "RUNNING";
  private readonly listeners = new Set<GameStateListener>();

  get state(): GameState {
    return this.current;
  }

  get capabilities(): StateCapabilities {
    return CAPABILITIES[this.current];
  }

  /** True only in RUNNING -- the one gate the existing game loop checks. */
  get isRunning(): boolean {
    return this.current === "RUNNING";
  }

  /** True while an encounter owns the screen, in any of its phases. */
  get isInEncounter(): boolean {
    return (
      this.current === "BOSS_INTRO" ||
      this.current === "BOSS_COMBAT" ||
      this.current === "BOSS_DEFEATED"
    );
  }

  canTransitionTo(next: GameState): boolean {
    return TRANSITIONS[this.current].includes(next);
  }

  /**
   * Attempt a transition. Returns false and does nothing for an illegal one
   * rather than throwing: a late timer firing after the player already died
   * is a normal race, not a programming error, and it must not be able to
   * drag the game out of GAME_OVER.
   */
  transitionTo(next: GameState): boolean {
    if (next === this.current || !this.canTransitionTo(next)) {
      return false;
    }
    const previous = this.current;
    this.current = next;
    for (const listener of this.listeners) {
      listener(next, previous);
    }
    return true;
  }

  onChange(listener: GameStateListener): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  /** Hard reset for Try Again. Does not notify listeners -- the restart path
   * rebuilds everything explicitly, and firing a transition here would make
   * it ambiguous whether listeners had already run. */
  reset(): void {
    this.current = "RUNNING";
  }
}

export function capabilitiesOf(state: GameState): StateCapabilities {
  return CAPABILITIES[state];
}
