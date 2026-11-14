import { describe, expect, it, vi } from "vitest";
import { GameStateMachine, capabilitiesOf, type GameState } from "../../src/game/gameState";

const ALL_STATES: GameState[] = [
  "RUNNING",
  "BOSS_INTRO",
  "BOSS_COMBAT",
  "BOSS_DEFEATED",
  "GAME_OVER",
];

describe("GameStateMachine capabilities", () => {
  it("starts in RUNNING with the full running game enabled", () => {
    const machine = new GameStateMachine();
    expect(machine.state).toBe("RUNNING");
    expect(machine.isRunning).toBe(true);
    expect(machine.capabilities.runningGameplay).toBe(true);
  });

  it("only RUNNING enables the existing running gameplay path", () => {
    // Brief §3: "Only RUNNING should ever run the existing gameplay code
    // path." This is the guard that keeps boss logic from leaking in.
    for (const state of ALL_STATES) {
      expect(capabilitiesOf(state).runningGameplay).toBe(state === "RUNNING");
    }
  });

  it("suspends obstacle spawning in every non-running state", () => {
    for (const state of ALL_STATES) {
      expect(capabilitiesOf(state).obstacleSpawn).toBe(state === "RUNNING");
    }
  });

  it("enables hand-seal gestures only in combat", () => {
    // Brief §6: gesture recognition must be inactive outside BOSS_COMBAT --
    // no CPU cost and no chance of misfires during normal running.
    for (const state of ALL_STATES) {
      expect(capabilitiesOf(state).gestureInput).toBe(state === "BOSS_COMBAT");
    }
  });

  it("keeps lane input available during combat but not during cutscenes", () => {
    expect(capabilitiesOf("BOSS_COMBAT").laneInput).toBe(true);
    expect(capabilitiesOf("BOSS_INTRO").laneInput).toBe(false);
    expect(capabilitiesOf("BOSS_DEFEATED").laneInput).toBe(false);
    expect(capabilitiesOf("GAME_OVER").laneInput).toBe(false);
  });

  it("disables jumping outside RUNNING", () => {
    for (const state of ALL_STATES) {
      expect(capabilitiesOf(state).jumpInput).toBe(state === "RUNNING");
    }
  });

  it("maps each state to the world-scroll behaviour the brief specifies", () => {
    expect(capabilitiesOf("RUNNING").worldScroll).toBe("on");
    expect(capabilitiesOf("BOSS_INTRO").worldScroll).toBe("decelerating");
    expect(capabilitiesOf("BOSS_COMBAT").worldScroll).toBe("stopped");
    expect(capabilitiesOf("BOSS_DEFEATED").worldScroll).toBe("accelerating");
    expect(capabilitiesOf("GAME_OVER").worldScroll).toBe("frozen");
  });
});

describe("GameStateMachine transitions", () => {
  it("follows the encounter happy path", () => {
    const machine = new GameStateMachine();
    expect(machine.transitionTo("BOSS_INTRO")).toBe(true);
    expect(machine.transitionTo("BOSS_COMBAT")).toBe(true);
    expect(machine.transitionTo("BOSS_DEFEATED")).toBe(true);
    expect(machine.transitionTo("RUNNING")).toBe(true);
  });

  it("allows GAME_OVER from running and from every encounter phase", () => {
    for (const from of ["RUNNING", "BOSS_INTRO", "BOSS_COMBAT", "BOSS_DEFEATED"] as GameState[]) {
      const machine = new GameStateMachine();
      // Walk to `from` legally.
      const path: Record<string, GameState[]> = {
        RUNNING: [],
        BOSS_INTRO: ["BOSS_INTRO"],
        BOSS_COMBAT: ["BOSS_INTRO", "BOSS_COMBAT"],
        BOSS_DEFEATED: ["BOSS_INTRO", "BOSS_COMBAT", "BOSS_DEFEATED"],
      };
      for (const step of path[from]!) machine.transitionTo(step);
      expect(machine.state).toBe(from);
      expect(machine.transitionTo("GAME_OVER")).toBe(true);
    }
  });

  it("rejects illegal transitions without throwing or changing state", () => {
    const machine = new GameStateMachine();
    // Can't skip the intro straight into combat.
    expect(machine.transitionTo("BOSS_COMBAT")).toBe(false);
    expect(machine.state).toBe("RUNNING");
  });

  it("only Try Again (RUNNING) can leave GAME_OVER", () => {
    const machine = new GameStateMachine();
    machine.transitionTo("GAME_OVER");

    // A late timer from the dead run must not be able to drag the game back
    // into an encounter -- that's a normal race, not a programming error.
    expect(machine.transitionTo("BOSS_COMBAT")).toBe(false);
    expect(machine.transitionTo("BOSS_INTRO")).toBe(false);
    expect(machine.transitionTo("BOSS_DEFEATED")).toBe(false);
    expect(machine.state).toBe("GAME_OVER");

    expect(machine.transitionTo("RUNNING")).toBe(true);
  });

  it("treats a transition to the current state as a no-op", () => {
    const machine = new GameStateMachine();
    expect(machine.transitionTo("RUNNING")).toBe(false);
    expect(machine.state).toBe("RUNNING");
  });

  it("notifies listeners with both the next and previous state", () => {
    const machine = new GameStateMachine();
    const listener = vi.fn();
    machine.onChange(listener);

    machine.transitionTo("BOSS_INTRO");

    expect(listener).toHaveBeenCalledWith("BOSS_INTRO", "RUNNING");
  });

  it("does not notify listeners for a rejected transition", () => {
    const machine = new GameStateMachine();
    const listener = vi.fn();
    machine.onChange(listener);

    machine.transitionTo("BOSS_COMBAT"); // illegal from RUNNING

    expect(listener).not.toHaveBeenCalled();
  });

  it("unsubscribes cleanly", () => {
    const machine = new GameStateMachine();
    const listener = vi.fn();
    const off = machine.onChange(listener);
    off();

    machine.transitionTo("BOSS_INTRO");

    expect(listener).not.toHaveBeenCalled();
  });

  it("isInEncounter covers exactly the three encounter phases", () => {
    const machine = new GameStateMachine();
    expect(machine.isInEncounter).toBe(false);
    machine.transitionTo("BOSS_INTRO");
    expect(machine.isInEncounter).toBe(true);
    machine.transitionTo("BOSS_COMBAT");
    expect(machine.isInEncounter).toBe(true);
    machine.transitionTo("BOSS_DEFEATED");
    expect(machine.isInEncounter).toBe(true);
    machine.transitionTo("RUNNING");
    expect(machine.isInEncounter).toBe(false);
  });

  it("reset() returns to RUNNING from anywhere", () => {
    const machine = new GameStateMachine();
    machine.transitionTo("GAME_OVER");
    machine.reset();
    expect(machine.state).toBe("RUNNING");
  });
});
