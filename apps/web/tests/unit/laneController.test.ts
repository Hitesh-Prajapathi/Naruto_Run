import { describe, expect, it } from "vitest";
import { LaneController } from "../../src/game/laneController";
import {
  JUMP_HEIGHT,
  JUMP_ANTICIPATION_S,
  JUMP_DURATION_S,
  JUMP_INPUT_BUFFER_S,
  JUMP_LANDING_S,
  LANE_CHANGE_DURATION_S,
  LANE_LEAN_MAX_RAD,
  LANE_X,
} from "../../src/config/gameConfig";

/** Total wall time of one complete jump, across all three phases. */
const FULL_JUMP_S = JUMP_ANTICIPATION_S + JUMP_DURATION_S + JUMP_LANDING_S;

describe("LaneController lane movement", () => {
  it("starts centered in the middle lane", () => {
    const controller = new LaneController();
    expect(controller.currentLane).toBe(1);
    expect(controller.currentX).toBeCloseTo(LANE_X[1], 5);
  });

  it("reaches the target lane's X exactly at LANE_CHANGE_DURATION_S", () => {
    const controller = new LaneController();
    controller.moveLane(1);
    controller.update(LANE_CHANGE_DURATION_S);
    expect(controller.currentX).toBeCloseTo(LANE_X[2], 5);
    expect(controller.currentLane).toBe(2);
  });

  it("is partway between lanes mid-transition", () => {
    const controller = new LaneController();
    controller.moveLane(1);
    controller.update(LANE_CHANGE_DURATION_S / 2);
    expect(controller.currentX).toBeGreaterThan(LANE_X[1]);
    expect(controller.currentX).toBeLessThan(LANE_X[2]);
    expect(controller.isTransitioningLane).toBe(true);
  });

  it("clamps at the rightmost lane and ignores a further right move", () => {
    const controller = new LaneController();
    controller.moveLane(1);
    controller.update(LANE_CHANGE_DURATION_S);
    controller.moveLane(1); // no-op, already at lane 2
    controller.update(LANE_CHANGE_DURATION_S);
    expect(controller.currentLane).toBe(2);
    expect(controller.currentX).toBeCloseTo(LANE_X[2], 5);
  });

  it("clamps at the leftmost lane symmetrically", () => {
    const controller = new LaneController();
    controller.moveLane(-1);
    controller.update(LANE_CHANGE_DURATION_S);
    controller.moveLane(-1); // no-op
    controller.update(LANE_CHANGE_DURATION_S);
    expect(controller.currentLane).toBe(0);
  });

  it("retargets smoothly from the current position when pressed again mid-transition", () => {
    const controller = new LaneController();
    controller.moveLane(1); // 1 -> 2
    controller.update(LANE_CHANGE_DURATION_S / 2);
    const midX = controller.currentX;

    controller.moveLane(-1); // 2 -> 1, retarget from wherever it currently is

    expect(controller.currentX).toBeCloseTo(midX, 5); // no snap at the instant of retargeting
    controller.update(LANE_CHANGE_DURATION_S);
    expect(controller.currentX).toBeCloseTo(LANE_X[1], 5);
  });

  it("produces the same end X regardless of step granularity (frame-rate independence)", () => {
    const coarse = new LaneController();
    coarse.moveLane(1);
    coarse.update(LANE_CHANGE_DURATION_S);

    const fine = new LaneController();
    fine.moveLane(1);
    const steps = 120;
    for (let i = 0; i < steps; i += 1) {
      fine.update(LANE_CHANGE_DURATION_S / steps);
    }

    expect(fine.currentX).toBeCloseTo(coarse.currentX, 5);
  });
});

describe("LaneController lane lean", () => {
  it("is zero when settled in a lane", () => {
    const controller = new LaneController();
    expect(controller.leanRad).toBe(0);
    controller.update(1);
    expect(controller.leanRad).toBe(0);
  });

  it("banks toward +X when moving right and toward -X when moving left", () => {
    const right = new LaneController();
    right.moveLane(1);
    right.update(LANE_CHANGE_DURATION_S / 2);
    expect(right.leanRad).toBeGreaterThan(0);

    const left = new LaneController();
    left.moveLane(-1);
    left.update(LANE_CHANGE_DURATION_S / 2);
    expect(left.leanRad).toBeLessThan(0);
  });

  it("peaks mid-transition and eases back to exactly zero on arrival", () => {
    const controller = new LaneController();
    controller.moveLane(1);

    controller.update(LANE_CHANGE_DURATION_S / 2);
    const peak = Math.abs(controller.leanRad);
    expect(peak).toBeCloseTo(LANE_LEAN_MAX_RAD, 4);

    controller.update(LANE_CHANGE_DURATION_S / 2);
    expect(controller.leanRad).toBe(0);
  });

  it("never exceeds the configured maximum bank angle", () => {
    const controller = new LaneController();
    controller.moveLane(1);
    for (let i = 0; i < 40; i += 1) {
      controller.update(LANE_CHANGE_DURATION_S / 40);
      expect(Math.abs(controller.leanRad)).toBeLessThanOrEqual(LANE_LEAN_MAX_RAD + 1e-9);
    }
  });
});

describe("LaneController jump phases", () => {
  it("enters anticipation (not airborne) on request, so the crouch reads before launch", () => {
    const controller = new LaneController();
    controller.requestJump();

    expect(controller.jumpPhase).toBe("anticipation");
    expect(controller.isAirborne).toBe(false);
    expect(controller.feetHeight).toBe(0);
  });

  it("crouches during anticipation with the feet still planted", () => {
    const controller = new LaneController();
    controller.requestJump();
    controller.update(JUMP_ANTICIPATION_S * 0.8);

    expect(controller.crouchAmount).toBeGreaterThan(0);
    expect(controller.feetHeight).toBe(0); // a crouch must not lift or sink the character
  });

  it("becomes airborne after the anticipation window and rises off the ground", () => {
    const controller = new LaneController();
    controller.requestJump();
    controller.update(JUMP_ANTICIPATION_S + JUMP_DURATION_S / 2);

    expect(controller.jumpPhase).toBe("airborne");
    expect(controller.isAirborne).toBe(true);
    expect(controller.feetHeight).toBeGreaterThan(0);
    expect(controller.crouchAmount).toBe(0);
  });

  it("absorbs the landing with a crouch, back on the ground", () => {
    const controller = new LaneController();
    controller.requestJump();
    controller.update(JUMP_ANTICIPATION_S + JUMP_DURATION_S + JUMP_LANDING_S / 2);

    expect(controller.jumpPhase).toBe("landing");
    expect(controller.isAirborne).toBe(false);
    expect(controller.feetHeight).toBe(0);
    expect(controller.crouchAmount).toBeGreaterThan(0);
  });

  it("returns fully to grounded after the whole sequence", () => {
    const controller = new LaneController();
    controller.requestJump();
    controller.update(FULL_JUMP_S + 0.01);

    expect(controller.jumpPhase).toBe("grounded");
    expect(controller.feetHeight).toBe(0);
    expect(controller.crouchAmount).toBe(0);
  });

  it("carries the remainder across phase boundaries instead of dropping it", () => {
    // One big step that lands mid-airborne must give the same height as many
    // small steps -- if a boundary crossing reset the phase clock to zero,
    // these would diverge.
    const coarse = new LaneController();
    coarse.requestJump();
    coarse.update(JUMP_ANTICIPATION_S + JUMP_DURATION_S / 2);

    const fine = new LaneController();
    fine.requestJump();
    const target = JUMP_ANTICIPATION_S + JUMP_DURATION_S / 2;
    const steps = 200;
    for (let i = 0; i < steps; i += 1) {
      fine.update(target / steps);
    }

    expect(fine.feetHeight).toBeCloseTo(coarse.feetHeight, 4);
  });

  it("keeps the jump arc within the configured height", () => {
    const controller = new LaneController();
    controller.requestJump();
    let peak = 0;
    for (let i = 0; i < 300; i += 1) {
      controller.update(FULL_JUMP_S / 300);
      peak = Math.max(peak, controller.feetHeight);
    }
    expect(peak).toBeGreaterThan(0.5);
    expect(peak).toBeLessThanOrEqual(JUMP_HEIGHT + 1e-6);
  });
});

describe("LaneController jump input buffering", () => {
  it("a jump requested mid-jump is buffered, not an immediate restart", () => {
    const controller = new LaneController();
    controller.requestJump();
    controller.update(JUMP_ANTICIPATION_S + JUMP_DURATION_S / 2);
    const heightBefore = controller.feetHeight;

    controller.requestJump();
    controller.update(0.001);

    expect(controller.hasBufferedJump).toBe(true);
    expect(controller.feetHeight).toBeLessThanOrEqual(heightBefore + 0.05);
  });

  it("a request buffered just before the sequence ends fires on landing", () => {
    const controller = new LaneController();
    controller.requestJump();
    controller.update(FULL_JUMP_S - 0.005);
    controller.requestJump(); // buffered a hair before grounding

    controller.update(0.01); // crosses back to grounded

    expect(controller.jumpPhase).toBe("anticipation"); // the buffered jump took over
    expect(controller.hasBufferedJump).toBe(false);
  });

  it("a buffered request older than JUMP_INPUT_BUFFER_S expires unconsumed", () => {
    // The full jump sequence comfortably exceeds the buffer window, so a
    // request made at the very start must be long stale by the time the
    // character is eligible again.
    expect(FULL_JUMP_S).toBeGreaterThan(JUMP_INPUT_BUFFER_S);

    const controller = new LaneController();
    controller.requestJump();
    controller.requestJump(); // buffered at age 0

    controller.update(FULL_JUMP_S + 0.01);

    expect(controller.hasBufferedJump).toBe(false);
    expect(controller.jumpPhase).toBe("grounded"); // no second jump triggered
  });

  it("does not buffer when the request is honoured immediately", () => {
    const controller = new LaneController();
    controller.requestJump();
    expect(controller.jumpPhase).toBe("anticipation");
    expect(controller.hasBufferedJump).toBe(false);
  });
});
