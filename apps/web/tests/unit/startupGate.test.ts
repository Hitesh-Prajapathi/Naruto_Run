/**
 * Feature Brief 05 §4 — the startup gate, driven without a camera.
 *
 * The gate is pure logic precisely so these rules can be checked here rather
 * than by standing in front of a webcam: the 2s sustained-detection hold, the
 * three distinguishable failures, the keyboard escape, and the manual
 * recalibrate.
 */

import { describe, expect, it } from "vitest";
import { StartupGate, type GateContext } from "../../src/game/startupGate";
import { DEFAULT_BODY_TUNING } from "../../src/input/bodyTuning";
import type { CameraSourceStatus } from "../../src/input/cameraPoseSource";
import { FixtureBuilder } from "../../src/input/poseFixtures";
import type { PoseSample } from "../../src/input/poseAdapter";

const GOOD_SAMPLE = new FixtureBuilder().add(1).build()[0] as PoseSample;
const ABSENT_SAMPLE = new FixtureBuilder().add(1, { personPresent: false }).build()[0] as PoseSample;

function context(overrides: Partial<GateContext> = {}): GateContext {
  return {
    status: "streaming",
    sample: GOOD_SAMPLE,
    permissionRequested: true,
    ...overrides,
  };
}

/** Advance the gate by `ms`, in 16ms frames. */
function run(gate: StartupGate, ms: number, ctx: GateContext = context()): void {
  for (let elapsed = 0; elapsed < ms; elapsed += 16) {
    gate.advance(16, ctx);
  }
}

describe("§4: the gate blocks the run until the player is detected", () => {
  it("starts blocking, at the permission prompt", () => {
    const gate = new StartupGate(DEFAULT_BODY_TUNING);
    const view = gate.advance(16, context({ permissionRequested: false, status: "idle" }));
    expect(gate.current).toBe("CAMERA_PERMISSION");
    expect(view.blocking).toBe(true);
    expect(view.action).toBe("Enable camera");
    // §4.1: never dead-end the player.
    expect(view.showKeyboardEscape).toBe(true);
  });

  it("walks the whole sequence to RUNNING (I-01)", () => {
    const gate = new StartupGate(DEFAULT_BODY_TUNING);
    gate.advance(16, context({ permissionRequested: false, status: "idle" }));

    gate.requestPermission();
    expect(gate.current).toBe("CAMERA_INIT");

    run(gate, 100);
    expect(gate.current).toBe("PLAYER_DETECTION");

    run(gate, DEFAULT_BODY_TUNING.detectionHoldMs + 100);
    expect(gate.current).toBe("CALIBRATION");

    run(gate, DEFAULT_BODY_TUNING.calibrationMs + 100);
    expect(gate.current).toBe("READY");

    run(gate, 3200);
    expect(gate.current).toBe("RUNNING");
    expect(gate.isBlocking).toBe(false);
    expect(gate.cameraActive).toBe(true);
  });

  it("requires sustained detection, not one good frame (§4.3)", () => {
    const gate = new StartupGate(DEFAULT_BODY_TUNING);
    gate.requestPermission();
    run(gate, 100);
    expect(gate.current).toBe("PLAYER_DETECTION");

    // Almost long enough, then the player drops out for a frame.
    run(gate, DEFAULT_BODY_TUNING.detectionHoldMs - 200);
    gate.advance(16, context({ sample: ABSENT_SAMPLE }));
    run(gate, 300);
    // The hold restarted, so it must not have advanced yet.
    expect(gate.current).toBe("PLAYER_DETECTION");

    run(gate, DEFAULT_BODY_TUNING.detectionHoldMs);
    expect(gate.current).toBe("CALIBRATION");
  });

  it("never advances while nobody is in frame", () => {
    const gate = new StartupGate(DEFAULT_BODY_TUNING);
    gate.requestPermission();
    run(gate, 10_000, context({ sample: ABSENT_SAMPLE }));
    expect(gate.current).toBe("PLAYER_DETECTION");
    expect(gate.isBlocking).toBe(true);
  });
});

describe("§4.2: the three failures are distinguishable", () => {
  const cases: Array<[CameraSourceStatus, string]> = [
    ["camera_error", "No camera found"],
    ["camera_denied", "Camera blocked"],
    ["service_unavailable", "Detection service not running"],
    ["stalled", "No frames received"],
  ];

  for (const [status, expectedTitle] of cases) {
    it(`${status} says "${expectedTitle}"`, () => {
      const gate = new StartupGate(DEFAULT_BODY_TUNING);
      gate.requestPermission();
      const view = gate.advance(16, context({ status }));
      expect(view.title).toBe(expectedTitle);
      expect(view.blocking).toBe(true);
      // Every failure offers a way forward.
      expect(view.action !== null || view.showKeyboardEscape).toBe(true);
    });
  }

  it("gives each failure a different message", () => {
    const gate = new StartupGate(DEFAULT_BODY_TUNING);
    gate.requestPermission();
    const bodies = cases.map(([status]) => gate.advance(16, context({ status })).body);
    expect(new Set(bodies).size).toBe(cases.length);
  });
});

describe("§4.3: detection guidance is specific", () => {
  const guidance = (sample: PoseSample | null): string => {
    const gate = new StartupGate(DEFAULT_BODY_TUNING);
    gate.requestPermission();
    gate.advance(16, context());
    return gate.advance(16, context({ sample })).body;
  };

  it("names the actual problem rather than saying 'not detected'", () => {
    expect(guidance(ABSENT_SAMPLE)).toContain("No one detected");

    const tooClose = new FixtureBuilder().add(1, { bodyScale: 0.9 }).build()[0] as PoseSample;
    expect(guidance(tooClose)).toContain("too close");

    const tooFar = new FixtureBuilder().add(1, { bodyScale: 0.1 }).build()[0] as PoseSample;
    expect(guidance(tooFar)).toContain("step closer");

    expect(guidance(GOOD_SAMPLE)).toContain("Got you");
  });
});

describe("§4.1 / Q7: the keyboard escape", () => {
  it("stops blocking immediately and stays out of the way (I-02)", () => {
    const gate = new StartupGate(DEFAULT_BODY_TUNING);
    gate.advance(16, context({ status: "camera_denied", permissionRequested: false }));
    gate.chooseKeyboard();
    expect(gate.current).toBe("KEYBOARD");
    expect(gate.isBlocking).toBe(false);
    // ...and body input stays off, since there is no camera.
    expect(gate.cameraActive).toBe(false);

    const view = gate.advance(16, context({ status: "camera_denied" }));
    expect(view.blocking).toBe(false);
  });

  it("is available from a service failure too (I-03)", () => {
    const gate = new StartupGate(DEFAULT_BODY_TUNING);
    gate.requestPermission();
    const view = gate.advance(16, context({ status: "service_unavailable" }));
    expect(view.showKeyboardEscape).toBe(true);
  });
});

describe("§4.4: manual recalibrate (I-07)", () => {
  it("re-enters calibration from RUNNING without restarting", () => {
    const gate = new StartupGate(DEFAULT_BODY_TUNING);
    gate.forceRunning();
    gate.recalibrate();
    expect(gate.current).toBe("CALIBRATION");

    run(gate, DEFAULT_BODY_TUNING.calibrationMs + 3400);
    expect(gate.current).toBe("RUNNING");
  });

  it("does nothing outside RUNNING, so it cannot derail the gate", () => {
    const gate = new StartupGate(DEFAULT_BODY_TUNING);
    gate.requestPermission();
    gate.recalibrate();
    expect(gate.current).toBe("CAMERA_INIT");
  });
});

describe("progress is reported for the states that hold", () => {
  it("reports detection and calibration progress, and nothing elsewhere", () => {
    const gate = new StartupGate(DEFAULT_BODY_TUNING);
    expect(gate.advance(16, context({ permissionRequested: false, status: "idle" })).progress).toBe(-1);

    gate.requestPermission();
    run(gate, 100);
    run(gate, DEFAULT_BODY_TUNING.detectionHoldMs / 2);
    const detection = gate.advance(16, context());
    expect(detection.progress).toBeGreaterThan(0.2);
    expect(detection.progress).toBeLessThan(1);
  });
});
