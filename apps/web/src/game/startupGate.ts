/**
 * The pre-game startup gate -- Feature Brief 05 §4.
 *
 * `BOOT → CAMERA_PERMISSION → CAMERA_INIT → PLAYER_DETECTION → CALIBRATION
 *  → READY → RUNNING`, plus a `KEYBOARD` terminal for the escape hatch.
 *
 * Pure logic, no DOM and no timers: it is fed elapsed time and a description
 * of the world, and returns which state it is in and what to say about it.
 * That keeps every rule -- 2s sustained detection, 3s calibration hold, the
 * three distinguishable failures -- testable without a camera.
 *
 * The settled contract (owner, Q7): the camera is the default path and the
 * gate blocks the run until the player is reliably detected, but a "Play
 * with keyboard" escape is always available and the keyboard stays live
 * throughout. The game never hard-requires a camera.
 */

import type { BodyTuning } from "../input/bodyTuning";
import type { CameraSourceStatus } from "../input/cameraPoseSource";
import type { PoseSample } from "../input/poseAdapter";

export type GateState =
  | "BOOT"
  | "CAMERA_PERMISSION"
  | "CAMERA_INIT"
  | "PLAYER_DETECTION"
  | "CALIBRATION"
  | "READY"
  | "RUNNING"
  | "KEYBOARD";

/** What the gate needs to know each frame. */
export interface GateContext {
  status: CameraSourceStatus;
  /** Newest pose sample, or null when none has arrived. */
  sample: PoseSample | null;
  /** True once the player has asked to start (the permission button). */
  permissionRequested: boolean;
}

export interface GateView {
  state: GateState;
  title: string;
  body: string;
  /** 0..1 for the states that have a hold; -1 when there is nothing to show. */
  progress: number;
  /** Show the primary action button with this label, or null. */
  action: string | null;
  /** Show the "Play with keyboard" escape (§4.1: never dead-end). */
  showKeyboardEscape: boolean;
  /** True while the gate owns the screen. */
  blocking: boolean;
}

/** Landmarks the game actually needs, per §4.3 -- not merely "a person". */
function hasRequiredLandmarks(sample: PoseSample): boolean {
  // `bodyScale` is the shoulder-to-shoulder distance, so a positive value
  // means both shoulders were present and measurable. Shoulders are all the
  // game needs: lean and jump are both derived from them, and requiring hips
  // as well would refuse to start for anyone at a desk-height webcam.
  return sample.personPresent && sample.bodyScale > 0.05;
}

export class StartupGate {
  private state: GateState = "BOOT";
  private stateElapsedMs = 0;
  /** Continuous ms the player has been properly detected. */
  private detectedForMs = 0;
  private lastGuidance = "";

  constructor(private tuning: BodyTuning) {}

  setTuning(tuning: BodyTuning): void {
    this.tuning = tuning;
  }

  get current(): GateState {
    return this.state;
  }

  /** True while the run must not start. */
  get isBlocking(): boolean {
    return this.state !== "RUNNING" && this.state !== "KEYBOARD";
  }

  /** True when body input should be consumed. */
  get cameraActive(): boolean {
    return this.state === "RUNNING";
  }

  /** The player chose the keyboard escape; the gate steps aside for good. */
  chooseKeyboard(): void {
    this.enter("KEYBOARD");
  }

  /** The player pressed "Enable camera". */
  requestPermission(): void {
    if (this.state === "BOOT" || this.state === "CAMERA_PERMISSION") {
      this.enter("CAMERA_INIT");
    }
  }

  /**
   * Re-run calibration without restarting -- §4.4 requires a manual
   * recalibrate key, because players drift.
   */
  recalibrate(): void {
    if (this.state === "RUNNING") {
      this.enter("CALIBRATION");
    }
  }

  /** Skip straight to playing, for tests and for the keyboard path. */
  forceRunning(): void {
    this.enter("RUNNING");
  }

  advance(dtMs: number, context: GateContext): GateView {
    this.stateElapsedMs += dtMs;

    switch (this.state) {
      case "BOOT":
        this.enter("CAMERA_PERMISSION");
        break;

      case "CAMERA_PERMISSION":
        if (context.permissionRequested) {
          this.enter("CAMERA_INIT");
        }
        break;

      case "CAMERA_INIT":
        // §4.2: three failures with completely different fixes, kept
        // distinguishable rather than collapsed into "camera error".
        if (context.status === "camera_denied" || context.status === "camera_error") {
          break; // stay put; the view below explains which
        }
        if (context.status === "streaming") {
          this.enter("PLAYER_DETECTION");
        }
        break;

      case "PLAYER_DETECTION": {
        if (context.sample && hasRequiredLandmarks(context.sample)) {
          this.detectedForMs += dtMs;
        } else {
          // §4.3: a single good frame is not enough, and a dropout resets
          // the hold rather than merely pausing it.
          this.detectedForMs = 0;
        }
        if (this.detectedForMs >= this.tuning.detectionHoldMs) {
          this.enter("CALIBRATION");
        }
        break;
      }

      case "CALIBRATION":
        if (this.stateElapsedMs >= this.tuning.calibrationMs) {
          this.enter("READY");
        }
        break;

      case "READY":
        if (this.stateElapsedMs >= 3000) {
          this.enter("RUNNING");
        }
        break;

      case "RUNNING":
      case "KEYBOARD":
      default:
        break;
    }

    return this.view(context);
  }

  private view(context: GateContext): GateView {
    switch (this.state) {
      case "BOOT":
      case "CAMERA_PERMISSION":
        return {
          state: this.state,
          title: "Play with your body",
          body:
            "Naruto follows you. Lean left or right to change lane, and jump to jump. " +
            "We need your camera to see you — the video never leaves this machine.",
          progress: -1,
          action: "Enable camera",
          showKeyboardEscape: true,
          blocking: true,
        };

      case "CAMERA_INIT":
        return {
          state: this.state,
          ...this.initMessage(context.status),
          progress: -1,
          showKeyboardEscape: true,
          blocking: true,
        };

      case "PLAYER_DETECTION":
        return {
          state: this.state,
          title: "Find your spot",
          body: this.detectionGuidance(context.sample),
          progress: Math.min(1, this.detectedForMs / this.tuning.detectionHoldMs),
          action: null,
          showKeyboardEscape: true,
          blocking: true,
        };

      case "CALIBRATION": {
        const remaining = Math.max(
          0,
          Math.ceil((this.tuning.calibrationMs - this.stateElapsedMs) / 1000),
        );
        return {
          state: this.state,
          title: "Hold still",
          body: `Stand square to the camera, arms relaxed. Measuring your neutral pose… ${remaining}`,
          progress: Math.min(1, this.stateElapsedMs / this.tuning.calibrationMs),
          action: null,
          showKeyboardEscape: true,
          blocking: true,
        };
      }

      case "READY": {
        const count = Math.max(1, 3 - Math.floor(this.stateElapsedMs / 1000));
        return {
          state: this.state,
          title: "Get ready",
          body: String(count),
          progress: Math.min(1, this.stateElapsedMs / 3000),
          action: null,
          showKeyboardEscape: false,
          blocking: true,
        };
      }

      case "RUNNING":
      case "KEYBOARD":
      default:
        return {
          state: this.state,
          title: "",
          body: "",
          progress: -1,
          action: null,
          showKeyboardEscape: false,
          blocking: false,
        };
    }
  }

  /**
   * §4.2 verbatim: "'camera not found,' 'detection service not running,'
   * and 'no frames received' have completely different fixes, and a generic
   * error will waste hours."
   */
  private initMessage(status: CameraSourceStatus): {
    title: string;
    body: string;
    action: string | null;
  } {
    switch (status) {
      case "camera_denied":
        return {
          title: "Camera blocked",
          body:
            "Your browser refused camera access. Click the camera icon in the address bar, " +
            "allow access for this page, then reload. Or play with the keyboard.",
          action: "Retry",
        };
      case "camera_error":
        return {
          title: "No camera found",
          body:
            "No camera device is available. Check that one is connected and not in use by " +
            "another app (video calls hold the camera exclusively).",
          action: "Retry",
        };
      case "service_unavailable":
        return {
          title: "Detection service not running",
          body:
            "The camera is working, but the recognition service isn't answering. " +
            "Start it in a terminal, then press Retry.",
          action: "Retry",
        };
      case "stalled":
        return {
          title: "No frames received",
          body:
            "Connected to the service, but no pose data is arriving. The service may have " +
            "stopped mid-run — check its terminal for errors.",
          action: "Retry",
        };
      case "waiting_for_frames":
      case "connecting":
        return {
          title: "Connecting",
          body: "Camera open. Waiting for the recognition service…",
          action: null,
        };
      case "requesting_camera":
        return {
          title: "Allow camera access",
          body: "Your browser is asking for permission. Choose Allow to continue.",
          action: null,
        };
      default:
        return { title: "Starting camera", body: "One moment…", action: null };
    }
  }

  /** §4.3: guidance must be specific and actionable. */
  private detectionGuidance(sample: PoseSample | null): string {
    if (!sample || !sample.personPresent) {
      this.lastGuidance = "No one detected — please stand in front of the camera.";
      return this.lastGuidance;
    }
    // Bands are shoulder width as a fraction of frame height. Roughly: 0.75
    // is close enough that a lean takes a shoulder out of frame, 0.15 is far
    // enough that landmark jitter starts to rival the movement being read.
    if (sample.bodyScale <= 0.05) {
      this.lastGuidance = "Partly out of frame — make sure both shoulders are visible.";
      return this.lastGuidance;
    }
    if (sample.bodyScale > 0.75) {
      this.lastGuidance = "You're too close — step back a little.";
      return this.lastGuidance;
    }
    if (sample.bodyScale < 0.15) {
      this.lastGuidance = "You're a long way off — step closer to the camera.";
      return this.lastGuidance;
    }
    this.lastGuidance = "Got you. Hold there…";
    return this.lastGuidance;
  }

  private enter(state: GateState): void {
    this.state = state;
    this.stateElapsedMs = 0;
    if (state === "PLAYER_DETECTION") {
      this.detectedForMs = 0;
    }
  }

  /** Ms spent in the current state -- the host drives calibration off this. */
  get elapsedInStateMs(): number {
    return this.stateElapsedMs;
  }
}
