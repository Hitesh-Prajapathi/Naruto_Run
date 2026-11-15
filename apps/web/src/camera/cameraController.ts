/**
 * Camera permission/device lifecycle, as its own state machine independent
 * of the transport connection. game_implementation_plan.md Phase B: "Camera
 * controller with the six states."
 *
 * This never mirrors the pixels it captures from -- the <video> element it
 * owns stays in raw camera orientation. Mirroring for display only happens
 * in ui/cameraPreview.ts via CSS, never here.
 */

import { Signal, type Unsubscribe } from "../transport/eventSource";

export const CAMERA_STATES = ["idle", "requesting", "ready", "denied", "ended", "error"] as const;
export type CameraState = (typeof CAMERA_STATES)[number];

export interface CameraControllerOptions {
  /** The raw, unmirrored <video> element frames are read from. */
  video: HTMLVideoElement;
  mediaDevices?: MediaDevices;
  constraints?: MediaStreamConstraints;
}

const DEFAULT_CONSTRAINTS: MediaStreamConstraints = {
  video: { width: { ideal: 640 }, height: { ideal: 360 }, frameRate: { ideal: 30 } },
  audio: false,
};

export class CameraController {
  private state: CameraState = "idle";
  private stream: MediaStream | null = null;
  private error: string | null = null;
  private readonly stateSignal = new Signal<CameraState>();

  constructor(private readonly options: CameraControllerOptions) {}

  get cameraState(): CameraState {
    return this.state;
  }

  get lastError(): string | null {
    return this.error;
  }

  get activeStream(): MediaStream | null {
    return this.stream;
  }

  onStateChange(handler: (state: CameraState) => void): Unsubscribe {
    return this.stateSignal.subscribe(handler);
  }

  async start(): Promise<void> {
    if (this.state === "requesting" || this.state === "ready") {
      return;
    }
    this.setState("requesting");
    const mediaDevices = this.options.mediaDevices ?? navigator.mediaDevices;
    if (!mediaDevices || typeof mediaDevices.getUserMedia !== "function") {
      this.error = "getUserMedia is not available in this browser";
      this.setState("error");
      return;
    }
    try {
      const stream = await mediaDevices.getUserMedia(this.options.constraints ?? DEFAULT_CONSTRAINTS);
      this.stream = stream;
      this.options.video.srcObject = stream;
      for (const track of stream.getVideoTracks()) {
        track.addEventListener("ended", () => this.handleTrackEnded());
      }
      try {
        await this.options.video.play();
      } catch {
        // Autoplay can reject before a user gesture on some browsers; the
        // stream is still attached and will play once one occurs. Not a
        // camera-acquisition failure.
      }
      this.setState("ready");
    } catch (caught) {
      this.error = caught instanceof Error ? caught.message : String(caught);
      const name = caught instanceof DOMException ? caught.name : "";
      if (name === "NotAllowedError" || name === "PermissionDeniedError" || name === "SecurityError") {
        this.setState("denied");
      } else {
        this.setState("error");
      }
    }
  }

  stop(): void {
    if (this.stream) {
      for (const track of this.stream.getTracks()) {
        track.stop();
      }
      this.stream = null;
    }
    this.options.video.srcObject = null;
    this.setState("ended");
  }

  private handleTrackEnded(): void {
    // A track can end on its own (device unplugged, OS revokes permission,
    // another app takes exclusive access) without stop() being called.
    if (this.state !== "ready") {
      return;
    }
    this.stream = null;
    this.setState("ended");
  }

  private setState(next: CameraState): void {
    this.state = next;
    this.stateSignal.emit(next);
  }
}
