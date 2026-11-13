/**
 * The live camera path -- Feature Brief 05 §3, step 6.
 *
 * Composes the pieces Phase B already built rather than writing new ones
 * (§2.2: "do not build a second transport"):
 *
 *   CameraController  getUserMedia + permission state machine
 *   FramePublisher    raw unmirrored capture -> JPEG -> binary frame
 *   WebSocketClient   frames up, snapshots and events down
 *   poseAdapter       pipeline space -> player space (the mirroring point)
 *
 * **Note the architecture is the reverse of what §3.2 sketches.** The brief
 * proposes a Python service that owns the camera; the transport that already
 * exists has the *browser* own the camera and Python do inference only.
 * §2.2 says reuse it, so this keeps that arrangement. Two consequences: the
 * permission prompt is a normal browser one (which is what §4.1 wants
 * anyway), and the latency chain has two stages §3.3's table omits -- JPEG
 * encode and frame upload -- both measured below.
 *
 * Threading (§3.4): inference already runs off the render thread by virtue
 * of being in another process. On this side the only per-frame work is a
 * canvas draw and a JPEG encode, and the publisher skips a tick rather than
 * queueing if the previous encode has not finished.
 */

import { CameraController, type CameraState } from "../camera/cameraController";
import { FramePublisher, type CaptureSurface } from "../camera/framePublisher";
import { WebSocketClient } from "../transport/websocketClient";
import type { ConnectionState } from "../transport/eventSource";
import type { StateSnapshotMessage } from "../transport/protocol";
import { toPoseSample, type PoseSample } from "./poseAdapter";
import {
  CV_CAPTURE_HEIGHT,
  CV_CAPTURE_WIDTH,
  CV_JPEG_QUALITY,
  CV_TARGET_FPS,
  CV_TRANSPORT_URL,
} from "../config/cvInputConfig";

/** Per-stage latency, in ms (§3.3). Every stage the real chain has. */
export interface LatencyBreakdown {
  /** Camera capture interval -- one frame period at the target rate. */
  captureMs: number;
  /** Canvas draw + JPEG encode, measured. */
  encodeMs: number;
  /** Frame upload + inference + snapshot down, measured as a round trip. */
  roundTripMs: number;
  /** Age of the newest snapshot when the game consumed it. */
  consumeMs: number;
  /** Sum -- the number §3.3 budgets at <120ms. */
  totalMs: number;
}

export type CameraSourceStatus =
  | "idle"
  | "requesting_camera"
  | "camera_denied"
  | "camera_error"
  | "connecting"
  | "waiting_for_frames"
  | "streaming"
  | "service_unavailable"
  | "stalled";

export interface CameraPoseSourceOptions {
  /** The <video> element the raw stream is attached to. */
  video: HTMLVideoElement;
  transportUrl?: string;
  onSample?: (sample: PoseSample) => void;
  onStatusChange?: (status: CameraSourceStatus) => void;
}

export class CameraPoseSource {
  private readonly camera: CameraController;
  private readonly client: WebSocketClient;
  private publisher: FramePublisher | null = null;
  private readonly unsubscribes: Array<() => void> = [];
  /** Guards `start()` against re-entry -- see the comment on that method. */
  private starting = false;

  private status: CameraSourceStatus = "idle";
  private lastSnapshotAtMs = 0;
  private lastSample: PoseSample | null = null;
  private snapshotCount = 0;

  private latency: LatencyBreakdown = {
    captureMs: 1000 / CV_TARGET_FPS,
    encodeMs: 0,
    roundTripMs: 0,
    consumeMs: 0,
    totalMs: 0,
  };
  private encodeMsEma = 0;
  private roundTripEma = 0;

  constructor(private readonly options: CameraPoseSourceOptions) {
    this.camera = new CameraController({ video: options.video });
    this.client = new WebSocketClient({
      url: options.transportUrl ?? CV_TRANSPORT_URL,
    });
  }

  get currentStatus(): CameraSourceStatus {
    return this.status;
  }

  get cameraState(): CameraState {
    return this.camera.cameraState;
  }

  get connectionState(): ConnectionState {
    return this.client.connectionState;
  }

  get lastCameraError(): string | null {
    return this.camera.lastError;
  }

  get latestSample(): PoseSample | null {
    return this.lastSample;
  }

  get snapshotsReceived(): number {
    return this.snapshotCount;
  }

  get latencyBreakdown(): LatencyBreakdown {
    return this.latency;
  }

  /**
   * Request the camera, connect, and start publishing.
   *
   * Resolves once the camera is open and the socket has been asked to
   * connect -- *not* once frames are flowing. The gate distinguishes those
   * three failures (§4.2), so it needs to observe them separately rather
   * than getting one merged promise rejection.
   *
   * **Idempotent, and it has to be.** This is wired to the gate's action
   * button, which is also the "Try again" button, so any player who retries a
   * failed camera start calls it more than once. It previously left the old
   * `FramePublisher` running when it overwrote the field, and pushed a second
   * copy of every subscription. Two clicks meant two publishers sending
   * 20fps each and every snapshot being handled twice; four meant 80fps of
   * JPEG into a service that can drain about 13, which backs up the socket
   * until frames arrive seconds stale and the player reads "not detected"
   * while standing in plain view. Restarting cleanly is the fix.
   */
  async start(): Promise<void> {
    // A second call while the first is still awaiting the camera would race
    // the teardown below and leak the very publisher it is trying to replace.
    if (this.starting) {
      return;
    }
    this.starting = true;
    try {
      await this.startInternal();
    } finally {
      this.starting = false;
    }
  }

  private async startInternal(): Promise<void> {
    // Drop anything a previous attempt left behind before building more.
    this.publisher?.stop();
    this.publisher = null;
    for (const unsubscribe of this.unsubscribes.splice(0)) unsubscribe();

    this.setStatus("requesting_camera");
    await this.camera.start();

    if (this.camera.cameraState === "denied") {
      this.setStatus("camera_denied");
      return;
    }
    if (this.camera.cameraState !== "ready") {
      this.setStatus("camera_error");
      return;
    }

    const stream = this.camera.activeStream;
    if (stream) {
      this.options.video.srcObject = stream;
      this.options.video.muted = true;
      this.options.video.playsInline = true;
      try {
        await this.options.video.play();
      } catch {
        // Autoplay can reject even when muted; the stream still renders.
      }
    }

    this.unsubscribes.push(
      this.client.onStateSnapshot((snapshot) => this.handleSnapshot(snapshot)),
    );
    this.unsubscribes.push(
      this.client.onConnectionStateChange((state: ConnectionState) => this.handleConnection(state)),
    );

    this.setStatus("connecting");
    this.client.connect();

    this.publisher = new FramePublisher({
      video: this.options.video,
      source: this.client,
      targetFps: CV_TARGET_FPS,
      jpegQuality: CV_JPEG_QUALITY,
      captureWidth: CV_CAPTURE_WIDTH,
      captureHeight: CV_CAPTURE_HEIGHT,
      surface: this.instrumentedSurface(),
    });
    this.publisher.start();
  }

  /**
   * A capture surface that times the draw + JPEG encode.
   *
   * §3.3 wants every stage instrumented, and this one is invisible from
   * outside the publisher -- it happens between "camera capture" and
   * "transport", two stages the brief's table does list. Measuring it is
   * also the only way to know whether lowering JPEG quality would help,
   * now that lowering *resolution* is known not to.
   */
  private instrumentedSurface(): CaptureSurface {
    const canvas = document.createElement("canvas");
    canvas.width = CV_CAPTURE_WIDTH;
    canvas.height = CV_CAPTURE_HEIGHT;
    const context = canvas.getContext("2d");
    if (!context) {
      throw new Error("2D canvas context is unavailable in this environment");
    }
    let drawStartedAt = 0;
    return {
      width: canvas.width,
      height: canvas.height,
      captureFrame: (video: HTMLVideoElement): void => {
        drawStartedAt = performance.now();
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
      },
      encodeJpeg: (quality: number): Promise<Blob | null> =>
        new Promise((resolve) => {
          canvas.toBlob((blob) => {
            const elapsed = performance.now() - drawStartedAt;
            this.encodeMsEma = this.encodeMsEma === 0 ? elapsed : this.encodeMsEma * 0.8 + elapsed * 0.2;
            resolve(blob);
          }, "image/jpeg", quality);
        }),
    };
  }

  private handleConnection(state: ConnectionState): void {
    if (state === "open") {
      this.setStatus(this.snapshotCount > 0 ? "streaming" : "waiting_for_frames");
    } else if (state === "closed" || state === "error") {
      // The socket retries with backoff on its own; what matters to the
      // player is that the service is not answering.
      this.setStatus("service_unavailable");
    } else if (state === "connecting") {
      this.setStatus("connecting");
    }
  }

  private handleSnapshot(snapshot: StateSnapshotMessage): void {
    const now = Date.now();
    this.snapshotCount += 1;
    this.lastSnapshotAtMs = now;

    // Round trip: the transport echoes the frame's capture timestamp back,
    // so the difference is upload + inference + downlink in one number.
    const capturedAt = snapshot.captured_at_ms;
    let roundTrip: number | undefined;
    if (typeof capturedAt === "number" && capturedAt > 0) {
      roundTrip = Math.max(0, now - capturedAt);
      this.roundTripEma = this.roundTripEma === 0 ? roundTrip : this.roundTripEma * 0.8 + roundTrip * 0.2;
    }

    // Releases the next frame, and feeds the publisher the latency it needs
    // to decide whether to keep sending freely -- see framePublisher's header.
    this.publisher?.noteResponse(roundTrip);

    const sample = toPoseSample(snapshot.output.body, capturedAt || now);
    this.lastSample = sample;
    this.latency = {
      captureMs: 1000 / CV_TARGET_FPS,
      encodeMs: this.encodeMsEma,
      roundTripMs: this.roundTripEma,
      consumeMs: 0,
      totalMs: 1000 / CV_TARGET_FPS + this.encodeMsEma + this.roundTripEma,
    };

    if (this.status !== "streaming") {
      this.setStatus("streaming");
    }
    this.options.onSample?.(sample);
  }

  /**
   * Called once a frame by the host. Detects a stalled stream -- the socket
   * can stay open while the service stops producing (§8: "events stop
   * arriving ... treat as disconnect").
   */
  tick(nowMs: number, staleAfterMs: number): void {
    if (this.status !== "streaming") return;
    if (nowMs - this.lastSnapshotAtMs > staleAfterMs) {
      this.setStatus("stalled");
    }
  }

  /** Age of the newest snapshot, for the debug overlay's consume stage. */
  noteConsumed(nowMs: number): void {
    if (this.lastSnapshotAtMs === 0) return;
    const consume = Math.max(0, nowMs - this.lastSnapshotAtMs);
    this.latency = {
      ...this.latency,
      consumeMs: consume,
      totalMs: this.latency.captureMs + this.latency.encodeMs + this.latency.roundTripMs + consume,
    };
  }

  private setStatus(status: CameraSourceStatus): void {
    if (this.status === status) return;
    this.status = status;
    this.options.onStatusChange?.(status);
  }

  /**
   * Ask the service to forget the person it is currently tracking.
   *
   * The pose model follows one player and is deliberately sticky about it, so
   * a second person stepping in front of the camera is not picked up while
   * the first is still being tracked -- see transport/pipeline.py. This is the
   * only way to hand the camera to somebody else without restarting the
   * service, so it is wired to the same key as recalibration: swapping player
   * and re-measuring their neutral pose are the same action to a player.
   */
  requestRedetect(): void {
    if (this.client.connectionState !== "open") {
      return;
    }
    this.client.sendControl("reset");
  }

  stop(): void {
    this.starting = false;
    this.publisher?.stop();
    this.publisher = null;
    for (const unsubscribe of this.unsubscribes.splice(0)) unsubscribe();
    this.client.disconnect();
    this.camera.stop();
    this.setStatus("idle");
  }
}
