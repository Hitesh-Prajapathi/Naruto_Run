/**
 * Captures frames from the raw (unmirrored) camera source and sends them as
 * bounded, sequenced JPEG frames through a PipelineEventSource.
 *
 * Invariant this file exists to enforce: submitted pixels keep raw camera
 * orientation. This never reads from a CSS-mirrored preview element -- only
 * from the <video> the caller passes in, which cameraController.ts also
 * never mirrors.
 *
 * Backpressure, and the part that used to be missing. Two separate queues can
 * form here, and only one of them was ever guarded:
 *
 *   1. *Encoding.* If the capture/encode of frame N is still running when the
 *      next tick fires, that tick is dropped rather than queued.
 *   2. *The round trip.* `sendFrame` only hands bytes to a WebSocket. It says
 *      nothing about whether the service has kept up. Sending at a fixed rate
 *      into a service that drains more slowly builds a backlog in the socket
 *      and in the service's read queue, and that backlog grows without limit:
 *      observed as sample ages climbing past two and a half seconds while the
 *      player stood in plain view and the game reported "not detected".
 *
 * The fix for (2) is *adaptive*, and it has to be, because the two regimes
 * want opposite behaviour. Measured against the real service:
 *
 *   - Healthy: sending faster than the service answers is a **win**. The
 *     service keeps only the newest frame and caps its own output at 15Hz, so
 *     oversending simply means the frame it reports on is fresher. Measured
 *     47ms of latency oversending, against 79ms when strictly self-clocked.
 *   - Congested: sending at a fixed rate is what builds the unbounded backlog
 *     in the first place, and latency climbs without limit -- observed past
 *     2.5 seconds, with the game showing "not detected" over a visible player.
 *
 * So the publisher sends freely at the target rate while round trips are
 * quick, and falls back to strict self-clocking -- one frame outstanding,
 * released by the next response -- once latency crosses `congestionMs`. The
 * healthy case keeps its lower latency and the congested case is bounded to
 * about one service time instead of growing forever.
 *
 * Self-clocking works even though the service deliberately drops frames,
 * because a response to *any* frame releases the next send. What it must not
 * do is wedge if a response never comes, so an outstanding frame is abandoned
 * after `responseTimeoutMs`.
 */

import { encodeFrame, type FrameEnvelope } from "../transport/protocol";
import type { PipelineEventSource } from "../transport/eventSource";

/** Only the piece of the Blob interface framePublisher actually calls.
 * Real Blob objects satisfy this structurally; tests can hand back a plain
 * object without needing a spec-complete Blob (jsdom's Blob, notably, does
 * not implement arrayBuffer()). */
export interface BlobLike {
  arrayBuffer(): Promise<ArrayBuffer>;
}

/** The minimal surface framePublisher needs from a canvas. Abstracted so
 * tests can supply a fake without requiring a real 2D canvas backend (jsdom
 * does not implement one without extra native dependencies). */
export interface CaptureSurface {
  readonly width: number;
  readonly height: number;
  captureFrame(video: HTMLVideoElement): void;
  encodeJpeg(quality: number): Promise<BlobLike | null>;
}

function createCanvasSurface(width: number, height: number): CaptureSurface {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    throw new Error("2D canvas context is unavailable in this environment");
  }
  return {
    width,
    height,
    captureFrame(video: HTMLVideoElement): void {
      ctx.drawImage(video, 0, 0, width, height);
    },
    encodeJpeg(quality: number): Promise<Blob | null> {
      return new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", quality));
    },
  };
}

// HTMLMediaElement.HAVE_CURRENT_DATA, inlined so tests can supply a plain
// object for `video` instead of a real <video> element.
const HAVE_CURRENT_DATA = 2;

export interface FramePublisherOptions {
  /** The raw, unmirrored source. Only `readyState` is read directly here;
   * pixels are read by the CaptureSurface, which casts to HTMLVideoElement
   * internally -- kept as just `readyState` so tests can inject a plain
   * object instead of a real <video> element. */
  video: Pick<HTMLVideoElement, "readyState">;
  source: PipelineEventSource;
  targetFps?: number;
  jpegQuality?: number;
  captureWidth?: number;
  captureHeight?: number;
  surface?: CaptureSurface;
  setIntervalFn?: typeof setInterval;
  clearIntervalFn?: typeof clearInterval;
  now?: () => number;
  /**
   * How long to wait for a response before assuming it is never coming and
   * sending anyway. Generous: too short and it degenerates into the fixed-rate
   * flooding it replaces, too long and a genuinely dropped response stalls
   * input for that whole period.
   */
  responseTimeoutMs?: number;
  /**
   * Round-trip latency, in ms, above which the publisher stops sending
   * freely and waits for responses instead. Comfortably below the staleness
   * limit that marks tracking lost, so throttling engages before the player
   * sees anything go wrong.
   */
  congestionMs?: number;
}

const DEFAULT_TARGET_FPS = 30;
const DEFAULT_JPEG_QUALITY = 0.75;
const DEFAULT_CAPTURE_WIDTH = 640;
const DEFAULT_CAPTURE_HEIGHT = 360;
const DEFAULT_RESPONSE_TIMEOUT_MS = 600;
const DEFAULT_CONGESTION_MS = 250;

export class FramePublisher {
  private readonly surface: CaptureSurface;
  private timer: ReturnType<typeof setInterval> | null = null;
  private sequence = 0;
  private capturing = false;
  private running = false;
  private droppedTicks = 0;
  private sentFrames = 0;
  /** When the outstanding frame was sent, or null when none is outstanding. */
  private awaitingSince: number | null = null;
  private waitedTicks = 0;
  private timedOutResponses = 0;
  private latencyEmaMs = 0;

  constructor(private readonly options: FramePublisherOptions) {
    this.surface =
      options.surface ??
      createCanvasSurface(
        options.captureWidth ?? DEFAULT_CAPTURE_WIDTH,
        options.captureHeight ?? DEFAULT_CAPTURE_HEIGHT,
      );
  }

  get isRunning(): boolean {
    return this.running;
  }

  get stats(): {
    sentFrames: number;
    droppedTicks: number;
    waitedTicks: number;
    timedOutResponses: number;
    latencyEmaMs: number;
    throttled: boolean;
  } {
    return {
      sentFrames: this.sentFrames,
      droppedTicks: this.droppedTicks,
      waitedTicks: this.waitedTicks,
      timedOutResponses: this.timedOutResponses,
      latencyEmaMs: this.latencyEmaMs,
      throttled: this.isThrottled,
    };
  }

  /**
   * Tell the publisher a result came back, releasing the next send.
   *
   * Called for any snapshot, not specifically the one matching the frame that
   * was sent -- the service keeps only the newest frame, so tying this to a
   * sequence number would wedge on every deliberate drop.
   */
  noteResponse(latencyMs?: number): void {
    this.awaitingSince = null;
    if (latencyMs !== undefined && Number.isFinite(latencyMs) && latencyMs >= 0) {
      // Smoothed, so one slow frame does not throttle the stream and one fast
      // frame does not release a genuinely congested one.
      this.latencyEmaMs =
        this.latencyEmaMs === 0 ? latencyMs : this.latencyEmaMs * 0.7 + latencyMs * 0.3;
    }
  }

  /** True while the publisher is holding back because the service is behind. */
  get isThrottled(): boolean {
    return this.latencyEmaMs > (this.options.congestionMs ?? DEFAULT_CONGESTION_MS);
  }

  start(): void {
    if (this.running) {
      return;
    }
    this.running = true;
    const setIntervalFn = this.options.setIntervalFn ?? setInterval;
    const fps = this.options.targetFps ?? DEFAULT_TARGET_FPS;
    this.timer = setIntervalFn(() => {
      void this.captureAndSend();
    }, Math.round(1000 / fps));
  }

  stop(): void {
    this.running = false;
    this.awaitingSince = null;
    this.latencyEmaMs = 0;
    if (this.timer !== null) {
      const clearIntervalFn = this.options.clearIntervalFn ?? clearInterval;
      clearIntervalFn(this.timer);
      this.timer = null;
    }
  }

  private async captureAndSend(): Promise<void> {
    if (!this.running || this.capturing) {
      if (this.running) {
        this.droppedTicks += 1;
      }
      return;
    }

    // Only let the service set the pace when it is visibly behind; see header.
    const nowFn = this.options.now ?? Date.now;
    if (this.isThrottled && this.awaitingSince !== null) {
      const timeout = this.options.responseTimeoutMs ?? DEFAULT_RESPONSE_TIMEOUT_MS;
      if (nowFn() - this.awaitingSince < timeout) {
        this.waitedTicks += 1;
        return;
      }
      this.timedOutResponses += 1;
      this.awaitingSince = null;
    }
    const { video, source } = this.options;
    if (video.readyState < HAVE_CURRENT_DATA) {
      return;
    }

    this.capturing = true;
    try {
      this.surface.captureFrame(video as HTMLVideoElement);
      const blob = await this.surface.encodeJpeg(this.options.jpegQuality ?? DEFAULT_JPEG_QUALITY);
      if (!blob || !this.running) {
        return;
      }
      const buffer = await blob.arrayBuffer();
      const now = this.options.now ?? Date.now;
      const envelope: FrameEnvelope = {
        version: 1,
        sequence: this.sequence,
        capturedAtMs: now(),
        width: this.surface.width,
        height: this.surface.height,
        payload: new Uint8Array(buffer),
      };
      this.sequence += 1;
      source.sendFrame(encodeFrame(envelope));
      this.sentFrames += 1;
      this.awaitingSince = envelope.capturedAtMs;
    } finally {
      this.capturing = false;
    }
  }
}
