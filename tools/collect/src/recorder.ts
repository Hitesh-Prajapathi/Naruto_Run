/** Camera + MediaRecorder wrappers. Not unit-testable without a browser; see tests/smoke.mjs. */
import { measureFps, PhaseClock } from "./timing";

export const REQUESTED = { width: 1280, height: 720, fps: 30 };
export const MIME_CANDIDATES = ["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm", "video/mp4"];

/** Preference order: vp9 webm, vp8 webm, bare webm, mp4 (Safari). */
export function pickMimeType(isSupported: (m: string) => boolean): string | null {
  return MIME_CANDIDATES.find((m) => isSupported(m)) ?? null;
}

export function browserMime(): string | null {
  if (typeof MediaRecorder === "undefined") return null;
  return pickMimeType((m) => MediaRecorder.isTypeSupported(m));
}

export interface CameraWarnings {
  lowFps: boolean;
  lowRes: boolean;
}
export function cameraWarnings(width: number, height: number, fps: number): CameraWarnings {
  // Phones report portrait (e.g. 720x1280): compare long/short sides, not width/height.
  const long = Math.max(width, height);
  const short = Math.min(width, height);
  return { lowFps: fps < 24, lowRes: long < 640 || short < 480 };
}

export async function openCamera(): Promise<MediaStream> {
  return navigator.mediaDevices.getUserMedia({
    audio: false,
    video: {
      facingMode: "user",
      width: { ideal: REQUESTED.width },
      height: { ideal: REQUESTED.height },
      frameRate: { ideal: REQUESTED.fps },
    },
  });
}

/** Counts presented frames of a <video> using requestVideoFrameCallback (falls back to track settings). */
export class FpsMeter {
  private frames = 0;
  private startedAt = 0;
  private handle = 0;
  private running = false;
  constructor(private video: HTMLVideoElement, private fallbackFps: number) {}

  start() {
    this.frames = 0;
    this.startedAt = performance.now();
    this.running = true;
    const v = this.video as HTMLVideoElement & {
      requestVideoFrameCallback?: (cb: () => void) => number;
    };
    if (!v.requestVideoFrameCallback) return;
    const tick = () => {
      if (!this.running) return;
      this.frames++;
      this.handle = v.requestVideoFrameCallback!(tick);
    };
    this.handle = v.requestVideoFrameCallback(tick);
  }
  stop(): number {
    this.running = false;
    const v = this.video as HTMLVideoElement & { cancelVideoFrameCallback?: (h: number) => void };
    v.cancelVideoFrameCallback?.(this.handle);
    const ms = performance.now() - this.startedAt;
    const hasRvfc = "requestVideoFrameCallback" in this.video;
    return hasRvfc && this.frames > 0 ? measureFps(this.frames, ms) : this.fallbackFps;
  }
}

export interface ActiveRecording {
  /** Resolves with the performance.now() captured in the MediaRecorder 'start' event. */
  started: Promise<void>;
  stop(): Promise<{ data: ArrayBuffer; mime: string; durationMs: number; fps: number }>;
  abort(): void;
}

export function startRecording(
  stream: MediaStream,
  mime: string,
  video: HTMLVideoElement,
  clock: PhaseClock,
  fallbackFps: number,
): ActiveRecording {
  const rec = new MediaRecorder(stream, { mimeType: mime, videoBitsPerSecond: 3_000_000 });
  const chunks: Blob[] = [];
  const meter = new FpsMeter(video, fallbackFps);
  let t0 = 0;
  let aborted = false;
  rec.ondataavailable = (e) => e.data.size && chunks.push(e.data);
  const started = new Promise<void>((resolve, reject) => {
    rec.onstart = () => {
      t0 = performance.now();
      clock.anchor(t0);
      meter.start();
      resolve();
    };
    rec.onerror = () => reject(new Error("MediaRecorder error"));
  });
  rec.start(1000);
  return {
    started,
    abort() {
      aborted = true;
      if (rec.state !== "inactive") rec.stop();
    },
    stop() {
      return new Promise((resolve, reject) => {
        rec.onstop = async () => {
          const durationMs = Math.round(performance.now() - t0);
          const fps = meter.stop();
          if (aborted) return reject(new Error("aborted"));
          const blob = new Blob(chunks, { type: mime });
          resolve({ data: await blob.arrayBuffer(), mime, durationMs, fps });
        };
        if (rec.state !== "inactive") rec.stop();
      });
    },
  };
}
