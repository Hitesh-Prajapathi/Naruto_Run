/**
 * Camera preview and tracking indicator -- Feature Brief 05 §7.1 and §7.2.
 *
 * **Bottom-right, and nowhere else.** §0.1 lists every occupied region and
 * marks the top-right as reserved for the seal reference prompt. Bottom-right
 * is the only free corner, so that is where this lives; nothing existing may
 * move by a pixel to make room for it.
 *
 * The preview is **CSS-mirrored** so it behaves like a mirror for the player
 * (§5.1). The skeleton is drawn from `PoseSample.displayLandmarks`, which the
 * pose adapter has *already* mirrored to match — see the long note in
 * `poseAdapter.ts`. Nothing here flips anything; if the skeleton ever appears
 * backwards over the video, the bug is in the adapter, not in this file.
 */

import type { PoseSample } from "../input/poseAdapter";
import type { TrackingStatus } from "../input/bodyInputSource";

/** MediaPipe Pose connections worth drawing: torso, arms, legs. */
const CONNECTIONS: ReadonlyArray<readonly [number, number]> = [
  [11, 12], [11, 23], [12, 24], [23, 24],
  [11, 13], [13, 15], [12, 14], [14, 16],
  [23, 25], [25, 27], [24, 26], [26, 28],
];

const STATUS_COLOR: Record<TrackingStatus, string> = {
  tracking: "#8fbf4d",
  weak: "#e8a33d",
  lost: "#e03131",
  no_signal: "#6b4423",
};

const STATUS_LABEL: Record<TrackingStatus, string> = {
  tracking: "tracking",
  weak: "weak signal",
  lost: "not detected",
  no_signal: "camera off",
};

export class CameraHud {
  readonly video: HTMLVideoElement;
  private readonly wrap: HTMLElement;
  private readonly canvas: HTMLCanvasElement;
  private readonly context: CanvasRenderingContext2D | null;
  private readonly indicator: HTMLElement;
  private readonly indicatorDot: HTMLElement;
  private readonly indicatorText: HTMLElement;
  private readonly toggleButton: HTMLButtonElement;

  private previewVisible = true;
  private lastStatus: TrackingStatus | null = null;

  constructor(private readonly root: HTMLElement) {
    root.innerHTML = `
      <div class="cam-indicator" data-indicator hidden>
        <span class="cam-dot" data-dot></span><span data-indicator-text>camera off</span>
      </div>
      <div class="cam-preview" data-wrap hidden>
        <video class="cam-video" data-video muted playsinline></video>
        <canvas class="cam-skeleton" data-skeleton width="180" height="135"></canvas>
        <div class="cam-ring" data-ring></div>
        <button class="cam-toggle" type="button" data-toggle title="Hide camera preview (C)">–</button>
      </div>
    `;
    this.wrap = root.querySelector("[data-wrap]")!;
    this.video = root.querySelector("[data-video]")!;
    this.canvas = root.querySelector("[data-skeleton]")!;
    this.context = this.canvas.getContext("2d");
    this.indicator = root.querySelector("[data-indicator]")!;
    this.indicatorDot = root.querySelector("[data-dot]")!;
    this.indicatorText = root.querySelector("[data-indicator-text]")!;
    this.toggleButton = root.querySelector("[data-toggle]")!;

    this.toggleButton.addEventListener("click", () => this.togglePreview());
    root.hidden = true;
  }

  /** Show the HUD at all. Called once camera input is actually enabled, so
   * that with the flag off nothing is added to the screen. */
  activate(): void {
    this.root.hidden = false;
    this.indicator.hidden = false;
    this.wrap.hidden = !this.previewVisible;
  }

  get isPreviewVisible(): boolean {
    return this.previewVisible;
  }

  /** §7.1: "toggleable — some players will want it gone." */
  togglePreview(): void {
    this.previewVisible = !this.previewVisible;
    this.wrap.hidden = !this.previewVisible;
    this.toggleButton.textContent = this.previewVisible ? "–" : "+";
    this.toggleButton.title = this.previewVisible
      ? "Hide camera preview (C)"
      : "Show camera preview (C)";
  }

  /**
   * @param sample newest pose, already in display space, or null.
   * @param status drives the ring colour and the persistent indicator.
   */
  render(sample: PoseSample | null, status: TrackingStatus): void {
    if (status !== this.lastStatus) {
      this.lastStatus = status;
      const colour = STATUS_COLOR[status];
      this.indicatorDot.style.background = colour;
      this.indicatorText.textContent = STATUS_LABEL[status];
      this.wrap.style.setProperty("--cam-ring", colour);
    }

    if (!this.previewVisible) return;
    const context = this.context;
    if (!context) return;

    const { width, height } = this.canvas;
    context.clearRect(0, 0, width, height);
    if (!sample || sample.displayLandmarks.length === 0) return;

    const points = sample.displayLandmarks;
    context.lineWidth = 2;
    context.strokeStyle = STATUS_COLOR[status];
    context.beginPath();
    for (const [from, to] of CONNECTIONS) {
      const a = points[from];
      const b = points[to];
      if (!a || !b) continue;
      context.moveTo(a[0] * width, a[1] * height);
      context.lineTo(b[0] * width, b[1] * height);
    }
    context.stroke();

    context.fillStyle = "#ffd9a0";
    for (const index of [11, 12, 23, 24]) {
      const point = points[index];
      if (!point) continue;
      context.beginPath();
      context.arc(point[0] * width, point[1] * height, 2.5, 0, Math.PI * 2);
      context.fill();
    }
  }

  dispose(): void {
    this.video.srcObject = null;
  }
}
