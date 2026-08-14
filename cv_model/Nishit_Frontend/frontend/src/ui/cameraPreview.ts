/**
 * Display-only mirrored preview. The mirroring itself is CSS
 * (`transform: scaleX(-1)` on #camera-preview, see styles/app.css) applied to
 * the *displayed* canvas only -- this module copies the raw <video> source
 * into that canvas unmirrored; framePublisher.ts reads from the same raw
 * <video> element, never from this canvas.
 */

export class CameraPreview {
  private readonly ctx: CanvasRenderingContext2D;
  private rafHandle: number | null = null;

  constructor(
    private readonly video: HTMLVideoElement,
    private readonly canvas: HTMLCanvasElement,
  ) {
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      throw new Error("2D canvas context is unavailable for the camera preview");
    }
    this.ctx = ctx;
  }

  start(): void {
    if (this.rafHandle !== null) {
      return;
    }
    const tick = (): void => {
      if (this.video.readyState >= this.video.HAVE_CURRENT_DATA) {
        this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
      }
      this.rafHandle = requestAnimationFrame(tick);
    };
    this.rafHandle = requestAnimationFrame(tick);
  }

  stop(): void {
    if (this.rafHandle !== null) {
      cancelAnimationFrame(this.rafHandle);
      this.rafHandle = null;
    }
  }
}
