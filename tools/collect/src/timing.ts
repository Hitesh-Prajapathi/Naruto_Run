/**
 * Phase-boundary bookkeeping.
 *
 * Anchor: performance.now() captured inside the MediaRecorder 'start' event handler = clip time 0.
 * Each prompt phase change calls mark() with performance.now() taken in the animation frame that
 * paints the new prompt. Expected accuracy is about +/- 1 camera frame (+/- 33 ms at 30 fps):
 *   - the 'start' event fires slightly before/after the first encoded frame's capture timestamp
 *     (browser dependent, up to ~1 frame);
 *   - a mark is taken at the rAF that paints the prompt, and the subject sees it up to one display
 *     frame (~16 ms) later; human reaction time (150-400 ms) dominates anyway, so these offsets are
 *     the PROMPT timing, not the true gesture onset. Downstream auto-trim should use landmark
 *     stability inside the HOLD window.
 * This is NOT verified against real browsers in this repository (see README).
 */
import type { MarkName, Seal } from "./protocol";

export interface SegmentOffsets {
  form_start: number;
  hold_start: number;
  hold_end: number;
  release_end: number;
}
export type LabeledSegment = SegmentOffsets & { label: Seal };

export class PhaseClock {
  private t0: number | null = null;
  private marks = new Map<string, number>();
  freeStart: number | null = null;
  freeEnd: number | null = null;

  /** Call from the MediaRecorder 'start' handler. */
  anchor(now: number): void {
    this.t0 = now;
  }
  get origin(): number {
    if (this.t0 === null) throw new Error("PhaseClock not anchored");
    return this.t0;
  }
  get anchored(): boolean {
    return this.t0 !== null;
  }
  /** ms since clip start, rounded to integer. */
  since(now: number): number {
    if (this.t0 === null) throw new Error("PhaseClock not anchored");
    return Math.max(0, Math.round(now - this.t0));
  }
  mark(segment: number, name: MarkName, now: number): number {
    const v = this.since(now);
    this.marks.set(`${segment}:${name}`, v);
    return v;
  }
  markFree(kind: "start" | "end", now: number): number {
    const v = this.since(now);
    if (kind === "start") this.freeStart = v;
    else this.freeEnd = v;
    return v;
  }
  segments(labels: Seal[]): LabeledSegment[] {
    return labels.map((label, i) => {
      const g = (n: MarkName) => {
        const v = this.marks.get(`${i}:${n}`);
        if (v === undefined) throw new Error(`missing mark ${i}:${n}`);
        return v;
      };
      return {
        label,
        form_start: g("form_start"),
        hold_start: g("hold_start"),
        hold_end: g("hold_end"),
        release_end: g("release_end"),
      };
    });
  }
}

/** Clip-level span: first form_start, first hold_start, last hold_end, last release_end. */
export function spanOffsets(segs: SegmentOffsets[]): SegmentOffsets {
  const first = segs[0];
  const last = segs[segs.length - 1];
  return {
    form_start: first.form_start,
    hold_start: first.hold_start,
    hold_end: last.hold_end,
    release_end: last.release_end,
  };
}

/** Frames / seconds, rounded to 0.1. 0 if the window is degenerate. */
export function measureFps(frames: number, windowMs: number): number {
  if (windowMs <= 0 || frames <= 0) return 0;
  return Math.round((frames / (windowMs / 1000)) * 10) / 10;
}
