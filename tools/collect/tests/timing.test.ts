import { describe, expect, it } from "vitest";
import { measureFps, PhaseClock, spanOffsets } from "../src/timing";
import { generatePlan } from "../src/protocol";
import { cameraWarnings, pickMimeType } from "../src/recorder";
import { simulateClip } from "./helpers";

describe("PhaseClock", () => {
  it("records integer ms offsets relative to the anchor", () => {
    const c = new PhaseClock();
    c.anchor(5000.4);
    expect(c.mark(0, "form_start", 8000.4)).toBe(3000);
    expect(Number.isInteger(c.mark(0, "hold_start", 10000.123))).toBe(true);
  });
  it("refuses to mark before the recorder started", () => {
    expect(() => new PhaseClock().mark(0, "form_start", 1)).toThrow();
  });
  it("requires all four marks per segment", () => {
    const c = new PhaseClock();
    c.anchor(0);
    c.mark(0, "form_start", 1);
    expect(() => c.segments(["rat"])).toThrow(/missing mark/);
  });
  it("seal clip offsets equal the nominal timeline, shifted by jitter", () => {
    const plan = generatePlan("t");
    const m = simulateClip(plan[0], 12);
    const s = m.segments![0];
    expect([s.form_start, s.hold_start, s.hold_end, s.release_end]).toEqual([3012, 5012, 8012, 9512]);
  });
  it("combo span uses first form/hold and last hold_end/release_end", () => {
    const combo = generatePlan("t").find((p) => p.block === "combo")!;
    const m = simulateClip(combo);
    expect(spanOffsets(m.segments!)).toEqual({ form_start: 3000, hold_start: 4500, hold_end: 9500, release_end: 10000 });
  });
  it("phase boundaries are strictly ordered", () => {
    for (const p of generatePlan("t").filter((x) => x.block !== "free")) {
      const segs = simulateClip(p).segments!;
      segs.forEach((s, i) => {
        expect(s.form_start).toBeLessThan(s.hold_start);
        expect(s.hold_start).toBeLessThan(s.hold_end);
        expect(s.hold_end).toBeLessThan(s.release_end);
        if (i > 0) expect(s.form_start).toBeGreaterThanOrEqual(segs[i - 1].release_end);
      });
    }
  });
});

describe("helpers", () => {
  it("measureFps", () => {
    expect(measureFps(60, 2000)).toBe(30);
    expect(measureFps(0, 2000)).toBe(0);
    expect(measureFps(10, 0)).toBe(0);
  });
  it("cameraWarnings: fps < 24 and resolution < 640x480, orientation-agnostic", () => {
    expect(cameraWarnings(1280, 720, 30)).toEqual({ lowFps: false, lowRes: false });
    expect(cameraWarnings(1280, 720, 15)).toEqual({ lowFps: true, lowRes: false });
    expect(cameraWarnings(320, 240, 30)).toEqual({ lowFps: false, lowRes: true });
    expect(cameraWarnings(720, 1280, 30).lowRes).toBe(false); // phone portrait
  });
  it("pickMimeType prefers vp9, then vp8, then mp4", () => {
    expect(pickMimeType(() => true)).toBe("video/webm;codecs=vp9");
    expect(pickMimeType((m) => m !== "video/webm;codecs=vp9")).toBe("video/webm;codecs=vp8");
    expect(pickMimeType((m) => m === "video/mp4")).toBe("video/mp4");
    expect(pickMimeType(() => false)).toBeNull();
  });
});
