// @vitest-environment node
import { afterAll, describe, expect, it } from "vitest";
import goldenRaw from "../../../../../tests/golden/decoder_v1.json";
import {
  SealDecoder,
  decoderConfigFromDict,
  defaultDecoderConfig,
  makeDecoderConfig,
  type DecoderConfig,
  type SealEvent,
} from "../../../src/recognition";
import { maxAbsDiff, report, track } from "./goldenTypes";

type Trace = ([number, number] | null)[];
interface Frame {
  t_ms?: number;
  probs?: number[];
  expected_next?: string[];
  reset?: boolean;
}
interface Case {
  name: string;
  config: Record<string, unknown>;
  frames: Frame[];
  expected: {
    events: SealEvent[];
    stats: { frames: number; nonmonotonic: number; gaps: number; events: number };
    trace: Trace;
    final_smoothed: number[];
    final_active: string | null;
  };
}
interface Golden {
  label_order: string[];
  default_config: Record<string, unknown>;
  cases: Case[];
}
const golden = goldenRaw as unknown as Golden;

afterAll(report);

describe("seal decoder: golden parity with Python", () => {
  it("default config equals Python's DecoderConfig()", () => {
    expect(defaultDecoderConfig()).toEqual(golden.default_config);
    expect(makeDecoderConfig()).toEqual(golden.default_config);
    expect(defaultDecoderConfig().labels).toEqual(golden.label_order);
  });

  it("covers the required stream families", () => {
    expect(golden.cases.length).toBeGreaterThanOrEqual(15);
    const stats = golden.cases.map((c) => c.expected.stats);
    expect(stats.some((s) => s.gaps > 0)).toBe(true);
    expect(stats.some((s) => s.nonmonotonic > 0)).toBe(true);
    expect(golden.cases.some((c) => c.config["smoother"] === "hmm")).toBe(true);
  });

  for (const c of golden.cases) {
    it(`decoder: ${c.name}`, () => {
      const dec = new SealDecoder(decoderConfigFromDict(c.config));
      expect(dec.config).toEqual(c.config);
      const events: SealEvent[] = [];
      const trace: Trace = [];
      for (const fr of c.frames) {
        if (fr.reset) {
          dec.reset();
          trace.push(null);
          continue;
        }
        events.push(
          ...dec.update(fr.probs as number[], fr.t_ms as number, {
            expectedNext: fr.expected_next,
          }),
        );
        const s = dec.smoothed;
        let top = 0;
        for (let i = 1; i < s.length; i += 1) if ((s[i] as number) > (s[top] as number)) top = i;
        trace.push([top, s[top] as number]);
      }
      // Events: label and times exact, confidence within 1e-9.
      expect(events.length).toBe(c.expected.events.length);
      events.forEach((e, i) => {
        const x = c.expected.events[i] as SealEvent;
        expect(e.label).toBe(x.label);
        expect(e.t_start_ms).toBe(x.t_start_ms);
        expect(e.t_emit_ms).toBe(x.t_emit_ms);
        const dev = Math.abs(e.confidence - x.confidence);
        track("decoder.confidence", dev);
        expect(dev).toBeLessThanOrEqual(1e-9);
      });
      expect(dec.stats).toEqual(c.expected.stats);
      expect(dec.activeLabel).toBe(c.expected.final_active);
      // Per-frame argmax identical; smoothed max and final vector within 1e-9.
      expect(trace.length).toBe(c.expected.trace.length);
      let worst = 0;
      trace.forEach((t, i) => {
        const x = c.expected.trace[i] ?? null;
        if (x === null || t === null) {
          expect(t).toBe(x);
          return;
        }
        expect(t[0]).toBe(x[0]);
        worst = Math.max(worst, Math.abs(t[1] - x[1]));
      });
      worst = Math.max(worst, maxAbsDiff(dec.smoothed, c.expected.final_smoothed));
      track("decoder.smoothed", worst);
      expect(worst).toBeLessThanOrEqual(1e-9);
    });
  }
});

describe("seal decoder: TS-native behaviour", () => {
  const labels = defaultDecoderConfig().labels;
  const onehot = (name: string, p = 0.95): number[] =>
    labels.map((l) => (l === name ? p : (1 - p) / (labels.length - 1)));

  it("emits one event for a held seal and none for a held idle", () => {
    const dec = new SealDecoder();
    const events: SealEvent[] = [];
    for (let i = 0; i < 40; i += 1) events.push(...dec.update(onehot("zero"), i * 33));
    expect(events).toHaveLength(0);
    for (let i = 40; i < 100; i += 1) events.push(...dec.update(onehot("tiger"), i * 33));
    expect(events).toHaveLength(1);
    expect(events[0]?.label).toBe("tiger");
    expect(dec.activeLabel).toBe("tiger");
  });

  it("ignores non-monotonic and duplicate timestamps without touching state", () => {
    const dec = new SealDecoder();
    dec.update(onehot("tiger"), 100);
    const before = Array.from(dec.smoothed);
    expect(dec.update(onehot("ram"), 100)).toEqual([]);
    expect(dec.update(onehot("ram"), 50)).toEqual([]);
    expect(dec.stats.nonmonotonic).toBe(2);
    expect(dec.stats.frames).toBe(1);
    expect(Array.from(dec.smoothed)).toEqual(before);
  });

  it("throws on bad input without mutating state", () => {
    const dec = new SealDecoder();
    const withNaN = onehot("tiger").map((v, i) => (i === 0 ? Number.NaN : v));
    expect(() => dec.update([1, 0], 0)).toThrow();
    expect(() => dec.update(withNaN, 0)).toThrow();
    expect(() => dec.update(onehot("tiger"), Number.NaN)).toThrow();
    expect(() => dec.update(onehot("tiger"), 0, { expectedNext: ["nope"] })).toThrow();
    expect(dec.stats.frames).toBe(0);
  });

  it("validates configuration like Python", () => {
    expect(() => makeDecoderConfig({ tau_on: 0.3 })).toThrow();
    expect(() => makeDecoderConfig({ labels: ["a", "a"] })).toThrow();
    expect(() => makeDecoderConfig({ idle_label: "missing" })).toThrow();
    expect(() => makeDecoderConfig({ hmm_p_stay: 0.01 })).toThrow();
    expect(() => decoderConfigFromDict({ bogus: 1 })).toThrow();
    const cfg: DecoderConfig = makeDecoderConfig({ smoother: "hmm" });
    expect(cfg.smoother).toBe("hmm");
  });

  it("reset clears refractory, stats and smoothing", () => {
    const dec = new SealDecoder();
    for (let i = 0; i < 40; i += 1) dec.update(onehot("tiger"), i * 33);
    expect(dec.stats.events).toBe(1);
    dec.reset();
    expect(dec.stats).toEqual({ frames: 0, nonmonotonic: 0, gaps: 0, events: 0 });
    expect(dec.smoothed.length).toBe(0);
    expect(dec.activeLabel).toBeNull();
  });
});
