// @vitest-environment node
import { afterAll, describe, expect, it } from "vitest";
import goldenRaw from "../../../../../tests/golden/calibration_v1.json";
import {
  StackedFusion,
  TemperatureScaler,
  VectorScaler,
  buildFusionFeatures,
  logSoftmax,
  softmax,
  type StackedFusionDict,
} from "../../../src/recognition";
import { maxAbsDiff, report, track } from "./goldenTypes";

type Matrix = number[][];
interface Golden {
  tolerance: number;
  cases: Record<string, unknown>[];
}
const golden = goldenRaw as unknown as Golden;
const TOL = golden.tolerance;

function find<T>(kind: string): T {
  return golden.cases.find((c) => c["kind"] === kind) as unknown as T;
}

function rowsDev(family: string, got: ArrayLike<number>[], want: Matrix): void {
  expect(got.length).toBe(want.length);
  let worst = 0;
  got.forEach((g, i) => {
    worst = Math.max(worst, maxAbsDiff(g, want[i] as number[]));
  });
  track(family, worst);
  expect(worst).toBeLessThanOrEqual(TOL);
}

afterAll(report);

describe("calibration: golden parity with Python", () => {
  it("temperature scaler", () => {
    const group = find<{
      cases: {
        scaler: { type: "temperature"; temperature: number; fitted: boolean };
        logits: Matrix;
        expected_proba: Matrix;
        expected_logits: Matrix;
      }[];
    }>("temperature");
    expect(group.cases.length).toBeGreaterThanOrEqual(5);
    for (const c of group.cases) {
      const ts = TemperatureScaler.fromDict(c.scaler);
      expect(ts.toDict()).toEqual(c.scaler);
      rowsDev(
        "calibration.temperature.proba",
        c.logits.map((r) => ts.transform(r)),
        c.expected_proba,
      );
      rowsDev(
        "calibration.temperature.logits",
        c.logits.map((r) => ts.transformLogits(r)),
        c.expected_logits,
      );
    }
  });

  it("vector scaler", () => {
    const c = find<{ scaler: Record<string, unknown>; logits: Matrix; expected_proba: Matrix }>("vector");
    const vs = VectorScaler.fromDict(c.scaler);
    rowsDev(
      "calibration.vector",
      c.logits.map((r) => vs.transform(r)),
      c.expected_proba,
    );
  });

  it("stacked fusion pipeline: features -> fusion logits -> proba -> temperature", () => {
    const c = find<{
      fusion: StackedFusionDict;
      branch_probs: Matrix[];
      meta: Matrix;
      expected_features: Matrix;
      expected_logits: Matrix;
      expected_proba: Matrix;
      temperature: { type: "temperature"; temperature: number };
      expected_calibrated_proba: Matrix;
    }>("stacked_fusion_pipeline");
    const sf = StackedFusion.fromDict(c.fusion);
    const ts = TemperatureScaler.fromDict(c.temperature);
    const feats = c.meta.map((meta, i) =>
      buildFusionFeatures(
        c.branch_probs.map((b) => b[i] as number[]),
        meta,
      ),
    );
    rowsDev("calibration.fusion.features", feats, c.expected_features);
    const logits = feats.map((f) => sf.predictLogits(f));
    rowsDev("calibration.fusion.logits", logits, c.expected_logits);
    rowsDev(
      "calibration.fusion.proba",
      feats.map((f) => sf.predictProba(f)),
      c.expected_proba,
    );
    rowsDev(
      "calibration.fusion.calibrated",
      logits.map((l) => ts.transform(l)),
      c.expected_calibrated_proba,
    );
  });

  it("stacked fusion without standardisation", () => {
    const c = find<{
      fusion: StackedFusionDict;
      features: Matrix;
      expected_logits: Matrix;
      expected_proba: Matrix;
    }>("stacked_fusion_no_standardize");
    const sf = StackedFusion.fromDict(c.fusion);
    rowsDev(
      "calibration.fusion.nostd.logits",
      c.features.map((f) => sf.predictLogits(f)),
      c.expected_logits,
    );
    rowsDev(
      "calibration.fusion.nostd.proba",
      c.features.map((f) => sf.predictProba(f)),
      c.expected_proba,
    );
  });
});

describe("calibration: TS-native behaviour", () => {
  it("softmax sums to one and is shift invariant, even for huge logits", () => {
    const p = softmax([1000, 1001, 999]);
    expect(Array.from(p).reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
    expect(Math.max(...logSoftmax([1000, 1001, 999]))).toBeLessThan(0);
    expect(maxAbsDiff(softmax([1, 2, 3]), softmax([101, 102, 103]))).toBeLessThan(1e-12);
  });

  it("rejects out-of-range temperature and wrong payload types", () => {
    expect(() => new TemperatureScaler(0.01)).toThrow();
    expect(() => new TemperatureScaler(25)).toThrow();
    expect(() => TemperatureScaler.fromDict({ type: "vector", temperature: 1 })).toThrow();
    expect(() => StackedFusion.fromDict({ type: "x" } as unknown as StackedFusionDict)).toThrow();
  });

  it("fusion rejects a feature row of the wrong length", () => {
    const sf = StackedFusion.fromDict({
      type: "stacked_fusion",
      l2: 0.01,
      standardize: true,
      mean: [0, 0],
      std: [1, 1],
      weights: [
        [1, 0],
        [0, 1],
      ],
      bias: [0, 0],
    });
    expect(() => sf.predictLogits([1, 2, 3])).toThrow();
    expect(Array.from(sf.predictLogits([2, 3]))).toEqual([2, 3]);
  });
});
