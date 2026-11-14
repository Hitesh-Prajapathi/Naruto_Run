// @vitest-environment node
import { afterAll, describe, expect, it } from "vitest";
import goldenRaw from "../../../../../tests/golden/features_v1.json";
import specRaw from "../../../../../docs/specs/landmark_features_v1.json";
import {
  FEATURE_DIM,
  FEATURE_NAMES,
  FEATURE_SPEC_VERSION,
  twoHandFeatures,
  type HandObservation,
} from "../../../src/recognition";
import { maxAbsDiff, report, revive, track } from "./goldenTypes";

interface GoldenHand {
  landmarks: number[][];
  handedness: string;
  score?: number;
}
interface GoldenCase {
  name: string;
  image_size: [number, number];
  hands: GoldenHand[];
  expected: number[] | null;
  expect_error: boolean;
}
interface Golden {
  spec_version: string;
  feature_dim: number;
  tolerance: number;
  cases: GoldenCase[];
}

const golden = revive(goldenRaw as unknown as Golden);
const spec = specRaw as unknown as {
  spec_version: string;
  feature_dim: number;
  features: { name: string }[];
};

function toObservations(hands: GoldenHand[]): HandObservation[] {
  return hands.map((h) => (h.score === undefined ? { landmarks: h.landmarks, handedness: h.handedness } : { ...h }));
}

afterAll(report);

describe("landmark features: golden parity with Python", () => {
  it("fixture and TS agree on spec version and dimensions", () => {
    expect(golden.spec_version).toBe(FEATURE_SPEC_VERSION);
    expect(golden.feature_dim).toBe(FEATURE_DIM);
    expect(spec.spec_version).toBe(FEATURE_SPEC_VERSION);
    expect(spec.feature_dim).toBe(FEATURE_DIM);
    expect(golden.cases.length).toBeGreaterThanOrEqual(40);
  });

  it("feature names match docs/specs/landmark_features_v1.json", () => {
    expect([...FEATURE_NAMES]).toEqual(spec.features.map((f) => f.name));
  });

  for (const c of golden.cases) {
    it(`features: ${c.name}`, () => {
      const hands = toObservations(c.hands);
      if (c.expect_error) {
        expect(() => twoHandFeatures(hands, c.image_size)).toThrow();
        return;
      }
      const out = twoHandFeatures(hands, c.image_size);
      const expected = c.expected as number[];
      expect(out).toBeInstanceOf(Float32Array);
      expect(out.length).toBe(FEATURE_DIM);
      const dev = maxAbsDiff(out, expected);
      track("features", dev);
      expect(dev).toBeLessThanOrEqual(golden.tolerance);
      // Discrete decisions must match exactly: palms_facing flag and meta block.
      for (const i of [79, 80, 81, 82]) expect(out[i]).toBe(expected[i]);
    });
  }
});
