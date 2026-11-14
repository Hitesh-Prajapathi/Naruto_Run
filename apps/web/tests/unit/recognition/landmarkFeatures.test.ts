// @vitest-environment node
import { describe, expect, it } from "vitest";
import goldenRaw from "../../../../../tests/golden/features_v1.json";
import {
  FEATURE_DIM,
  assignSlots,
  canonicalizeHand,
  landmarksFromObjects,
  twoHandFeatures,
  type HandObservation,
} from "../../../src/recognition";
import { maxAbsDiff, revive } from "./goldenTypes";

interface Hand {
  landmarks: number[][];
  handedness: string;
  score?: number;
}
interface Case {
  name: string;
  image_size: [number, number];
  hands: Hand[];
}
const cases = revive((goldenRaw as unknown as { cases: Case[] }).cases);
const byName = (name: string): Case => cases.find((c) => c.name === name) as Case;
const firstHand = (name: string): Hand => byName(name).hands[0] as Hand;
const SIZE: [number, number] = [640, 480];

describe("landmark features: TS-native behaviour", () => {
  it("returns 84 zeros for no hands", () => {
    const out = twoHandFeatures([], SIZE);
    expect(out.length).toBe(FEATURE_DIM);
    expect(Array.from(out).every((v) => v === 0)).toBe(true);
  });

  it("is invariant to translation and uniform scale of the pixel-space input", () => {
    const hand = firstHand("single_right_curl0.6");
    const ref = twoHandFeatures([hand], SIZE);
    const moved = hand.landmarks.map((p) => [
      (p[0] as number) * 0.5 + 0.2,
      (p[1] as number) * 0.5 + 0.1,
      (p[2] as number) * 0.5,
    ]);
    // 0.5 scale on x,y,z in normalised space is a uniform pixel-space scale (z uses width).
    // The aspect ratio differs between x and y scaling only through the image size, which
    // is unchanged, so this is a similarity transform.
    const out = twoHandFeatures([{ ...hand, landmarks: moved }], SIZE);
    expect(maxAbsDiff(ref, out)).toBeLessThan(1e-4);
  });

  it("a mirrored right hand labelled Left gives the same per-hand features", () => {
    const hand = firstHand("single_right_curl0.6");
    const mirrored = hand.landmarks.map((p) => [1 - (p[0] as number), p[1] as number, p[2] as number]);
    const right = twoHandFeatures([hand], SIZE);
    const left = twoHandFeatures([{ ...hand, landmarks: mirrored, handedness: "Left" }], SIZE);
    expect(maxAbsDiff(right.slice(25, 50), left.slice(0, 25))).toBeLessThan(1e-4);
  });

  it("same-label hands are split by mean x; degenerate hands are dropped", () => {
    const c = byName("same_label_right_right_swapped");
    const [first, second] = c.hands as [Hand, Hand];
    const [l, r] = assignSlots([first, second] as HandObservation[], c.image_size);
    expect(l).toBe(second);
    expect(r).toBe(first);
    const nan = byName("one_valid_one_nan");
    const [l2, r2] = assignSlots(nan.hands as HandObservation[], nan.image_size);
    expect(l2).not.toBeNull();
    expect(r2).toBeNull();
  });

  it("canonicalisation puts the middle MCP at (0, 1, 0)", () => {
    const hand = firstHand("single_left_curl0.5");
    const ch = canonicalizeHand(hand.landmarks, "Left", SIZE);
    expect(ch.valid).toBe(true);
    expect(ch.mirrored).toBe(true);
    const mid = ch.points[9] as [number, number, number];
    expect(mid[0]).toBeCloseTo(0, 12);
    expect(mid[1]).toBeCloseTo(1, 12);
    expect(mid[2]).toBeCloseTo(0, 12);
  });

  it("throws on malformed landmarks and unknown handedness", () => {
    const lm = firstHand("single_right_curl0.6").landmarks;
    expect(() => twoHandFeatures([{ landmarks: lm.slice(0, 20), handedness: "Right" }], SIZE)).toThrow();
    expect(() => twoHandFeatures([{ landmarks: lm, handedness: "Both" }], SIZE)).toThrow();
  });

  it("landmarksFromObjects adapts MediaPipe-style points", () => {
    expect(landmarksFromObjects([{ x: 1, y: 2, z: 3 }])).toEqual([[1, 2, 3]]);
  });
});
