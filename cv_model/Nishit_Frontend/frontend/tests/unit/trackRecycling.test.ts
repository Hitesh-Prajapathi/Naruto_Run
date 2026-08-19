import { describe, expect, it } from "vitest";
import { advanceTilePositions, initialTilePositions } from "../../src/scene/trackRecycling";

describe("initialTilePositions", () => {
  it("produces evenly spaced positions starting at 0 and extending forward", () => {
    expect(initialTilePositions(4, 40)).toEqual([0, -40, -80, -120]);
  });
});

describe("advanceTilePositions", () => {
  it("shifts every tile by deltaZ when nothing crosses the threshold", () => {
    const positions = [0, -40, -80];
    const next = advanceTilePositions(positions, 5, 40, 6);
    expect(next).toEqual([5, -35, -75]);
  });

  it("wraps a tile that crosses the recycle threshold back by one full span", () => {
    const positions = [4, -36, -76]; // span = 3*40 = 120
    const next = advanceTilePositions(positions, 5, 40, 6);
    // 4 + 5 = 9 > threshold(6) -> 9 - 120 = -111
    expect(next).toEqual([-111, -31, -71]);
  });

  it("keeps the strip continuous over many steps (no gaps or overlaps)", () => {
    let positions = initialTilePositions(5, 40);
    const tileLength = 40;
    const threshold = 6;

    for (let i = 0; i < 500; i += 1) {
      positions = advanceTilePositions(positions, 1.3, tileLength, threshold);
      const sorted = [...positions].sort((a, b) => b - a);
      for (let j = 1; j < sorted.length; j += 1) {
        expect(sorted[j - 1]! - sorted[j]!).toBeCloseTo(tileLength, 5);
      }
    }
  });

  it("never leaves a tile past the recycle threshold by more than one delta step", () => {
    let positions = initialTilePositions(4, 40);
    const deltaZ = 2;
    for (let i = 0; i < 200; i += 1) {
      positions = advanceTilePositions(positions, deltaZ, 40, 6);
      for (const z of positions) {
        expect(z).toBeLessThanOrEqual(6 + deltaZ);
      }
    }
  });
});
