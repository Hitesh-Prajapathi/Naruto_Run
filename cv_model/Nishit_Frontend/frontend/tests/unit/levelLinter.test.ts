import { describe, expect, it } from "vitest";
import { assertLevelSegmentValid, lintLevelSegment } from "../../src/game/levelLinter";
import { SEGMENT_1 } from "../../src/config/levelDefinition";
import type { LevelSegment } from "../../src/config/levelDefinition";

function segment(obstacles: LevelSegment["obstacles"]): LevelSegment {
  return { id: "test-segment", durationSeconds: 60, obstacles };
}

describe("lintLevelSegment against the real authored level", () => {
  it("SEGMENT_1 has no lint issues", () => {
    expect(lintLevelSegment(SEGMENT_1)).toEqual([]);
  });

  it("assertLevelSegmentValid does not throw for SEGMENT_1", () => {
    expect(() => assertLevelSegmentValid(SEGMENT_1)).not.toThrow();
  });
});

describe("lintLevelSegment rule: no obstacle may span every lane", () => {
  it("flags an obstacle covering all 3 lanes", () => {
    const issues = lintLevelSegment(segment([{ atSeconds: 3, typeId: "boulder", lanes: [0, 1, 2] }]));
    expect(issues.map((i) => i.rule)).toContain("lane-span");
  });

  it("does not flag a 2-lane obstacle", () => {
    const issues = lintLevelSegment(segment([{ atSeconds: 3, typeId: "thornWall", lanes: [1, 2] }]));
    expect(issues.map((i) => i.rule)).not.toContain("lane-span");
  });
});

describe("lintLevelSegment rule: minimum gap between consecutive obstacles", () => {
  it("flags two obstacles closer than the general minimum gap", () => {
    const issues = lintLevelSegment(
      segment([
        { atSeconds: 3.0, typeId: "boulder", lanes: [0] },
        { atSeconds: 3.5, typeId: "boulder", lanes: [1] }, // 0.5s apart, well under ~1.57s
      ]),
    );
    expect(issues.map((i) => i.rule)).toContain("min-gap");
  });

  it("does not flag obstacles at exactly the generous authored spacing", () => {
    const issues = lintLevelSegment(
      segment([
        { atSeconds: 3.0, typeId: "boulder", lanes: [0] },
        { atSeconds: 7.0, typeId: "boulder", lanes: [1] },
      ]),
    );
    expect(issues).toEqual([]);
  });
});

describe("lintLevelSegment rule: jump/lane mixed-type gap", () => {
  it("flags a jump obstacle immediately followed by a lane obstacle inside 1.2s", () => {
    // Gap chosen between MIN_MIXED_TYPE_GAP_S(1.2) and MIN_OBSTACLE_GAP_S(~1.57)
    // so only the mixed-type rule (not the general min-gap rule) is exercised.
    const issues = lintLevelSegment(
      segment([
        { atSeconds: 3.0, typeId: "log", lanes: [0] }, // jump
        { atSeconds: 4.0, typeId: "boulder", lanes: [1] }, // lane, 1.0s later
      ]),
    );
    expect(issues.map((i) => i.rule)).toContain("mixed-type-gap");
  });

  it("does not flag two same-avoidance-type obstacles at that same short gap", () => {
    const issues = lintLevelSegment(
      segment([
        { atSeconds: 3.0, typeId: "log", lanes: [0] }, // jump
        { atSeconds: 4.0, typeId: "pit", lanes: [1] }, // also jump
      ]),
    );
    expect(issues.map((i) => i.rule)).not.toContain("mixed-type-gap");
  });
});

describe("lintLevelSegment structural checks", () => {
  it("flags an obstacle with no lanes", () => {
    const issues = lintLevelSegment(segment([{ atSeconds: 1, typeId: "boulder", lanes: [] }]));
    expect(issues.map((i) => i.rule)).toContain("empty-lanes");
  });

  it("flags a duplicated lane index", () => {
    const issues = lintLevelSegment(segment([{ atSeconds: 1, typeId: "boulder", lanes: [1, 1] }]));
    expect(issues.map((i) => i.rule)).toContain("duplicate-lane");
  });

  it("flags an out-of-range lane index", () => {
    // @ts-expect-error -- intentionally invalid input to prove the linter catches it
    const issues = lintLevelSegment(segment([{ atSeconds: 1, typeId: "boulder", lanes: [5] }]));
    expect(issues.map((i) => i.rule)).toContain("lane-range");
  });

  it("flags a negative spawn time", () => {
    const issues = lintLevelSegment(segment([{ atSeconds: -1, typeId: "boulder", lanes: [0] }]));
    expect(issues.map((i) => i.rule)).toContain("negative-time");
  });
});

describe("assertLevelSegmentValid", () => {
  it("throws with every rule violation listed", () => {
    const broken = segment([{ atSeconds: -1, typeId: "boulder", lanes: [0, 1, 2] }]);
    expect(() => assertLevelSegmentValid(broken)).toThrowError(/lane-span/);
    expect(() => assertLevelSegmentValid(broken)).toThrowError(/negative-time/);
  });
});
