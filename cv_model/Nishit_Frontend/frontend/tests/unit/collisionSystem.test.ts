import { describe, expect, it } from "vitest";
import { aabbFromGroundCenter, aabbOverlap, checkPlayerCollisions } from "../../src/game/collisionSystem";

describe("aabbFromGroundCenter", () => {
  it("centers the box horizontally and bases it at center.y", () => {
    const box = aabbFromGroundCenter({ x: 2, y: 0, z: -10 }, { x: 4, y: 3, z: 2 });
    expect(box).toEqual({ minX: 0, maxX: 4, minY: 0, maxY: 3, minZ: -11, maxZ: -9 });
  });

  it("supports a non-zero ground y (e.g. a player mid-jump)", () => {
    const box = aabbFromGroundCenter({ x: 0, y: 1.4, z: 0 }, { x: 1, y: 2, z: 1 });
    expect(box.minY).toBe(1.4);
    expect(box.maxY).toBe(3.4);
  });
});

describe("aabbOverlap", () => {
  it("detects overlapping boxes", () => {
    const a = aabbFromGroundCenter({ x: 0, y: 0, z: 0 }, { x: 2, y: 2, z: 2 });
    const b = aabbFromGroundCenter({ x: 1, y: 0, z: 0 }, { x: 2, y: 2, z: 2 });
    expect(aabbOverlap(a, b)).toBe(true);
  });

  it("detects non-overlapping boxes separated on X", () => {
    const a = aabbFromGroundCenter({ x: 0, y: 0, z: 0 }, { x: 1, y: 1, z: 1 });
    const b = aabbFromGroundCenter({ x: 5, y: 0, z: 0 }, { x: 1, y: 1, z: 1 });
    expect(aabbOverlap(a, b)).toBe(false);
  });

  it("a player high enough in a jump clears a low obstacle", () => {
    const log = aabbFromGroundCenter({ x: 0, y: 0, z: -50 }, { x: 3, y: 0.6, z: 0.8 });
    const jumpingPlayer = aabbFromGroundCenter({ x: 0, y: 1.4, z: -50 }, { x: 0.7, y: 1.7, z: 0.7 });
    expect(aabbOverlap(log, jumpingPlayer)).toBe(false);
  });

  it("a grounded player does not clear the same low obstacle", () => {
    const log = aabbFromGroundCenter({ x: 0, y: 0, z: -50 }, { x: 3, y: 0.6, z: 0.8 });
    const groundedPlayer = aabbFromGroundCenter({ x: 0, y: 0, z: -50 }, { x: 0.7, y: 1.7, z: 0.7 });
    expect(aabbOverlap(log, groundedPlayer)).toBe(true);
  });

  it("touching-but-not-overlapping boxes count as overlapping (inclusive bounds)", () => {
    const a = aabbFromGroundCenter({ x: 0, y: 0, z: 0 }, { x: 2, y: 2, z: 2 });
    const b = aabbFromGroundCenter({ x: 2, y: 0, z: 0 }, { x: 2, y: 2, z: 2 }); // shares the x=1 edge
    expect(aabbOverlap(a, b)).toBe(true);
  });
});

describe("checkPlayerCollisions", () => {
  it("returns a hit for every overlapping obstacle, none for the rest", () => {
    const player = aabbFromGroundCenter({ x: 0, y: 0, z: -50 }, { x: 0.7, y: 1.7, z: 0.7 });
    const hits = checkPlayerCollisions(player, [
      { id: 1, typeId: "boulder", aabb: aabbFromGroundCenter({ x: 0, y: 0, z: -50 }, { x: 1.6, y: 1.8, z: 1.6 }) },
      { id: 2, typeId: "boulder", aabb: aabbFromGroundCenter({ x: 2.2, y: 0, z: -50 }, { x: 1.6, y: 1.8, z: 1.6 }) },
    ]);
    expect(hits).toEqual([{ obstacleId: 1, typeId: "boulder" }]);
  });

  it("returns an empty array when nothing overlaps", () => {
    const player = aabbFromGroundCenter({ x: 0, y: 0, z: 0 }, { x: 0.7, y: 1.7, z: 0.7 });
    expect(checkPlayerCollisions(player, [])).toEqual([]);
  });
});
