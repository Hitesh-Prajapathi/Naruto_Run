/**
 * Pure math for the ground-tile conveyor-belt trick: N tiles laid end to end
 * covering a fixed span, each frame shifted by the world scroll delta; any
 * tile that scrolls past the recycle threshold jumps back by exactly one
 * full span, landing seamlessly at the far end. No tile is ever created or
 * destroyed after the initial N -- see objectPool.ts's doc comment for why
 * that matters for the Phase-C "no allocation spikes" gate.
 *
 * Framework-agnostic (no Three.js) so this is unit-testable directly.
 */

export function advanceTilePositions(
  positions: readonly number[],
  deltaZ: number,
  tileLength: number,
  recycleThresholdZ: number,
): number[] {
  const span = positions.length * tileLength;
  return positions.map((z) => {
    const moved = z + deltaZ;
    return moved > recycleThresholdZ ? moved - span : moved;
  });
}

/** Initial evenly-spaced tile positions covering one full span, starting at
 * z = 0 and extending in the -z (forward) direction. */
export function initialTilePositions(tileCount: number, tileLength: number): number[] {
  // `index === 0 ? 0 : ...` rather than `-index * tileLength` for index 0:
  // the latter produces -0, which is numerically identical but a needless
  // surprise in equality assertions and debug output.
  return Array.from({ length: tileCount }, (_, index) => (index === 0 ? 0 : -index * tileLength));
}
