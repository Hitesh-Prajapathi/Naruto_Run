/**
 * AABB collision detection with a forgiving player hitbox.
 * game_implementation_plan.md §4.5: the collision box is deliberately
 * smaller than the visual character model.
 *
 * Pure and framework-agnostic on purpose (plain {x,y,z} vectors in, plain
 * booleans/records out) so it's testable without a scene graph. The scene
 * layer is responsible for reading Object3D positions and calling in here,
 * never the reverse.
 *
 * Phase C scope: detect overlap only. Applying damage/HP is Phase F's
 * battle system -- this module has no concept of health.
 */

export interface Vector3Like {
  x: number;
  y: number;
  z: number;
}

export interface AABB {
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
  minZ: number;
  maxZ: number;
}

/** Builds an AABB centered horizontally on `center`, with its base at
 * `center.y` (ground level) rather than centered vertically -- obstacles and
 * the player are both placed with y=0 as their feet/base. */
export function aabbFromGroundCenter(center: Vector3Like, size: Vector3Like): AABB {
  const halfX = size.x / 2;
  const halfZ = size.z / 2;
  return {
    minX: center.x - halfX,
    maxX: center.x + halfX,
    minY: center.y,
    maxY: center.y + size.y,
    minZ: center.z - halfZ,
    maxZ: center.z + halfZ,
  };
}

export function aabbOverlap(a: AABB, b: AABB): boolean {
  return (
    a.minX <= b.maxX &&
    a.maxX >= b.minX &&
    a.minY <= b.maxY &&
    a.maxY >= b.minY &&
    a.minZ <= b.maxZ &&
    a.maxZ >= b.minZ
  );
}

export interface CollidableObstacle {
  id: number;
  typeId: string;
  aabb: AABB;
}

export interface CollisionHit {
  obstacleId: number;
  typeId: string;
}

/** Returns every currently-active obstacle whose AABB overlaps the player's.
 * Order matches `obstacles`' input order; callers that only care about "hit
 * or not" should just check `.length > 0`. */
export function checkPlayerCollisions(
  playerAabb: AABB,
  obstacles: readonly CollidableObstacle[],
): CollisionHit[] {
  const hits: CollisionHit[] = [];
  for (const obstacle of obstacles) {
    if (aabbOverlap(playerAabb, obstacle.aabb)) {
      hits.push({ obstacleId: obstacle.id, typeId: obstacle.typeId });
    }
  }
  return hits;
}
