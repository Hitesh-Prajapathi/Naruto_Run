/**
 * Track, lane, camera, and obstacle-catalog constants for Phase C
 * (environment/map) onward. Numbers match game_implementation_plan.md §4.2
 * ("Track and lanes") and §4.3 ("Obstacles") exactly -- this file is the one
 * place those numbers live; scene/game code reads from here rather than
 * hardcoding them again.
 *
 * World scroll speed is held constant in Phase C. The plan's optional
 * naruto_run "×1.35 speed boost" is a body-movement/gameplay decision that
 * belongs to Phase E (wiring real CV input) and is deliberately not modeled
 * here yet -- see game_implementation_plan.md §1.1.
 */

/** Left, center, right lane X positions, in that index order. */
export const LANE_X = [-2.2, 0, 2.2] as const;
export type LaneIndex = 0 | 1 | 2;
export const LANE_COUNT = LANE_X.length;
export const LANE_WIDTH = 2.2; // spacing used to size the ground/props, not just the X positions
export const TRACK_WIDTH = LANE_WIDTH * LANE_COUNT;

export const LANE_CHANGE_DURATION_S = 0.18;

/** Constant world scroll speed, in world units per second. */
export const BASE_RUN_SPEED_U_S = 14;

/** Obstacles are spawned this far ahead of the player (negative Z, forward). */
export const OBSTACLE_SPAWN_Z = -90;
/** An obstacle is recycled once it's this far behind the player. */
export const OBSTACLE_DESPAWN_Z = 6;

/** Minimum world-unit gap the level linter enforces between any two
 * consecutive obstacles -- game_implementation_plan.md §4.3: "Min gap between
 * obstacles: 22 u (≈1.6 s)". */
export const MIN_OBSTACLE_GAP_U = 22;
export const MIN_OBSTACLE_GAP_S = MIN_OBSTACLE_GAP_U / BASE_RUN_SPEED_U_S;

/** Minimum seconds required between a jump-avoidance obstacle and a
 * lane-avoidance obstacle specifically (game_implementation_plan.md §5
 * obstacle table note). In practice MIN_OBSTACLE_GAP_S already exceeds this,
 * but the linter checks both named rules independently rather than relying
 * on that being true forever if the constants above ever change. */
export const MIN_MIXED_TYPE_GAP_S = 1.2;

/** No obstacle placement may span every lane -- there would be no valid
 * player input. */
export const MAX_OBSTACLE_LANE_SPAN = LANE_COUNT - 1;

/**
 * Camera rig -- naruto_run_agent_brief.md §C. Behind and above the
 * character, pitched down, so the player can actually see approaching
 * obstacles (the old ground-level camera made the character occlude the
 * path ahead, which was a gameplay failure, not just a visual one).
 */
export const CAMERA_POSITION = { x: 0, y: 3.0, z: 6.0 } as const;
/** The camera aims at a point this far ahead of the player. Together with
 * CAMERA_POSITION this gives ~12° of downward pitch and, critically, keeps
 * the whole character inside the frame: an earlier, closer rig (z=4.5 aimed
 * 8 units ahead) pushed the character's feet ~26° below the view axis,
 * right against the bottom edge, so they were clipped off-screen. */
export const CAMERA_LOOK_AHEAD_Z = -4;
export const CAMERA_LOOK_AT_HEIGHT = 0.9;
export const CAMERA_FOV_DEG = 65;
export const CAMERA_NEAR = 0.1;
export const CAMERA_FAR = 300;
/** Camera tracks only part of the player's lateral offset, and arrives late
 * (see CAMERA_FOLLOW_LAMBDA) -- a full 1:1 lateral follow makes lane changes
 * nearly invisible because the character never moves within the frame. */
export const CAMERA_LATERAL_FOLLOW = 0.5;
/** Exponential-damping constant for frame-rate-independent smoothing:
 * position.lerp(target, 1 - exp(-lambda * dt)). Higher = snappier. */
export const CAMERA_FOLLOW_LAMBDA = 6;

/**
 * Ground tiles recycle one at a time, each only once it has fully scrolled
 * past GROUND_RECYCLE_THRESHOLD_Z -- so right after any single recycle
 * event, the belt's nearest edge is temporarily GROUND_TILE_LENGTH short of
 * that threshold (a "sawtooth" dip that repeats every tile-length of
 * travel). GROUND_TILE_LENGTH is deliberately much smaller than an
 * obstacle's despawn distance (and GROUND_RECYCLE_THRESHOLD_Z deliberately
 * separate from OBSTACLE_DESPAWN_Z) so that worst-case dip still lands past
 * the camera -- reusing OBSTACLE_DESPAWN_Z (6) with 40-unit tiles produced a
 * dip reaching all the way back to -34, i.e. no visible ground under large
 * stretches of the track. See trackRecycling.ts for the recycling math and
 * jungleTrack.test.ts / the Playwright scene screenshot for how this is
 * verified.
 */
/**
 * Width of the grass terrain that flanks the dirt path. The path itself is
 * only TRACK_WIDTH (~6.6u) across; without this, everything placed beside it
 * -- the entire treeline -- stood over empty space with the sky showing
 * through underneath. Wide enough to reach past the far tree band and the
 * fog's far plane, so no edge is ever visible.
 */
export const TERRAIN_WIDTH = 260;

export const GROUND_TILE_LENGTH = 10;
/** Recycle once a tile scrolls this far past the player -- must clear
 * CAMERA_POSITION.z (8) by more than one GROUND_TILE_LENGTH so the
 * worst-case sawtooth dip (threshold - GROUND_TILE_LENGTH) still lands
 * beyond the camera. */
export const GROUND_RECYCLE_THRESHOLD_Z = 20;
/** Enough tiles that even the worst-case dip still reaches back past
 * OBSTACLE_SPAWN_Z, plus one spare tile of margin. */
export const GROUND_TILE_COUNT =
  Math.ceil((GROUND_RECYCLE_THRESHOLD_Z - OBSTACLE_SPAWN_Z) / GROUND_TILE_LENGTH) + 1;

/**
 * World scale contract -- naruto_run_agent_brief.md §P0.4: **1 unit = 1
 * metre**, character 1.7-1.8u, and every other asset scaled to it. The
 * brief's acceptance criterion is a *ratio* ("character height is roughly
 * 1/3 to 1/4 of nearby tree height"), so the fix is to grow the trees to
 * match a realistically-sized human rather than shrink the human. (This
 * supersedes an earlier verbal request to shrink the character to 0.8u --
 * same underlying complaint, "character is too big relative to the world",
 * but the brief's approach keeps a physically meaningful unit scale that
 * every future asset can be authored against. Flagged in the report.)
 */
export const PLAYER_HEIGHT = 1.75;
export const PLAYER_EYE_HEIGHT = 1.6;
/** Forgiving player collision box -- game_implementation_plan.md §4.5:
 * smaller than the visual model on purpose. */
export const PLAYER_COLLISION_SIZE = { x: 0.6, y: 1.6, z: 0.6 } as const;

/** Tree trunk-to-canopy total height range (brief: 6-15u). */
export const TREE_HEIGHT_MIN = 6;
export const TREE_HEIGHT_MAX = 14;
export const ROCK_SIZE_MIN = 0.4;
export const ROCK_SIZE_MAX = 1.2;

export type ObstacleAvoidance = "lane" | "jump";

export interface ObstacleTypeDef {
  id: string;
  displayName: string;
  avoidance: ObstacleAvoidance;
  /** Collision box in world units, centered on the obstacle's spawn point at
   * ground level (y=0 is the box's bottom, matching how it's placed in scene). */
  size: { x: number; y: number; z: number };
  color: number;
  shape: ObstacleShape;
  /** Rim/marking colour that separates a hazard from decorative scenery. */
  accent: number;
}

/**
 * Jungle-themed obstacle catalog. Sizes are in the 1u = 1m contract above,
 * so heights are readable against a 1.75m character: a jumpable log clears
 * at knee height, a lane-blocker stands at chest height or taller.
 *
 * `shape` drives the mesh built in obstacleSpawner -- brief §H requires each
 * hazard's *silhouette* to communicate jump-vs-dodge at a distance, which a
 * uniform grey box cannot. `accent` is the rim/marking colour that separates
 * hazards from decorative scenery.
 */
export type ObstacleShape = "log" | "boulder" | "pit" | "barrier";

export const OBSTACLE_TYPES = {
  boulder: {
    id: "boulder",
    displayName: "Boulder",
    avoidance: "lane",
    size: { x: 1.5, y: 1.5, z: 1.5 },
    color: 0x6d6862,
    shape: "boulder",
    accent: 0x3a3733,
  },
  log: {
    id: "log",
    displayName: "Fallen Log",
    avoidance: "jump",
    size: { x: 3.2, y: 0.55, z: 0.7 },
    color: 0x6b4423,
    shape: "log",
    accent: 0xc9a227,
  },
  pit: {
    id: "pit",
    displayName: "Mud Pit",
    avoidance: "jump",
    size: { x: 2.2, y: 0.06, z: 2.0 },
    color: 0x241a10,
    shape: "pit",
    accent: 0x7a5a2e,
  },
  thornWall: {
    id: "thornWall",
    displayName: "Thorn Vine Wall",
    avoidance: "lane",
    size: { x: 3.2, y: 1.9, z: 0.5 },
    color: 0x2f5d33,
    shape: "barrier",
    accent: 0x8fbf4d,
  },
} as const satisfies Record<string, ObstacleTypeDef>;

export type ObstacleTypeId = keyof typeof OBSTACLE_TYPES;

/**
 * Character model + animation asset paths. Config-driven rather than
 * hardcoded in the loader, per naruto_run_agent_brief.md §7 -- swapping the
 * character should be an edit here, not a refactor. (Note: the run clip is
 * retargeted onto whatever rig MODEL_URL provides, so a swap also needs the
 * bone-name map in scene/characterRig.ts updated to match the new rig.)
 */
export const CHARACTER_MODEL_URL = "/models/naruto/Naruto.dae";
export const CHARACTER_RUN_CLIP_URL = "/models/naruto/run.fbx";

/** Jump arc, owned by game/laneController.ts. Brief §D asks for 0.55-0.75s
 * airtime with a fast rise / slower hang / fast fall -- see laneController's
 * jump curve, which is deliberately not a symmetric physics parabola. */
export const JUMP_DURATION_S = 0.62;
export const JUMP_HEIGHT = 1.5;
/** Crouch anticipation before launch and compression on landing (brief §D). */
export const JUMP_ANTICIPATION_S = 0.07;
export const JUMP_LANDING_S = 0.12;
/** Peak squash-and-stretch at full crouch: the body compresses this
 * fraction on Y (and widens by roughly half of it on X/Z). Applied as a
 * model scale with the feet pinned to the ground, so the character never
 * sinks through the path. */
export const JUMP_CROUCH_SQUASH = 0.13;

/** Lane-change bank/lean, brief §D: "the highest ratio of feels-good to
 * work-required on this whole list". Roll into the direction of travel and
 * ease back out. */
export const LANE_LEAN_MAX_RAD = (13 * Math.PI) / 180;

/**
 * Run-cycle cadence matching (brief §A): the animation must advance in
 * proportion to world scroll speed or the feet slide and the character
 * reads as skating. This is the clip playback rate that looks correct at
 * exactly BASE_RUN_SPEED_U_S; playerCharacter scales it linearly with
 * actual speed from there.
 */
export const RUN_CLIP_TIMESCALE_AT_BASE_SPEED = 1.35;
/** A jump requested while already airborne is remembered for this long and
 * fires the instant the character lands, instead of being dropped -- the
 * "forgiveness window" from game_implementation_plan.md Phase D. This is
 * input buffering (not classic ledge coyote-time, which doesn't apply here
 * since jumps aren't ledge-triggered); it exists for the same reason: a
 * player's press timing shouldn't have to be frame-perfect, which matters
 * doubly once Phase E's CV pipeline adds its own ~150-200ms latency on top. */
export const JUMP_INPUT_BUFFER_S = 0.15;
