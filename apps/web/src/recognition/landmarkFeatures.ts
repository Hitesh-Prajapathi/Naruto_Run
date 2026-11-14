/**
 * Two-hand landmark feature extraction -- TypeScript port of
 * `src/narutocv/features/landmarks.py` (spec: `docs/specs/landmark_features_v1.md`).
 *
 * This is a line-for-line port: same operand order, same eps guards, same tie-break rules.
 * All arithmetic is float64 (`number`); only the final 84-d vector is rounded to float32
 * (a `Float32Array` store rounds to nearest-even, exactly like numpy's `astype(np.float32)`).
 * Parity with Python is enforced by `tests/golden/features_v1.json` (see `docs/specs/parity.md`).
 *
 * No dependencies, no DOM, no MediaPipe import.
 */

export const FEATURE_SPEC_VERSION = "1.0.0";

export const NUM_LANDMARKS = 21;
export const FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"] as const;
/** Landmark chains (wrist, then the four joints of each finger). */
export const FINGER_CHAINS: readonly (readonly number[])[] = [
  [0, 1, 2, 3, 4],
  [0, 5, 6, 7, 8],
  [0, 9, 10, 11, 12],
  [0, 13, 14, 15, 16],
  [0, 17, 18, 19, 20],
];
/** Proximal bone (start, end) used for the signed abduction angle of each finger. */
export const ABDUCTION_BONES: readonly (readonly [number, number])[] = [
  [2, 3],
  [5, 6],
  [9, 10],
  [13, 14],
  [17, 18],
];
export const FINGERTIPS: readonly number[] = [4, 8, 12, 16, 20];
export const PALM_CENTER_LANDMARKS: readonly number[] = [0, 5, 9, 13, 17];

const WRIST = 0;
const INDEX_MCP = 5;
const MIDDLE_MCP = 9;
const PINKY_MCP = 17;

/** Absolute floor (pixel units) below which a palm is considered degenerate. */
export const EPS_PALM = 1e-6;
/** Relative floor for the lateral palm axis before the deterministic fallback applies. */
export const EPS_AXIS_REL = 1e-6;
/** Minimum cosine for a palm normal to count as pointing toward / away from the other palm. */
export const FACING_COS_MIN = 0.2;

export const SLOT_NAMES = ["L", "R"] as const; // slot 0 = Left, slot 1 = Right
export const PER_HAND_DIM = 25;
export const INTER_HAND_DIM = 30;
export const META_DIM = 4;
export const FEATURE_DIM = 2 * PER_HAND_DIM + INTER_HAND_DIM + META_DIM; // 84

export type Vec3 = [number, number, number];
type Frame = [Vec3, Vec3, Vec3];

/** 21 landmarks, each `(x, y, z)`: image-normalised x/y, MediaPipe z (same scale as x). */
export type LandmarkInput = ArrayLike<ArrayLike<number>>;

/** One detected hand as reported by the landmarker (`world` landmarks are unused by v1). */
export interface HandObservation {
  landmarks: LandmarkInput;
  /** "Left" or "Right" (case-insensitive, surrounding whitespace ignored). */
  handedness: string;
  /** Handedness score in [0, 1]; defaults to 1. */
  score?: number;
}

export interface CanonicalHand {
  /** 21 points in the canonical right-hand palm frame (wrist at origin, unit = palm size). */
  points: Vec3[];
  palmSize: number;
  mirrored: boolean;
  /** Rows x, y, z of the palm frame in the *original* (un-mirrored) pixel space. */
  frame: Frame;
  /** Wrist position in original pixel space. */
  wrist: Vec3;
  /** False for degenerate input (all numeric fields are then zero). */
  valid: boolean;
}

// -- small vector helpers (plain arithmetic, same operation order as the numpy code) -------

function dot(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function cross(a: Vec3, b: Vec3): Vec3 {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

function sub(a: Vec3, b: Vec3): Vec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function norm(a: Vec3): number {
  return Math.sqrt(dot(a, a));
}

function unit(v: Vec3): Vec3 {
  const n = norm(v);
  if (n < 1e-300) return [0, 0, 0];
  return [v[0] / n, v[1] / n, v[2] / n];
}

/** Unsigned angle in [0, pi] via `atan2(|u x v|, u . v)` (stable near 0 and pi). */
function angle(u: Vec3, v: Vec3): number {
  return Math.atan2(norm(cross(u, v)), dot(u, v));
}

function identityFrame(): Frame {
  return [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ];
}

/** Python `str.strip().lower()` followed by the Left/Right check. */
export function isLeft(handedness: string): boolean {
  const key = handedness.trim().toLowerCase();
  if (key === "left") return true;
  if (key === "right") return false;
  throw new Error(`handedness must be 'Left' or 'Right', got ${JSON.stringify(handedness)}`);
}

/** Validate shape `(21, 3)` and copy to tuples (throws like Python's `_as_landmarks`). */
function asLandmarks(lm: LandmarkInput): Vec3[] {
  const out: Vec3[] = [];
  let ok = lm.length === NUM_LANDMARKS;
  for (let i = 0; ok && i < NUM_LANDMARKS; i += 1) {
    const row = lm[i] as ArrayLike<number>;
    if (row.length !== 3) {
      ok = false;
      break;
    }
    out.push([Number(row[0]), Number(row[1]), Number(row[2])]);
  }
  if (!ok) throw new Error("landmarks must have shape (21, 3)");
  return out;
}

/** Scale normalised `(x, y, z)` to pixels: `(x*w, y*h, z*w)`. */
export function toPixelSpace(lm: LandmarkInput, imageSize: readonly [number, number]): Vec3[] {
  const [w, h] = imageSize;
  return asLandmarks(lm).map((p): Vec3 => [p[0] * w, p[1] * h, p[2] * w]);
}

function degenerate(): CanonicalHand {
  return {
    points: Array.from({ length: NUM_LANDMARKS }, (): Vec3 => [0, 0, 0]),
    palmSize: 0,
    mirrored: false,
    frame: identityFrame(),
    wrist: [0, 0, 0],
    valid: false,
  };
}

/**
 * Canonicalise one hand (spec section 2).
 *
 * `lm` is in pixel space unless `imageSize` is given, in which case it is treated as
 * image-normalised and converted with {@link toPixelSpace}. Left hands are mirrored
 * (`x -> -x`) into the right-hand frame before the palm frame is built.
 */
export function canonicalizeHand(
  lm: LandmarkInput,
  handedness: string,
  imageSize?: readonly [number, number],
): CanonicalHand {
  const pts = imageSize !== undefined ? toPixelSpace(lm, imageSize) : asLandmarks(lm);
  const mirrored = isLeft(handedness);
  for (const p of pts) {
    if (!Number.isFinite(p[0]) || !Number.isFinite(p[1]) || !Number.isFinite(p[2])) {
      return degenerate();
    }
  }
  const mx = mirrored ? -1 : 1;
  const mPts = pts.map((p): Vec3 => [p[0] * mx, p[1], p[2]]);

  const wrist = mPts[WRIST] as Vec3;
  const yRaw = sub(mPts[MIDDLE_MCP] as Vec3, wrist);
  const palmSize = Math.sqrt(dot(yRaw, yRaw));
  if (palmSize < EPS_PALM) return degenerate();
  const yAxis: Vec3 = [yRaw[0] / palmSize, yRaw[1] / palmSize, yRaw[2] / palmSize];

  let xRaw = sub(mPts[PINKY_MCP] as Vec3, mPts[INDEX_MCP] as Vec3);
  const proj = dot(xRaw, yAxis);
  xRaw = [xRaw[0] - proj * yAxis[0], xRaw[1] - proj * yAxis[1], xRaw[2] - proj * yAxis[2]];
  if (Math.sqrt(dot(xRaw, xRaw)) < EPS_AXIS_REL * palmSize) {
    // Deterministic fallback: standard basis vector with the smallest |y| component
    // (lowest index on ties), orthogonalised against y.
    let k = 0;
    for (let i = 1; i < 3; i += 1) {
      if (Math.abs(yAxis[i] as number) < Math.abs(yAxis[k] as number)) k = i;
    }
    const e: Vec3 = [0, 0, 0];
    e[k] = 1;
    const p2 = dot(e, yAxis);
    xRaw = [e[0] - p2 * yAxis[0], e[1] - p2 * yAxis[1], e[2] - p2 * yAxis[2]];
  }
  const xAxis = unit(xRaw);
  const zAxis = cross(xAxis, yAxis);
  const rot: Frame = [xAxis, yAxis, zAxis]; // rows = axes (mirrored space)

  const points = mPts.map((p): Vec3 => {
    const d = sub(p, wrist);
    return [dot(d, rot[0]) / palmSize, dot(d, rot[1]) / palmSize, dot(d, rot[2]) / palmSize];
  });

  // Frame expressed in the original (un-mirrored) space: reflect axes back (column x * mx).
  const frame = rot.map((r): Vec3 => [r[0] * mx, r[1], r[2]]) as Frame;
  return {
    points,
    palmSize,
    mirrored,
    frame,
    wrist: [...(pts[WRIST] as Vec3)] as Vec3,
    valid: true,
  };
}

function perHandFeatures(hand: CanonicalHand, out: Float64Array, offset: number): void {
  const p = hand.points;
  const at = (i: number): Vec3 => p[i] as Vec3;
  let i = offset;
  for (const chain of FINGER_CHAINS) {
    // 15 flexion angles
    for (let k = 1; k < 4; k += 1) {
      const incoming = sub(at(chain[k] as number), at(chain[k - 1] as number));
      const outgoing = sub(at(chain[k + 1] as number), at(chain[k] as number));
      out[i] = angle(incoming, outgoing);
      i += 1;
    }
  }
  for (const [start, end] of ABDUCTION_BONES) {
    // 5 signed abduction angles in the palm plane
    const d = sub(at(end), at(start));
    out[i] = Math.atan2(d[0], d[1]);
    i += 1;
  }
  const center = meanPoint(PALM_CENTER_LANDMARKS.map(at));
  for (const tip of FINGERTIPS) {
    // 5 tip-to-palm-centre distances (palm-size units)
    const d = sub(at(tip), center);
    out[i] = Math.sqrt(dot(d, d));
    i += 1;
  }
}

/** Mean over rows, summed left to right then divided (numpy `mean(axis=0)` on 5 rows). */
function meanPoint(pts: Vec3[]): Vec3 {
  let sx = 0;
  let sy = 0;
  let sz = 0;
  for (const p of pts) {
    sx += p[0];
    sy += p[1];
    sz += p[2];
  }
  const n = pts.length;
  return [sx / n, sy / n, sz / n];
}

function meanX(lm: Vec3[]): number {
  let s = 0;
  for (const p of lm) s += p[0];
  return s / lm.length;
}

type Slot = HandObservation | null;

/**
 * Assign up to two hands to the (Left, Right) slots (spec section 3).
 *
 * 1. Degenerate hands are dropped; if more than two remain, the two highest-scoring are
 *    kept (ties: input order), preserving input order.
 * 2. One hand: slot by its handedness label.
 * 3. Two hands with different labels: slot by label.
 * 4. Two hands claiming the same label: the hand with the smaller mean landmark x
 *    (image-normalised) takes the Left slot (equal x: input order).
 */
export function assignSlots(hands: readonly HandObservation[], imageSize: readonly [number, number]): [Slot, Slot] {
  let valid: HandObservation[] = hands.filter((h) => canonicalizeHand(h.landmarks, h.handedness, imageSize).valid);
  if (valid.length > 2) {
    const order = valid
      .map((h, j) => ({ negScore: -Number(h.score ?? 1), j }))
      .sort((a, b) => (a.negScore < b.negScore ? -1 : a.negScore > b.negScore ? 1 : a.j - b.j))
      .slice(0, 2)
      .map((o) => o.j)
      .sort((a, b) => a - b);
    valid = order.map((j) => valid[j] as HandObservation);
  }
  if (valid.length === 0) return [null, null];
  if (valid.length === 1) {
    const only = valid[0] as HandObservation;
    return isLeft(only.handedness) ? [only, null] : [null, only];
  }
  const a = valid[0] as HandObservation;
  const b = valid[1] as HandObservation;
  if (isLeft(a.handedness) !== isLeft(b.handedness)) {
    return isLeft(a.handedness) ? [a, b] : [b, a];
  }
  const xa = meanX(asLandmarks(a.landmarks));
  const xb = meanX(asLandmarks(b.landmarks));
  return xa <= xb ? [a, b] : [b, a];
}

/** Mean of the given landmarks mapped back to original pixel space (points, not mean). */
function originalSpace(hand: CanonicalHand, indices: readonly number[]): Vec3[] {
  return indices.map((idx): Vec3 => {
    const c = hand.points[idx] as Vec3;
    const x = c[0] * hand.palmSize;
    const y = c[1] * hand.palmSize;
    const z = c[2] * hand.palmSize;
    const [fx, fy, fz] = hand.frame;
    return [
      hand.wrist[0] + (x * fx[0] + y * fy[0] + z * fz[0]),
      hand.wrist[1] + (x * fx[1] + y * fy[1] + z * fz[1]),
      hand.wrist[2] + (x * fx[2] + y * fy[2] + z * fz[2]),
    ];
  });
}

function interHandFeatures(left: CanonicalHand, right: CanonicalHand, out: Float64Array, offset: number): void {
  const meanPalm = 0.5 * (left.palmSize + right.palmSize);
  // Wrist-to-wrist vector (Left - Right) in the Right-slot palm frame (original space).
  const wv = sub(left.wrist, right.wrist);
  const w: Vec3 = [wv[0] / meanPalm, wv[1] / meanPalm, wv[2] / meanPalm];
  out[offset] = dot(right.frame[0], w);
  out[offset + 1] = dot(right.frame[1], w);
  out[offset + 2] = dot(right.frame[2], w);
  // 5x5 fingertip distances: row = Left-slot finger, column = Right-slot finger.
  const lt = originalSpace(left, FINGERTIPS);
  const rt = originalSpace(right, FINGERTIPS);
  let i = offset + 3;
  for (const a of lt) {
    for (const b of rt) {
      const d = sub(a, b);
      out[i] = Math.sqrt(dot(d, d)) / meanPalm;
      i += 1;
    }
  }
  const nL = left.frame[2];
  const nR = right.frame[2];
  out[offset + 28] = angle(nL, nR);
  const cL = meanPoint(originalSpace(left, PALM_CENTER_LANDMARKS));
  const cR = meanPoint(originalSpace(right, PALM_CENTER_LANDMARKS));
  const u = unit(sub(cR, cL));
  const facing = dot(nL, u) > FACING_COS_MIN && dot(nR, u) < -FACING_COS_MIN;
  out[offset + 29] = facing ? 1 : 0;
}

function buildNames(): string[] {
  const names: string[] = [];
  for (const slot of SLOT_NAMES) {
    for (const finger of FINGER_NAMES) {
      for (let k = 0; k < 3; k += 1) names.push(`${slot}.flex.${finger}.${k}`);
    }
    for (const finger of FINGER_NAMES) names.push(`${slot}.abd.${finger}`);
    for (const finger of FINGER_NAMES) names.push(`${slot}.tip_palm.${finger}`);
  }
  for (const axis of ["x", "y", "z"]) names.push(`inter.wrist_vec.${axis}`);
  for (const fl of FINGER_NAMES) {
    for (const fr of FINGER_NAMES) names.push(`inter.tip_dist.L${fl}_R${fr}`);
  }
  names.push("inter.normal_angle", "inter.palms_facing");
  names.push("meta.hand_count", "meta.present_L", "meta.present_R", "meta.mean_score");
  return names;
}

export const FEATURE_NAMES: readonly string[] = buildNames();

/**
 * Fixed-length (`FEATURE_DIM`) float32 feature vector for up to two hands.
 *
 * `imageSize` is `[width, height]` in pixels.
 * Layout: `[Left per-hand (25) | Right per-hand (25) | inter-hand (30) | meta (4)]`.
 * Missing hands contribute zeros plus a zero presence flag.
 *
 * Throws on a malformed landmark array or an unknown handedness label (like Python).
 */
export function twoHandFeatures(hands: readonly HandObservation[], imageSize: readonly [number, number]): Float32Array {
  const [leftObs, rightObs] = assignSlots(hands, imageSize);
  const vec = new Float64Array(FEATURE_DIM);
  const present = [false, false];
  const canon: (CanonicalHand | null)[] = [null, null];
  const scores: number[] = [];
  [leftObs, rightObs].forEach((obs, slot) => {
    if (obs === null) return;
    // The slot (not the raw label) decides mirroring, so the tie-break is authoritative.
    const ch = canonicalizeHand(obs.landmarks, slot === 0 ? "Left" : "Right", imageSize);
    canon[slot] = ch;
    present[slot] = true;
    perHandFeatures(ch, vec, slot * PER_HAND_DIM);
    const s = Number(obs.score ?? 1);
    scores.push(Number.isFinite(s) ? s : 0);
  });
  const base = 2 * PER_HAND_DIM;
  const cl = canon[0] ?? null;
  const cr = canon[1] ?? null;
  if (cl !== null && cr !== null) interHandFeatures(cl, cr, vec, base);
  const meta = base + INTER_HAND_DIM;
  vec[meta] = (present[0] ? 1 : 0) + (present[1] ? 1 : 0);
  vec[meta + 1] = present[0] ? 1 : 0;
  vec[meta + 2] = present[1] ? 1 : 0;
  let total = 0;
  for (const s of scores) total += s;
  vec[meta + 3] = scores.length > 0 ? total / scores.length : 0;
  return Float32Array.from(vec); // round to float32 (nearest-even), as numpy astype does
}

/** Convenience adapter for MediaPipe `{x, y, z}` landmark objects. */
export function landmarksFromObjects(points: readonly { x: number; y: number; z: number }[]): Vec3[] {
  return points.map((p): Vec3 => [p.x, p.y, p.z]);
}
