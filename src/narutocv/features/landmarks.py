"""Two-hand landmark feature extraction (spec: ``docs/specs/landmark_features_v1.md``).

Pure numpy, deterministic, no MediaPipe import.  Everything here is meant to be
ported 1:1 to TypeScript, so only plain arithmetic (dot/cross/atan2/sqrt) is used.

Pipeline:  raw MediaPipe landmarks -> pixel space -> mirror (left slot) ->
palm frame -> canonical coordinates -> per-hand / inter-hand / meta features.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

FEATURE_SPEC_VERSION = "1.0.1"

NUM_LANDMARKS = 21
FINGER_NAMES: tuple[str, ...] = ("thumb", "index", "middle", "ring", "pinky")
# Landmark chains (wrist, then the four joints of each finger).
FINGER_CHAINS: tuple[tuple[int, ...], ...] = (
    (0, 1, 2, 3, 4),
    (0, 5, 6, 7, 8),
    (0, 9, 10, 11, 12),
    (0, 13, 14, 15, 16),
    (0, 17, 18, 19, 20),
)
# Proximal bone (start, end) used for the signed abduction angle of each finger.
ABDUCTION_BONES: tuple[tuple[int, int], ...] = ((2, 3), (5, 6), (9, 10), (13, 14), (17, 18))
FINGERTIPS: tuple[int, ...] = (4, 8, 12, 16, 20)
PALM_CENTER_LANDMARKS: tuple[int, ...] = (0, 5, 9, 13, 17)

WRIST = 0
INDEX_MCP = 5
MIDDLE_MCP = 9
PINKY_MCP = 17

#: Absolute floor (pixel units) below which a palm is considered degenerate.
EPS_PALM = 1e-6
#: Relative floor for the lateral palm axis before the deterministic fallback applies.
EPS_AXIS_REL = 1e-6
#: Minimum cosine for a palm normal to count as pointing toward / away from the other palm.
FACING_COS_MIN = 0.2

SLOT_NAMES: tuple[str, str] = ("L", "R")  # slot 0 = Left, slot 1 = Right
PER_HAND_DIM = 25
INTER_HAND_DIM = 30
META_DIM = 4
FEATURE_DIM = 2 * PER_HAND_DIM + INTER_HAND_DIM + META_DIM  # 84

Array = NDArray[np.float64]


@dataclass(frozen=True)
class HandObservation:
    """One detected hand as reported by the landmarker.

    ``landmarks`` is ``(21, 3)`` with image-normalised ``x, y`` and MediaPipe ``z``
    (same scale as ``x``).  ``world`` (``(21, 3)`` metres) is accepted for forward
    compatibility but unused by feature spec v1.
    """

    landmarks: ArrayLike
    handedness: str
    score: float = 1.0
    world: ArrayLike | None = None


@dataclass(frozen=True)
class CanonicalHand:
    """A hand in its palm frame.

    ``points`` is ``(21, 3)`` in the canonical right-hand palm frame (wrist at the
    origin, unit = palm size).  ``frame`` has rows ``x, y, z`` of the palm frame
    expressed in the *original* (un-mirrored) pixel space, so ``frame[2]`` is the
    physical palm normal.  ``wrist`` is the original-space wrist position.
    ``valid`` is False for degenerate input (all numeric fields are then zero).
    """

    points: Array
    palm_size: float
    mirrored: bool
    frame: Array
    wrist: Array
    valid: bool


def _as_landmarks(lm: ArrayLike) -> Array:
    arr = np.asarray(lm, dtype=np.float64)
    if arr.shape != (NUM_LANDMARKS, 3):
        raise ValueError(f"landmarks must have shape (21, 3), got {arr.shape}")
    return arr


def to_pixel_space(lm: ArrayLike, image_size: tuple[int, int]) -> Array:
    """Scale normalised ``(x, y, z)`` to pixels: ``(x*w, y*h, z*w)``."""
    arr = _as_landmarks(lm)
    width, height = float(image_size[0]), float(image_size[1])
    return arr * np.array([width, height, width], dtype=np.float64)


def _unit(v: Array) -> Array:
    n = math.sqrt(float(np.dot(v, v)))
    if n < 1e-300:
        return np.zeros(3, dtype=np.float64)
    return v / n


def _angle(u: Array, v: Array) -> float:
    """Unsigned angle in [0, pi] via ``atan2(|u x v|, u . v)`` (stable near 0 and pi)."""
    cross = np.cross(u, v)
    return math.atan2(math.sqrt(float(np.dot(cross, cross))), float(np.dot(u, v)))


def _is_left(handedness: str) -> bool:
    key = handedness.strip().lower()
    if key == "left":
        return True
    if key == "right":
        return False
    raise ValueError(f"handedness must be 'Left' or 'Right', got {handedness!r}")


def _degenerate() -> CanonicalHand:
    return CanonicalHand(
        points=np.zeros((NUM_LANDMARKS, 3), dtype=np.float64),
        palm_size=0.0,
        mirrored=False,
        frame=np.eye(3, dtype=np.float64),
        wrist=np.zeros(3, dtype=np.float64),
        valid=False,
    )


def canonicalize_hand(
    lm: ArrayLike,
    handedness: str,
    *,
    image_size: tuple[int, int] | None = None,
) -> CanonicalHand:
    """Canonicalise one hand (see spec section 2).

    ``lm`` is in pixel space unless ``image_size`` is given, in which case it is
    treated as image-normalised and converted with :func:`to_pixel_space`.
    Left hands are mirrored (``x -> -x``) into the right-hand frame *before* the
    palm frame is built, and ``CanonicalHand.mirrored`` records that.
    """
    pts = to_pixel_space(lm, image_size) if image_size is not None else _as_landmarks(lm)
    mirrored = _is_left(handedness)
    if not bool(np.all(np.isfinite(pts))):
        return _degenerate()
    mirror = np.array([-1.0 if mirrored else 1.0, 1.0, 1.0], dtype=np.float64)
    m_pts = pts * mirror

    wrist = m_pts[WRIST]
    y_raw = m_pts[MIDDLE_MCP] - wrist
    palm_size = math.sqrt(float(np.dot(y_raw, y_raw)))
    if palm_size < EPS_PALM:
        return _degenerate()
    y_axis = y_raw / palm_size

    x_raw = m_pts[PINKY_MCP] - m_pts[INDEX_MCP]
    x_raw = x_raw - float(np.dot(x_raw, y_axis)) * y_axis
    if math.sqrt(float(np.dot(x_raw, x_raw))) < EPS_AXIS_REL * palm_size:
        # Deterministic fallback: standard basis vector with the smallest |y| component
        # (lowest index on ties), orthogonalised against y.
        k = int(np.argmin(np.abs(y_axis)))
        e = np.zeros(3, dtype=np.float64)
        e[k] = 1.0
        x_raw = e - float(np.dot(e, y_axis)) * y_axis
    x_axis = _unit(x_raw)
    z_axis = np.cross(x_axis, y_axis)
    rot = np.stack([x_axis, y_axis, z_axis])  # rows = axes (mirrored space)
    canonical = ((m_pts - wrist) @ rot.T) / palm_size

    # Frame expressed in the original (un-mirrored) space: reflect axes back.
    frame = rot * mirror
    return CanonicalHand(
        points=canonical,
        palm_size=palm_size,
        mirrored=mirrored,
        frame=frame,
        wrist=pts[WRIST].copy(),
        valid=True,
    )


def _per_hand_features(hand: CanonicalHand) -> Array:
    p = hand.points
    out = np.zeros(PER_HAND_DIM, dtype=np.float64)
    i = 0
    for chain in FINGER_CHAINS:  # 15 flexion angles
        for k in range(1, 4):
            incoming = p[chain[k]] - p[chain[k - 1]]
            outgoing = p[chain[k + 1]] - p[chain[k]]
            out[i] = _angle(incoming, outgoing)
            i += 1
    for start, end in ABDUCTION_BONES:  # 5 signed abduction angles in the palm plane
        d = p[end] - p[start]
        out[i] = math.atan2(float(d[0]), float(d[1]))
        i += 1
    center = p[list(PALM_CENTER_LANDMARKS)].mean(axis=0)
    for tip in FINGERTIPS:  # 5 tip-to-palm-centre distances (palm-size units)
        d = p[tip] - center
        out[i] = math.sqrt(float(np.dot(d, d)))
        i += 1
    return out


def _mean_x(pts: Array) -> float:
    return float(pts[:, 0].mean())


def _rank_score(score: float) -> float:
    """Score used to rank hands: non-finite values count as 0.0 (total order, spec 1.0.1)."""
    value = float(score)
    return value if math.isfinite(value) else 0.0


def assign_slots(
    hands: Sequence[HandObservation], image_size: tuple[int, int]
) -> tuple[HandObservation | None, HandObservation | None]:
    """Assign up to two hands to the (Left, Right) slots.

    1. Hands with non-finite or degenerate landmarks are dropped; if more than two
       remain, the two highest-scoring are kept (ties: input order).  A non-finite
       score (NaN, +/-inf) ranks as 0.0, so the order is total and deterministic.
    2. One hand: slot by its handedness label.
    3. Two hands with different labels: slot by label.
    4. Two hands claiming the *same* label: the hand with the smaller mean landmark
       x (image-normalised) takes the Left slot (equal x: input order).  MediaPipe
       assumes a mirrored (selfie) image, so Left == smaller x.  The tie-break exists
       because in mirrored / interlocked poses the landmarker often labels both
       hands identically; without it both would land in one slot and the other
       slot's features would be lost.
    """
    valid: list[HandObservation] = []
    for h in hands:
        if canonicalize_hand(h.landmarks, h.handedness, image_size=image_size).valid:
            valid.append(h)
    if len(valid) > 2:
        order = sorted(range(len(valid)), key=lambda j: (-_rank_score(valid[j].score), j))[:2]
        valid = [valid[j] for j in sorted(order)]
    if not valid:
        return None, None
    if len(valid) == 1:
        return (valid[0], None) if _is_left(valid[0].handedness) else (None, valid[0])
    a, b = valid
    if _is_left(a.handedness) != _is_left(b.handedness):
        return (a, b) if _is_left(a.handedness) else (b, a)
    xa = _mean_x(_as_landmarks(a.landmarks))
    xb = _mean_x(_as_landmarks(b.landmarks))
    return (a, b) if xa <= xb else (b, a)


def _canonical_for_slot(
    obs: HandObservation, slot: int, image_size: tuple[int, int]
) -> CanonicalHand:
    # The slot (not the raw label) decides mirroring, so the tie-break is authoritative.
    return canonicalize_hand(obs.landmarks, "Left" if slot == 0 else "Right", image_size=image_size)


def _original_space(hand: CanonicalHand, indices: Sequence[int]) -> Array:
    """Mean of the given landmarks (or each, if one index) back in original pixel space."""
    pts = hand.points[list(indices)] * hand.palm_size
    return hand.wrist + pts @ hand.frame


def _inter_hand_features(left: CanonicalHand, right: CanonicalHand) -> Array:
    out = np.zeros(INTER_HAND_DIM, dtype=np.float64)
    mean_palm = 0.5 * (left.palm_size + right.palm_size)
    # Wrist-to-wrist vector (Left - Right) in the Right-slot palm frame (original space).
    w = (left.wrist - right.wrist) / mean_palm
    out[0:3] = right.frame @ w
    # 5x5 fingertip distances: row = Left-slot finger, column = Right-slot finger.
    lt = _original_space(left, FINGERTIPS)
    rt = _original_space(right, FINGERTIPS)
    diff = lt[:, None, :] - rt[None, :, :]
    out[3:28] = (np.sqrt(np.sum(diff * diff, axis=2)) / mean_palm).reshape(-1)
    n_l, n_r = left.frame[2], right.frame[2]
    out[28] = _angle(n_l, n_r)
    c_l = _original_space(left, PALM_CENTER_LANDMARKS).mean(axis=0)
    c_r = _original_space(right, PALM_CENTER_LANDMARKS).mean(axis=0)
    u = _unit(c_r - c_l)
    facing = float(np.dot(n_l, u)) > FACING_COS_MIN and float(np.dot(n_r, u)) < -FACING_COS_MIN
    out[29] = 1.0 if facing else 0.0
    return out


def _build_names() -> tuple[str, ...]:
    names: list[str] = []
    for slot in SLOT_NAMES:
        for finger in FINGER_NAMES:
            names.extend(f"{slot}.flex.{finger}.{k}" for k in range(3))
        names.extend(f"{slot}.abd.{finger}" for finger in FINGER_NAMES)
        names.extend(f"{slot}.tip_palm.{finger}" for finger in FINGER_NAMES)
    names.extend(f"inter.wrist_vec.{axis}" for axis in "xyz")
    for fl in FINGER_NAMES:
        for fr in FINGER_NAMES:
            names.append(f"inter.tip_dist.L{fl}_R{fr}")
    names.extend(["inter.normal_angle", "inter.palms_facing"])
    names.extend(["meta.hand_count", "meta.present_L", "meta.present_R", "meta.mean_score"])
    return tuple(names)


FEATURE_NAMES: tuple[str, ...] = _build_names()
assert len(FEATURE_NAMES) == FEATURE_DIM


def feature_names() -> list[str]:
    return list(FEATURE_NAMES)


def two_hand_features(
    hands: Sequence[HandObservation], *, image_size: tuple[int, int]
) -> NDArray[np.float32]:
    """Fixed-length (``FEATURE_DIM``) float32 feature vector for up to two hands.

    ``image_size`` is ``(width, height)`` in pixels.
    Layout: ``[Left per-hand (25) | Right per-hand (25) | inter-hand (30) | meta (4)]``.
    Missing hands contribute zeros plus a zero presence flag.
    """
    left_obs, right_obs = assign_slots(hands, image_size)
    vec = np.zeros(FEATURE_DIM, dtype=np.float64)
    present = [False, False]
    canon: list[CanonicalHand | None] = [None, None]
    scores: list[float] = []
    for slot, obs in enumerate((left_obs, right_obs)):
        if obs is None:
            continue
        ch = _canonical_for_slot(obs, slot, image_size)
        canon[slot] = ch
        present[slot] = True
        vec[slot * PER_HAND_DIM : (slot + 1) * PER_HAND_DIM] = _per_hand_features(ch)
        s = float(obs.score)
        scores.append(s if math.isfinite(s) else 0.0)
    base = 2 * PER_HAND_DIM
    cl, cr = canon
    if cl is not None and cr is not None:
        vec[base : base + INTER_HAND_DIM] = _inter_hand_features(cl, cr)
    meta = base + INTER_HAND_DIM
    vec[meta] = float(sum(present))
    vec[meta + 1] = 1.0 if present[0] else 0.0
    vec[meta + 2] = 1.0 if present[1] else 0.0
    vec[meta + 3] = float(sum(scores) / len(scores)) if scores else 0.0
    return vec.astype(np.float32)


def feature_spec() -> dict[str, Any]:
    """Machine-readable description of every feature (for the TypeScript port)."""
    feats: list[dict[str, Any]] = []

    def add(name: str, formula: str, landmarks: list[int], **extra: Any) -> None:
        feats.append(
            {"index": len(feats), "name": name, "formula": formula, "landmarks": landmarks, **extra}
        )

    for slot in SLOT_NAMES:
        for fi, finger in enumerate(FINGER_NAMES):
            chain = FINGER_CHAINS[fi]
            for k in range(1, 4):
                add(
                    f"{slot}.flex.{finger}.{k - 1}",
                    "bone_angle",
                    [chain[k - 1], chain[k], chain[k + 1]],
                    scope=slot,
                )
        for fi, finger in enumerate(FINGER_NAMES):
            add(f"{slot}.abd.{finger}", "abduction_angle", list(ABDUCTION_BONES[fi]), scope=slot)
        for fi, finger in enumerate(FINGER_NAMES):
            add(
                f"{slot}.tip_palm.{finger}",
                "tip_palm_dist",
                [FINGERTIPS[fi]],
                scope=slot,
                palm_center=list(PALM_CENTER_LANDMARKS),
            )
    for axis in "xyz":
        add(f"inter.wrist_vec.{axis}", "wrist_vec_shared_frame", [WRIST], scope="inter")
    for fi, fl in enumerate(FINGER_NAMES):
        for fj, fr in enumerate(FINGER_NAMES):
            add(
                f"inter.tip_dist.L{fl}_R{fr}",
                "tip_tip_dist",
                [FINGERTIPS[fi], FINGERTIPS[fj]],
                scope="inter",
            )
    palm = [WRIST, INDEX_MCP, MIDDLE_MCP, PINKY_MCP]
    add("inter.normal_angle", "palm_normal_angle", palm, scope="inter")
    add("inter.palms_facing", "palm_facing", palm, scope="inter")
    add("meta.hand_count", "hand_count", [], scope="meta")
    add("meta.present_L", "presence_left", [], scope="meta")
    add("meta.present_R", "presence_right", [], scope="meta")
    add("meta.mean_score", "mean_handedness_score", [], scope="meta")
    return {
        "spec_version": FEATURE_SPEC_VERSION,
        "dtype": "float32",
        "feature_dim": FEATURE_DIM,
        "constants": {
            "eps_palm": EPS_PALM,
            "eps_axis_rel": EPS_AXIS_REL,
            "facing_cos_min": FACING_COS_MIN,
            "palm_size_landmarks": [WRIST, MIDDLE_MCP],
            "pixel_scale": "x*width, y*height, z*width",
        },
        "features": feats,
    }


def export_feature_spec(path: str | Path) -> dict[str, Any]:
    """Write :func:`feature_spec` as JSON to ``path`` and return it."""
    spec = feature_spec()
    if len(spec["features"]) != FEATURE_DIM:
        raise AssertionError("feature spec does not match FEATURE_DIM")
    Path(path).write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    return spec
