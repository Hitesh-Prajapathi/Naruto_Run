"""Property and unit tests for ``narutocv.features.landmarks``."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from narutocv.features.landmarks import (
    FEATURE_DIM,
    FEATURE_NAMES,
    FEATURE_SPEC_VERSION,
    PER_HAND_DIM,
    HandObservation,
    assign_slots,
    canonicalize_hand,
    export_feature_spec,
    feature_names,
    feature_spec,
    to_pixel_space,
    two_hand_features,
)
from numpy.typing import NDArray

SIZE = (640, 480)
TOL = 1e-4
BASES = [0.0, -0.35, -0.12, 0.12, 0.35]  # x of the 5 finger bases (thumb..pinky)


def make_hand(
    seed: int, curl: float = 0.5, origin: tuple[float, float] = (0.5, 0.6)
) -> NDArray[np.float64]:
    """Plausible right-hand-like template in image-normalised coordinates.

    Image y points down, so fingers extend toward negative y.  Per-landmark noise is
    bounded so the palm frame stays well conditioned.
    """
    rng = np.random.default_rng(seed)
    pts = np.zeros((21, 3))
    pts[0] = (0.0, 0.0, 0.0)
    lengths = [
        (0.06, 0.05, 0.04),
        (0.05, 0.035, 0.03),
        (0.055, 0.04, 0.03),
        (0.05, 0.035, 0.03),
        (0.04, 0.03, 0.025),
    ]
    mcp_x = [0.04, 0.03, 0.0, -0.03, -0.055]
    mcp_y = [-0.03, -0.09, -0.1, -0.095, -0.08]
    for f in range(5):
        base = np.array([mcp_x[f], mcp_y[f], 0.0])
        idx = 1 + 4 * f if f > 0 else 1
        if f == 0:
            pts[1] = (0.02, -0.015, 0.0)
            base = pts[1]
            idx = 2
        else:
            pts[idx] = base
            idx = idx + 1
        pos = base.copy()
        angle = 0.0
        for j, ln in enumerate(lengths[f]):
            angle += curl * (0.4 + 0.3 * j) * (0.5 if f == 0 else 1.0)
            direction = np.array(
                [0.15 * (f - 2) * (0.3 if f else -2.0), -math.cos(angle), -math.sin(angle)]
            )
            direction /= np.linalg.norm(direction)
            pos = pos + ln * direction
            pts[idx + j] = pos
    pts += rng.uniform(-0.002, 0.002, size=pts.shape)
    pts[:, 0] += origin[0]
    pts[:, 1] += origin[1]
    return pts


def transform(
    lm: NDArray[np.float64],
    *,
    theta: float = 0.0,
    scale: float = 1.0,
    shift: tuple[float, float] = (0.0, 0.0),
    size: tuple[int, int] = SIZE,
) -> NDArray[np.float64]:
    """Similarity transform applied in pixel space, returned as normalised landmarks."""
    px = to_pixel_space(lm, size)
    c, s = math.cos(theta), math.sin(theta)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    out = (px @ rot.T) * scale
    out[:, 0] += shift[0] * size[0]
    out[:, 1] += shift[1] * size[1]
    result: NDArray[np.float64] = out / np.array([size[0], size[1], size[0]], dtype=np.float64)
    return result


def mirror(lm: NDArray[np.float64]) -> NDArray[np.float64]:
    out = lm.copy()
    out[:, 0] = 1.0 - out[:, 0]
    return out


hand_params = st.tuples(
    st.integers(0, 10_000),
    st.floats(0.0, 1.2),
)
transforms = st.tuples(
    st.floats(-math.pi, math.pi),
    st.floats(0.4, 2.5),
    st.floats(-0.3, 0.3),
    st.floats(-0.3, 0.3),
)


@settings(max_examples=60, deadline=None)
@given(hand_params, transforms)
def test_single_hand_invariant_to_similarity(
    hp: tuple[int, float], tf: tuple[float, float, float, float]
) -> None:
    seed, curl = hp
    theta, scale, tx, ty = tf
    lm = make_hand(seed, curl)
    ref = two_hand_features([HandObservation(lm, "Right", 0.9)], image_size=SIZE)
    moved = two_hand_features(
        [HandObservation(transform(lm, theta=theta, scale=scale, shift=(tx, ty)), "Right", 0.9)],
        image_size=SIZE,
    )
    assert np.allclose(ref, moved, atol=TOL)


@settings(max_examples=60, deadline=None)
@given(hand_params, hand_params, transforms)
def test_two_hand_invariant_to_similarity(
    hp1: tuple[int, float], hp2: tuple[int, float], tf: tuple[float, float, float, float]
) -> None:
    theta, scale, tx, ty = tf
    left = make_hand(hp1[0], hp1[1], origin=(0.3, 0.6))
    right = make_hand(hp2[0], hp2[1], origin=(0.7, 0.55))
    hands = [HandObservation(left, "Left", 0.8), HandObservation(right, "Right", 0.9)]
    ref = two_hand_features(hands, image_size=SIZE)
    kw = {"theta": theta, "scale": scale, "shift": (tx, ty)}
    hands_t = [
        HandObservation(transform(left, **kw), "Left", 0.8),  # type: ignore[arg-type]
        HandObservation(transform(right, **kw), "Right", 0.9),  # type: ignore[arg-type]
    ]
    moved = two_hand_features(hands_t, image_size=SIZE)
    assert np.allclose(ref, moved, atol=TOL)


@settings(max_examples=50, deadline=None)
@given(hand_params, transforms)
def test_invariance_in_non_square_image(
    hp: tuple[int, float], tf: tuple[float, float, float, float]
) -> None:
    seed, curl = hp
    theta, scale, tx, ty = tf
    size = (1280, 360)
    lm = make_hand(seed, curl)
    ref = two_hand_features([HandObservation(lm, "Right")], image_size=size)
    moved = two_hand_features(
        [
            HandObservation(
                transform(lm, theta=theta, scale=scale, shift=(tx, ty), size=size), "Right"
            )
        ],
        image_size=size,
    )
    assert np.allclose(ref, moved, atol=TOL)


@settings(max_examples=50, deadline=None)
@given(hand_params)
def test_mirrored_right_hand_labelled_left_is_identical(hp: tuple[int, float]) -> None:
    seed, curl = hp
    right = make_hand(seed, curl)
    left = mirror(right)
    c_r = canonicalize_hand(right, "Right", image_size=SIZE)
    c_l = canonicalize_hand(left, "Left", image_size=SIZE)
    assert c_l.mirrored and not c_r.mirrored
    assert np.allclose(c_r.points, c_l.points, atol=1e-9)
    f_r = two_hand_features([HandObservation(right, "Right")], image_size=SIZE)
    f_l = two_hand_features([HandObservation(left, "Left")], image_size=SIZE)
    # right slot of f_r == left slot of f_l
    assert np.allclose(f_r[PER_HAND_DIM : 2 * PER_HAND_DIM], f_l[:PER_HAND_DIM], atol=TOL)


def test_vector_length_fixed_for_any_hand_count() -> None:
    a, b, c = make_hand(1), make_hand(2, origin=(0.2, 0.5)), make_hand(3, origin=(0.8, 0.5))
    cases = [
        [],
        [HandObservation(a, "Right")],
        [HandObservation(a, "Left")],
        [HandObservation(a, "Left"), HandObservation(b, "Right")],
        [HandObservation(a, "Right"), HandObservation(b, "Right"), HandObservation(c, "Left")],
    ]
    for hands in cases:
        v = two_hand_features(hands, image_size=SIZE)
        assert v.shape == (FEATURE_DIM,)
        assert v.dtype == np.float32
        assert np.all(np.isfinite(v))
    assert (
        len(FEATURE_NAMES) == FEATURE_DIM == len(feature_names()) == len(feature_spec()["features"])
    )
    assert len(set(FEATURE_NAMES)) == FEATURE_DIM


def test_missing_hands_zero_with_mask() -> None:
    names = {n: i for i, n in enumerate(FEATURE_NAMES)}
    none = two_hand_features([], image_size=SIZE)
    assert not none.any()
    one = two_hand_features([HandObservation(make_hand(1), "Right", 0.7)], image_size=SIZE)
    assert one[names["meta.hand_count"]] == 1.0
    assert one[names["meta.present_L"]] == 0.0 and one[names["meta.present_R"]] == 1.0
    assert one[names["meta.mean_score"]] == np.float32(0.7)
    assert not one[:PER_HAND_DIM].any()
    assert one[PER_HAND_DIM : 2 * PER_HAND_DIM].any()
    inter = slice(2 * PER_HAND_DIM, 2 * PER_HAND_DIM + 30)
    assert not one[inter].any()
    two = two_hand_features(
        [
            HandObservation(make_hand(1, origin=(0.3, 0.6)), "Left", 0.6),
            HandObservation(make_hand(2, origin=(0.7, 0.6)), "Right", 1.0),
        ],
        image_size=SIZE,
    )
    assert two[names["meta.hand_count"]] == 2.0
    assert two[names["meta.mean_score"]] == np.float32(0.8)
    assert two[inter].any()


def test_degenerate_inputs_are_finite() -> None:
    zeros = np.zeros((21, 3))
    same = np.tile(np.array([0.4, 0.5, 0.1]), (21, 1))
    collinear = np.zeros((21, 3))
    collinear[:, 1] = np.linspace(0.0, 0.5, 21)  # all on a line: lateral axis undefined
    flat_x = make_hand(5)
    flat_x[:, 0] = 0.5
    nan_hand = make_hand(6)
    nan_hand[3, 1] = np.nan
    inf_hand = make_hand(7)
    inf_hand[0, 0] = np.inf
    for lm in (zeros, same, collinear, flat_x, nan_hand, inf_hand):
        for label in ("Left", "Right"):
            ch = canonicalize_hand(lm, label, image_size=SIZE)
            assert np.all(np.isfinite(ch.points)) and np.all(np.isfinite(ch.frame))
            v = two_hand_features(
                [HandObservation(lm, label), HandObservation(make_hand(1), "Right")],
                image_size=SIZE,
            )
            assert np.all(np.isfinite(v))
            v2 = two_hand_features(
                [HandObservation(lm, label), HandObservation(lm, label)], image_size=SIZE
            )
            assert np.all(np.isfinite(v2))
    # degenerate hands do not count as present
    v = two_hand_features([HandObservation(zeros, "Right")], image_size=SIZE)
    assert v[FEATURE_NAMES.index("meta.hand_count")] == 0.0
    # degenerate image size must not produce NaN either
    assert np.all(
        np.isfinite(two_hand_features([HandObservation(make_hand(1), "Right")], image_size=(0, 0)))
    )


@settings(max_examples=40, deadline=None)
@given(
    st.lists(st.floats(-1e3, 1e3, allow_nan=False), min_size=63, max_size=63),
    st.sampled_from(["Left", "Right"]),
)
def test_arbitrary_finite_input_never_nan(values: list[float], label: str) -> None:
    lm = np.asarray(values).reshape(21, 3)
    v = two_hand_features(
        [HandObservation(lm, label), HandObservation(lm[::-1].copy(), label)], image_size=SIZE
    )
    assert np.all(np.isfinite(v))


def test_tie_break_same_handedness_by_x() -> None:
    a = make_hand(1, origin=(0.25, 0.6))
    b = make_hand(2, origin=(0.75, 0.6))
    for order in ([a, b], [b, a]):
        hands = [HandObservation(x, "Right") for x in order]
        left, right = assign_slots(hands, SIZE)
        assert left is not None and right is not None
        assert np.asarray(left.landmarks)[:, 0].mean() < np.asarray(right.landmarks)[:, 0].mean()
    v1 = two_hand_features(
        [HandObservation(a, "Right"), HandObservation(b, "Right")], image_size=SIZE
    )
    v2 = two_hand_features(
        [HandObservation(b, "Right"), HandObservation(a, "Right")], image_size=SIZE
    )
    assert np.array_equal(v1, v2)
    # different labels: labels win, even against position
    left, right = assign_slots([HandObservation(a, "Right"), HandObservation(b, "Left")], SIZE)
    assert left is not None and right is not None
    assert np.asarray(left.landmarks)[:, 0].mean() > np.asarray(right.landmarks)[:, 0].mean()


def test_more_than_two_hands_keeps_best_scores() -> None:
    hands = [
        HandObservation(make_hand(1, origin=(0.2, 0.5)), "Left", 0.3),
        HandObservation(make_hand(2, origin=(0.5, 0.5)), "Right", 0.9),
        HandObservation(make_hand(3, origin=(0.8, 0.5)), "Left", 0.8),
    ]
    left, right = assign_slots(hands, SIZE)
    assert left is hands[2] and right is hands[1]


def test_palms_facing_indicator() -> None:
    """Mirror-symmetric pair: rotating one way makes palms face, the other way face away."""
    size = (640, 640)
    base = make_hand(1, curl=0.2)
    centred = base - base[0]
    facing_idx = FEATURE_NAMES.index("inter.palms_facing")
    angle_idx = FEATURE_NAMES.index("inter.normal_angle")
    results = []
    for a in (math.radians(90), math.radians(-90)):
        ry = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
        right = centred @ ry.T
        left = right.copy()
        left[:, 0] = -left[:, 0]
        right = right + np.array([0.7, 0.6, 0.0])
        left = left + np.array([0.3, 0.6, 0.0])
        v = two_hand_features(
            [HandObservation(left, "Left"), HandObservation(right, "Right")], image_size=size
        )
        results.append(float(v[facing_idx]))
        assert abs(float(v[angle_idx]) - math.pi) < 0.3  # normals are (anti)parallel
    assert sorted(results) == [0.0, 1.0]


def test_export_feature_spec_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "spec.json"
    spec = export_feature_spec(path)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded == spec
    assert loaded["spec_version"] == FEATURE_SPEC_VERSION
    assert [f["name"] for f in loaded["features"]] == list(FEATURE_NAMES)
    assert [f["index"] for f in loaded["features"]] == list(range(FEATURE_DIM))
    assert all(
        isinstance(f["formula"], str) and isinstance(f["landmarks"], list)
        for f in loaded["features"]
    )
