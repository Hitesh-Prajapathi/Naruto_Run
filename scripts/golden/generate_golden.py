"""Generate the golden parity fixtures in ``tests/golden/`` (see ``docs/specs/parity.md``).

Run with ``uv run python scripts/golden/generate_golden.py``.  The fixtures are produced by
the Python reference implementations (``narutocv.features.landmarks``,
``narutocv.temporal.decoder``, ``narutocv.models.calibration``) and consumed by both
``tests/golden/test_golden_up_to_date.py`` (drift check) and the TypeScript parity tests in
``apps/web/tests/unit/recognition/``.

Everything here is deterministic: seeded ``numpy.random.default_rng`` generators only, inputs
rounded to a fixed number of decimals, no wall-clock or environment dependence.  Non-finite
floats cannot be expressed in JSON, so they are written as the strings ``"NaN"``,
``"Infinity"`` and ``"-Infinity"`` (the TypeScript tests revive them).
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from narutocv.features.landmarks import (
    FEATURE_DIM,
    FEATURE_SPEC_VERSION,
    HandObservation,
    two_hand_features,
)
from narutocv.models.calibration import (
    StackedFusion,
    TemperatureScaler,
    VectorScaler,
    build_fusion_features,
)
from narutocv.temporal.decoder import DEFAULT_LABELS, DecoderConfig, SealDecoder
from numpy.typing import NDArray

GOLDEN_DIR = Path(__file__).resolve().parents[2] / "tests" / "golden"
FIXTURE_NAMES = ("features_v1.json", "decoder_v1.json", "calibration_v1.json")

#: Absolute tolerance each fixture family is *compared* with by the drift test.  Values the
#: Python side computes may differ in the last bits across platforms (BLAS, libm), so the
#: drift test cannot demand bit equality.  Tolerances are far below the parity tolerances.
FEATURES_DRIFT_TOL = 1e-6  # float32 output
DECODER_DRIFT_TOL = 1e-9
CALIBRATION_DRIFT_TOL = 1e-9

Json = Any
Array = NDArray[np.float64]


# --------------------------------------------------------------------------------------
# JSON helpers
# --------------------------------------------------------------------------------------
def _encode(obj: Json) -> Json:
    """Make ``obj`` strict-JSON safe (non-finite floats become strings)."""
    if isinstance(obj, float):
        if math.isnan(obj):
            return "NaN"
        if math.isinf(obj):
            return "Infinity" if obj > 0 else "-Infinity"
        return obj
    if isinstance(obj, dict):
        return {k: _encode(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_encode(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _encode(obj.tolist())
    if isinstance(obj, np.floating):
        return _encode(float(obj))
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def dumps_fixture(doc: dict[str, Json]) -> str:
    """Serialise with one case per line (stable, reviewable diffs)."""
    doc = _encode(doc)
    cases = doc.pop("cases")
    head = json.dumps(doc, allow_nan=False, separators=(",", ":"))
    lines = [json.dumps(c, allow_nan=False, separators=(",", ":")) for c in cases]
    return head[:-1] + ',"cases":[\n' + ",\n".join(lines) + "\n]}\n"


# --------------------------------------------------------------------------------------
# Features fixture
# --------------------------------------------------------------------------------------
SIZE = (640, 480)


def _template(curl: float) -> Array:
    """Plausible right-hand template in image-normalised coordinates (image y points down)."""
    pts = np.zeros((21, 3))
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
        if f == 0:
            pts[1] = (0.02, -0.015, 0.0)
            base = pts[1].copy()
            idx = 2
        else:
            base = np.array([mcp_x[f], mcp_y[f], 0.0])
            pts[1 + 4 * f] = base
            idx = 2 + 4 * f
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
    return pts


def _rot3(yaw: float, pitch: float, roll: float) -> Array:
    cy, sy, cp, sp, cr, sr = (
        math.cos(yaw),
        math.sin(yaw),
        math.cos(pitch),
        math.sin(pitch),
        math.cos(roll),
        math.sin(roll),
    )
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
    rz = np.array([[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]])
    out: Array = rz @ rx @ ry
    return out


def make_hand(
    seed: int,
    curl: float = 0.5,
    origin: tuple[float, float] = (0.5, 0.6),
    *,
    size: tuple[int, int] = SIZE,
    yaw: float = 0.0,
    pitch: float = 0.0,
    roll: float = 0.0,
    scale: float = 1.0,
    noise: float = 0.002,
    z_noise: float = 0.004,
    ndigits: int = 7,
) -> Array:
    """A seeded, perturbed, rotated template; returns normalised landmarks rounded to ``ndigits``."""
    rng = np.random.default_rng(seed)
    pts = _template(curl)
    pts = pts + rng.uniform(-noise, noise, size=pts.shape)
    pts[:, 2] += rng.uniform(-z_noise, z_noise, size=21)
    # Rotate in pixel space so the rotation is a physical one on non-square images.
    w, h = float(size[0]), float(size[1])
    px = pts * np.array([w, h, w])
    px = (px @ _rot3(yaw, pitch, roll).T) * scale
    pts = px / np.array([w, h, w])
    pts[:, 0] += origin[0]
    pts[:, 1] += origin[1]
    return np.round(pts, ndigits)


def _hand_dict(
    lm: Array | list[list[float]], handedness: str, score: float | None = 0.9
) -> dict[str, Json]:
    d: dict[str, Json] = {"landmarks": np.asarray(lm).tolist(), "handedness": handedness}
    if score is not None:
        d["score"] = float(score)
    return d


def _features_case(
    name: str, note: str, hands: list[dict[str, Json]], size: tuple[int, int] = SIZE
) -> dict[str, Json]:
    case: dict[str, Json] = {
        "name": name,
        "note": note,
        "image_size": [int(size[0]), int(size[1])],
        "hands": hands,
    }
    try:
        obs = [HandObservation(h["landmarks"], h["handedness"], h.get("score", 1.0)) for h in hands]
        case["expected"] = two_hand_features(obs, image_size=size).astype(np.float64).tolist()
        case["expect_error"] = False
    except ValueError:
        case["expected"] = None
        case["expect_error"] = True
    return case


def build_features_fixture() -> dict[str, Json]:
    cases: list[dict[str, Json]] = []

    def add(
        name: str, note: str, hands: list[dict[str, Json]], size: tuple[int, int] = SIZE
    ) -> None:
        cases.append(_features_case(name, note, hands, size))

    # --- single hands --------------------------------------------------------------
    for i, curl in enumerate((0.0, 0.3, 0.6, 0.9, 1.2)):
        lm = make_hand(100 + i, curl, yaw=0.1 * i, roll=0.15 * i)
        add(f"single_right_curl{curl}", "one right hand", [_hand_dict(lm, "Right", 0.95)])
    for i, curl in enumerate((0.1, 0.5, 0.8, 1.1)):
        lm = make_hand(200 + i, curl, origin=(0.4, 0.55), pitch=0.2 * i, roll=-0.3 * i)
        add(f"single_left_curl{curl}", "one left hand", [_hand_dict(lm, "Left", 0.85)])

    # --- non-square / extreme image sizes ------------------------------------------
    base = make_hand(300, 0.4, yaw=0.3, pitch=-0.2, roll=0.4)
    for size in ((1280, 720), (480, 640), (1920, 1080), (100, 1000), (1000, 100), (1, 1)):
        add(
            f"image_size_{size[0]}x{size[1]}",
            "same landmarks, other aspect",
            [_hand_dict(base, "Right", 0.9)],
            size,
        )

    # --- two hands, different labels -----------------------------------------------
    for i in range(5):
        left = make_hand(400 + i, 0.2 + 0.2 * i, origin=(0.3, 0.6), yaw=0.2 * i, roll=0.1)
        right = make_hand(450 + i, 1.0 - 0.15 * i, origin=(0.7, 0.55), yaw=-0.2 * i, roll=-0.1)
        size = [(640, 480), (1280, 720), (480, 640), (800, 800), (1920, 1080)][i]
        add(
            f"two_hands_LR_{i}",
            "Left + Right, distinct labels",
            [_hand_dict(left, "Left", 0.8), _hand_dict(right, "Right", 0.9)],
            size,
        )
    left = make_hand(410, 0.4, origin=(0.3, 0.6))
    right = make_hand(460, 0.7, origin=(0.7, 0.55))
    add(
        "two_hands_RL_input_order",
        "Right listed before Left",
        [_hand_dict(right, "Right", 0.9), _hand_dict(left, "Left", 0.8)],
    )

    # --- palms facing: search deterministic poses until the flag is 1 and 0 --------
    facing_found = 0
    for seed in range(1, 200):
        rng = np.random.default_rng(seed)
        yl, yr = rng.uniform(-2.2, -0.9), rng.uniform(-2.2, -0.9)
        left = make_hand(500 + seed, 0.3, origin=(0.3, 0.55), yaw=yl, roll=0.05)
        right = make_hand(600 + seed, 0.3, origin=(0.68, 0.55), yaw=yr, roll=-0.05)
        hands = [_hand_dict(left, "Left", 0.9), _hand_dict(right, "Right", 0.9)]
        probe = _features_case("probe", "", hands)
        if probe["expected"][-5] == 1.0:  # meta is last 4, so palms_facing is index 79
            cases.append(
                {
                    **probe,
                    "name": f"palms_facing_{facing_found}",
                    "note": "palm normals pointing at each other",
                }
            )
            facing_found += 1
            if facing_found == 3:
                break
    assert facing_found == 3, "could not construct palms-facing poses"
    add(
        "palms_not_facing_coplanar",
        "flat co-planar hands: dot products ~0, flag must be 0",
        [
            _hand_dict(make_hand(701, 0.2, origin=(0.3, 0.6), noise=0.0, z_noise=0.0), "Left"),
            _hand_dict(make_hand(702, 0.2, origin=(0.7, 0.6), noise=0.0, z_noise=0.0), "Right"),
        ],
    )

    # --- same-label tie-break ------------------------------------------------------
    a = make_hand(800, 0.3, origin=(0.3, 0.6))
    b = make_hand(801, 0.6, origin=(0.7, 0.6))
    add(
        "same_label_right_right_ordered",
        "both Right, first has smaller x",
        [_hand_dict(a, "Right", 0.9), _hand_dict(b, "Right", 0.8)],
    )
    add(
        "same_label_right_right_swapped",
        "both Right, first has larger x",
        [_hand_dict(b, "Right", 0.9), _hand_dict(a, "Right", 0.8)],
    )
    add(
        "same_label_left_left_ordered",
        "both Left, first has smaller x",
        [_hand_dict(a, "Left", 0.9), _hand_dict(b, "Left", 0.8)],
    )
    add(
        "same_label_left_left_swapped",
        "both Left, first has larger x",
        [_hand_dict(b, "Left", 0.9), _hand_dict(a, "Left", 0.8)],
    )
    add(
        "same_label_identical_hands",
        "two identical hands, equal mean x: input order wins",
        [_hand_dict(a, "Right", 0.9), _hand_dict(a, "Right", 0.7)],
    )
    shifted = a.copy()
    shifted[:, 1] = np.round(shifted[:, 1] - 0.2, 7)
    add(
        "same_label_equal_mean_x",
        "same x column, different y: equal means, input order wins",
        [_hand_dict(shifted, "Left", 0.9), _hand_dict(a, "Left", 0.7)],
    )

    # --- more than two hands -------------------------------------------------------
    h = [make_hand(900 + i, 0.2 + 0.2 * i, origin=(0.2 + 0.25 * i, 0.6)) for i in range(3)]
    add(
        "three_hands_by_score",
        "lowest score dropped",
        [
            _hand_dict(h[0], "Left", 0.6),
            _hand_dict(h[1], "Right", 0.95),
            _hand_dict(h[2], "Left", 0.9),
        ],
    )
    add(
        "three_hands_tied_scores",
        "equal scores: earlier input kept",
        [
            _hand_dict(h[0], "Left", 0.8),
            _hand_dict(h[1], "Right", 0.8),
            _hand_dict(h[2], "Right", 0.8),
        ],
    )
    nan_hand = np.full((21, 3), math.nan)
    add(
        "three_hands_one_degenerate",
        "degenerate hand dropped before the top-2 cut",
        [
            _hand_dict(nan_hand, "Left", 0.99),
            _hand_dict(h[1], "Right", 0.5),
            _hand_dict(h[2], "Left", 0.4),
        ],
    )

    # --- no hands / degenerate -----------------------------------------------------
    add("no_hands", "empty input", [])
    add(
        "one_valid_one_nan",
        "NaN hand ignored",
        [_hand_dict(h[0], "Left", 0.9), _hand_dict(nan_hand, "Right", 0.9)],
    )
    add(
        "two_degenerate",
        "both hands unusable",
        [_hand_dict(nan_hand, "Left", 0.9), _hand_dict(np.zeros((21, 3)), "Right", 0.9)],
    )
    inf_hand = h[1].copy()
    inf_hand[4, 0] = math.inf
    add(
        "inf_landmark",
        "one infinite coordinate makes the hand degenerate",
        [_hand_dict(inf_hand, "Right", 0.9)],
    )
    add("all_zero_landmarks", "palm size 0", [_hand_dict(np.zeros((21, 3)), "Right", 0.9)])
    sub_eps = h[1].copy()  # middle MCP (9) closer than 1e-6 px to the wrist
    sub_eps[9] = sub_eps[0] + np.array([5e-7 / 640.0, 0.0, 0.0])
    add("palm_below_eps", "palm size 5e-7 px < EPS_PALM", [_hand_dict(sub_eps, "Right", 0.9)])
    tiny = make_hand(1000, 0.5, origin=(0.0, 0.0), scale=4e-8, noise=0.0, z_noise=0.0, ndigits=18)
    add(
        "tiny_hand_above_eps",
        "whole hand ~2e-6 px (valid, scale invariant)",
        [_hand_dict(tiny, "Right", 0.9)],
    )

    # fallback lateral axis: pinky MCP == index MCP, and parallel to the palm axis
    fb = h[1].copy()
    fb[17] = fb[5]
    add(
        "lateral_axis_zero_fallback",
        "x_raw == 0 -> deterministic fallback axis",
        [_hand_dict(fb, "Right", 0.9)],
    )
    par = h[1].copy()
    par[5] = par[0] + 0.3 * (par[9] - par[0])
    par[17] = par[0] + 0.9 * (par[9] - par[0])
    add(
        "lateral_axis_parallel_fallback",
        "index->pinky parallel to wrist->middle",
        [_hand_dict(par, "Left", 0.9)],
    )
    # palm axis exactly along +x in pixels: |y| = (1, 0, 0), argmin tie -> index 1
    tie_x = h[1].copy()
    tie_x[0] = (0.5, 0.5, 0.1)
    tie_x[9] = (0.5 + 64.0 / 640.0, 0.5, 0.1)
    tie_x[17] = tie_x[5]
    add(
        "fallback_argmin_tie_x_axis",
        "y = (1,0,0): lowest-index tie -> e = (0,1,0)",
        [_hand_dict(tie_x, "Right", 0.9)],
    )
    tie_z = h[1].copy()
    tie_z[0] = (0.5, 0.5, 0.0)
    tie_z[9] = (0.5, 0.5, 0.1)
    tie_z[17] = tie_z[5]
    add(
        "fallback_argmin_tie_z_axis",
        "y = (0,0,1): lowest-index tie -> e = (1,0,0)",
        [_hand_dict(tie_z, "Left", 0.9)],
    )

    # --- scores, handedness strings, errors ---------------------------------------
    add(
        "nan_score",
        "non-finite score counts as 0 in mean_score",
        [_hand_dict(h[0], "Left", math.nan), _hand_dict(h[1], "Right", 0.5)],
    )
    add("inf_score_single", "infinite score counts as 0", [_hand_dict(h[1], "Right", math.inf)])
    add(
        "missing_score_defaults_to_one",
        "no score field: default 1.0",
        [_hand_dict(h[1], "Right", None)],
    )
    add(
        "handedness_lowercase",
        "labels are case-insensitive",
        [_hand_dict(h[0], "left", 0.7), _hand_dict(h[1], "right", 0.8)],
    )
    add("handedness_whitespace", "labels are stripped", [_hand_dict(h[1], " RIGHT ", 0.8)])
    add("handedness_invalid_error", "unknown label raises", [_hand_dict(h[1], "Middle", 0.8)])

    # --- geometry oddities ---------------------------------------------------------
    flat = [
        _hand_dict(make_hand(1100 + i, 0.5, origin=(0.3 + 0.4 * i, 0.6), z_noise=0.0), lab)
        for i, lab in enumerate(("Left", "Right"))
    ]
    for hd in flat:
        arr = np.asarray(hd["landmarks"])
        arr[:, 2] = 0.0
        hd["landmarks"] = arr.tolist()
    add("planar_hands_z_zero", "all z = 0 (planar hands)", flat)
    off = make_hand(1200, 0.4, origin=(1.1, -0.1))
    add("hand_outside_image", "coordinates outside [0, 1]", [_hand_dict(off, "Right", 0.9)])

    assert FEATURE_DIM == 84
    return {
        "spec_version": FEATURE_SPEC_VERSION,
        "feature_dim": FEATURE_DIM,
        "tolerance": 1e-5,
        "encoding": "non-finite floats are the strings NaN / Infinity / -Infinity",
        "cases": cases,
    }


# --------------------------------------------------------------------------------------
# Decoder fixture
# --------------------------------------------------------------------------------------
LABELS = list(DEFAULT_LABELS)
K = len(LABELS)
IDLE = LABELS.index("zero")


def _round_probs(p: Array) -> list[float]:
    return [float(v) for v in np.round(p, 5)]


def _probs(
    rng: np.random.Generator,
    dominant: int,
    level: float,
    *,
    n_classes: int = K,
    noise: float = 0.0,
) -> list[float]:
    """Probability vector with ``level`` mass on ``dominant`` (jittered by ``noise``)."""
    lv = float(np.clip(level + (rng.normal(0.0, noise) if noise > 0 else 0.0), 0.01, 0.995))
    rest = np.ones(n_classes) if noise == 0.0 else rng.dirichlet(np.full(n_classes, 2.0))
    rest = rest.copy()
    rest[dominant] = 0.0
    rest = rest / rest.sum() * (1.0 - lv)
    rest[dominant] = lv
    return _round_probs(rest)


Segment = tuple[float, float, int, float]  # (start_ms, end_ms, class index, level)


def _signal(
    rng: np.random.Generator,
    segments: list[Segment],
    *,
    fps: float,
    duration_ms: float,
    t0: float = 0.0,
    idle: int = IDLE,
    idle_level: float = 0.9,
    n_classes: int = K,
    noise: float = 0.0,
    drop: Callable[[int, float], bool] | None = None,
    jitter_ms: float = 0.0,
) -> list[dict[str, Json]]:
    frames: list[dict[str, Json]] = []
    step = 1000.0 / fps
    n = int(duration_ms / step)
    for i in range(n):
        rel = i * step
        if drop is not None and drop(i, rel):
            continue
        cls, lvl = idle, idle_level
        for s, e, c, lv in segments:
            if s <= rel < e:
                cls, lvl = c, lv
        t = t0 + rel + (rng.uniform(-jitter_ms, jitter_ms) if jitter_ms and i else 0.0)
        frames.append(
            {"t_ms": float(t), "probs": _probs(rng, cls, lvl, n_classes=n_classes, noise=noise)}
        )
    return frames


def _run_stream(
    name: str, note: str, config: DecoderConfig, frames: list[dict[str, Json]]
) -> dict[str, Json]:
    dec = SealDecoder(config)
    events: list[dict[str, Json]] = []
    trace: list[Json] = []
    for fr in frames:
        if fr.get("reset"):
            dec.reset()
            trace.append(None)
            continue
        out = dec.update(fr["probs"], fr["t_ms"], expected_next=fr.get("expected_next"))
        events.extend(e.to_dict() for e in out)
        s = dec.smoothed
        trace.append([int(np.argmax(s)), float(s.max())])
    return {
        "name": name,
        "note": note,
        "config": config.to_dict(),
        "frames": frames,
        "expected": {
            "events": events,
            "stats": {
                "frames": dec.stats.frames,
                "nonmonotonic": dec.stats.nonmonotonic,
                "gaps": dec.stats.gaps,
                "events": dec.stats.events,
            },
            "trace": trace,
            "final_smoothed": dec.smoothed.tolist(),
            "final_active": dec.active_label,
        },
    }


def build_decoder_fixture() -> dict[str, Json]:
    ix = {name: i for i, name in enumerate(LABELS)}
    tiger, ram, monkey, snake = ix["tiger"], ix["ram"], ix["monkey"], ix["snake"]
    cases: list[dict[str, Json]] = []

    def add(name: str, note: str, frames: list[dict[str, Json]], **cfg: Json) -> dict[str, Json]:
        case = _run_stream(name, note, DecoderConfig(**cfg), frames)
        cases.append(case)
        return case

    clean = [(300.0, 900.0, tiger, 0.92)]
    r = np.random.default_rng(1)
    add(
        "ema_clean_seal_30fps",
        "one clean seal, deterministic signal",
        _signal(r, clean, fps=30, duration_ms=1500),
    )
    add(
        "ema_clean_seal_60fps", "same signal at 60 fps", _signal(r, clean, fps=60, duration_ms=1500)
    )
    add(
        "ema_clean_seal_15fps", "same signal at 15 fps", _signal(r, clean, fps=15, duration_ms=1500)
    )
    add(
        "ema_noisy_seal_nonzero_start",
        "noisy 30 fps stream starting at t=1000 ms",
        _signal(
            np.random.default_rng(2),
            clean,
            fps=30,
            duration_ms=1500,
            t0=1000.0,
            noise=0.04,
            jitter_ms=3.0,
        ),
    )
    add(
        "ema_sequence_direct_switches",
        "tiger -> ram -> monkey with no idle in between",
        _signal(
            np.random.default_rng(3),
            [(200, 700, tiger, 0.9), (700, 1200, ram, 0.88), (1200, 1800, monkey, 0.93)],
            fps=30,
            duration_ms=2200,
            noise=0.03,
        ),
    )
    add(
        "ema_refractory_deferral",
        "direct switch faster than the dwell+refractory window",
        _signal(
            np.random.default_rng(4),
            [(100, 500, tiger, 0.95), (500, 900, ram, 0.95)],
            fps=60,
            duration_ms=1300,
            noise=0.0,
        ),
        refractory_ms=600.0,
        min_dwell_ms=60.0,
    )
    # Raw level wobbles around the smoothed threshold.
    wob = np.random.default_rng(5)
    flick: list[dict[str, Json]] = []
    for i in range(90):
        level = 0.62 + 0.08 * math.sin(i * 1.3) + wob.normal(0.0, 0.03)
        flick.append({"t_ms": i * 1000.0 / 30.0, "probs": _probs(wob, snake, level, noise=0.02)})
    add(
        "ema_flicker_around_tau_on",
        "level hovers around tau_on",
        flick,
        half_life_ms=40.0,
        min_dwell_ms=150.0,
    )
    add(
        "ema_hold_then_release_reenter",
        "seal held, released below tau_off, then re-entered",
        _signal(
            np.random.default_rng(6),
            [(100, 700, tiger, 0.95), (1100, 1700, tiger, 0.95)],
            fps=30,
            duration_ms=2200,
        ),
    )
    add(
        "ema_slow_decay_no_reentry",
        "level sags between tau_off and tau_on: no second event while active",
        _signal(
            np.random.default_rng(7),
            [(100, 600, tiger, 0.95), (600, 1100, tiger, 0.5), (1100, 1600, tiger, 0.95)],
            fps=30,
            duration_ms=1800,
        ),
    )

    # --- gaps / timestamps ---------------------------------------------------------
    add(
        "ema_gap_during_held_seal",
        "300 ms stall mid-hold: gap counted, no duplicate event",
        _signal(
            np.random.default_rng(8),
            [(100, 1500, tiger, 0.93)],
            fps=30,
            duration_ms=1700,
            drop=lambda i, t: 700 <= t < 1000,
        ),
    )
    add(
        "ema_gap_during_dwell",
        "gap resets the pending dwell",
        _signal(
            np.random.default_rng(9),
            [(100, 1500, ram, 0.93)],
            fps=30,
            duration_ms=1800,
            drop=lambda i, t: 300 <= t < 560,
        ),
        min_dwell_ms=200.0,
    )
    add(
        "ema_dropped_frames_below_max_gap",
        "every third frame dropped (<= max_gap)",
        _signal(
            np.random.default_rng(10), clean, fps=60, duration_ms=1500, drop=lambda i, t: i % 3 == 0
        ),
    )
    ts = _signal(np.random.default_rng(11), clean, fps=30, duration_ms=1500)
    ts.insert(
        14, {"t_ms": ts[13]["t_ms"], "probs": _probs(np.random.default_rng(1), ix["boar"], 0.9)}
    )
    ts.insert(
        20,
        {"t_ms": ts[18]["t_ms"] - 5.0, "probs": _probs(np.random.default_rng(1), ix["dog"], 0.9)},
    )
    ts.insert(
        30, {"t_ms": ts[29]["t_ms"], "probs": _probs(np.random.default_rng(1), ix["bird"], 0.9)}
    )
    add("ema_nonmonotonic_timestamps", "equal and backwards timestamps are ignored", ts)
    seq = _signal(np.random.default_rng(12), clean, fps=30, duration_ms=1500)
    seq.insert(20, {"reset": True})
    for fr in seq[21:]:
        fr["t_ms"] = float(fr["t_ms"]) - float(seq[19]["t_ms"])  # restart the clock at ~0
    add("ema_reset_midstream", "reset() clears refractory/stats; clock restarts", seq)

    # --- grammar prior -------------------------------------------------------------
    pr = np.random.default_rng(13)
    amb: list[dict[str, Json]] = []
    for i in range(60):
        p = np.full(K, 0.02)
        p[ram] = 0.36 + 0.02 * pr.standard_normal()
        p[tiger] = 0.40
        p[IDLE] = 0.1
        p = np.clip(p, 0.001, None)
        fr: dict[str, Json] = {"t_ms": i * 1000.0 / 30.0, "probs": _round_probs(p / p.sum())}
        if 10 <= i < 50:
            fr["expected_next"] = ["ram"]
        amb.append(fr)
    add("prior_on_resolves_ambiguity", "ram boosted by the grammar prior", amb, use_prior=True)
    add("prior_off_ignores_expected_next", "same stream, prior disabled", amb, use_prior=False)
    idle_boost = [dict(f, expected_next=["zero", "tiger"]) for f in amb]
    add(
        "prior_idle_never_boosted",
        "idle named in expected_next keeps weight 1",
        idle_boost,
        prior_boost=5.0,
    )

    # --- idle / open set -----------------------------------------------------------
    add(
        "ema_zero_dominant_never_emits",
        "weak seal evidence under an idle-dominated signal",
        _signal(
            np.random.default_rng(14),
            [(200, 1200, tiger, 0.45)],
            fps=30,
            duration_ms=1500,
            idle_level=0.5,
            noise=0.03,
        ),
    )
    add(
        "ema_open_set_reject",
        "flat distribution: max below tau_reject",
        [
            {
                "t_ms": i * 33.0,
                "probs": _round_probs(np.random.default_rng(i).dirichlet(np.ones(K) * 20.0)),
            }
            for i in range(60)
        ],
    )
    odd = _signal(np.random.default_rng(15), clean, fps=30, duration_ms=1500)
    odd[5]["probs"] = [0.0] * K  # all-zero -> uniform
    odd[9]["probs"] = [-0.2 if j == 3 else p for j, p in enumerate(odd[9]["probs"])]
    odd[12]["probs"] = [float(p) * 7.0 for p in odd[12]["probs"]]  # unnormalised
    add("ema_zero_negative_unnormalised_probs", "clipping and renormalisation of raw probs", odd)
    add(
        "ema_idle_label_none",
        "idle_label = null: no class is treated as idle",
        _signal(
            np.random.default_rng(16),
            [(0, 700, ix["zero"], 0.9)],
            fps=30,
            duration_ms=1200,
            idle=tiger,
        ),
        idle_label=None,
    )

    # --- HMM smoother --------------------------------------------------------------
    add(
        "hmm_clean_seal_30fps",
        "HMM smoother, one clean seal",
        _signal(np.random.default_rng(17), clean, fps=30, duration_ms=1500),
        smoother="hmm",
    )
    add(
        "hmm_clean_seal_60fps",
        "HMM smoother, 60 fps",
        _signal(np.random.default_rng(17), clean, fps=60, duration_ms=1500),
        smoother="hmm",
    )
    add(
        "hmm_noisy_sequence",
        "HMM smoother, noisy seal sequence",
        _signal(
            np.random.default_rng(18),
            [(200, 800, tiger, 0.85), (900, 1500, monkey, 0.8)],
            fps=30,
            duration_ms=1900,
            noise=0.08,
            jitter_ms=2.0,
        ),
        smoother="hmm",
        hmm_p_stay=0.9,
    )
    add(
        "hmm_gap_resets_belief",
        "HMM smoother across a >max_gap stall",
        _signal(
            np.random.default_rng(19),
            [(100, 1500, ram, 0.9)],
            fps=30,
            duration_ms=1700,
            drop=lambda i, t: 500 <= t < 800,
        ),
        smoother="hmm",
        hmm_ref_ms=20.0,
    )
    zero_hmm = _signal(
        np.random.default_rng(20), [(100, 900, tiger, 0.9)], fps=30, duration_ms=1200
    )
    zero_hmm[8]["probs"] = [0.0 if j != IDLE else 1.0 for j in range(K)]  # exact zeros in tiger
    add(
        "hmm_exact_zero_probabilities",
        "zeros in the emission term (0 ** x = 0)",
        zero_hmm,
        smoother="hmm",
    )

    # --- custom label set ----------------------------------------------------------
    k4 = ["a", "b", "c", "idle"]
    add(
        "custom_labels_k4",
        "4 classes, non-default thresholds",
        _signal(
            np.random.default_rng(21),
            [(100, 600, 1, 0.8), (700, 1200, 2, 0.75)],
            fps=45,
            duration_ms=1500,
            idle=3,
            n_classes=4,
            noise=0.03,
        ),
        labels=tuple(k4),
        idle_label="idle",
        tau_on=0.55,
        tau_off=0.35,
        tau_reject=0.25,
        half_life_ms=60.0,
        min_dwell_ms=90.0,
        refractory_ms=100.0,
        max_gap_ms=120.0,
        prior_boost=2.0,
        hmm_p_stay=0.7,
    )

    return {
        "label_order": LABELS,
        "default_config": DecoderConfig().to_dict(),
        "tolerances": {"confidence": 1e-9, "times": "exact"},
        "cases": cases,
    }


# --------------------------------------------------------------------------------------
# Calibration fixture
# --------------------------------------------------------------------------------------
def _round(a: Array, nd: int = 6) -> Array:
    out: Array = np.round(a, nd)
    return out


def build_calibration_fixture() -> dict[str, Json]:
    rng = np.random.default_rng(2024)

    # ---- temperature scaler -------------------------------------------------------
    logits = _round(rng.normal(0.0, 3.0, size=(8, K)))
    logits[1] = 0.0  # uniform
    logits[2, 4] = 60.0  # extreme but finite (numerical stability)
    logits[3] = -80.0
    logits[3, 7] = -75.0
    temp_cases = []
    for t in (0.05, 0.7, 1.0, 1.7, 20.0):
        ts = TemperatureScaler(t)
        ts.fitted = t != 1.0
        temp_cases.append(
            {
                "scaler": ts.to_dict(),
                "logits": logits.tolist(),
                "expected_proba": ts.transform(logits).tolist(),
                "expected_logits": ts.transform_logits(logits).tolist(),
            }
        )

    # ---- vector scaler ------------------------------------------------------------
    vs = VectorScaler(K)
    vs.scale = _round(rng.uniform(0.5, 2.0, size=K))
    vs.bias = _round(rng.normal(0.0, 0.3, size=K))
    vec = {
        "scaler": vs.to_dict(),
        "logits": logits.tolist(),
        "expected_proba": vs.transform(logits).tolist(),
    }

    # ---- stacked fusion + full pipeline ------------------------------------------
    n_branches, meta_dim, n_classes = 2, 2, K
    d = n_branches * K + meta_dim
    sf = StackedFusion(l2=0.01, standardize=True)
    sf.mean = _round(rng.normal(-2.0, 0.5, size=d))
    sf.std = _round(rng.uniform(0.3, 2.0, size=d))
    sf.std[3] = 1.0  # a constant-feature column as produced by fit()
    sf.weights = _round(rng.normal(0.0, 0.4, size=(d, n_classes)))
    sf.bias = _round(rng.normal(0.0, 0.2, size=n_classes))

    def branch(n: int) -> Array:
        raw = rng.dirichlet(np.ones(K) * 0.3, size=n)
        raw[0] = 1.0 / K
        raw[1] = np.eye(K)[2]  # exact 0 / 1 entries hit the eps clip
        return _round(raw, 8)

    n = 7
    probs_a, probs_b = branch(n), branch(n)
    meta = _round(np.column_stack([rng.integers(0, 3, size=n), rng.uniform(0.0, 1.0, size=n)]))
    feats = build_fusion_features([probs_a, probs_b], meta)
    ts_pipe = TemperatureScaler(1.35)
    ts_pipe.fitted = True
    fused_logits = sf.predict_logits(feats)
    fusion = {
        "fusion": sf.to_dict(),
        "branch_probs": [probs_a.tolist(), probs_b.tolist()],
        "meta": meta.tolist(),
        "expected_features": feats.tolist(),
        "expected_logits": fused_logits.tolist(),
        "expected_proba": sf.predict_proba(feats).tolist(),
        "temperature": ts_pipe.to_dict(),
        "expected_calibrated_proba": ts_pipe.transform(fused_logits).tolist(),
    }
    fusion_no_std = StackedFusion(l2=0.1, standardize=False)
    fusion_no_std.mean = np.zeros(d)
    fusion_no_std.std = np.ones(d)
    fusion_no_std.weights = sf.weights.copy()
    fusion_no_std.bias = sf.bias.copy()
    fusion_nostd_case = {
        "fusion": fusion_no_std.to_dict(),
        "features": feats.tolist(),
        "expected_logits": fusion_no_std.predict_logits(feats).tolist(),
        "expected_proba": fusion_no_std.predict_proba(feats).tolist(),
    }
    return {
        "tolerance": 1e-9,
        "cases": [
            {"kind": "temperature", "cases": temp_cases},
            {"kind": "vector", **vec},
            {"kind": "stacked_fusion_pipeline", **fusion},
            {"kind": "stacked_fusion_no_standardize", **fusion_nostd_case},
        ],
    }


# --------------------------------------------------------------------------------------
# Entry points
# --------------------------------------------------------------------------------------
def build_all() -> dict[str, dict[str, Json]]:
    return {
        "features_v1.json": build_features_fixture(),
        "decoder_v1.json": build_decoder_fixture(),
        "calibration_v1.json": build_calibration_fixture(),
    }


def write_all(out_dir: Path = GOLDEN_DIR) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, doc in build_all().items():
        path = out_dir / name
        path.write_text(dumps_fixture(doc), encoding="utf-8")
        written.append(path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=GOLDEN_DIR, help="output directory")
    args = parser.parse_args()
    for path in write_all(args.out):
        print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
