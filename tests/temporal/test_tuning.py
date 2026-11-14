"""Tests for ``narutocv.temporal.tuning``."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from narutocv.temporal.decoder import DecoderConfig, SealEvent
from narutocv.temporal.tuning import (
    GTEvent,
    LabeledStream,
    OperatingPoint,
    evaluate_config,
    grid_search,
    make_stream,
    match_events,
    pareto_frontier,
    select_operating_point,
)

LABELS = DecoderConfig().labels
K = len(LABELS)
ZERO = LABELS.index("zero")


def ev(label: str, t_emit: float, t_start: float | None = None) -> SealEvent:
    return SealEvent(label, t_emit - 150.0 if t_start is None else t_start, t_emit, 0.9)


def test_match_true_positive_and_tolerance_window() -> None:
    gt = [GTEvent("dog", 1000, 1500)]
    assert match_events([ev("dog", 1200)], gt, 300).matches == ((0, 0),)
    assert match_events([ev("dog", 1000)], gt, 300).matches == ((0, 0),)  # inclusive start
    assert match_events([ev("dog", 1800)], gt, 300).matches == ((0, 0),)  # end + tolerance
    late = match_events([ev("dog", 1801)], gt, 300)
    assert late.matches == () and late.false_positives == (0,) and late.misses == (0,)
    early = match_events([ev("dog", 999)], gt, 300)
    assert early.false_positives == (0,) and early.misses == (0,)


def test_match_false_positive_false_negative_and_wrong_label() -> None:
    gt = [GTEvent("dog", 1000, 1500), GTEvent("ram", 3000, 3400)]
    pred = [ev("dog", 1200), ev("tiger", 2000), ev("rat", 3100)]
    res = match_events(pred, gt)
    assert res.matches == ((0, 0),)
    assert res.false_positives == (1, 2)  # spurious tiger; wrong label rat
    assert res.misses == (1,)  # ram never detected


def test_match_double_emission_is_one_tp_one_fp() -> None:
    gt = [GTEvent("dog", 1000, 2000)]
    res = match_events([ev("dog", 1200), ev("dog", 1700)], gt)
    assert res.matches == ((0, 0),)
    assert res.false_positives == (1,)
    assert res.misses == ()


def test_match_is_one_to_one_greedy_by_time() -> None:
    gt = [GTEvent("dog", 1000, 1500), GTEvent("dog", 1400, 2000)]
    # unsorted input: the earlier emission takes the earlier interval
    res = match_events([ev("dog", 1450), ev("dog", 1300)], gt)
    assert res.matches == ((0, 1), (1, 0))
    assert res.false_positives == () and res.misses == ()
    # three predictions, two intervals -> one FP
    res3 = match_events([ev("dog", 1100), ev("dog", 1450), ev("dog", 1600)], gt)
    assert len(res3.matches) == 2 and len(res3.false_positives) == 1


def test_match_empty_inputs() -> None:
    assert match_events([], []).matches == ()
    assert match_events([], [GTEvent("dog", 0, 1)]).misses == (0,)
    assert match_events([ev("dog", 5)], []).false_positives == (0,)
    with pytest.raises(ValueError):
        GTEvent("dog", 5, 1)


@settings(max_examples=60, deadline=None)
@given(
    st.lists(st.tuples(st.sampled_from(["dog", "ram"]), st.floats(0, 5000)), max_size=12),
    st.lists(
        st.tuples(st.sampled_from(["dog", "ram"]), st.floats(0, 5000), st.floats(0, 800)),
        max_size=8,
    ),
)
def test_match_partition_properties(
    preds: list[tuple[str, float]], gts: list[tuple[str, float, float]]
) -> None:
    pred = [ev(label, t) for label, t in preds]
    gt = [GTEvent(label, s, s + d) for label, s, d in gts]
    res = match_events(pred, gt, 300.0)
    assert len(res.matches) + len(res.false_positives) == len(pred)
    assert len(res.matches) + len(res.misses) == len(gt)
    assert len({p for p, _ in res.matches}) == len(res.matches)
    assert len({g for _, g in res.matches}) == len(res.matches)
    for p, g in res.matches:
        assert pred[p].label == gt[g].label
        assert gt[g].start_ms <= pred[p].t_emit_ms <= gt[g].end_ms + 300.0


def build_stream(
    seed: int, fps: float = 30.0, minutes: float = 1.0, noise: float = 0.25
) -> tuple[list[GTEvent], LabeledStream]:
    """Noisy stream with 8 true seals plus short spurious bursts."""
    rng = np.random.default_rng(seed)
    dt = 1000.0 / fps
    t = np.arange(0.0, minutes * 60_000.0, dt)
    probs = np.zeros((t.size, K))
    probs[:, ZERO] = 1.0
    gts: list[GTEvent] = []
    for n in range(8):
        label = LABELS[n % 4]
        start = 3000.0 + n * 6500.0
        gts.append(GTEvent(label, start, start + 700.0))
        mask = (t >= start) & (t < start + 700.0)
        probs[mask] = 0.0
        probs[mask, LABELS.index(label)] = 0.85
        probs[mask, ZERO] = 0.15
    for _ in range(10):  # brief, confident-looking spurious bursts (40-90 ms)
        s = rng.uniform(1000.0, minutes * 60_000.0 - 1000.0)
        m = (t >= s) & (t < s + rng.uniform(40.0, 90.0))
        probs[m] = 0.0
        probs[m, LABELS.index("snake")] = 0.9
        probs[m, ZERO] = 0.1
    probs += rng.uniform(0, noise * 0.05, size=probs.shape)
    probs /= probs.sum(axis=1, keepdims=True)
    return gts, make_stream(probs, t, gts)


def test_evaluate_config_counts() -> None:
    gts, stream = build_stream(0)
    pt = evaluate_config([stream], DecoderConfig())
    assert pt.n_events == 8
    assert pt.recall == pytest.approx(pt.n_detected / 8)
    assert pt.recall == 1.0
    assert 0.0 < pt.median_ttd_ms < 500.0
    assert pt.fa_per_min == pytest.approx(pt.n_false / 1.0, rel=0.02)
    assert pt.fa_per_min == 0.0  # 40-90 ms bursts are filtered by smoothing + dwell
    with pytest.raises(ValueError):
        evaluate_config([make_stream(np.full((5, K), 1 / K), np.arange(5.0), [])], DecoderConfig())


def test_grid_search_and_pareto_frontier() -> None:
    gts, stream = build_stream(1)
    points = grid_search(
        [stream],
        tau_on=[0.5, 0.7, 0.95],
        tau_off=[0.3, 0.5],
        min_dwell_ms=[0.0, 120.0, 300.0],
        half_life_ms=[20.0, 80.0],
    )
    assert len(points) == 5 * 3 * 2  # (on, off) pairs with off < on: 5 of 6
    assert all(p.config.tau_reject <= p.config.tau_off for p in points)
    front = pareto_frontier(points)
    assert 1 <= len(front) <= len(points)
    for a in front:
        for b in front:
            if a is not b:
                dominated = (
                    a.fa_per_min <= b.fa_per_min
                    and a.recall >= b.recall
                    and a.median_ttd_ms <= b.median_ttd_ms
                )
                assert not dominated
    # sharp thresholds sacrifice recall (0.95 is unreachable with p=0.85), and
    # permissive thresholds incur false activations: both extremes appear in the sweep.
    assert min(p.recall for p in points) == 0.0
    assert max(p.fa_per_min for p in points) > 0.0
    best = select_operating_point(points, max_fa_per_min=0.5)
    assert best is not None and best.fa_per_min <= 0.5 and best.recall == 1.0
    assert select_operating_point(points, max_fa_per_min=-1.0) is None


def op(fa: float, rec: float, ttd: float) -> OperatingPoint:
    return OperatingPoint(DecoderConfig(), fa, rec, ttd, 10, 0, 0)


def test_pareto_handles_inf_ties_and_duplicates() -> None:
    pts = [
        op(0.0, 0.0, math.inf),  # no detections, no FA: still non-dominated (lowest FA)
        op(0.0, 0.0, math.inf),  # duplicate
        op(0.5, 0.9, 300.0),
        op(0.5, 0.9, 300.0),  # duplicate
        op(0.6, 0.9, 300.0),  # dominated by (0.5, 0.9, 300)
        op(0.2, 0.9, math.inf),  # inf ttd but better FA than (0.5,...): kept
        op(1.0, 1.0, 400.0),
        op(1.0, 0.95, 400.0),  # dominated by previous
    ]
    front = pareto_frontier(pts)
    keys = [(p.fa_per_min, p.recall, p.median_ttd_ms) for p in front]
    assert keys == [
        (0.0, 0.0, math.inf),
        (0.2, 0.9, math.inf),
        (0.5, 0.9, 300.0),
        (1.0, 1.0, 400.0),
    ]
    assert pareto_frontier([]) == []
