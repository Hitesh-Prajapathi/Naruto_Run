"""Tests for ``narutocv.temporal.decoder``."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from narutocv.temporal.decoder import (
    DecoderConfig,
    EmaSmoother,
    HMMSmoother,
    SealDecoder,
    SealEvent,
)
from numpy.typing import NDArray

LABELS = DecoderConfig().labels
K = len(LABELS)
A = LABELS.index("dog")
B = LABELS.index("ram")
ZERO = LABELS.index("zero")


def vec(label: int, p: float, rest: int = ZERO) -> NDArray[np.float64]:
    """Probability vector with ``p`` on ``label`` and the remainder on ``rest``."""
    v = np.zeros(K)
    v[label] = p
    v[rest] += 1.0 - p
    return v


def run(
    signal: Callable[[float], NDArray[np.float64]],
    fps: float,
    duration_ms: float,
    config: DecoderConfig | None = None,
    **kwargs: object,
) -> list[SealEvent]:
    dec = SealDecoder(config)
    events: list[SealEvent] = []
    dt = 1000.0 / fps
    n = int(duration_ms / dt)
    for i in range(n + 1):
        t = i * dt
        events += dec.update(signal(t), t, **kwargs)  # type: ignore[arg-type]
    return events


def pulse(
    start: float, end: float, p: float = 0.9, label: int = A
) -> Callable[[float], NDArray[np.float64]]:
    def f(t: float) -> NDArray[np.float64]:
        return vec(label, p if start <= t < end else 0.01)

    return f


def lag_to_threshold(cfg: DecoderConfig, p: float, floor: float = 0.01) -> float:
    """Time for the EMA to climb from ``floor`` to ``tau_on`` toward target ``p``."""
    frac = (p - cfg.tau_on) / (p - floor)
    return cfg.half_life_ms * math.log2(1.0 / frac)


def test_clean_seal_emits_exactly_once_with_bounded_latency() -> None:
    cfg = DecoderConfig()
    onset, fps = 300.0, 30.0
    frame = 1000.0 / fps
    events = run(pulse(onset, onset + 500.0), fps, 1500.0)
    assert len(events) == 1
    ev = events[0]
    assert ev.label == "dog"
    lag = lag_to_threshold(cfg, 0.9)
    # The EMA treats each sample as held over the preceding interval (anticipation <= 1 frame).
    assert onset + lag - frame <= ev.t_start_ms <= onset + lag + frame
    assert ev.t_emit_ms - ev.t_start_ms >= cfg.min_dwell_ms
    assert ev.t_emit_ms <= onset + lag + cfg.min_dwell_ms + 2 * frame
    assert 0.6 <= ev.confidence <= 0.9


def test_event_serialisation_and_idempotent_reset() -> None:
    dec = SealDecoder()
    events: list[SealEvent] = []
    for i in range(40):
        events += dec.update(vec(A, 0.9), i * 33.3)
    assert len(events) == 1
    assert set(events[0].to_dict()) == {"label", "t_start_ms", "t_emit_ms", "confidence"}
    dec.reset()
    assert dec.stats.frames == 0 and dec.active_label is None
    again: list[SealEvent] = []
    for i in range(40):  # timestamps restart from 0 after reset: must be accepted
        again += dec.update(vec(A, 0.9), i * 33.3)
    assert [e.label for e in again] == ["dog"]
    assert again[0].t_emit_ms == events[0].t_emit_ms


def test_same_seal_needs_drop_below_tau_off_to_reenter() -> None:
    # Long hold: one event only.
    assert len(run(pulse(100, 3000), 60, 3500)) == 1
    # Dip that stays above tau_off (0.4 smoothed): still one event, no re-trigger.

    def dip(t: float) -> NDArray[np.float64]:
        p = 0.55 if 1000 <= t < 1300 else 0.9
        return vec(A, p)

    assert len(run(dip, 60, 2500)) == 1
    # Real release (drop to idle) then a second seal after the refractory: two events.
    two = run(lambda t: vec(A, 0.9 if (t < 600 or 1200 <= t < 1800) else 0.01), 60, 2500)
    assert [e.label for e in two] == ["dog", "dog"]
    assert two[1].t_emit_ms > two[0].t_emit_ms


def test_flicker_around_tau_on_does_not_chatter() -> None:
    cfg = DecoderConfig()
    for fps in (30.0, 60.0):
        for hi, lo in [(0.7, 0.5), (0.8, 0.4), (0.65, 0.55)]:
            period = 2.0 * 1000.0 / fps

            def f(
                t: float, hi: float = hi, lo: float = lo, period: float = period
            ) -> NDArray[np.float64]:
                return vec(A, hi if (t % period) < period / 2 else lo)

            events = run(f, fps, 4000.0)
            assert len(events) <= 1, (fps, hi, lo, events)

    # flicker about tau_off while active: still no extra events
    def g(t: float) -> NDArray[np.float64]:
        if t < 600:
            return vec(A, 0.9)
        return vec(A, 0.5 if int(t / 33.3) % 2 else 0.3)

    assert len(run(g, 30.0, 4000.0, cfg)) == 1


def test_fps_independence_30_vs_60() -> None:
    cfg = DecoderConfig()
    sig = lambda t: vec(A, 0.9 if 300 <= t < 900 else 0.01) if t < 1500 else vec(B, 0.9)  # noqa: E731
    e30 = run(sig, 30.0, 2500.0)
    e60 = run(sig, 60.0, 2500.0)
    assert [e.label for e in e30] == [e.label for e in e60] == ["dog", "ram"]
    # +-1 frame of the coarser rate on the continuous-time quantities; dwell is measured on
    # sample timestamps so t_emit may add one more coarse frame of quantisation.
    frame = 1000.0 / 30.0
    for a, b in zip(e30, e60, strict=True):
        assert abs(a.t_start_ms - b.t_start_ms) <= frame + 1e-6
        assert abs(a.t_emit_ms - b.t_emit_ms) <= 2 * frame + 1e-6
    assert cfg.min_dwell_ms > 0


def test_dropped_frames_handled() -> None:
    rng = np.random.default_rng(0)
    fps, onset = 60.0, 300.0
    dt = 1000.0 / fps
    times = [i * dt for i in range(int(2000 / dt))]
    keep = [t for t in times if rng.random() > 0.4 or t < 100]
    dec = SealDecoder()
    events: list[SealEvent] = []
    sig = pulse(onset, onset + 700)
    for t in keep:
        events += dec.update(sig(t), t)
    assert len(events) == 1
    lag = lag_to_threshold(dec.config, 0.9)
    assert abs(events[0].t_start_ms - (onset + lag)) <= 3 * dt + 20  # a few dropped frames
    assert dec.stats.gaps == 0


def test_long_gap_breaks_dwell_but_not_active_seal() -> None:
    cfg = DecoderConfig()
    dec = SealDecoder(cfg)
    events: list[SealEvent] = []
    # Pending dwell is interrupted by a stall: dwell restarts after the gap.
    for t in (0.0, 33.0, 66.0, 99.0, 132.0):
        events += dec.update(vec(A, 0.01), t)
    t = 165.0
    events += dec.update(vec(A, 0.95), t)  # first sample after reset-like gap-free step
    events += dec.update(vec(A, 0.95), t + 33)
    assert events == []
    events += dec.update(vec(A, 0.95), t + 33 + 500)  # 500 ms stall > max_gap
    assert dec.stats.gaps == 1
    assert events == []  # dwell restarted at the stall boundary
    for i in range(1, 8):
        events += dec.update(vec(A, 0.95), t + 533 + 33 * i)
    assert len(events) == 1
    assert events[0].t_start_ms >= t + 533  # started after the gap
    # An active seal survives a stall: no duplicate.
    n = len(events)
    events += dec.update(vec(A, 0.95), t + 533 + 33 * 8 + 600)
    assert len(events) == n and dec.active_label == "dog"


def test_nonmonotonic_timestamps_ignored_and_counted() -> None:
    dec = SealDecoder()
    dec.update(vec(A, 0.9), 100.0)
    before = dec.smoothed
    assert dec.update(vec(B, 0.99), 100.0) == []  # equal
    assert dec.update(vec(B, 0.99), 50.0) == []  # backwards
    assert dec.stats.nonmonotonic == 2
    assert np.array_equal(before, dec.smoothed)
    assert dec.stats.frames == 1
    with pytest.raises(ValueError):
        dec.update(np.ones(K - 1), 200.0)
    with pytest.raises(ValueError):
        dec.update(np.full(K, np.nan), 200.0)


def borderline(t: float) -> NDArray[np.float64]:
    """Raw p(dog) = 0.5 (below tau_on) with the rest spread evenly."""
    v = np.full(K, 0.5 / (K - 1))
    v[A] = 0.5
    return v


def test_grammar_prior_raises_recall_on_borderline_signal() -> None:
    base = run(borderline, 30.0, 2000.0)
    assert base == []
    with_prior = run(borderline, 30.0, 2000.0, expected_next=["dog"])
    assert [e.label for e in with_prior] == ["dog"]
    off = DecoderConfig(use_prior=False)
    disabled = run(borderline, 30.0, 2000.0, off, expected_next=["dog"])
    assert disabled == base


def test_prior_disabled_is_bit_identical_to_none() -> None:
    cfg = DecoderConfig(use_prior=False)
    d1, d2 = SealDecoder(cfg), SealDecoder(cfg)
    rng = np.random.default_rng(3)
    for i in range(100):
        p = rng.dirichlet(np.ones(K))
        e1 = d1.update(p, i * 33.0, expected_next=["dog", "ram"])
        e2 = d2.update(p, i * 33.0)
        assert e1 == e2
        assert np.array_equal(d1.smoothed, d2.smoothed)
    # unknown label only raises when the prior is active
    SealDecoder(cfg).update(borderline(0), 0.0, expected_next=["nope"])
    with pytest.raises(ValueError):
        SealDecoder().update(borderline(0), 0.0, expected_next=["nope"])


def test_zero_dominated_sequence_yields_no_events() -> None:
    rng = np.random.default_rng(1)
    dec = SealDecoder()
    for i in range(600):
        p = np.full(K, 0.02)
        p[ZERO] = 0.85 + 0.1 * rng.random()
        p = p / p.sum()
        assert dec.update(p, i * 16.7) == []
    # prior cannot resurrect idle-dominated frames
    dec2 = SealDecoder()
    for i in range(600):
        p = np.full(K, 0.01)
        p[ZERO] = 0.88
        p = p / p.sum()
        assert dec2.update(p, i * 16.7, expected_next=["dog"]) == []


def test_open_set_reject_flat_distribution() -> None:
    dec = SealDecoder()
    flat = np.full(K, 1.0 / K)
    for i in range(300):
        assert dec.update(flat, i * 33.0) == []
    # max prob below tau_reject while a seal is active ends it
    d = SealDecoder()
    for i in range(30):
        d.update(vec(A, 0.9), i * 33.0)
    assert d.active_label == "dog"
    for i in range(30, 80):
        d.update(flat, i * 33.0)
    assert d.active_label is None


def test_refractory_defers_next_seal() -> None:
    cfg = DecoderConfig(refractory_ms=600.0)

    def f(t: float) -> NDArray[np.float64]:
        if t < 400:
            return vec(A, 0.95)
        return vec(B, 0.95)

    ev = run(f, 60.0, 2000.0, cfg)
    assert [e.label for e in ev] == ["dog", "ram"]
    assert ev[1].t_emit_ms - ev[0].t_emit_ms >= 600.0 - 1e-9
    assert ev[1].t_start_ms < ev[1].t_emit_ms - 100  # deferred, not restarted


def test_hmm_smoother_alternative() -> None:
    cfg = DecoderConfig(smoother="hmm")
    ev = run(pulse(300, 800), 30.0, 1500.0, cfg)
    assert [e.label for e in ev] == ["dog"]
    # frame-rate independence of the predict step
    s30, s60 = HMMSmoother(K), HMMSmoother(K)
    p = vec(A, 0.9)
    q = vec(B, 0.9)
    s30.update(q, 0.0)
    s60.update(q, 0.0)
    r1 = s30.update(p, 33.3)
    s60.update(p, 16.65)
    r2 = s60.update(p, 16.65)
    assert abs(r1[A] - r2[A]) < 0.15  # approx. equal (untempered: 0.36)
    assert abs(float(r1.sum()) - 1.0) < 1e-12
    # zero-likelihood observation must not produce NaN
    h = HMMSmoother(K)
    h.update(vec(A, 1.0, rest=A), 0.0)
    out = h.update(vec(B, 1.0, rest=B), 1e6)
    assert np.all(np.isfinite(out))


def test_ema_alpha_is_half_life_based() -> None:
    sm = EmaSmoother(80.0)
    sm.update(np.array([0.0, 1.0]), 0.0)
    out = sm.update(np.array([1.0, 0.0]), 80.0)
    assert out[0] == pytest.approx(0.5)  # one half-life closes half the gap
    # splitting the interval gives the same result for a constant input
    a = EmaSmoother(80.0)
    a.update(np.array([0.0, 1.0]), 0.0)
    a.update(np.array([1.0, 0.0]), 40.0)
    out2 = a.update(np.array([1.0, 0.0]), 40.0)
    assert out2[0] == pytest.approx(0.5)


@pytest.mark.parametrize(
    "field",
    [
        "half_life_ms",
        "hmm_ref_ms",
        "min_dwell_ms",
        "refractory_ms",
        "max_gap_ms",
        "prior_boost",
        "tau_on",
        "tau_off",
        "tau_reject",
        "hmm_p_stay",
    ],
)
@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_config_rejects_non_finite_numbers(field: str, bad: float) -> None:
    with pytest.raises(ValueError, match=f"{field} must be finite"):
        DecoderConfig(**{field: bad})  # type: ignore[arg-type]


def test_spec_version_is_a_patch_bump() -> None:
    from narutocv.temporal.decoder import DECODER_SPEC_VERSION

    assert DECODER_SPEC_VERSION == "1.0.1"


def test_config_roundtrip_and_validation() -> None:
    cfg = DecoderConfig(tau_on=0.7, tau_off=0.35, half_life_ms=60.0, smoother="hmm")
    assert DecoderConfig.from_dict(cfg.to_dict()) == cfg
    import json

    assert DecoderConfig.from_dict(json.loads(json.dumps(cfg.to_dict()))) == cfg
    with pytest.raises(ValueError):
        DecoderConfig(tau_on=0.4, tau_off=0.5)
    with pytest.raises(ValueError):
        DecoderConfig(tau_reject=0.5)
    with pytest.raises(ValueError):
        DecoderConfig(idle_label="nope")
    with pytest.raises(ValueError):
        DecoderConfig.from_dict({"bogus": 1})
    d = DecoderConfig()
    assert (d.half_life_ms, d.tau_on, d.tau_off, d.min_dwell_ms, d.refractory_ms) == (
        80.0,
        0.6,
        0.4,
        120.0,
        250.0,
    )


@settings(max_examples=40, deadline=None)
@given(
    st.lists(st.integers(0, K - 1), min_size=5, max_size=60),
    st.lists(st.floats(1.0, 80.0), min_size=60, max_size=60),
    st.integers(0, 10_000),
)
def test_decoder_invariants_random_streams(labels: list[int], dts: list[float], seed: int) -> None:
    """Events are ordered, non-idle, respect dwell and refractory, never NaN."""
    rng = np.random.default_rng(seed)
    cfg = DecoderConfig()
    dec = SealDecoder(cfg)
    t = 0.0
    events: list[SealEvent] = []
    for i, lab in enumerate(labels * 3):
        t += dts[i % len(dts)]
        p = rng.dirichlet(np.ones(K) * 0.3) * 0.3
        p[lab] += 0.7
        events += dec.update(p / p.sum(), t)
    assert np.all(np.isfinite(dec.smoothed))
    for ev in events:
        assert ev.label != "zero"
        assert ev.t_emit_ms - ev.t_start_ms >= cfg.min_dwell_ms - 1e-9
        assert 0.0 <= ev.confidence <= 1.0
    for a, b in zip(events, events[1:], strict=False):
        assert b.t_emit_ms - a.t_emit_ms >= cfg.refractory_ms - 1e-9
