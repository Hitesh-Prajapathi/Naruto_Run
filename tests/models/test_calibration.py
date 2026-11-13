"""Tests for ``narutocv.models.calibration``."""

from __future__ import annotations

import json

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.typing import NDArray
from scipy.optimize import check_grad

from narutocv.models.calibration import (
    StackedFusion,
    TemperatureScaler,
    VectorScaler,
    build_fusion_features,
    expected_calibration_error,
    log_softmax,
    softmax,
)

K = 6


def synth(n: int, true_temp: float, seed: int) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """True logits z; labels ~ softmax(z); *observed* logits are true_temp * z."""
    rng = np.random.default_rng(seed)
    z = rng.normal(0.0, 1.5, size=(n, K))
    p = softmax(z)
    cum = p.cumsum(axis=1)
    u = rng.random((n, 1))
    labels = (u > cum).sum(axis=1).astype(np.int64)
    labels = np.minimum(labels, K - 1)
    return true_temp * z, labels


def test_temperature_recovers_known_value() -> None:
    logits, labels = synth(30_000, 2.5, seed=0)
    scaler = TemperatureScaler().fit(logits, labels)
    assert scaler.fitted
    assert abs(scaler.temperature - 2.5) / 2.5 < 0.10


def test_temperature_respects_bounds() -> None:
    # perfectly separable -> NLL keeps falling with sharper logits; T pinned at lower bound
    rng = np.random.default_rng(1)
    labels = rng.integers(0, K, size=500)
    logits = np.full((500, K), -5.0)
    logits[np.arange(500), labels] = 5.0
    t = TemperatureScaler().fit(logits, labels).temperature
    assert 0.05 <= t <= 20.0 and t < 0.1
    with pytest.raises(ValueError):
        TemperatureScaler(100.0)


def test_ece_reduced_on_overconfident_heldout() -> None:
    logits, labels = synth(40_000, 3.0, seed=2)
    fit_l, fit_y = logits[:20_000], labels[:20_000]
    test_l, test_y = logits[20_000:], labels[20_000:]
    before = expected_calibration_error(softmax(test_l), test_y)
    scaler = TemperatureScaler().fit(fit_l, fit_y)
    after = expected_calibration_error(scaler.transform(test_l), test_y)
    assert before > 0.1
    assert after < 0.03
    assert after < before / 3


def test_ece_basic_cases() -> None:
    probs = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    assert expected_calibration_error(probs, np.array([0, 0, 1, 1])) == pytest.approx(0.0)
    assert expected_calibration_error(probs, np.array([1, 1, 0, 0])) == pytest.approx(1.0)
    assert expected_calibration_error(np.zeros((0, 2)), np.zeros(0)) == 0.0


def test_temperature_json_roundtrip() -> None:
    logits, labels = synth(2_000, 1.7, seed=3)
    a = TemperatureScaler().fit(logits, labels)
    payload = json.loads(json.dumps(a.to_dict()))
    b = TemperatureScaler.from_dict(payload)
    assert b.to_dict() == a.to_dict()
    assert b.temperature == a.temperature
    assert np.array_equal(a.transform(logits), b.transform(logits))


def test_vector_scaler_improves_and_roundtrips() -> None:
    logits, labels = synth(20_000, 2.0, seed=4)
    # class-dependent distortion that a single temperature cannot undo
    distorted = logits * np.linspace(0.6, 1.6, K) + np.linspace(-0.5, 0.5, K)
    tr_l, tr_y, te_l, te_y = (
        distorted[:10_000],
        labels[:10_000],
        distorted[10_000:],
        labels[10_000:],
    )
    t = TemperatureScaler().fit(tr_l, tr_y)
    v = VectorScaler(K).fit(tr_l, tr_y)
    nll_t = -np.mean(log_softmax(te_l / t.temperature)[np.arange(len(te_y)), te_y])
    nll_v = -np.mean(np.log(v.transform(te_l)[np.arange(len(te_y)), te_y]))
    assert nll_v < nll_t
    v2 = VectorScaler.from_dict(json.loads(json.dumps(v.to_dict())))
    assert np.array_equal(v.transform(te_l), v2.transform(te_l))
    assert v2.to_dict() == v.to_dict()


def _fusion_data(n: int, seed: int) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, K, size=n)

    def branch(noise: float) -> NDArray[np.float64]:
        z = rng.normal(0, noise, size=(n, K))
        z[np.arange(n), y] += 2.0
        return softmax(z)

    meta = np.stack([rng.integers(0, 3, size=n), rng.random(n)], axis=1).astype(np.float64)
    return build_fusion_features([branch(1.5), branch(1.0)], meta), y.astype(np.int64)


def test_fusion_learns_and_beats_single_branch() -> None:
    x, y = _fusion_data(6_000, 5)
    xt, yt = _fusion_data(3_000, 6)
    model = StackedFusion(l2=1e-3).fit(x, y, n_classes=K)
    proba = model.predict_proba(xt)
    assert proba.shape == (3_000, K)
    assert np.allclose(proba.sum(axis=1), 1.0)
    acc_fused = float((proba.argmax(axis=1) == yt).mean())
    acc_branch0 = float((xt[:, :K].argmax(axis=1) == yt).mean())
    assert acc_fused > acc_branch0
    assert acc_fused > 0.8


def test_fusion_json_roundtrip_exact() -> None:
    x, y = _fusion_data(1_500, 7)
    model = StackedFusion(l2=1e-2).fit(x, y, n_classes=K)
    payload = json.loads(json.dumps(model.to_dict()))
    clone = StackedFusion.from_dict(payload)
    assert clone.to_dict() == model.to_dict()
    assert np.array_equal(model.predict_proba(x), clone.predict_proba(x))
    assert np.array_equal(model.predict_logits(x), clone.predict_logits(x))


def test_fusion_gradient_matches_numeric() -> None:
    x, y = _fusion_data(200, 8)
    model = StackedFusion(l2=0.05)
    n, d = x.shape
    onehot = np.zeros((n, K))
    onehot[np.arange(n), y] = 1.0
    xs = (x - x.mean(axis=0)) / x.std(axis=0)
    rng = np.random.default_rng(0)
    theta = rng.normal(0, 0.1, size=d * K + K)

    def f(t: NDArray[np.float64]) -> float:
        return model._objective(t, xs, onehot)[0]

    def g(t: NDArray[np.float64]) -> NDArray[np.float64]:
        return model._objective(t, xs, onehot)[1]

    assert check_grad(f, g, theta) < 1e-5


def test_fusion_constant_column_and_validation() -> None:
    x, y = _fusion_data(300, 9)
    x = np.concatenate([x, np.ones((300, 1))], axis=1)  # zero-variance column
    model = StackedFusion().fit(x, y, n_classes=K)
    assert np.all(np.isfinite(model.predict_proba(x)))
    with pytest.raises(ValueError):
        model.predict_proba(x[:, :3])


@settings(max_examples=30, deadline=None)
@given(st.floats(0.2, 8.0), st.integers(0, 1000))
def test_temperature_scaling_preserves_argmax(temp: float, seed: int) -> None:
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(50, K))
    scaler = TemperatureScaler(float(np.clip(temp, 0.05, 20.0)))
    assert np.array_equal(scaler.transform(logits).argmax(axis=1), logits.argmax(axis=1))
