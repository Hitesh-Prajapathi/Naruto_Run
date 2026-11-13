"""Probability calibration and stacked fusion (numpy + scipy only).

* :class:`TemperatureScaler` -- single global temperature, fit by NLL.
* :class:`VectorScaler` -- per-class scale and bias (optional, needs more data).
* :func:`expected_calibration_error` -- top-label ECE.
* :class:`StackedFusion` -- multinomial logistic regression over concatenated
  per-branch log-probabilities and meta features; inference is a matrix multiply
  plus softmax so it ports directly to TypeScript (weights live in JSON).

All ``to_dict`` payloads are plain JSON types (``calibration.json`` in the model bundle).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize, minimize_scalar

Array = NDArray[np.float64]

T_MIN = 0.05
T_MAX = 20.0


def _as_logits(logits: ArrayLike) -> Array:
    arr = np.asarray(logits, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"logits must be 2-D (N, K), got shape {arr.shape}")
    return arr


def _as_labels(labels: ArrayLike, n_samples: int, n_classes: int) -> NDArray[np.int64]:
    y = np.asarray(labels)
    if y.shape != (n_samples,):
        raise ValueError(f"labels must have shape ({n_samples},), got {y.shape}")
    y_int = y.astype(np.int64)
    if y_int.size and (y_int.min() < 0 or y_int.max() >= n_classes):
        raise ValueError("labels out of range for the number of classes")
    return y_int


def log_softmax(logits: ArrayLike) -> Array:
    """Numerically stable row-wise log-softmax."""
    z = np.asarray(logits, dtype=np.float64)
    shifted = z - z.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def softmax(logits: ArrayLike) -> Array:
    """Numerically stable row-wise softmax."""
    return np.exp(log_softmax(logits))


def negative_log_likelihood(probs: ArrayLike, labels: ArrayLike, eps: float = 1e-12) -> float:
    p = np.asarray(probs, dtype=np.float64)
    y = _as_labels(labels, p.shape[0], p.shape[1])
    return float(-np.mean(np.log(np.clip(p[np.arange(p.shape[0]), y], eps, 1.0))))


def expected_calibration_error(probs: ArrayLike, labels: ArrayLike, n_bins: int = 15) -> float:
    """Top-label expected calibration error with equal-width confidence bins.

    ``ECE = sum_b (n_b / N) * |acc_b - conf_b|`` where a sample falls into bin
    ``b`` if ``lo < confidence <= hi`` (the first bin also includes 0).
    """
    p = np.asarray(probs, dtype=np.float64)
    if p.ndim != 2:
        raise ValueError("probs must be 2-D (N, K)")
    y = _as_labels(labels, p.shape[0], p.shape[1])
    if p.shape[0] == 0:
        return 0.0
    conf = p.max(axis=1)
    correct = (p.argmax(axis=1) == y).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        mask = (conf > lo) & (conf <= hi) if b > 0 else (conf >= lo) & (conf <= hi)
        if mask.any():
            ece += float(mask.mean()) * abs(float(correct[mask].mean()) - float(conf[mask].mean()))
    return ece


class TemperatureScaler:
    """``p = softmax(logits / T)`` with ``T`` fit by minimising NLL, ``T in [0.05, 20]``."""

    def __init__(self, temperature: float = 1.0) -> None:
        if not T_MIN <= temperature <= T_MAX:
            raise ValueError(f"temperature must be within [{T_MIN}, {T_MAX}]")
        self.temperature = float(temperature)
        self.fitted = False

    def fit(self, logits: ArrayLike, labels: ArrayLike) -> TemperatureScaler:
        z = _as_logits(logits)
        y = _as_labels(labels, z.shape[0], z.shape[1])
        rows = np.arange(z.shape[0])

        def nll(beta: float) -> float:  # beta = 1 / T; the NLL is convex in beta
            return float(-np.mean(log_softmax(z * beta)[rows, y]))

        res = minimize_scalar(
            nll,
            bounds=(1.0 / T_MAX, 1.0 / T_MIN),
            method="bounded",
            options={"xatol": 1e-10},
        )
        self.temperature = float(np.clip(1.0 / res.x, T_MIN, T_MAX))
        self.fitted = True
        return self

    def transform(self, logits: ArrayLike) -> Array:
        """Calibrated probabilities."""
        return softmax(np.asarray(logits, dtype=np.float64) / self.temperature)

    predict_proba = transform

    def transform_logits(self, logits: ArrayLike) -> Array:
        return np.asarray(logits, dtype=np.float64) / self.temperature

    def to_dict(self) -> dict[str, Any]:
        return {"type": "temperature", "temperature": self.temperature, "fitted": self.fitted}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TemperatureScaler:
        if data.get("type") != "temperature":
            raise ValueError("not a temperature scaler payload")
        obj = cls(float(data["temperature"]))
        obj.fitted = bool(data.get("fitted", True))
        return obj


class VectorScaler:
    """``p = softmax(a * logits + b)`` with per-class ``a in [0.05, 20]`` and bias ``b``.

    Optional richer variant of :class:`TemperatureScaler`; the L2 term pulls
    ``(a, b)`` toward the identity ``(1, 0)`` so small calibration sets stay stable.
    """

    def __init__(self, n_classes: int, l2: float = 1e-3) -> None:
        self.n_classes = n_classes
        self.l2 = l2
        self.scale: Array = np.ones(n_classes)
        self.bias: Array = np.zeros(n_classes)

    def fit(self, logits: ArrayLike, labels: ArrayLike) -> VectorScaler:
        z = _as_logits(logits)
        if z.shape[1] != self.n_classes:
            raise ValueError("logits have the wrong number of classes")
        y = _as_labels(labels, z.shape[0], self.n_classes)
        n, k = z.shape
        onehot = np.zeros((n, k))
        onehot[np.arange(n), y] = 1.0

        def objective(theta: Array) -> tuple[float, Array]:
            a, b = theta[:k], theta[k:]
            ls = log_softmax(z * a + b)
            loss = -float(np.mean(ls[np.arange(n), y]))
            loss += 0.5 * self.l2 * (float(np.sum((a - 1.0) ** 2)) + float(np.sum(b**2)))
            diff = (np.exp(ls) - onehot) / n
            grad_a = np.sum(diff * z, axis=0) + self.l2 * (a - 1.0)
            grad_b = np.sum(diff, axis=0) + self.l2 * b
            return loss, np.concatenate([grad_a, grad_b])

        theta0 = np.concatenate([np.ones(k), np.zeros(k)])
        bounds = [(T_MIN, T_MAX)] * k + [(None, None)] * k
        res = minimize(objective, theta0, jac=True, method="L-BFGS-B", bounds=bounds)
        self.scale, self.bias = res.x[:k].copy(), res.x[k:].copy()
        return self

    def transform(self, logits: ArrayLike) -> Array:
        return softmax(np.asarray(logits, dtype=np.float64) * self.scale + self.bias)

    predict_proba = transform

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "vector",
            "n_classes": self.n_classes,
            "l2": self.l2,
            "scale": [float(v) for v in self.scale],
            "bias": [float(v) for v in self.bias],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VectorScaler:
        if data.get("type") != "vector":
            raise ValueError("not a vector scaler payload")
        obj = cls(int(data["n_classes"]), float(data["l2"]))
        obj.scale = np.asarray(data["scale"], dtype=np.float64)
        obj.bias = np.asarray(data["bias"], dtype=np.float64)
        return obj


def log_prob_features(probs: ArrayLike, eps: float = 1e-6) -> Array:
    """``log(clip(p, eps, 1))`` -- the per-branch input representation for fusion."""
    return np.log(np.clip(np.asarray(probs, dtype=np.float64), eps, 1.0))


def build_fusion_features(
    branch_probs: Sequence[ArrayLike],
    meta: ArrayLike | None = None,
    eps: float = 1e-6,
) -> Array:
    """Concatenate per-branch log-probs (each ``(N, K_b)``) and meta features ``(N, M)``.

    Typical meta columns: ``hand_count`` and a landmark-quality score.
    """
    parts = [log_prob_features(p, eps) for p in branch_probs]
    if meta is not None:
        m = np.asarray(meta, dtype=np.float64)
        if m.ndim == 1:
            m = m[:, None]
        parts.append(m)
    return np.concatenate(parts, axis=1)


class StackedFusion:
    """Multinomial logistic regression (softmax regression) with L2, fit by L-BFGS.

    Objective: ``mean NLL + 0.5 * l2 * ||W||^2`` (bias unregularised) on features
    standardised with the training mean/std (stored, so inference is
    ``softmax(((x - mean) / std) @ W + b)``).
    """

    def __init__(self, l2: float = 1e-2, standardize: bool = True) -> None:
        self.l2 = l2
        self.standardize = standardize
        self.mean: Array = np.zeros(0)
        self.std: Array = np.ones(0)
        self.weights: Array = np.zeros((0, 0))
        self.bias: Array = np.zeros(0)

    @property
    def n_classes(self) -> int:
        return int(self.bias.shape[0])

    def _standardized(self, x: ArrayLike) -> Array:
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != self.weights.shape[0]:
            raise ValueError("feature matrix has the wrong shape")
        return (arr - self.mean) / self.std

    def _objective(self, theta: Array, xs: Array, onehot: Array) -> tuple[float, Array]:
        n, d = xs.shape
        k = onehot.shape[1]
        w = theta[: d * k].reshape(d, k)
        b = theta[d * k :]
        ls = log_softmax(xs @ w + b)
        loss = -float(np.sum(onehot * ls)) / n + 0.5 * self.l2 * float(np.sum(w * w))
        diff = (np.exp(ls) - onehot) / n
        grad_w = xs.T @ diff + self.l2 * w
        grad_b = diff.sum(axis=0)
        return loss, np.concatenate([grad_w.ravel(), grad_b])

    def fit(self, x: ArrayLike, y: ArrayLike, n_classes: int | None = None) -> StackedFusion:
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError("x must be 2-D (N, D)")
        n, d = arr.shape
        labels = np.asarray(y).astype(np.int64)
        k = int(n_classes if n_classes is not None else labels.max() + 1)
        labels = _as_labels(labels, n, k)
        if self.standardize:
            self.mean = arr.mean(axis=0)
            std = arr.std(axis=0)
            self.std = np.where(std < 1e-8, 1.0, std)
        else:
            self.mean = np.zeros(d)
            self.std = np.ones(d)
        xs = (arr - self.mean) / self.std
        onehot = np.zeros((n, k))
        onehot[np.arange(n), labels] = 1.0
        theta0 = np.zeros(d * k + k)
        res = minimize(
            self._objective,
            theta0,
            args=(xs, onehot),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 500},
        )
        self.weights = res.x[: d * k].reshape(d, k).copy()
        self.bias = res.x[d * k :].copy()
        return self

    def predict_logits(self, x: ArrayLike) -> Array:
        """Fused logits (feed to :class:`TemperatureScaler`)."""
        return self._standardized(x) @ self.weights + self.bias

    def predict_proba(self, x: ArrayLike) -> Array:
        return softmax(self.predict_logits(x))

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "stacked_fusion",
            "l2": self.l2,
            "standardize": self.standardize,
            "mean": [float(v) for v in self.mean],
            "std": [float(v) for v in self.std],
            "weights": [[float(v) for v in row] for row in self.weights],
            "bias": [float(v) for v in self.bias],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StackedFusion:
        if data.get("type") != "stacked_fusion":
            raise ValueError("not a stacked fusion payload")
        obj = cls(float(data["l2"]), bool(data["standardize"]))
        obj.mean = np.asarray(data["mean"], dtype=np.float64)
        obj.std = np.asarray(data["std"], dtype=np.float64)
        obj.weights = np.asarray(data["weights"], dtype=np.float64)
        obj.bias = np.asarray(data["bias"], dtype=np.float64)
        return obj
