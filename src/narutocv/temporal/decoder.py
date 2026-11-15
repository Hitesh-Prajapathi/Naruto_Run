"""Online seal decoder: time-aware smoothing + open-set reject + hysteresis + dwell.

Spec (with pseudocode for the TypeScript port): ``docs/specs/seal_decoder_v1.md``.
Input is one calibrated probability vector per frame plus a timestamp in
milliseconds; output is at most one :class:`SealEvent` per seal entry.  All timing
is in milliseconds, never frames, so behaviour is frame-rate independent.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Literal, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

Array = NDArray[np.float64]

#: Version of ``docs/specs/seal_decoder_v1.md``.  1.0.1: configuration validation rejects
#: non-finite numbers (NaN / inf) in both languages.
DECODER_SPEC_VERSION = "1.0.1"

DEFAULT_LABELS: tuple[str, ...] = (
    "bird",
    "boar",
    "dog",
    "dragon",
    "hare",
    "horse",
    "monkey",
    "ox",
    "ram",
    "rat",
    "snake",
    "tiger",
    "zero",
)


@dataclass(frozen=True)
class SealEvent:
    """One recognised seal.

    ``t_start_ms``: first update of the continuous run with smoothed p >= tau_on.
    ``t_emit_ms``: timestamp of the update that emitted the event.
    ``confidence``: smoothed probability of ``label`` at emission.
    """

    label: str
    t_start_ms: float
    t_emit_ms: float
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_NUMERIC_FIELDS = (
    "half_life_ms",
    "tau_on",
    "tau_off",
    "tau_reject",
    "min_dwell_ms",
    "refractory_ms",
    "max_gap_ms",
    "prior_boost",
    "hmm_p_stay",
    "hmm_ref_ms",
)


@dataclass(frozen=True)
class DecoderConfig:
    labels: tuple[str, ...] = DEFAULT_LABELS
    idle_label: str | None = "zero"
    smoother: Literal["ema", "hmm"] = "ema"
    half_life_ms: float = 80.0
    tau_on: float = 0.6
    tau_off: float = 0.4
    tau_reject: float = 0.3
    min_dwell_ms: float = 120.0
    refractory_ms: float = 250.0
    max_gap_ms: float = 200.0
    prior_boost: float = 3.0
    use_prior: bool = True
    hmm_p_stay: float = 0.95
    hmm_ref_ms: float = 33.3

    def __post_init__(self) -> None:
        k = len(self.labels)
        if k < 2 or len(set(self.labels)) != k:
            raise ValueError("labels must be at least two unique names")
        if self.idle_label is not None and self.idle_label not in self.labels:
            raise ValueError("idle_label must be one of labels")
        if self.smoother not in ("ema", "hmm"):
            raise ValueError("smoother must be 'ema' or 'hmm'")
        for name in _NUMERIC_FIELDS:
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if not 0.0 < self.tau_reject <= self.tau_off < self.tau_on <= 1.0:
            raise ValueError("require 0 < tau_reject <= tau_off < tau_on <= 1")
        # Written as `not (x > 0)` so that a NaN can never slip through a comparison.
        if not (self.half_life_ms > 0.0 and self.hmm_ref_ms > 0.0):
            raise ValueError("half_life_ms and hmm_ref_ms must be positive")
        if not (self.min_dwell_ms >= 0.0 and self.refractory_ms >= 0.0):
            raise ValueError("min_dwell_ms and refractory_ms must be >= 0")
        if not self.max_gap_ms > 0.0:
            raise ValueError("max_gap_ms must be positive")
        if not self.prior_boost >= 1.0:
            raise ValueError("prior_boost must be >= 1")
        if not 1.0 / k < self.hmm_p_stay < 1.0:
            raise ValueError("hmm_p_stay must be in (1/K, 1)")

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["labels"] = list(self.labels)
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DecoderConfig:
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown DecoderConfig keys: {sorted(unknown)}")
        kwargs = dict(data)
        if "labels" in kwargs:
            kwargs["labels"] = tuple(kwargs["labels"])
        return cls(**kwargs)


class Smoother(Protocol):
    """Online filter over class probabilities."""

    def reset(self) -> None: ...

    def update(self, probs: Array, dt_ms: float) -> Array:
        """Fold in one observation ``dt_ms`` after the previous one (ignored on the first)."""
        ...


class EmaSmoother:
    """Time-aware exponential moving average: ``alpha = 1 - 2**(-dt / half_life)``."""

    def __init__(self, half_life_ms: float) -> None:
        self.half_life_ms = half_life_ms
        self.state: Array | None = None

    def reset(self) -> None:
        self.state = None

    def update(self, probs: Array, dt_ms: float) -> Array:
        if self.state is None:
            self.state = probs.copy()
        else:
            alpha = 1.0 - 2.0 ** (-dt_ms / self.half_life_ms)
            self.state = self.state + alpha * (probs - self.state)
        return self.state


class HMMSmoother:
    """Online forward filter with a sticky transition matrix.

    The per-reference-step transition is ``p_stay`` on the diagonal and
    ``(1 - p_stay) / (K - 1)`` elsewhere.  Because this matrix is
    ``lam * I + (1 - lam) / K * ones`` with ``lam = (K * p_stay - 1) / (K - 1)``, its
    ``n``-th power (``n = dt / ref_ms``, fractional allowed) acts on a belief ``b``
    as ``lam**n * b + (1 - lam**n) / K``.  That makes the filter frame-rate
    independent.  The calibrated posterior is used as the emission term, raised to
    the power ``dt / ref_ms`` (a tempered likelihood) so that the evidence gathered per
    unit time does not depend on how many frames arrive in that time.
    """

    def __init__(self, n_classes: int, p_stay: float = 0.95, ref_ms: float = 33.3) -> None:
        if not 1.0 / n_classes < p_stay < 1.0:
            raise ValueError("p_stay must be in (1/K, 1)")
        self.n_classes = n_classes
        self.p_stay = p_stay
        self.ref_ms = ref_ms
        self.lam = (n_classes * p_stay - 1.0) / (n_classes - 1.0)
        self.state: Array | None = None

    def reset(self) -> None:
        self.state = None

    def update(self, probs: Array, dt_ms: float) -> Array:
        if self.state is None:
            belief = probs.copy()
        else:
            decay = self.lam ** (dt_ms / self.ref_ms)
            predicted = decay * self.state + (1.0 - decay) / self.n_classes
            belief = predicted * probs ** (dt_ms / self.ref_ms)
        total = float(belief.sum())
        if total <= 1e-300:
            belief = np.full(self.n_classes, 1.0 / self.n_classes)
        else:
            belief = belief / total
        self.state = belief
        return belief


@dataclass
class DecoderStats:
    frames: int = 0
    nonmonotonic: int = 0
    gaps: int = 0
    events: int = 0


@dataclass
class _State:
    last_t: float | None = None
    active: int | None = None
    pending: int | None = None
    pending_start: float = 0.0
    last_emit_t: float | None = None
    smoothed: Array = field(default_factory=lambda: np.zeros(0))


class SealDecoder:
    """Streaming decoder; call :meth:`update` once per frame."""

    def __init__(self, config: DecoderConfig | None = None) -> None:
        self.config = config or DecoderConfig()
        self._index = {name: i for i, name in enumerate(self.config.labels)}
        self._idle = (
            self._index[self.config.idle_label] if self.config.idle_label is not None else None
        )
        self._smoother: Smoother
        if self.config.smoother == "hmm":
            self._smoother = HMMSmoother(
                len(self.config.labels), self.config.hmm_p_stay, self.config.hmm_ref_ms
            )
        else:
            self._smoother = EmaSmoother(self.config.half_life_ms)
        self.stats = DecoderStats()
        self._s = _State()

    # -- introspection -------------------------------------------------------------
    @property
    def smoothed(self) -> Array:
        return self._s.smoothed.copy()

    @property
    def active_label(self) -> str | None:
        return None if self._s.active is None else self.config.labels[self._s.active]

    def reset(self) -> None:
        """Forget all state (smoothing, hysteresis, refractory) and counters."""
        self._smoother.reset()
        self.stats = DecoderStats()
        self._s = _State()

    # -- core ----------------------------------------------------------------------
    def _prepare(self, probs: ArrayLike) -> Array:
        p = np.asarray(probs, dtype=np.float64)
        if p.shape != (len(self.config.labels),):
            raise ValueError(f"probs must have shape ({len(self.config.labels)},), got {p.shape}")
        if not bool(np.all(np.isfinite(p))):
            raise ValueError("probs must be finite")
        p = np.clip(p, 0.0, None)
        total = float(p.sum())
        if total <= 0.0:
            return np.full(p.shape, 1.0 / p.size)
        return p / total

    def _apply_prior(self, p: Array, expected_next: Sequence[str]) -> Array:
        weights = np.ones_like(p)
        for name in expected_next:
            if name not in self._index:
                raise ValueError(f"unknown label in expected_next: {name!r}")
            idx = self._index[name]
            if idx != self._idle:
                weights[idx] = self.config.prior_boost
        boosted = p * weights
        total = float(boosted.sum())
        return boosted / total if total > 0.0 else p

    def update(
        self,
        probs: ArrayLike,
        t_ms: float,
        *,
        expected_next: Sequence[str] | None = None,
    ) -> list[SealEvent]:
        cfg = self.config
        st = self._s
        p = self._prepare(probs)
        if not math.isfinite(t_ms):
            raise ValueError("t_ms must be finite")
        if st.last_t is not None and t_ms <= st.last_t:
            self.stats.nonmonotonic += 1
            return []
        if cfg.use_prior and expected_next:
            p = self._apply_prior(p, expected_next)

        if st.last_t is None:
            dt = 0.0
        else:
            dt = t_ms - st.last_t
            if dt > cfg.max_gap_ms:
                # Continuity broken: restart smoothing, drop any dwell in progress.
                self.stats.gaps += 1
                self._smoother.reset()
                st.pending = None
        st.last_t = t_ms
        self.stats.frames += 1
        s = self._smoother.update(p, dt)
        st.smoothed = s

        top = int(np.argmax(s))
        # An active seal ends only when its own smoothed probability decays below tau_off
        # (pure hysteresis: an idle-dominated frame above tau_off does not end it).
        if st.active is not None and float(s[st.active]) < cfg.tau_off:
            st.active = None
        # Open-set reject: nothing confident, or idle is the arg-max -> no candidate.
        if float(s[top]) < cfg.tau_reject or top == self._idle:
            st.pending = None
            return []

        # Candidate: the top class, if non-idle and above tau_on.
        candidate = top if float(s[top]) >= cfg.tau_on else None
        if candidate is None:
            st.pending = None
            return []
        if st.active == candidate:
            return []
        if st.active is not None:
            st.active = None  # direct switch to a different, confident seal
        if st.pending != candidate:
            st.pending = candidate
            st.pending_start = t_ms
        if t_ms - st.pending_start < cfg.min_dwell_ms:
            return []
        if st.last_emit_t is not None and t_ms - st.last_emit_t < cfg.refractory_ms:
            return []  # deferred: stays pending, emits after the refractory period
        event = SealEvent(
            label=cfg.labels[candidate],
            t_start_ms=st.pending_start,
            t_emit_ms=t_ms,
            confidence=float(s[candidate]),
        )
        st.active = candidate
        st.pending = None
        st.last_emit_t = t_ms
        self.stats.events += 1
        return [event]
