"""Decoder tuning: event matching, operating-point evaluation, grid search, Pareto front.

Event matching rule (spec: ``docs/specs/seal_decoder_v1.md`` section 6): a predicted
event matches a ground-truth interval if the labels are equal and
``gt.start_ms <= pred.t_emit_ms <= gt.end_ms + tolerance_ms``; matching is
one-to-one and greedy by time (predictions in ``t_emit`` order, each taking the
earliest-starting still-unmatched eligible interval).  Unmatched predictions are
false activations (a double emission yields one TP and one FP); unmatched
intervals are misses.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from statistics import median

import numpy as np
from narutocv.temporal.decoder import DecoderConfig, SealDecoder, SealEvent
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class GTEvent:
    """Ground-truth seal interval (milliseconds on the stream clock)."""

    label: str
    start_ms: float
    end_ms: float

    def __post_init__(self) -> None:
        if self.end_ms < self.start_ms:
            raise ValueError("GTEvent end_ms must be >= start_ms")


@dataclass(frozen=True)
class MatchResult:
    matches: tuple[tuple[int, int], ...]  # (pred index, gt index)
    false_positives: tuple[int, ...]  # unmatched pred indices
    misses: tuple[int, ...]  # unmatched gt indices


def match_events(
    predicted: Sequence[SealEvent],
    ground_truth: Sequence[GTEvent],
    tolerance_ms: float = 300.0,
) -> MatchResult:
    """One-to-one greedy-by-time matching (see module docstring)."""
    order = sorted(range(len(predicted)), key=lambda i: (predicted[i].t_emit_ms, i))
    taken: set[int] = set()
    matches: list[tuple[int, int]] = []
    fps: list[int] = []
    for pi in order:
        pred = predicted[pi]
        best: int | None = None
        for gi, gt in enumerate(ground_truth):
            if gi in taken or gt.label != pred.label:
                continue
            if gt.start_ms <= pred.t_emit_ms <= gt.end_ms + tolerance_ms and (
                best is None or (gt.start_ms, gi) < (ground_truth[best].start_ms, best)
            ):
                best = gi
        if best is None:
            fps.append(pi)
        else:
            taken.add(best)
            matches.append((pi, best))
    misses = tuple(gi for gi in range(len(ground_truth)) if gi not in taken)
    return MatchResult(tuple(sorted(matches)), tuple(sorted(fps)), misses)


@dataclass(frozen=True)
class LabeledStream:
    """A recording: per-frame calibrated probs, timestamps and ground-truth events.

    ``expected_next`` (optional) gives the grammar prior for each frame.
    """

    probs: NDArray[np.float64]
    t_ms: NDArray[np.float64]
    events: tuple[GTEvent, ...]
    expected_next: tuple[Sequence[str] | None, ...] | None = None

    def __post_init__(self) -> None:
        if self.probs.ndim != 2 or self.probs.shape[0] != self.t_ms.shape[0]:
            raise ValueError("probs must be (T, K) and t_ms (T,)")
        if self.expected_next is not None and len(self.expected_next) != self.t_ms.shape[0]:
            raise ValueError("expected_next must have one entry per frame")

    @property
    def duration_ms(self) -> float:
        if self.t_ms.shape[0] < 2:
            return 0.0
        return float(self.t_ms[-1] - self.t_ms[0])


def make_stream(
    probs: ArrayLike,
    t_ms: ArrayLike,
    events: Iterable[GTEvent],
    expected_next: Sequence[Sequence[str] | None] | None = None,
) -> LabeledStream:
    return LabeledStream(
        np.asarray(probs, dtype=np.float64),
        np.asarray(t_ms, dtype=np.float64),
        tuple(events),
        None if expected_next is None else tuple(expected_next),
    )


def decode_stream(stream: LabeledStream, config: DecoderConfig) -> list[SealEvent]:
    decoder = SealDecoder(config)
    out: list[SealEvent] = []
    for i in range(stream.t_ms.shape[0]):
        nxt = None if stream.expected_next is None else stream.expected_next[i]
        out.extend(decoder.update(stream.probs[i], float(stream.t_ms[i]), expected_next=nxt))
    return out


@dataclass(frozen=True)
class OperatingPoint:
    config: DecoderConfig
    fa_per_min: float
    recall: float
    median_ttd_ms: float  # inf when no event was detected
    n_events: int
    n_false: int
    n_detected: int


def evaluate_config(
    streams: Sequence[LabeledStream],
    config: DecoderConfig,
    tolerance_ms: float = 300.0,
) -> OperatingPoint:
    """FA/min = false activations / total stream minutes; recall over all GT events;
    time-to-detect = ``t_emit - gt.start`` of matched events (median, ``inf`` if none)."""
    total_gt = 0
    false_total = 0
    detected = 0
    ttds: list[float] = []
    minutes = 0.0
    for stream in streams:
        pred = decode_stream(stream, config)
        res = match_events(pred, stream.events, tolerance_ms)
        total_gt += len(stream.events)
        false_total += len(res.false_positives)
        detected += len(res.matches)
        minutes += stream.duration_ms / 60_000.0
        ttds.extend(pred[pi].t_emit_ms - stream.events[gi].start_ms for pi, gi in res.matches)
    if total_gt == 0:
        raise ValueError("no ground-truth events to evaluate against")
    return OperatingPoint(
        config=config,
        fa_per_min=false_total / minutes if minutes > 0.0 else (math.inf if false_total else 0.0),
        recall=detected / total_gt,
        median_ttd_ms=float(median(ttds)) if ttds else math.inf,
        n_events=total_gt,
        n_false=false_total,
        n_detected=detected,
    )


def grid_search(
    streams: Sequence[LabeledStream],
    base: DecoderConfig | None = None,
    *,
    tau_on: Sequence[float] = (0.5, 0.6, 0.7, 0.8),
    tau_off: Sequence[float] = (0.3, 0.4, 0.5),
    min_dwell_ms: Sequence[float] = (60.0, 120.0, 200.0),
    half_life_ms: Sequence[float] = (40.0, 80.0, 120.0),
    tolerance_ms: float = 300.0,
) -> list[OperatingPoint]:
    """Evaluate every valid combination (``tau_off < tau_on``); ``tau_reject`` is
    clamped to ``min(base.tau_reject, tau_off)``."""
    base = base or DecoderConfig()
    points: list[OperatingPoint] = []
    for on, off, dwell, hl in itertools.product(tau_on, tau_off, min_dwell_ms, half_life_ms):
        if not off < on:
            continue
        cfg = replace(
            base,
            tau_on=on,
            tau_off=off,
            tau_reject=min(base.tau_reject, off),
            min_dwell_ms=dwell,
            half_life_ms=hl,
        )
        points.append(evaluate_config(streams, cfg, tolerance_ms))
    return points


def _dominates(a: OperatingPoint, b: OperatingPoint) -> bool:
    """``a`` dominates ``b``: no worse on (FA/min down, recall up, TTD down), better on one."""
    no_worse = (
        a.fa_per_min <= b.fa_per_min and a.recall >= b.recall and a.median_ttd_ms <= b.median_ttd_ms
    )
    better = (
        a.fa_per_min < b.fa_per_min or a.recall > b.recall or a.median_ttd_ms < b.median_ttd_ms
    )
    return no_worse and better


def pareto_frontier(points: Sequence[OperatingPoint]) -> list[OperatingPoint]:
    """Non-dominated points, sorted by FA/min then recall (desc) then TTD.

    ``inf`` TTD is worse than any finite value; points with identical
    (FA/min, recall, TTD) are deduplicated, keeping the first."""
    front: list[OperatingPoint] = []
    seen: set[tuple[float, float, float]] = set()
    for i, p in enumerate(points):
        if any(_dominates(q, p) for j, q in enumerate(points) if j != i):
            continue
        key = (p.fa_per_min, p.recall, p.median_ttd_ms)
        if key in seen:
            continue
        seen.add(key)
        front.append(p)
    return sorted(front, key=lambda p: (p.fa_per_min, -p.recall, p.median_ttd_ms))


def select_operating_point(
    points: Sequence[OperatingPoint], max_fa_per_min: float = 0.5
) -> OperatingPoint | None:
    """Highest recall with ``fa_per_min <= max_fa_per_min`` (ties: lower TTD)."""
    ok = [p for p in points if p.fa_per_min <= max_fa_per_min]
    if not ok:
        return None
    return min(ok, key=lambda p: (-p.recall, p.median_ttd_ms, p.fa_per_min))
