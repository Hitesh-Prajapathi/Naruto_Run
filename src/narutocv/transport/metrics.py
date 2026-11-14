"""A tiny, dependency-free Prometheus text exposition (format 0.0.4).

Hand-rolled on purpose: the transport only needs counters, gauges and fixed
bucket histograms, and keeping it in-tree avoids a new runtime dependency.
All metrics are thread-safe because some observations originate on worker
threads.

Cardinality rule: label values must come from small, closed sets (drop
reasons, stage names from the pipeline's `timings_ms` keys). Never label by
connection id, client address, or anything a client controls.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Iterable, Sequence

CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"

# Milliseconds. Covers a ~1 ms stage up to a badly stalled 1 s frame.
LATENCY_BUCKETS_MS: tuple[float, ...] = (
    1.0,
    2.0,
    5.0,
    10.0,
    20.0,
    35.0,
    50.0,
    75.0,
    100.0,
    150.0,
    250.0,
    500.0,
    1000.0,
)


def _escape_label_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _format_value(value: float) -> str:
    if math.isinf(value):
        return "+Inf" if value > 0 else "-Inf"
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return repr(float(value))


def _label_block(names: Sequence[str], values: Sequence[str], extra: str = "") -> str:
    parts = [f'{n}="{_escape_label_value(v)}"' for n, v in zip(names, values, strict=True)]
    if extra:
        parts.append(extra)
    return "{" + ",".join(parts) + "}" if parts else ""


class _Metric:
    kind = "untyped"

    def __init__(self, name: str, help_text: str, labelnames: Sequence[str] = ()) -> None:
        self.name = name
        self.help = help_text
        self.labelnames = tuple(labelnames)
        self._lock = threading.Lock()

    def _key(self, labels: dict[str, str]) -> tuple[str, ...]:
        if set(labels) != set(self.labelnames):
            raise ValueError(f"{self.name} expects labels {self.labelnames}, got {tuple(labels)}")
        return tuple(str(labels[name]) for name in self.labelnames)

    def _header(self) -> list[str]:
        return [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} {self.kind}"]

    def render(self) -> list[str]:  # pragma: no cover - abstract
        raise NotImplementedError


class Counter(_Metric):
    kind = "counter"

    def __init__(self, name: str, help_text: str, labelnames: Sequence[str] = ()) -> None:
        super().__init__(name, help_text, labelnames)
        self._values: dict[tuple[str, ...], float] = {}

    def touch(self, **labels: str) -> None:
        """Create the series at 0 so it is exported before the first event."""
        with self._lock:
            self._values.setdefault(self._key(labels), 0.0)

    def inc(self, amount: float = 1.0, **labels: str) -> None:
        if amount < 0:
            raise ValueError("counters can only increase")
        key = self._key(labels)
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + amount

    def set_total(self, value: float, **labels: str) -> None:
        """Mirror a monotonic total owned elsewhere (e.g. scheduler stats)."""
        key = self._key(labels)
        with self._lock:
            self._values[key] = max(self._values.get(key, 0.0), float(value))

    def value(self, **labels: str) -> float:
        with self._lock:
            return self._values.get(self._key(labels), 0.0)

    def render(self) -> list[str]:
        lines = self._header()
        with self._lock:
            items = sorted(self._values.items())
        for key, value in items:
            lines.append(f"{self.name}{_label_block(self.labelnames, key)} {_format_value(value)}")
        return lines


class Gauge(_Metric):
    kind = "gauge"

    def __init__(self, name: str, help_text: str, labelnames: Sequence[str] = ()) -> None:
        super().__init__(name, help_text, labelnames)
        self._values: dict[tuple[str, ...], float] = {}

    def set(self, value: float, **labels: str) -> None:
        key = self._key(labels)
        with self._lock:
            self._values[key] = float(value)

    def inc(self, amount: float = 1.0, **labels: str) -> None:
        key = self._key(labels)
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + amount

    def dec(self, amount: float = 1.0, **labels: str) -> None:
        self.inc(-amount, **labels)

    def value(self, **labels: str) -> float:
        with self._lock:
            return self._values.get(self._key(labels), 0.0)

    def render(self) -> list[str]:
        lines = self._header()
        with self._lock:
            items = sorted(self._values.items())
        for key, value in items:
            lines.append(f"{self.name}{_label_block(self.labelnames, key)} {_format_value(value)}")
        return lines


class Histogram(_Metric):
    kind = "histogram"

    def __init__(
        self,
        name: str,
        help_text: str,
        labelnames: Sequence[str] = (),
        buckets: Iterable[float] = LATENCY_BUCKETS_MS,
    ) -> None:
        super().__init__(name, help_text, labelnames)
        self.buckets = tuple(sorted(float(b) for b in buckets))
        # key -> [per-bucket counts (non-cumulative, last slot is +Inf), sum]
        self._series: dict[tuple[str, ...], list] = {}

    def observe(self, value: float, **labels: str) -> None:
        if math.isnan(value):
            return
        key = self._key(labels)
        with self._lock:
            series = self._series.setdefault(key, [[0] * (len(self.buckets) + 1), 0.0])
            index = len(self.buckets)
            for i, bound in enumerate(self.buckets):
                if value <= bound:
                    index = i
                    break
            series[0][index] += 1
            series[1] += value

    def count(self, **labels: str) -> int:
        with self._lock:
            series = self._series.get(self._key(labels))
            return sum(series[0]) if series else 0

    def render(self) -> list[str]:
        lines = self._header()
        with self._lock:
            snapshot = {k: (list(v[0]), v[1]) for k, v in self._series.items()}
        inf_label = 'le="+Inf"'
        for key in sorted(snapshot):
            counts, total = snapshot[key]
            cumulative = 0
            for bound, count in zip(self.buckets, counts, strict=False):
                cumulative += count
                le = f'le="{_format_value(bound)}"'
                lines.append(
                    f"{self.name}_bucket{_label_block(self.labelnames, key, le)} {cumulative}"
                )
            cumulative += counts[-1]
            lines.append(
                f"{self.name}_bucket{_label_block(self.labelnames, key, inf_label)} {cumulative}"
            )
            block = _label_block(self.labelnames, key)
            lines.append(f"{self.name}_sum{block} {_format_value(total)}")
            lines.append(f"{self.name}_count{block} {cumulative}")
        return lines


class MetricsRegistry:
    """Owns every metric the transport exports."""

    def __init__(self) -> None:
        self._metrics: list[_Metric] = []

    def _register(self, metric):
        if any(m.name == metric.name for m in self._metrics):
            raise ValueError(f"metric {metric.name} already registered")
        self._metrics.append(metric)
        return metric

    def counter(self, name: str, help_text: str, labelnames: Sequence[str] = ()) -> Counter:
        return self._register(Counter(name, help_text, labelnames))

    def gauge(self, name: str, help_text: str, labelnames: Sequence[str] = ()) -> Gauge:
        return self._register(Gauge(name, help_text, labelnames))

    def histogram(
        self,
        name: str,
        help_text: str,
        labelnames: Sequence[str] = (),
        buckets: Iterable[float] = LATENCY_BUCKETS_MS,
    ) -> Histogram:
        return self._register(Histogram(name, help_text, labelnames, buckets))

    def render(self) -> str:
        lines: list[str] = []
        for metric in self._metrics:
            lines.extend(metric.render())
        return "\n".join(lines) + "\n"
