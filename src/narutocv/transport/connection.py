"""Per-client transport state: sequencing guards and bounded delivery queues.

Kept free of aiohttp/asyncio so the backpressure and ordering guarantees can
be unit tested as plain data structures (see test_transport_backpressure.py).
"""

from __future__ import annotations

import collections
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Deque, Optional


class TokenBucket:
    """Classic token bucket: `rate` tokens/second, holding at most `burst`.

    `try_consume` never blocks and never raises; a False return means "over the
    cap right now", which callers turn into a counted drop. The clock is
    injectable so tests do not need to sleep.
    """

    def __init__(
        self,
        rate: float,
        burst: int,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if rate <= 0 or burst < 1:
            raise ValueError("rate must be positive and burst at least 1")
        self.rate = float(rate)
        self.burst = float(burst)
        self._clock = clock
        self._tokens = float(burst)
        self._updated = clock()

    def try_consume(self, tokens: float = 1.0) -> bool:
        now = self._clock()
        elapsed = max(0.0, now - self._updated)
        self._updated = now
        self._tokens = min(self.burst, self._tokens + elapsed * self.rate)
        if self._tokens >= tokens:
            self._tokens -= tokens
            return True
        return False


class StaleFrameError(ValueError):
    """Raised when a frame sequence does not advance the client's stream."""


@dataclass
class FrameSequenceGuard:
    """Rejects frames that arrive out of order or duplicate an old sequence.

    The browser's framePublisher only ever holds one outgoing frame and
    replaces it before sending (game_implementation_plan.md §4 framePublisher
    contract), so sequences should already be monotonic by the time they
    arrive. A reordered or duplicate delivery at the transport layer is
    treated as stale input and dropped rather than fed to recognition.
    """

    last_sequence: Optional[int] = None

    def accept(self, sequence: int) -> None:
        if self.last_sequence is not None and sequence <= self.last_sequence:
            raise StaleFrameError(
                f"frame sequence {sequence} is not newer than {self.last_sequence}"
            )
        self.last_sequence = sequence

    def reset(self) -> None:
        self.last_sequence = None


class EventQueueOverflow(RuntimeError):
    """Raised when a client cannot keep up with ordered pipeline events.

    Per docs/archive/context.md step-6 backpressure rules, events are never dropped or
    reordered to keep delivery honest. A client that cannot consume the
    bounded buffer is disconnected instead (server.py), and this exception is
    the signal that triggers that disconnect.
    """


@dataclass
class BoundedEventQueue:
    """Ordered, bounded outbound event queue for one client connection."""

    max_size: int = 256
    _items: Deque[dict] = field(default_factory=collections.deque)

    def __post_init__(self) -> None:
        if self.max_size < 1:
            raise ValueError("max_size must be positive")

    def push(self, item: dict) -> None:
        if len(self._items) >= self.max_size:
            raise EventQueueOverflow(f"event queue exceeded {self.max_size} buffered events")
        self._items.append(item)

    def pop_all(self) -> list[dict]:
        drained = list(self._items)
        self._items.clear()
        return drained

    def __len__(self) -> int:
        return len(self._items)


@dataclass
class LatestSnapshotSlot:
    """Keeps only the newest state snapshot pending delivery to one client.

    This is what keeps state_snapshot traffic capped at a send-side rate
    (docs/archive/context.md: "Keep only the newest unsent state snapshot per client")
    regardless of how fast recognition produces frames.
    """

    _pending: Optional[dict] = None

    def replace(self, snapshot: dict) -> None:
        self._pending = snapshot

    def take(self) -> Optional[dict]:
        pending = self._pending
        self._pending = None
        return pending

    @property
    def has_pending(self) -> bool:
        return self._pending is not None


class ClientSession:
    """Bundles the per-connection guards a single browser client owns."""

    def __init__(
        self,
        *,
        event_queue_size: int = 256,
        connection_id: Optional[str] = None,
        frame_bucket: Optional[TokenBucket] = None,
        control_bucket: Optional[TokenBucket] = None,
        error_bucket: Optional[TokenBucket] = None,
    ) -> None:
        self.connection_id = connection_id or uuid.uuid4().hex[:12]
        self.frame_bucket = frame_bucket
        self.control_bucket = control_bucket
        self.error_bucket = error_bucket
        self.invalid_messages = 0
        self.frame_guard = FrameSequenceGuard()
        self.events = BoundedEventQueue(max_size=event_queue_size)
        self.snapshot = LatestSnapshotSlot()
        self.is_controller = False

    def reset_stream_state(self) -> None:
        """Clear per-stream guards on reset/reconnect.

        Only the frame sequence guard resets here: old evidence in the
        recognition pipeline itself is cleared via
        PipelineRuntimeController.reset(), not by this transport-only state.
        """
        self.frame_guard.reset()
