"""Latest-frame scheduling for real-time NarutoCV recognition."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional, Protocol


class FrameRuntime(Protocol):
    def process_frame(
        self, frame: Any, *, captured_at_ms: Optional[int] = None
    ) -> Any: ...


class SchedulerState(str, Enum):
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"


class SchedulerStateError(RuntimeError):
    """Raised when a scheduler operation is invalid for its lifecycle."""


@dataclass(frozen=True)
class SchedulerStats:
    submitted_frames: int
    processed_frames: int
    dropped_frames: int
    failed_frames: int
    superseded_results: int
    pending_frame: bool
    processing_frame: bool


@dataclass(frozen=True)
class ScheduledResult:
    capture_sequence: int
    captured_at_ms: int
    frame: Any
    runtime_frame: Optional[Any]
    error: Optional[Exception]
    queue_wait_ms: float
    worker_elapsed_ms: float
    end_to_end_ms: float


@dataclass(frozen=True)
class _PendingFrame:
    capture_sequence: int
    captured_at_ms: int
    submitted_at: float
    frame: Any


class LatestFrameScheduler:
    """Process one frame while retaining at most the newest pending frame."""

    def __init__(
        self,
        runtime: FrameRuntime,
        *,
        clock: Callable[[], float] = time.monotonic,
        thread_name: str = "narutocv-latest-frame",
    ) -> None:
        self.runtime = runtime
        self.clock = clock
        self.thread_name = thread_name
        self._condition = threading.Condition()
        self._state = SchedulerState.CREATED
        self._thread: Optional[threading.Thread] = None
        self._pending: Optional[_PendingFrame] = None
        self._latest_result: Optional[ScheduledResult] = None
        self._processing = False
        self._stop_requested = False
        self._drain_on_stop = False
        self._next_capture_sequence = 0
        self._submitted_frames = 0
        self._processed_frames = 0
        self._dropped_frames = 0
        self._failed_frames = 0
        self._superseded_results = 0

    @property
    def state(self) -> SchedulerState:
        with self._condition:
            return self._state

    def start(self) -> LatestFrameScheduler:
        with self._condition:
            if self._state != SchedulerState.CREATED:
                raise SchedulerStateError("scheduler can only be started once")
            self._state = SchedulerState.RUNNING
            self._thread = threading.Thread(
                target=self._worker,
                name=self.thread_name,
                daemon=True,
            )
            self._thread.start()
        return self

    def submit(self, frame: Any, *, captured_at_ms: Optional[int] = None) -> int:
        submitted_at = self.clock()
        timestamp_ms = (
            int(submitted_at * 1000.0)
            if captured_at_ms is None
            else int(captured_at_ms)
        )
        if timestamp_ms < 0:
            raise ValueError("captured_at_ms cannot be negative")
        with self._condition:
            if self._state != SchedulerState.RUNNING or self._stop_requested:
                raise SchedulerStateError("scheduler is not accepting frames")
            capture_sequence = self._next_capture_sequence
            self._next_capture_sequence += 1
            self._submitted_frames += 1
            if self._pending is not None:
                self._dropped_frames += 1
            self._pending = _PendingFrame(
                capture_sequence,
                timestamp_ms,
                submitted_at,
                frame,
            )
            self._condition.notify_all()
            return capture_sequence

    def poll_result(self) -> Optional[ScheduledResult]:
        with self._condition:
            result = self._latest_result
            self._latest_result = None
            return result

    def clear_pending(self) -> bool:
        """Discard a queued frame before a reset or calibration boundary."""
        with self._condition:
            if self._pending is None:
                return False
            self._pending = None
            self._dropped_frames += 1
            self._condition.notify_all()
            return True

    def wait_for_result(self, timeout: Optional[float] = None) -> Optional[ScheduledResult]:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while self._latest_result is None:
                if self._state == SchedulerState.STOPPED:
                    return None
                remaining = (
                    None if deadline is None else max(0.0, deadline - time.monotonic())
                )
                if remaining == 0.0:
                    return None
                self._condition.wait(remaining)
            result = self._latest_result
            self._latest_result = None
            return result

    def wait_until_idle(self, timeout: Optional[float] = None) -> bool:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while self._pending is not None or self._processing:
                remaining = (
                    None if deadline is None else max(0.0, deadline - time.monotonic())
                )
                if remaining == 0.0:
                    return False
                self._condition.wait(remaining)
            return True

    def stats(self) -> SchedulerStats:
        with self._condition:
            return SchedulerStats(
                submitted_frames=self._submitted_frames,
                processed_frames=self._processed_frames,
                dropped_frames=self._dropped_frames,
                failed_frames=self._failed_frames,
                superseded_results=self._superseded_results,
                pending_frame=self._pending is not None,
                processing_frame=self._processing,
            )

    def stop(self, *, drain: bool = False, timeout: float = 5.0) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        with self._condition:
            if self._state == SchedulerState.STOPPED:
                return
            if self._state == SchedulerState.CREATED:
                self._state = SchedulerState.STOPPED
                return
            self._state = SchedulerState.STOPPING
            self._stop_requested = True
            self._drain_on_stop = drain
            if not drain and self._pending is not None:
                self._pending = None
                self._dropped_frames += 1
            thread = self._thread
            self._condition.notify_all()
        if thread is not None:
            thread.join(timeout)
            if thread.is_alive():
                raise TimeoutError("scheduler worker did not stop in time")

    def _publish(self, result: ScheduledResult) -> None:
        if self._latest_result is not None:
            self._superseded_results += 1
        self._latest_result = result
        self._condition.notify_all()

    def _worker(self) -> None:
        while True:
            with self._condition:
                while self._pending is None and not self._stop_requested:
                    self._condition.wait()
                if self._stop_requested and (
                    not self._drain_on_stop or self._pending is None
                ):
                    self._state = SchedulerState.STOPPED
                    self._condition.notify_all()
                    return
                pending = self._pending
                self._pending = None
                self._processing = True

            assert pending is not None
            started_at = self.clock()
            runtime_frame = None
            error = None
            try:
                runtime_frame = self.runtime.process_frame(
                    pending.frame,
                    captured_at_ms=pending.captured_at_ms,
                )
            except Exception as caught:
                error = caught
            completed_at = self.clock()
            result = ScheduledResult(
                capture_sequence=pending.capture_sequence,
                captured_at_ms=pending.captured_at_ms,
                frame=pending.frame,
                runtime_frame=runtime_frame,
                error=error,
                queue_wait_ms=max(0.0, (started_at - pending.submitted_at) * 1000.0),
                worker_elapsed_ms=max(0.0, (completed_at - started_at) * 1000.0),
                end_to_end_ms=max(0.0, (completed_at - pending.submitted_at) * 1000.0),
            )
            with self._condition:
                if error is None:
                    self._processed_frames += 1
                else:
                    self._failed_frames += 1
                self._processing = False
                self._publish(result)

    def __enter__(self) -> LatestFrameScheduler:
        return self.start()

    def __exit__(self, *_error: object) -> None:
        self.stop()
