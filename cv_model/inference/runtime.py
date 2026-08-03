"""Lifecycle controller around the locked NarutoCV recognition pipeline."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable, Optional, Protocol

from .events import (
    DispatchReport,
    EventCallback,
    PipelineEventDispatcher,
    PipelineEventType,
    combine_dispatch_reports,
)
from .output_schema import PipelineOutputSerializer, PipelineOutputV1


class RuntimeState(str, Enum):
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    CLOSED = "CLOSED"


class RuntimeStateError(RuntimeError):
    """Raised when a runtime command is invalid for the current lifecycle."""


class RecognitionPipeline(Protocol):
    hand: Any
    attacks: Any

    def process(self, frame: Any, timestamp: float) -> Any: ...

    def reset(self) -> None: ...

    def close(self) -> None: ...


PipelineFactory = Callable[[], RecognitionPipeline]


@dataclass(frozen=True)
class RuntimeConfig:
    include_geometry: bool = False
    capture_failure_reset_threshold: int = 3
    default_calibration_seconds: float = 3.0

    def __post_init__(self) -> None:
        if self.capture_failure_reset_threshold < 1:
            raise ValueError("capture_failure_reset_threshold must be positive")
        if self.default_calibration_seconds <= 0:
            raise ValueError("default_calibration_seconds must be positive")


@dataclass(frozen=True)
class CalibrationOutcome:
    adjustments: dict[str, tuple[float, float]] = field(default_factory=dict)
    dispatch: DispatchReport = field(default_factory=DispatchReport)


@dataclass(frozen=True)
class RuntimeFrame:
    result: Any
    output: PipelineOutputV1
    dispatch: DispatchReport
    calibration_active: bool
    calibration_remaining_ms: int
    calibration_adjustments: dict[str, tuple[float, float]] = field(
        default_factory=dict
    )


def _default_pipeline_factory() -> RecognitionPipeline:
    # Keep MediaPipe and ONNX imports lazy for schema/event-only consumers.
    from .combined_pipeline import CombinedNarutoPipeline

    return CombinedNarutoPipeline()


class PipelineRuntimeController:
    """Own pipeline state, frame identity, commands, and external event delivery."""

    def __init__(
        self,
        *,
        pipeline_factory: Optional[PipelineFactory] = None,
        config: RuntimeConfig = RuntimeConfig(),
        dispatcher: Optional[PipelineEventDispatcher] = None,
        serializer: Optional[PipelineOutputSerializer] = None,
        clock: Callable[[], float] = time.monotonic,
        session_id: Optional[str] = None,
    ) -> None:
        self.config = config
        self.dispatcher = dispatcher or PipelineEventDispatcher()
        self.serializer = serializer or PipelineOutputSerializer(
            include_geometry=config.include_geometry
        )
        self.clock = clock
        self.session_id = session_id or uuid.uuid4().hex
        if not self.session_id:
            raise ValueError("session_id cannot be empty")
        self._pipeline_factory = pipeline_factory or _default_pipeline_factory
        self._pipeline: Optional[RecognitionPipeline] = None
        self._state = RuntimeState.CREATED
        self._next_frame_id = 0
        self._last_timestamp_ms = -1
        self._capture_failures = 0
        self._calibration_deadline_ms: Optional[int] = None
        self._calibration_samples: list[Any] = []
        self._last_calibration_adjustments: dict[str, tuple[float, float]] = {}

    @property
    def state(self) -> RuntimeState:
        return self._state

    @property
    def pipeline(self) -> RecognitionPipeline:
        if self._pipeline is None:
            raise RuntimeStateError("runtime has not been started")
        return self._pipeline

    @property
    def calibration_active(self) -> bool:
        return self._calibration_deadline_ms is not None

    @property
    def calibration_sample_count(self) -> int:
        return len(self._calibration_samples)

    @property
    def last_calibration_adjustments(self) -> dict[str, tuple[float, float]]:
        return dict(self._last_calibration_adjustments)

    def start(self) -> PipelineRuntimeController:
        if self._state == RuntimeState.CLOSED:
            raise RuntimeStateError("a closed runtime cannot be restarted")
        if self._state == RuntimeState.CREATED:
            self._pipeline = self._pipeline_factory()
            self._state = RuntimeState.RUNNING
        return self

    def subscribe(
        self,
        callback: EventCallback,
        event_types: Optional[Iterable[PipelineEventType | str]] = None,
    ) -> int:
        return self.dispatcher.subscribe(callback, event_types)

    def unsubscribe(self, subscription_id: int) -> bool:
        return self.dispatcher.unsubscribe(subscription_id)

    def _require_running(self) -> None:
        if self._state != RuntimeState.RUNNING:
            raise RuntimeStateError("runtime must be running")

    def _timestamp_ms(self, supplied: Optional[int] = None) -> int:
        candidate = int(self.clock() * 1000.0) if supplied is None else int(supplied)
        if candidate < 0:
            raise ValueError("captured_at_ms cannot be negative")
        timestamp_ms = max(candidate, self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms
        return timestamp_ms

    def _reset_hand_calibration_state(self) -> None:
        self.pipeline.hand.filter.reset()
        self.pipeline.attacks.reset()

    def _complete_calibration(self, captured_at_ms: int) -> CalibrationOutcome:
        adjustments = self.pipeline.hand.calibrate_neutral(
            tuple(self._calibration_samples)
        )
        self.pipeline.reset()
        self._calibration_deadline_ms = None
        self._calibration_samples = []
        self._last_calibration_adjustments = dict(adjustments)
        dispatch = self.dispatcher.dispatch_reset(
            session_id=self.session_id,
            frame_id=self._next_frame_id,
            captured_at_ms=captured_at_ms,
            reason="calibration_complete",
        )
        return CalibrationOutcome(dict(adjustments), dispatch)

    def process_frame(
        self, frame: Any, *, captured_at_ms: Optional[int] = None
    ) -> RuntimeFrame:
        self._require_running()
        timestamp_ms = self._timestamp_ms(captured_at_ms)
        completed = CalibrationOutcome()
        if (
            self._calibration_deadline_ms is not None
            and timestamp_ms >= self._calibration_deadline_ms
        ):
            completed = self._complete_calibration(timestamp_ms)

        frame_id = self._next_frame_id
        self._next_frame_id += 1
        result = self.pipeline.process(frame, timestamp_ms / 1000.0)
        output = self.serializer.to_dict(
            result,
            session_id=self.session_id,
            frame_id=frame_id,
            captured_at_ms=timestamp_ms,
        )
        self._capture_failures = 0

        if self.calibration_active:
            self._calibration_samples.append(result.hand.raw)
            self._reset_hand_calibration_state()
            frame_dispatch = DispatchReport()
        else:
            frame_dispatch = self.dispatcher.dispatch_frame(output)
        dispatch = combine_dispatch_reports(completed.dispatch, frame_dispatch)
        remaining = (
            max(0, self._calibration_deadline_ms - timestamp_ms)
            if self._calibration_deadline_ms is not None
            else 0
        )
        return RuntimeFrame(
            result=result,
            output=output,
            dispatch=dispatch,
            calibration_active=self.calibration_active,
            calibration_remaining_ms=remaining,
            calibration_adjustments=completed.adjustments,
        )

    def reset(
        self,
        *,
        reason: str = "manual",
        captured_at_ms: Optional[int] = None,
    ) -> DispatchReport:
        self._require_running()
        if not reason:
            raise ValueError("reset reason cannot be empty")
        timestamp_ms = self._timestamp_ms(captured_at_ms)
        self.pipeline.reset()
        self._calibration_deadline_ms = None
        self._calibration_samples = []
        return self.dispatcher.dispatch_reset(
            session_id=self.session_id,
            frame_id=self._next_frame_id,
            captured_at_ms=timestamp_ms,
            reason=reason,
        )

    def begin_neutral_calibration(
        self,
        *,
        duration_seconds: Optional[float] = None,
        captured_at_ms: Optional[int] = None,
    ) -> DispatchReport:
        self._require_running()
        duration = (
            self.config.default_calibration_seconds
            if duration_seconds is None
            else duration_seconds
        )
        if duration <= 0:
            raise ValueError("calibration duration must be positive")
        timestamp_ms = self._timestamp_ms(captured_at_ms)
        self._calibration_samples = []
        self._last_calibration_adjustments = {}
        self._calibration_deadline_ms = timestamp_ms + int(round(duration * 1000.0))
        # Match the locked camera behavior: begin from a completely clean
        # hand, body, and attack state, then suppress hand/attack evidence
        # while neutral samples are collected.
        self.pipeline.reset()
        return self.dispatcher.dispatch_reset(
            session_id=self.session_id,
            frame_id=self._next_frame_id,
            captured_at_ms=timestamp_ms,
            reason="calibration_started",
        )

    def finish_neutral_calibration(
        self, *, captured_at_ms: Optional[int] = None
    ) -> CalibrationOutcome:
        self._require_running()
        if not self.calibration_active:
            raise RuntimeStateError("neutral calibration is not active")
        return self._complete_calibration(self._timestamp_ms(captured_at_ms))

    def cancel_neutral_calibration(
        self, *, captured_at_ms: Optional[int] = None
    ) -> DispatchReport:
        self._require_running()
        if not self.calibration_active:
            raise RuntimeStateError("neutral calibration is not active")
        timestamp_ms = self._timestamp_ms(captured_at_ms)
        self.pipeline.reset()
        self._calibration_deadline_ms = None
        self._calibration_samples = []
        return self.dispatcher.dispatch_reset(
            session_id=self.session_id,
            frame_id=self._next_frame_id,
            captured_at_ms=timestamp_ms,
            reason="calibration_cancelled",
        )

    def handle_capture_failure(
        self, *, captured_at_ms: Optional[int] = None
    ) -> DispatchReport:
        self._require_running()
        self._capture_failures += 1
        if self._capture_failures < self.config.capture_failure_reset_threshold:
            return DispatchReport()
        timestamp_ms = self._timestamp_ms(captured_at_ms)
        self._capture_failures = 0
        self.pipeline.reset()
        self._calibration_deadline_ms = None
        self._calibration_samples = []
        return self.dispatcher.dispatch_reset(
            session_id=self.session_id,
            frame_id=self._next_frame_id,
            captured_at_ms=timestamp_ms,
            reason="camera_recovery",
        )

    def close(self) -> None:
        if self._state == RuntimeState.CLOSED:
            return
        if self._pipeline is not None:
            self._pipeline.close()
        self._pipeline = None
        self._calibration_deadline_ms = None
        self._calibration_samples = []
        self._state = RuntimeState.CLOSED

    def __enter__(self) -> PipelineRuntimeController:
        return self.start()

    def __exit__(self, *_error: object) -> None:
        self.close()
