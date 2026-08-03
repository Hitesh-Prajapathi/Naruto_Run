"""Sparse, typed event dispatch for serialized NarutoCV pipeline outputs."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Iterable, Mapping, Optional, TypedDict

from .output_schema import SCHEMA_VERSION, PipelineOutputV1, validate_pipeline_output_v1


class PipelineEventType(str, Enum):
    HAND_SEAL = "HAND_SEAL"
    BODY_MOVEMENT = "BODY_MOVEMENT"
    ATTACK_TRIGGERED = "ATTACK_TRIGGERED"
    QUEUE_CLEARED = "QUEUE_CLEARED"
    PIPELINE_RESET = "PIPELINE_RESET"


class PipelineEventV1(TypedDict):
    schema_version: str
    event_id: str
    event_sequence: int
    event_type: str
    session_id: str
    frame_id: int
    captured_at_ms: int
    payload: dict[str, Any]


EventCallback = Callable[[PipelineEventV1], None]


@dataclass(frozen=True)
class DispatchFailure:
    subscription_id: int
    event_type: PipelineEventType
    error: Exception


@dataclass(frozen=True)
class DispatchReport:
    events: tuple[PipelineEventV1, ...] = ()
    failures: tuple[DispatchFailure, ...] = ()

    @property
    def succeeded(self) -> bool:
        return not self.failures


def combine_dispatch_reports(*reports: DispatchReport) -> DispatchReport:
    return DispatchReport(
        events=tuple(event for report in reports for event in report.events),
        failures=tuple(failure for report in reports for failure in report.failures),
    )


class PipelineEventDispatcher:
    """Derive sparse events and deliver them to isolated synchronous callbacks."""

    def __init__(self) -> None:
        self._subscriptions: dict[
            int, tuple[EventCallback, Optional[frozenset[PipelineEventType]]]
        ] = {}
        self._next_subscription_id = 1
        self._next_event_sequence = 0
        self._lock = threading.RLock()

    def subscribe(
        self,
        callback: EventCallback,
        event_types: Optional[Iterable[PipelineEventType | str]] = None,
    ) -> int:
        if not callable(callback):
            raise TypeError("callback must be callable")
        selected = (
            frozenset(PipelineEventType(value) for value in event_types)
            if event_types is not None
            else None
        )
        with self._lock:
            subscription_id = self._next_subscription_id
            self._next_subscription_id += 1
            self._subscriptions[subscription_id] = (callback, selected)
        return subscription_id

    def unsubscribe(self, subscription_id: int) -> bool:
        with self._lock:
            return self._subscriptions.pop(subscription_id, None) is not None

    def _new_event(
        self,
        event_type: PipelineEventType,
        *,
        session_id: str,
        frame_id: int,
        captured_at_ms: int,
        payload: Mapping[str, Any],
    ) -> PipelineEventV1:
        with self._lock:
            sequence = self._next_event_sequence
            self._next_event_sequence += 1
        return {
            "schema_version": SCHEMA_VERSION,
            "event_id": f"{session_id}:{sequence:08d}",
            "event_sequence": sequence,
            "event_type": event_type.value,
            "session_id": session_id,
            "frame_id": frame_id,
            "captured_at_ms": captured_at_ms,
            "payload": dict(payload),
        }

    def _deliver(self, events: Iterable[PipelineEventV1]) -> DispatchReport:
        event_tuple = tuple(events)
        failures: list[DispatchFailure] = []
        with self._lock:
            subscriptions = tuple(self._subscriptions.items())
        for event in event_tuple:
            event_type = PipelineEventType(event["event_type"])
            for subscription_id, (callback, selected) in subscriptions:
                if selected is not None and event_type not in selected:
                    continue
                try:
                    callback(event)
                except Exception as error:  # Keep recognition alive if a consumer fails.
                    failures.append(
                        DispatchFailure(subscription_id, event_type, error)
                    )
        return DispatchReport(event_tuple, tuple(failures))

    def dispatch_frame(self, output: PipelineOutputV1) -> DispatchReport:
        """Emit sparse events in HAND, BODY, ATTACK, QUEUE order."""
        validate_pipeline_output_v1(output)
        common = {
            "session_id": output["session_id"],
            "frame_id": output["frame_id"],
            "captured_at_ms": output["captured_at_ms"],
        }
        events: list[PipelineEventV1] = []

        accepted_seal = output["queue"]["accepted_seal"]
        if accepted_seal is not None:
            events.append(
                self._new_event(
                    PipelineEventType.HAND_SEAL,
                    payload={
                        "seal": accepted_seal,
                        "stable_label": output["hand"]["stable_label"],
                        "confidence": output["hand"]["raw"]["confidence"],
                    },
                    **common,
                )
            )

        movement = output["body"]["emitted_movement"]
        if movement is not None:
            events.append(
                self._new_event(
                    PipelineEventType.BODY_MOVEMENT,
                    payload={
                        "movement": movement,
                        "stable_label": output["body"]["stable_label"],
                        "metrics": dict(output["body"]["metrics"]),
                    },
                    **common,
                )
            )

        attack = output["attack"]
        if attack is not None:
            events.append(
                self._new_event(
                    PipelineEventType.ATTACK_TRIGGERED,
                    payload=dict(attack),
                    **common,
                )
            )

        clear_reasons = []
        if output["queue"]["timeout_cleared"]:
            clear_reasons.append("timeout")
        if output["queue"]["max_length_cleared"]:
            clear_reasons.append("max_length")
        if output["queue"]["cooldown_suppressed"]:
            clear_reasons.append("cooldown_suppressed")
        if attack is not None:
            clear_reasons.append("attack_triggered")
        if clear_reasons:
            events.append(
                self._new_event(
                    PipelineEventType.QUEUE_CLEARED,
                    payload={"reasons": clear_reasons},
                    **common,
                )
            )

        return self._deliver(events)

    def dispatch_reset(
        self,
        *,
        session_id: str,
        frame_id: int,
        captured_at_ms: int,
        reason: str,
    ) -> DispatchReport:
        if not reason:
            raise ValueError("reset reason cannot be empty")
        event = self._new_event(
            PipelineEventType.PIPELINE_RESET,
            session_id=session_id,
            frame_id=frame_id,
            captured_at_ms=captured_at_ms,
            payload={"reason": reason},
        )
        return self._deliver((event,))
