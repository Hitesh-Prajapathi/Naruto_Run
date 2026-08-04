from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from cv_model.inference.combined_pipeline import (
    AttackEvent,
    BodyResult,
    Classification,
    FrameResult,
    HandResult,
    QueueUpdate,
)
from cv_model.inference.events import PipelineEventDispatcher, PipelineEventType
from cv_model.inference.output_schema import PipelineOutputSerializer
from cv_model.inference.runtime import (
    PipelineRuntimeController,
    RuntimeConfig,
    RuntimeState,
    RuntimeStateError,
)


EVENT_SCHEMA_PATH = (
    Path(__file__).resolve().parents[1]
    / "schemas"
    / "pipeline_event_v1.schema.json"
)


def _result(*, include_body: bool = True, include_attack: bool = True) -> FrameResult:
    attack = (
        AttackEvent("ikazuchi", "IKAZUCHI / LIGHTNING", 2.0)
        if include_attack
        else None
    )
    hand = HandResult(
        raw=Classification("dog", 0.90, "zero", 0.05),
        accepted_label="dog",
        rejection_reason=None,
        stable_label="dog",
        emitted_seal="dog",
        classification_bbox=(0, 0, 2, 2),
        hand_bbox=None,
    )
    body = BodyResult(
        raw_label="jumping" if include_body else "idle",
        stable_label="jumping" if include_body else "idle",
        emitted_movement="jumping" if include_body else None,
        metrics={"jump_height": 0.06},
    )
    queue = QueueUpdate(
        attack=attack,
        accepted_seal="dog",
        duplicate_ignored=False,
        timeout_cleared=False,
        max_length_cleared=False,
        cooldown_suppressed=False,
        queue=(),
    )
    return FrameResult(
        hand=hand,
        body=body,
        attack=attack,
        seal_history=(),
        queue_update=queue,
        processing_ms=10.0,
    )


def _output(*, include_body: bool = True, include_attack: bool = True) -> dict:
    return PipelineOutputSerializer().to_dict(
        _result(include_body=include_body, include_attack=include_attack),
        session_id="events-test",
        frame_id=3,
        captured_at_ms=2000,
    )


class PipelineEventDispatcherTests(unittest.TestCase):
    def test_frame_events_have_deterministic_order_and_identity(self) -> None:
        dispatcher = PipelineEventDispatcher()
        received = []
        attacks = []
        dispatcher.subscribe(received.append)
        dispatcher.subscribe(attacks.append, [PipelineEventType.ATTACK_TRIGGERED])

        report = dispatcher.dispatch_frame(_output())

        self.assertTrue(report.succeeded)
        self.assertEqual(
            [event["event_type"] for event in report.events],
            [
                "HAND_SEAL",
                "BODY_MOVEMENT",
                "ATTACK_TRIGGERED",
                "QUEUE_CLEARED",
            ],
        )
        self.assertEqual(
            [event["event_sequence"] for event in report.events], [0, 1, 2, 3]
        )
        self.assertEqual(report.events[0]["event_id"], "events-test:00000000")
        self.assertEqual(received, list(report.events))
        self.assertEqual([event["event_type"] for event in attacks], ["ATTACK_TRIGGERED"])
        self.assertEqual(
            report.events[-1]["payload"]["reasons"], ["attack_triggered"]
        )

    def test_failing_subscriber_does_not_block_other_consumers(self) -> None:
        dispatcher = PipelineEventDispatcher()
        delivered = []

        def fail(_event: dict) -> None:
            raise RuntimeError("consumer unavailable")

        dispatcher.subscribe(fail)
        dispatcher.subscribe(delivered.append)
        report = dispatcher.dispatch_frame(
            _output(include_body=False, include_attack=False)
        )

        self.assertEqual(len(report.events), 1)
        self.assertEqual(len(report.failures), 1)
        self.assertEqual(delivered, list(report.events))
        self.assertEqual(report.failures[0].event_type, PipelineEventType.HAND_SEAL)

    def test_reset_event_and_unsubscribe(self) -> None:
        dispatcher = PipelineEventDispatcher()
        received = []
        subscription_id = dispatcher.subscribe(received.append)

        report = dispatcher.dispatch_reset(
            session_id="reset-test",
            frame_id=4,
            captured_at_ms=2500,
            reason="manual",
        )

        self.assertEqual(report.events[0]["event_type"], "PIPELINE_RESET")
        self.assertEqual(report.events[0]["payload"], {"reason": "manual"})
        self.assertTrue(dispatcher.unsubscribe(subscription_id))
        self.assertFalse(dispatcher.unsubscribe(subscription_id))

    def test_machine_readable_event_schema_lists_every_event_type(self) -> None:
        schema = json.loads(EVENT_SCHEMA_PATH.read_text(encoding="utf-8"))

        self.assertEqual(schema["properties"]["schema_version"]["const"], "1.0.0")
        self.assertEqual(
            set(schema["properties"]["event_type"]["enum"]),
            {event_type.value for event_type in PipelineEventType},
        )
        self.assertFalse(schema["additionalProperties"])

    def test_queue_clear_event_reports_every_state_reason(self) -> None:
        dispatcher = PipelineEventDispatcher()
        output = _output(include_body=False, include_attack=False)
        output["queue"]["accepted_seal"] = None
        output["queue"]["timeout_cleared"] = True
        output["queue"]["max_length_cleared"] = True
        output["queue"]["cooldown_suppressed"] = True

        report = dispatcher.dispatch_frame(output)

        self.assertEqual([event["event_type"] for event in report.events], ["QUEUE_CLEARED"])
        self.assertEqual(
            report.events[0]["payload"]["reasons"],
            ["timeout", "max_length", "cooldown_suppressed"],
        )


class _ResetCounter:
    def __init__(self) -> None:
        self.reset_count = 0

    def reset(self) -> None:
        self.reset_count += 1


class _FakeHandControl:
    def __init__(self) -> None:
        self.filter = _ResetCounter()
        self.calibrated_sample_counts: list[int] = []

    def calibrate_neutral(
        self, samples: tuple[Classification, ...]
    ) -> dict[str, tuple[float, float]]:
        self.calibrated_sample_counts.append(len(samples))
        return {"ox": (0.80, 0.25)} if samples else {}


class _FakePipeline:
    def __init__(self) -> None:
        self.hand = _FakeHandControl()
        self.attacks = _ResetCounter()
        self.process_timestamps: list[float] = []
        self.reset_count = 0
        self.close_count = 0

    def process(self, _frame: np.ndarray, timestamp: float) -> FrameResult:
        self.process_timestamps.append(timestamp)
        return _result()

    def reset(self) -> None:
        self.reset_count += 1

    def close(self) -> None:
        self.close_count += 1


class PipelineRuntimeControllerTests(unittest.TestCase):
    def make_runtime(
        self, *, failure_threshold: int = 3
    ) -> tuple[PipelineRuntimeController, _FakePipeline]:
        pipeline = _FakePipeline()
        runtime = PipelineRuntimeController(
            pipeline_factory=lambda: pipeline,
            config=RuntimeConfig(
                capture_failure_reset_threshold=failure_threshold,
                default_calibration_seconds=3.0,
            ),
            session_id="runtime-test",
            clock=lambda: 1.0,
        )
        return runtime, pipeline

    def test_lifecycle_frame_ids_and_monotonic_timestamps(self) -> None:
        runtime, pipeline = self.make_runtime()
        with self.assertRaises(RuntimeStateError):
            runtime.process_frame(np.zeros((2, 2, 3), dtype=np.uint8))

        runtime.start()
        first = runtime.process_frame(
            np.zeros((2, 2, 3), dtype=np.uint8), captured_at_ms=100
        )
        second = runtime.process_frame(
            np.zeros((2, 2, 3), dtype=np.uint8), captured_at_ms=100
        )

        self.assertEqual(first.output["frame_id"], 0)
        self.assertEqual(second.output["frame_id"], 1)
        self.assertEqual(first.output["captured_at_ms"], 100)
        self.assertEqual(second.output["captured_at_ms"], 101)
        self.assertEqual(pipeline.process_timestamps, [0.1, 0.101])
        self.assertEqual(len(first.dispatch.events), 4)
        self.assertIn("serialization_ms", first.timings_ms)
        self.assertIn("callback_delivery_ms", first.timings_ms)
        self.assertIn("runtime_total_ms", first.timings_ms)
        self.assertGreaterEqual(first.dispatch.callback_delivery_ms, 0.0)
        runtime.close()
        runtime.close()
        self.assertEqual(runtime.state, RuntimeState.CLOSED)
        self.assertEqual(pipeline.close_count, 1)
        with self.assertRaises(RuntimeStateError):
            runtime.start()

    def test_model_initialization_failure_is_clear_and_retryable(self) -> None:
        pipeline = _FakePipeline()
        attempts = iter((RuntimeError("model missing"), pipeline))

        def factory() -> _FakePipeline:
            result = next(attempts)
            if isinstance(result, Exception):
                raise result
            return result

        runtime = PipelineRuntimeController(
            pipeline_factory=factory,
            session_id="startup-recovery-test",
        )

        with self.assertRaisesRegex(
            RuntimeError, "failed to initialize the recognition pipeline"
        ):
            runtime.start()
        self.assertEqual(runtime.state, RuntimeState.CREATED)

        runtime.start()
        self.assertEqual(runtime.state, RuntimeState.RUNNING)
        runtime.close()

    def test_close_failure_still_leaves_terminal_clean_state(self) -> None:
        pipeline = _FakePipeline()

        def fail_close() -> None:
            pipeline.close_count += 1
            raise RuntimeError("native close failed")

        pipeline.close = fail_close  # type: ignore[method-assign]
        runtime = PipelineRuntimeController(
            pipeline_factory=lambda: pipeline,
            session_id="close-failure-test",
        ).start()

        with self.assertRaisesRegex(RuntimeError, "native close failed"):
            runtime.close()
        self.assertEqual(runtime.state, RuntimeState.CLOSED)
        with self.assertRaises(RuntimeStateError):
            _ = runtime.pipeline
        runtime.close()
        self.assertEqual(pipeline.close_count, 1)

    def test_manual_reset_is_dispatched(self) -> None:
        runtime, pipeline = self.make_runtime()
        received = []
        runtime.subscribe(received.append, [PipelineEventType.PIPELINE_RESET])
        runtime.start()

        report = runtime.reset(reason="manual", captured_at_ms=50)

        self.assertEqual(pipeline.reset_count, 1)
        self.assertEqual(report.events[0]["payload"], {"reason": "manual"})
        self.assertEqual(received, list(report.events))

    def test_hand_temporal_reset_preserves_body_pipeline_state(self) -> None:
        runtime, pipeline = self.make_runtime()
        runtime.start()

        report = runtime.reset_hand_temporal_state(
            reason="evaluation_action_boundary",
            captured_at_ms=75,
        )

        self.assertEqual(pipeline.hand.filter.reset_count, 1)
        self.assertEqual(pipeline.attacks.reset_count, 1)
        self.assertEqual(pipeline.reset_count, 0)
        self.assertEqual(
            report.events[0]["payload"],
            {"reason": "evaluation_action_boundary"},
        )

    def test_capture_failure_threshold_resets_recognition_state(self) -> None:
        runtime, pipeline = self.make_runtime(failure_threshold=2)
        runtime.start()

        first = runtime.handle_capture_failure(captured_at_ms=10)
        second = runtime.handle_capture_failure(captured_at_ms=11)

        self.assertEqual(first.events, ())
        self.assertEqual(pipeline.reset_count, 1)
        self.assertEqual(second.events[0]["payload"]["reason"], "camera_recovery")

    def test_calibration_suppresses_frame_events_and_finishes_automatically(self) -> None:
        runtime, pipeline = self.make_runtime()
        runtime.start()
        runtime.begin_neutral_calibration(
            duration_seconds=0.010,
            captured_at_ms=100,
        )

        calibrating = runtime.process_frame(
            np.zeros((2, 2, 3), dtype=np.uint8), captured_at_ms=105
        )
        completed = runtime.process_frame(
            np.zeros((2, 2, 3), dtype=np.uint8), captured_at_ms=110
        )

        self.assertTrue(calibrating.calibration_active)
        self.assertEqual(calibrating.dispatch.events, ())
        self.assertEqual(runtime.calibration_sample_count, 0)
        self.assertFalse(completed.calibration_active)
        self.assertEqual(completed.calibration_adjustments, {"ox": (0.80, 0.25)})
        self.assertEqual(pipeline.hand.calibrated_sample_counts, [1])
        self.assertEqual(pipeline.reset_count, 2)
        self.assertEqual(
            completed.dispatch.events[0]["payload"]["reason"],
            "calibration_complete",
        )
        self.assertEqual(len(completed.dispatch.events), 5)

    def test_calibration_can_be_finished_or_cancelled_explicitly(self) -> None:
        runtime, pipeline = self.make_runtime()
        runtime.start()
        runtime.begin_neutral_calibration(duration_seconds=1.0, captured_at_ms=100)
        runtime.process_frame(
            np.zeros((2, 2, 3), dtype=np.uint8), captured_at_ms=110
        )

        outcome = runtime.finish_neutral_calibration(captured_at_ms=120)

        self.assertEqual(outcome.adjustments, {"ox": (0.80, 0.25)})
        self.assertEqual(outcome.dispatch.events[0]["payload"]["reason"], "calibration_complete")
        runtime.begin_neutral_calibration(duration_seconds=1.0, captured_at_ms=130)
        cancelled = runtime.cancel_neutral_calibration(captured_at_ms=140)
        self.assertEqual(cancelled.events[0]["payload"]["reason"], "calibration_cancelled")
        self.assertFalse(runtime.calibration_active)
        self.assertEqual(pipeline.reset_count, 4)


if __name__ == "__main__":
    unittest.main()
