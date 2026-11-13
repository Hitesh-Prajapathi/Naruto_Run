from __future__ import annotations

import threading
import unittest
from types import SimpleNamespace

import numpy as np

from cv_model.inference.combined_pipeline import HandSignRecognizer
from cv_model.inference.scheduler import (
    LatestFrameScheduler,
    SchedulerState,
    SchedulerStateError,
)


class _BlockingRuntime:
    def __init__(self) -> None:
        self.first_started = threading.Event()
        self.release_first = threading.Event()
        self.processed: list[int] = []

    def process_frame(
        self, frame: int, *, captured_at_ms: int | None = None
    ) -> SimpleNamespace:
        if frame == 0:
            self.first_started.set()
            if not self.release_first.wait(2.0):
                raise TimeoutError("test did not release first frame")
        self.processed.append(frame)
        return SimpleNamespace(frame=frame, captured_at_ms=captured_at_ms)


class _RecoveringRuntime:
    def __init__(self) -> None:
        self.calls = 0

    def process_frame(
        self, frame: str, *, captured_at_ms: int | None = None
    ) -> SimpleNamespace:
        self.calls += 1
        if frame == "bad":
            raise ValueError("invalid frame")
        return SimpleNamespace(frame=frame, captured_at_ms=captured_at_ms)


class LatestFrameSchedulerTests(unittest.TestCase):
    def test_only_newest_pending_frame_is_processed(self) -> None:
        runtime = _BlockingRuntime()
        scheduler = LatestFrameScheduler(runtime).start()
        try:
            scheduler.submit(0, captured_at_ms=100)
            self.assertTrue(runtime.first_started.wait(1.0))
            scheduler.submit(1, captured_at_ms=110)
            scheduler.submit(2, captured_at_ms=120)
            runtime.release_first.set()

            self.assertTrue(scheduler.wait_until_idle(2.0))
            result = scheduler.poll_result()
            stats = scheduler.stats()

            self.assertEqual(runtime.processed, [0, 2])
            self.assertIsNotNone(result)
            self.assertEqual(result.capture_sequence, 2)
            self.assertEqual(result.runtime_frame.frame, 2)
            self.assertEqual(stats.submitted_frames, 3)
            self.assertEqual(stats.processed_frames, 2)
            self.assertEqual(stats.dropped_frames, 1)
            self.assertEqual(stats.superseded_results, 1)
            self.assertGreaterEqual(result.queue_wait_ms, 0.0)
            self.assertGreaterEqual(result.end_to_end_ms, result.worker_elapsed_ms)
        finally:
            scheduler.stop()
        self.assertEqual(scheduler.state, SchedulerState.STOPPED)

    def test_pending_frame_can_be_cleared_at_reset_boundary(self) -> None:
        runtime = _BlockingRuntime()
        scheduler = LatestFrameScheduler(runtime).start()
        try:
            scheduler.submit(0)
            self.assertTrue(runtime.first_started.wait(1.0))
            scheduler.submit(1)
            self.assertTrue(scheduler.clear_pending())
            self.assertFalse(scheduler.clear_pending())
            runtime.release_first.set()
            self.assertTrue(scheduler.wait_until_idle(2.0))

            self.assertEqual(runtime.processed, [0])
            self.assertEqual(scheduler.stats().dropped_frames, 1)
        finally:
            scheduler.stop()

    def test_worker_reports_failure_and_accepts_later_frames(self) -> None:
        runtime = _RecoveringRuntime()
        scheduler = LatestFrameScheduler(runtime).start()
        try:
            scheduler.submit("bad")
            failed = scheduler.wait_for_result(1.0)
            scheduler.submit("good")
            recovered = scheduler.wait_for_result(1.0)

            self.assertIsInstance(failed.error, ValueError)
            self.assertIsNone(failed.runtime_frame)
            self.assertIsNone(recovered.error)
            self.assertEqual(recovered.runtime_frame.frame, "good")
            self.assertEqual(scheduler.stats().failed_frames, 1)
            self.assertEqual(scheduler.stats().processed_frames, 1)
        finally:
            scheduler.stop()

    def test_scheduler_cannot_restart_or_accept_after_stop(self) -> None:
        scheduler = LatestFrameScheduler(_RecoveringRuntime()).start()
        with self.assertRaises(SchedulerStateError):
            scheduler.start()
        scheduler.stop()
        with self.assertRaises(SchedulerStateError):
            scheduler.submit("late")


class _FakeSession:
    def run(self, _outputs: list[str], _inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        return [np.asarray([[0.05, 0.90, 0.05]], dtype=np.float32)]


class PerformanceInstrumentationTests(unittest.TestCase):
    def test_crop_timing_separates_preprocess_onnx_and_postprocess(self) -> None:
        recognizer = object.__new__(HandSignRecognizer)
        recognizer.session = _FakeSession()
        recognizer.input_name = "images"
        recognizer.output_name = "output0"
        recognizer.classes = ("dog", "ox", "zero")
        image = np.zeros((80, 120, 3), dtype=np.uint8)

        prediction, timings = recognizer._classify_crop_timed(image)

        self.assertEqual(prediction.label, "ox")
        self.assertEqual(
            set(timings),
            {"preprocess_ms", "onnx_ms", "postprocess_ms"},
        )
        self.assertTrue(all(value >= 0.0 for value in timings.values()))


if __name__ == "__main__":
    unittest.main()
