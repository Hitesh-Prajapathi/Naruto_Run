"""A reset must genuinely forget the person the model was tracking.

MediaPipe's pose landmarker tracks one player and is sticky about it, so the
only way a second person is ever seen is if the landmarker itself is rebuilt.
`CombinedNarutoPipeline.reset()` clears the filters and history but leaves the
landmarker -- and therefore the tracking -- untouched, which is why detection
worked for the first person in a session and then reported "not detected" for
the next one.

These tests exercise the swap in isolation. Building a real
`SessionScopedPipeline` loads MediaPipe and an ONNX session, which is far too
slow for a unit test, so the instance is constructed field-by-field. The logic
under test is entirely about which object ends up in `self.body` and what
happens to the old one.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

from cv_model.Nishit_Frontend.transport import pipeline as pipeline_module
from cv_model.Nishit_Frontend.transport.pipeline import SessionScopedPipeline


class FakeRecognizer:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def make_pipeline() -> SessionScopedPipeline:
    """A SessionScopedPipeline with no models behind it."""
    instance = object.__new__(SessionScopedPipeline)
    instance.body = FakeRecognizer()
    instance._pose_model_path = Path("pose.task")
    instance.config = mock.Mock()
    return instance


class SessionScopedPipelineTests(unittest.TestCase):
    def test_rebuild_replaces_the_recognizer(self) -> None:
        instance = make_pipeline()
        original = instance.body
        replacement = FakeRecognizer()

        with mock.patch.object(
            pipeline_module, "BodyMovementRecognizer", return_value=replacement
        ):
            instance._rebuild_body()

        self.assertIs(instance.body, replacement)
        self.assertIsNot(instance.body, original)

    def test_rebuild_closes_the_old_recognizer(self) -> None:
        # Landmarkers hold native resources; leaking one per reset would grow
        # the process for the whole session.
        instance = make_pipeline()
        original = instance.body

        with mock.patch.object(
            pipeline_module, "BodyMovementRecognizer", return_value=FakeRecognizer()
        ):
            instance._rebuild_body()

        self.assertTrue(original.closed)

    def test_a_failed_rebuild_keeps_the_working_recognizer(self) -> None:
        # Still tracking the previous person is bad; no recognition at all is
        # worse, and would take the whole session down.
        instance = make_pipeline()
        original = instance.body

        with mock.patch.object(
            pipeline_module, "BodyMovementRecognizer", side_effect=RuntimeError("no model")
        ):
            with self.assertLogs(pipeline_module.logger, level="ERROR"):
                instance._rebuild_body()

        self.assertIs(instance.body, original)
        self.assertFalse(original.closed)

    def test_reset_rebuilds_as_well_as_clearing_state(self) -> None:
        # The regression itself: reset used to clear the filters and leave the
        # landmarker, so "start again" did not restart the part that was stuck.
        instance = make_pipeline()
        replacement = FakeRecognizer()

        with mock.patch.object(
            pipeline_module.CombinedNarutoPipeline, "reset"
        ) as base_reset, mock.patch.object(
            pipeline_module, "BodyMovementRecognizer", return_value=replacement
        ):
            instance.reset()

        base_reset.assert_called_once()
        self.assertIs(instance.body, replacement)


class BodyOnlyModeTests(unittest.TestCase):
    """Skipping hand inference must not change the shape of the output.

    The serialiser validates snapshots strictly, so a body-only frame still
    has to carry a complete, valid hand section -- just an empty one.
    """

    def make(self) -> SessionScopedPipeline:
        instance = object.__new__(SessionScopedPipeline)
        instance.body = mock.Mock()
        instance.hand = mock.Mock()
        instance.attacks = mock.Mock()
        instance._pose_model_path = Path("pose.task")
        instance.config = mock.Mock()
        instance._body_only = True
        instance._idle_hand = mock.sentinel.idle_hand
        instance.body.process.return_value = mock.Mock(timings_ms={"body_total_ms": 9.0})
        instance.attacks.update_detailed.return_value = mock.Mock(
            attack=None, queue=()
        )
        return instance

    def test_hand_recognition_is_not_run(self) -> None:
        instance = self.make()
        instance.process(mock.Mock(), 1.0)
        instance.hand.process.assert_not_called()
        instance.body.process.assert_called_once()

    def test_output_still_carries_a_hand_section(self) -> None:
        instance = self.make()
        result = instance.process(mock.Mock(), 1.0)
        self.assertIs(result.hand, mock.sentinel.idle_hand)
        self.assertEqual(result.timings_ms["hand_total_ms"], 0.0)
        self.assertIn("recognition_total_ms", result.timings_ms)

    def test_the_attack_queue_still_gets_updated(self) -> None:
        # It has timeouts of its own; starving it would leave stale state if
        # hand input is ever switched back on.
        instance = self.make()
        instance.process(mock.Mock(), 5.0)
        instance.attacks.update_detailed.assert_called_once_with(None, 5.0)

    def test_full_mode_delegates_to_the_base_pipeline(self) -> None:
        instance = self.make()
        instance._body_only = False
        with mock.patch.object(
            pipeline_module.CombinedNarutoPipeline, "process", return_value=mock.sentinel.full
        ) as base:
            self.assertIs(instance.process(mock.sentinel.frame, 2.0), mock.sentinel.full)
        base.assert_called_once()


if __name__ == "__main__":
    unittest.main()
