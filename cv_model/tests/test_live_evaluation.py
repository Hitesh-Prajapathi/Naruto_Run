from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from cv_model.inference.evaluation import (
    AttackAttemptEvaluator,
    AttackEvaluationSession,
    LabelAttemptEvaluator,
    LabelEvaluationSession,
    write_evaluation_reports,
)


def _output(
    *,
    captured_at_ms: int = 1000,
    hand_raw: str = "zero",
    hand_accepted: str = "zero",
    hand_stable: str = "zero",
    hand_emitted: str | None = None,
    hand_rejection: str | None = None,
    body_raw: str = "idle",
    body_stable: str = "idle",
    body_emitted: str | None = None,
    accepted_seal: str | None = None,
    duplicate: bool = False,
    timeout: bool = False,
    max_length: bool = False,
    cooldown: bool = False,
    attack: str | None = None,
) -> dict:
    return {
        "captured_at_ms": captured_at_ms,
        "hand": {
            "raw": {
                "label": hand_raw,
                "confidence": 0.85,
                "margin": 0.50,
            },
            "accepted_label": hand_accepted,
            "stable_label": hand_stable,
            "emitted_seal": hand_emitted,
            "rejection_reason": hand_rejection,
        },
        "body": {
            "raw_label": body_raw,
            "stable_label": body_stable,
            "emitted_movement": body_emitted,
        },
        "queue": {
            "accepted_seal": accepted_seal,
            "duplicate_ignored": duplicate,
            "timeout_cleared": timeout,
            "max_length_cleared": max_length,
            "cooldown_suppressed": cooldown,
        },
        "attack": (
            {"name": attack, "display_name": attack.upper()}
            if attack is not None
            else None
        ),
    }


class LabelAttemptEvaluatorTests(unittest.TestCase):
    def test_hand_attempt_records_event_success_latency_and_threshold_evidence(self) -> None:
        attempt = LabelAttemptEvaluator("hand", "dog", 1, action_started_at_ms=1100)
        attempt.record(
            _output(
                hand_raw="dog",
                hand_accepted="dog",
                hand_stable="dog",
                hand_emitted="dog",
                hand_rejection="confidence<0.80",
            ),
            phase="prepare",
        )
        attempt.record(
            _output(
                captured_at_ms=1250,
                hand_raw="dog",
                hand_accepted="dog",
                hand_stable="dog",
                hand_emitted="dog",
            ),
            phase="action",
        )

        result = attempt.finish()

        self.assertTrue(result["success"])
        self.assertEqual(result["system_prediction"], "dog")
        self.assertEqual(result["detection_latency_ms"], 150)
        self.assertEqual(result["prepare_false_positive_events"], ["dog"])
        self.assertEqual(result["expected_confidence"]["count"], 1)
        self.assertEqual(result["rejection_reasons"], {})

    def test_first_event_and_eventual_detection_are_reported_separately(self) -> None:
        attempt = LabelAttemptEvaluator("hand", "bird", 1, action_started_at_ms=1000)
        attempt.record(
            _output(
                captured_at_ms=1100,
                hand_raw="horse",
                hand_accepted="horse",
                hand_stable="horse",
                hand_emitted="horse",
            ),
            phase="action",
        )
        attempt.record(
            _output(
                captured_at_ms=1200,
                hand_raw="bird",
                hand_accepted="bird",
                hand_stable="bird",
                hand_emitted="bird",
            ),
            phase="action",
        )
        attempt.record(
            _output(
                captured_at_ms=1300,
                hand_raw="bird",
                hand_accepted="bird",
                hand_stable="bird",
                hand_emitted="bird",
            ),
            phase="action",
        )

        result = attempt.finish()

        self.assertFalse(result["first_event_success"])
        self.assertTrue(result["target_event_detected"])
        self.assertEqual(result["unexpected_action_events"], ["horse"])
        self.assertEqual(result["duplicate_emissions"], 1)

    def test_action_start_can_be_reanchored_after_temporal_reset(self) -> None:
        attempt = LabelAttemptEvaluator("hand", "dog", 1, action_started_at_ms=1000)
        attempt.start_action(2000)
        attempt.record(
            _output(
                captured_at_ms=2250,
                hand_raw="dog",
                hand_accepted="dog",
                hand_stable="dog",
                hand_emitted="dog",
            ),
            phase="action",
        )

        self.assertEqual(attempt.finish()["detection_latency_ms"], 250)

    def test_hand_session_reports_confusion_precision_and_recall(self) -> None:
        dog = LabelAttemptEvaluator("hand", "dog", 1, action_started_at_ms=1000)
        dog.record(
            _output(
                captured_at_ms=1100,
                hand_raw="dog",
                hand_accepted="dog",
                hand_stable="dog",
                hand_emitted="dog",
            ),
            phase="action",
        )
        rat = LabelAttemptEvaluator("hand", "rat", 1, action_started_at_ms=2000)
        rat.record(
            _output(
                captured_at_ms=2100,
                hand_raw="ram",
                hand_accepted="ram",
                hand_stable="ram",
                hand_emitted="ram",
            ),
            phase="action",
        )
        session = LabelEvaluationSession("hand")
        session.add(dog.finish())
        session.add(rat.finish())

        summary = session.summary()

        self.assertEqual(summary["confusion"]["rat"], {"ram": 1})
        self.assertEqual(summary["metrics"]["dog"]["recall"], 1.0)
        self.assertEqual(summary["metrics"]["rat"]["recall"], 0.0)
        self.assertEqual(summary["metrics"]["ram"]["false_positives"], 1)
        self.assertEqual(summary["metrics"]["dog"]["eventual_recall"], 1.0)

    def test_neutral_body_attempt_uses_stable_idle_prediction(self) -> None:
        attempt = LabelAttemptEvaluator("body", "idle", 1, action_started_at_ms=1000)
        attempt.record(
            _output(captured_at_ms=1010, body_raw="idle", body_stable="idle"),
            phase="action",
        )

        result = attempt.finish()

        self.assertTrue(result["success"])
        self.assertEqual(result["system_prediction"], "idle")


class AttackAttemptEvaluatorTests(unittest.TestCase):
    def test_successful_attack_preserves_accepted_sequence_and_queue_details(self) -> None:
        attempt = AttackAttemptEvaluator("shippu", 1)
        for seal in ("bird", "ram"):
            attempt.record(
                _output(
                    hand_raw=seal,
                    hand_accepted=seal,
                    hand_stable=seal,
                    hand_emitted=seal,
                    accepted_seal=seal,
                )
            )
        attempt.record(
            _output(
                hand_raw="rat",
                hand_accepted="rat",
                hand_stable="rat",
                hand_emitted="rat",
                accepted_seal="rat",
                attack="shippu",
            )
        )

        result = attempt.finish()

        self.assertTrue(result["success"])
        self.assertIsNone(result["failure_reason"])
        self.assertEqual(result["accepted_seals"], ["bird", "ram", "rat"])
        self.assertEqual(result["missing_expected_seals"], [])

    def test_timeout_failure_is_explained_and_aggregated(self) -> None:
        attempt = AttackAttemptEvaluator("homura", 1)
        attempt.record(
            _output(
                hand_raw="tiger",
                hand_accepted="tiger",
                hand_stable="tiger",
                hand_emitted="tiger",
                accepted_seal="tiger",
            )
        )
        attempt.record(_output(timeout=True))
        result = attempt.finish()
        session = AttackEvaluationSession()
        session.add(result)

        summary = session.summary()

        self.assertEqual(result["failure_reason"], "seal_timeout")
        self.assertEqual(result["missing_expected_seals"], ["dragon", "horse"])
        self.assertEqual(
            summary["metrics"]["homura"]["failure_reasons"],
            {"seal_timeout": 1},
        )

    def test_summary_and_attempt_reports_contain_no_image_data(self) -> None:
        attempt = AttackAttemptEvaluator("ikazuchi", 1)
        attempt.record(
            _output(
                hand_raw="dog",
                hand_accepted="dog",
                hand_stable="dog",
                hand_emitted="dog",
                accepted_seal="dog",
                attack="ikazuchi",
            )
        )
        session = AttackEvaluationSession()
        session.add(attempt.finish())
        summary = session.summary()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            json_path = root / "summary.json"
            csv_path = root / "attempts.csv"
            write_evaluation_reports(
                summary,
                json_path=json_path,
                csv_path=csv_path,
            )
            saved_summary = json.loads(json_path.read_text())
            with csv_path.open(newline="", encoding="utf-8") as csv_file:
                rows = list(csv.DictReader(csv_file))

        self.assertEqual(saved_summary["attempt_count"], 1)
        self.assertEqual(rows[0]["expected_sequence"], "dog")
        self.assertNotIn("frame", rows[0]["details_json"])


if __name__ == "__main__":
    unittest.main()
