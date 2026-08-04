from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from cv_model.inference.diagnostics import (
    BenchmarkCollector,
    SessionDiagnosticsRecorder,
    build_frame_diagnostic,
    format_benchmark_summary,
)
from cv_model.inference.scheduler import ScheduledResult, SchedulerStats


def _stats(
    *,
    submitted: int = 1,
    processed: int = 1,
    dropped: int = 0,
    failed: int = 0,
    superseded: int = 0,
) -> SchedulerStats:
    return SchedulerStats(
        submitted_frames=submitted,
        processed_frames=processed,
        dropped_frames=dropped,
        failed_frames=failed,
        superseded_results=superseded,
        pending_frame=False,
        processing_frame=False,
    )


def _scheduled(
    *,
    frame_id: int = 7,
    runtime_ms: float = 20.0,
    end_to_end_ms: float = 24.0,
) -> ScheduledResult:
    output = {
        "schema_version": "1.0.0",
        "session_id": "session-test",
        "frame_id": frame_id,
        "captured_at_ms": 1234 + frame_id,
        "processing_ms": 18.5,
        "hand": {
            "raw": {
                "label": "dog",
                "confidence": 0.91,
                "second_label": "ox",
                "second_confidence": 0.04,
                "margin": 0.87,
            },
            "center": {
                "label": "dog",
                "confidence": 0.91,
                "second_label": "ox",
                "second_confidence": 0.04,
                "margin": 0.87,
            },
            "roi": None,
            "accepted_label": "dog",
            "rejection_reason": None,
            "stable_label": "dog",
            "emitted_seal": "dog",
            "fusion_status": "center_only",
            "detected_hand_count": 2,
            "geometry": None,
        },
        "body": {
            "raw_label": "neutral",
            "stable_label": "neutral",
            "emitted_movement": None,
            "metrics": {},
            "geometry": None,
        },
        "queue": {
            "seals": [],
            "accepted_seal": "dog",
            "duplicate_ignored": False,
            "timeout_cleared": False,
            "max_length_cleared": False,
            "cooldown_suppressed": False,
        },
        "attack": {
            "name": "ikazuchi",
            "display_name": "Ikazuchi",
            "recognized_at_ms": 1241,
        },
    }
    event = {
        "schema_version": "1.0.0",
        "event_id": "session-test:00000000",
        "event_sequence": 0,
        "event_type": "HAND_SEAL",
        "session_id": "session-test",
        "frame_id": frame_id,
        "captured_at_ms": 1234 + frame_id,
        "payload": {"seal": "dog"},
    }
    runtime_frame = SimpleNamespace(
        output=output,
        timings_ms={
            "hand_total_ms": 8.0,
            "body_total_ms": 6.0,
            "runtime_total_ms": runtime_ms,
        },
        dispatch=SimpleNamespace(events=(event,)),
    )
    return ScheduledResult(
        capture_sequence=frame_id,
        captured_at_ms=1234 + frame_id,
        frame=object(),
        runtime_frame=runtime_frame,
        error=None,
        queue_wait_ms=1.0,
        worker_elapsed_ms=runtime_ms,
        end_to_end_ms=end_to_end_ms,
    )


class BenchmarkCollectorTests(unittest.TestCase):
    def test_summary_contains_rates_and_interpolated_latency_statistics(self) -> None:
        collector = BenchmarkCollector()
        for index, value in enumerate((10.0, 20.0, 30.0, 40.0)):
            collector.record(
                _scheduled(frame_id=index, runtime_ms=value, end_to_end_ms=value + 4)
            )

        summary = collector.summary(
            initial_stats=_stats(submitted=10, processed=8, dropped=2),
            final_stats=_stats(submitted=50, processed=38, dropped=12, superseded=1),
            elapsed_seconds=2.0,
            requested_duration_seconds=2.0,
            warmup_seconds=0.5,
        )

        self.assertEqual(summary["frames"]["submitted"], 40)
        self.assertEqual(summary["frames"]["processed"], 30)
        self.assertEqual(summary["frames"]["dropped"], 10)
        self.assertEqual(summary["frames"]["effective_fps"], 15.0)
        runtime = summary["timings_ms"]["runtime_total_ms"]
        self.assertEqual(runtime["mean"], 25.0)
        self.assertEqual(runtime["median"], 25.0)
        self.assertAlmostEqual(runtime["p95"], 38.5)
        self.assertEqual(runtime["max"], 40.0)
        self.assertIn("runtime_total_ms", format_benchmark_summary(summary))


class SessionDiagnosticsRecorderTests(unittest.TestCase):
    def test_frame_record_contains_metadata_but_not_camera_pixels(self) -> None:
        record = build_frame_diagnostic(
            _scheduled(),
            _stats(),
            recorded_at_ms=9999,
        )

        self.assertEqual(record["hand"]["emitted_seal"], "dog")
        self.assertEqual(record["attack"]["name"], "ikazuchi")
        self.assertEqual(record["events"][0]["event_type"], "HAND_SEAL")
        self.assertEqual(record["timings_ms"]["scheduler_end_to_end_ms"], 24.0)
        self.assertNotIn("frame", record)

    def test_jsonl_and_csv_are_streamed_with_session_summaries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jsonl_path = root / "session.jsonl"
            csv_path = root / "session.csv"
            recorder = SessionDiagnosticsRecorder(
                jsonl_path=jsonl_path,
                csv_path=csv_path,
                clock_ms=lambda: 5000,
            )
            recorder.record(_scheduled(), _stats())
            recorder.close(summary={"processed": 1})

            jsonl_records = [
                json.loads(line) for line in jsonl_path.read_text().splitlines()
            ]
            with csv_path.open(newline="", encoding="utf-8") as csv_file:
                csv_records = list(csv.DictReader(csv_file))

        self.assertEqual(
            [record["record_type"] for record in jsonl_records],
            ["frame", "session_summary"],
        )
        self.assertEqual(
            [record["record_type"] for record in csv_records],
            ["frame", "session_summary"],
        )
        self.assertEqual(csv_records[0]["hand_raw_label"], "dog")
        self.assertEqual(csv_records[0]["runtime_total_ms"], "20.0")
        self.assertEqual(csv_records[0]["scheduler_end_to_end_ms"], "24.0")
        self.assertEqual(json.loads(csv_records[1]["summary_json"]), {"processed": 1})

    def test_same_path_for_both_formats_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.data"
            with self.assertRaises(ValueError):
                SessionDiagnosticsRecorder(jsonl_path=path, csv_path=path)


if __name__ == "__main__":
    unittest.main()
