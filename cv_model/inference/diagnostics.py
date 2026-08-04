"""Benchmark aggregation and metadata-only session reports for NarutoCV."""

from __future__ import annotations

import csv
import json
import math
import statistics
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from .scheduler import ScheduledResult, SchedulerStats


DIAGNOSTICS_VERSION = "1.0.0"
RUNTIME_TIMING_KEYS = (
    "hand_center_preprocess_ms",
    "hand_center_onnx_ms",
    "hand_center_postprocess_ms",
    "hand_landmarker_preprocess_ms",
    "hand_landmarker_ms",
    "hand_roi_preprocess_ms",
    "hand_roi_onnx_ms",
    "hand_roi_postprocess_ms",
    "hand_postprocess_ms",
    "hand_total_ms",
    "pose_preprocess_ms",
    "pose_landmarker_ms",
    "body_postprocess_ms",
    "body_total_ms",
    "attack_queue_ms",
    "pipeline_overhead_ms",
    "recognition_total_ms",
    "runtime_recognition_ms",
    "serialization_ms",
    "event_derivation_ms",
    "callback_delivery_ms",
    "event_dispatch_total_ms",
    "runtime_overhead_ms",
    "runtime_total_ms",
)
SCHEDULER_TIMING_KEYS = (
    "scheduler_queue_wait_ms",
    "scheduler_worker_ms",
    "scheduler_end_to_end_ms",
)


def scheduled_timings_ms(scheduled: ScheduledResult) -> dict[str, float]:
    """Combine runtime phase timings with scheduler latency measurements."""
    runtime_frame = scheduled.runtime_frame
    if runtime_frame is None:
        return {}
    return {
        **runtime_frame.timings_ms,
        "scheduler_queue_wait_ms": scheduled.queue_wait_ms,
        "scheduler_worker_ms": scheduled.worker_elapsed_ms,
        "scheduler_end_to_end_ms": scheduled.end_to_end_ms,
    }


def _timing_statistics(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    position = (len(ordered) - 1) * 0.95
    lower = math.floor(position)
    upper = math.ceil(position)
    p95 = ordered[lower]
    if upper != lower:
        p95 += (ordered[upper] - ordered[lower]) * (position - lower)
    return {
        "count": len(ordered),
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "p95": p95,
        "max": ordered[-1],
    }


class BenchmarkCollector:
    """Collect timing samples and produce a deterministic benchmark summary."""

    def __init__(self) -> None:
        self._samples: dict[str, list[float]] = {}
        self.collected_results = 0

    def record(self, scheduled: ScheduledResult) -> None:
        if scheduled.error is not None or scheduled.runtime_frame is None:
            return
        self.collected_results += 1
        for name, raw_value in scheduled_timings_ms(scheduled).items():
            value = float(raw_value)
            if not math.isfinite(value) or value < 0:
                continue
            self._samples.setdefault(name, []).append(value)

    def summary(
        self,
        *,
        initial_stats: SchedulerStats,
        final_stats: SchedulerStats,
        elapsed_seconds: float,
        requested_duration_seconds: float,
        warmup_seconds: float,
    ) -> dict[str, Any]:
        if elapsed_seconds <= 0:
            raise ValueError("elapsed_seconds must be positive")

        def delta(name: str) -> int:
            return max(
                0,
                int(getattr(final_stats, name)) - int(getattr(initial_stats, name)),
            )

        submitted = delta("submitted_frames")
        processed = delta("processed_frames")
        dropped = delta("dropped_frames")
        failed = delta("failed_frames")
        superseded = delta("superseded_results")
        return {
            "diagnostics_version": DIAGNOSTICS_VERSION,
            "benchmark": {
                "requested_duration_seconds": float(requested_duration_seconds),
                "warmup_seconds": float(warmup_seconds),
                "elapsed_seconds": float(elapsed_seconds),
            },
            "frames": {
                "submitted": submitted,
                "processed": processed,
                "dropped": dropped,
                "failed": failed,
                "superseded_results": superseded,
                "collected_results": self.collected_results,
                "capture_fps": submitted / elapsed_seconds,
                "effective_fps": processed / elapsed_seconds,
                "drop_rate": dropped / submitted if submitted else 0.0,
            },
            "timings_ms": {
                name: _timing_statistics(values)
                for name, values in sorted(self._samples.items())
                if values
            },
        }


def format_benchmark_summary(summary: Mapping[str, Any]) -> str:
    """Render the complete aggregate benchmark as a terminal-friendly table."""
    benchmark = summary["benchmark"]
    frames = summary["frames"]
    lines = [
        "NarutoCV live-camera benchmark",
        (
            f"Duration: {benchmark['elapsed_seconds']:.2f}s "
            f"(requested {benchmark['requested_duration_seconds']:.2f}s, "
            f"warmup {benchmark['warmup_seconds']:.2f}s)"
        ),
        (
            f"Frames: submitted={frames['submitted']} processed={frames['processed']} "
            f"dropped={frames['dropped']} failed={frames['failed']} "
            f"collected={frames['collected_results']}"
        ),
        (
            f"Rates: capture={frames['capture_fps']:.2f} FPS "
            f"effective={frames['effective_fps']:.2f} FPS "
            f"dropped={frames['drop_rate']:.1%}"
        ),
        "",
        f"{'Timing':36} {'Mean':>9} {'Median':>9} {'P95':>9} {'Max':>9}",
        "-" * 78,
    ]
    for name, values in summary["timings_ms"].items():
        lines.append(
            f"{name:36} {values['mean']:9.2f} {values['median']:9.2f} "
            f"{values['p95']:9.2f} {values['max']:9.2f}"
        )
    return "\n".join(lines)


def build_frame_diagnostic(
    scheduled: ScheduledResult,
    scheduler_stats: SchedulerStats,
    *,
    recorded_at_ms: Optional[int] = None,
) -> dict[str, Any]:
    """Create a JSON-safe metadata record without retaining a camera image."""
    if scheduled.error is not None:
        raise ValueError("cannot record a failed scheduled result as a frame")
    runtime_frame = scheduled.runtime_frame
    if runtime_frame is None:
        raise ValueError("scheduled result does not contain a runtime frame")
    output = runtime_frame.output
    return {
        "record_type": "frame",
        "diagnostics_version": DIAGNOSTICS_VERSION,
        "recorded_at_ms": (
            int(time.time() * 1000.0)
            if recorded_at_ms is None
            else int(recorded_at_ms)
        ),
        "session_id": output["session_id"],
        "frame_id": output["frame_id"],
        "capture_sequence": scheduled.capture_sequence,
        "captured_at_ms": output["captured_at_ms"],
        "processing_ms": output["processing_ms"],
        "hand": dict(output["hand"]),
        "body": dict(output["body"]),
        "queue": dict(output["queue"]),
        "attack": dict(output["attack"]) if output["attack"] is not None else None,
        "events": [dict(event) for event in runtime_frame.dispatch.events],
        "timings_ms": scheduled_timings_ms(scheduled),
        "scheduler": asdict(scheduler_stats),
    }


CSV_FIELDS = (
    "record_type",
    "diagnostics_version",
    "recorded_at_ms",
    "session_id",
    "frame_id",
    "capture_sequence",
    "captured_at_ms",
    "processing_ms",
    "hand_raw_label",
    "hand_raw_confidence",
    "hand_raw_margin",
    "hand_accepted_label",
    "hand_rejection_reason",
    "hand_stable_label",
    "hand_emitted_seal",
    "body_raw_label",
    "body_stable_label",
    "body_emitted_movement",
    "queue_seals",
    "queue_accepted_seal",
    "queue_duplicate_ignored",
    "queue_timeout_cleared",
    "queue_max_length_cleared",
    "queue_cooldown_suppressed",
    "attack_name",
    "attack_display_name",
    "event_types",
    "submitted_frames",
    "processed_frames",
    "dropped_frames",
    "failed_frames",
    "superseded_results",
    "timings_json",
    "events_json",
    "summary_json",
) + RUNTIME_TIMING_KEYS + SCHEDULER_TIMING_KEYS


def _csv_frame(record: Mapping[str, Any]) -> dict[str, Any]:
    hand = record["hand"]
    raw = hand["raw"]
    body = record["body"]
    queue = record["queue"]
    attack = record["attack"] or {}
    scheduler = record["scheduler"]
    events = record["events"]
    row = {
        "record_type": "frame",
        "diagnostics_version": record["diagnostics_version"],
        "recorded_at_ms": record["recorded_at_ms"],
        "session_id": record["session_id"],
        "frame_id": record["frame_id"],
        "capture_sequence": record["capture_sequence"],
        "captured_at_ms": record["captured_at_ms"],
        "processing_ms": record["processing_ms"],
        "hand_raw_label": raw["label"],
        "hand_raw_confidence": raw["confidence"],
        "hand_raw_margin": raw["margin"],
        "hand_accepted_label": hand["accepted_label"],
        "hand_rejection_reason": hand["rejection_reason"],
        "hand_stable_label": hand["stable_label"],
        "hand_emitted_seal": hand["emitted_seal"],
        "body_raw_label": body["raw_label"],
        "body_stable_label": body["stable_label"],
        "body_emitted_movement": body["emitted_movement"],
        "queue_seals": ">".join(queue["seals"]),
        "queue_accepted_seal": queue["accepted_seal"],
        "queue_duplicate_ignored": queue["duplicate_ignored"],
        "queue_timeout_cleared": queue["timeout_cleared"],
        "queue_max_length_cleared": queue["max_length_cleared"],
        "queue_cooldown_suppressed": queue["cooldown_suppressed"],
        "attack_name": attack.get("name"),
        "attack_display_name": attack.get("display_name"),
        "event_types": "|".join(event["event_type"] for event in events),
        "submitted_frames": scheduler["submitted_frames"],
        "processed_frames": scheduler["processed_frames"],
        "dropped_frames": scheduler["dropped_frames"],
        "failed_frames": scheduler["failed_frames"],
        "superseded_results": scheduler["superseded_results"],
        "timings_json": json.dumps(record["timings_ms"], separators=(",", ":")),
        "events_json": json.dumps(events, separators=(",", ":")),
        "summary_json": "",
    }
    row.update(
        {
            name: record["timings_ms"].get(name, "")
            for name in RUNTIME_TIMING_KEYS + SCHEDULER_TIMING_KEYS
        }
    )
    return row


class SessionDiagnosticsRecorder:
    """Stream frame metadata to optional JSONL and CSV files."""

    def __init__(
        self,
        *,
        jsonl_path: Optional[Path] = None,
        csv_path: Optional[Path] = None,
        clock_ms: Callable[[], int] = lambda: int(time.time() * 1000.0),
    ) -> None:
        if jsonl_path is None and csv_path is None:
            raise ValueError("at least one diagnostics output path is required")
        if jsonl_path is not None and csv_path is not None:
            if jsonl_path.resolve() == csv_path.resolve():
                raise ValueError("JSONL and CSV reports must use different files")
        self.jsonl_path = jsonl_path
        self.csv_path = csv_path
        self.clock_ms = clock_ms
        self._jsonl_file = None
        self._csv_file = None
        self._csv_writer: Optional[csv.DictWriter] = None
        self._closed = False
        try:
            if jsonl_path is not None:
                jsonl_path.parent.mkdir(parents=True, exist_ok=True)
                self._jsonl_file = jsonl_path.open("w", encoding="utf-8")
            if csv_path is not None:
                csv_path.parent.mkdir(parents=True, exist_ok=True)
                self._csv_file = csv_path.open("w", encoding="utf-8", newline="")
                self._csv_writer = csv.DictWriter(
                    self._csv_file,
                    fieldnames=CSV_FIELDS,
                )
                self._csv_writer.writeheader()
        except Exception:
            self.close()
            raise

    def record(
        self,
        scheduled: ScheduledResult,
        scheduler_stats: SchedulerStats,
    ) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("diagnostics recorder is closed")
        record = build_frame_diagnostic(
            scheduled,
            scheduler_stats,
            recorded_at_ms=self.clock_ms(),
        )
        if self._jsonl_file is not None:
            self._jsonl_file.write(json.dumps(record, separators=(",", ":")) + "\n")
            self._jsonl_file.flush()
        if self._csv_writer is not None and self._csv_file is not None:
            self._csv_writer.writerow(_csv_frame(record))
            self._csv_file.flush()
        return record

    def close(self, *, summary: Optional[Mapping[str, Any]] = None) -> None:
        if self._closed:
            return
        if summary is not None:
            summary_record = {
                "record_type": "session_summary",
                "diagnostics_version": DIAGNOSTICS_VERSION,
                "recorded_at_ms": self.clock_ms(),
                "summary": dict(summary),
            }
            if self._jsonl_file is not None:
                self._jsonl_file.write(
                    json.dumps(summary_record, separators=(",", ":")) + "\n"
                )
            if self._csv_writer is not None:
                row = {field: "" for field in CSV_FIELDS}
                row.update(
                    {
                        "record_type": "session_summary",
                        "diagnostics_version": DIAGNOSTICS_VERSION,
                        "recorded_at_ms": summary_record["recorded_at_ms"],
                        "summary_json": json.dumps(summary, separators=(",", ":")),
                    }
                )
                self._csv_writer.writerow(row)
        if self._jsonl_file is not None:
            self._jsonl_file.close()
        if self._csv_file is not None:
            self._csv_file.close()
        self._closed = True

    def __enter__(self) -> SessionDiagnosticsRecorder:
        return self

    def __exit__(self, *_error: object) -> None:
        self.close()
