#!/usr/bin/env python3
"""Run a bounded, reproducible NarutoCV benchmark against a live camera."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import cv2

from inference.combined_pipeline import (
    DEFAULT_HAND_LANDMARKER,
    DEFAULT_HAND_MODEL,
    DEFAULT_POSE_MODEL,
    CombinedNarutoPipeline,
)
from inference.diagnostics import (
    BenchmarkCollector,
    SessionDiagnosticsRecorder,
    format_benchmark_summary,
)
from inference.runtime import PipelineRuntimeController
from inference.scheduler import LatestFrameScheduler, ScheduledResult


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the combined hand, body, and attack pipeline on a live camera."
        )
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Measured session duration in seconds (default: 30)",
    )
    parser.add_argument(
        "--warmup",
        type=float,
        default=3.0,
        help="Unmeasured model/camera warmup in seconds (default: 3)",
    )
    parser.add_argument("--hand-confidence", type=float, default=None)
    parser.add_argument("--hand-model", type=Path, default=DEFAULT_HAND_MODEL)
    parser.add_argument("--hand-landmarker", type=Path, default=DEFAULT_HAND_LANDMARKER)
    parser.add_argument("--pose-model", type=Path, default=DEFAULT_POSE_MODEL)
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional path for the aggregate benchmark JSON.",
    )
    parser.add_argument(
        "--report-jsonl",
        type=Path,
        default=None,
        help="Optional metadata-only per-frame JSONL report.",
    )
    parser.add_argument(
        "--report-csv",
        type=Path,
        default=None,
        help="Optional metadata-only per-frame CSV report.",
    )
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="Load models and validate arguments without opening the camera.",
    )
    args = parser.parse_args()
    if args.duration <= 0:
        parser.error("--duration must be positive")
    if args.warmup < 0:
        parser.error("--warmup cannot be negative")
    outputs = [
        path.resolve()
        for path in (args.summary_json, args.report_jsonl, args.report_csv)
        if path is not None
    ]
    if len(outputs) != len(set(outputs)):
        parser.error("benchmark output paths must refer to different files")
    return args


def _accept_result(
    scheduled: Optional[ScheduledResult],
    collector: BenchmarkCollector,
    scheduler: LatestFrameScheduler,
    recorder: Optional[SessionDiagnosticsRecorder],
) -> None:
    if scheduled is None:
        return
    if scheduled.error is not None:
        raise RuntimeError(f"scheduled recognition failed: {scheduled.error}")
    collector.record(scheduled)
    if recorder is not None:
        recorder.record(scheduled, scheduler.stats())


def _run_capture_window(
    camera: cv2.VideoCapture,
    scheduler: LatestFrameScheduler,
    *,
    duration_seconds: float,
    collector: Optional[BenchmarkCollector] = None,
    recorder: Optional[SessionDiagnosticsRecorder] = None,
) -> None:
    deadline = time.perf_counter() + duration_seconds
    consecutive_failures = 0
    while time.perf_counter() < deadline:
        ok, frame = camera.read()
        if not ok:
            consecutive_failures += 1
            if consecutive_failures >= 3:
                raise RuntimeError("camera frame read failed three consecutive times")
            continue
        consecutive_failures = 0
        scheduler.submit(frame)
        scheduled = scheduler.poll_result()
        if collector is not None:
            _accept_result(scheduled, collector, scheduler, recorder)
        elif scheduled is not None and scheduled.error is not None:
            raise RuntimeError(f"scheduled recognition failed: {scheduled.error}")


def main() -> int:
    args = parse_args()
    print("Loading hand-sign and pose models...")
    runtime = PipelineRuntimeController(
        pipeline_factory=lambda: CombinedNarutoPipeline(
            hand_model=args.hand_model,
            hand_landmarker=args.hand_landmarker,
            pose_model=args.pose_model,
            hand_confidence=args.hand_confidence,
        )
    ).start()
    if args.self_check:
        print("Benchmark self-check passed: models and arguments are valid.")
        runtime.close()
        return 0

    try:
        recorder = (
            SessionDiagnosticsRecorder(
                jsonl_path=args.report_jsonl,
                csv_path=args.report_csv,
            )
            if args.report_jsonl is not None or args.report_csv is not None
            else None
        )
    except Exception:
        runtime.close()
        raise
    camera = cv2.VideoCapture(args.camera)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not camera.isOpened():
        if recorder is not None:
            recorder.close()
        runtime.close()
        print(f"Error: camera {args.camera} could not be opened.")
        return 2

    scheduler = LatestFrameScheduler(runtime).start()
    summary = None
    try:
        if args.warmup:
            print(f"Warming up for {args.warmup:.1f} seconds...")
            _run_capture_window(
                camera,
                scheduler,
                duration_seconds=args.warmup,
            )
            scheduler.clear_pending()
            if not scheduler.wait_until_idle(timeout=5.0):
                raise TimeoutError("recognition worker did not finish warmup")
            warmup_result = scheduler.poll_result()
            if warmup_result is not None and warmup_result.error is not None:
                raise RuntimeError(
                    f"scheduled recognition failed: {warmup_result.error}"
                )

        collector = BenchmarkCollector()
        initial_stats = scheduler.stats()
        started_at = time.perf_counter()
        print(f"Measuring for {args.duration:.1f} seconds...")
        _run_capture_window(
            camera,
            scheduler,
            duration_seconds=args.duration,
            collector=collector,
            recorder=recorder,
        )
        scheduler.stop(drain=True, timeout=5.0)
        _accept_result(scheduler.poll_result(), collector, scheduler, recorder)
        elapsed_seconds = time.perf_counter() - started_at
        final_stats = scheduler.stats()
        summary = collector.summary(
            initial_stats=initial_stats,
            final_stats=final_stats,
            elapsed_seconds=elapsed_seconds,
            requested_duration_seconds=args.duration,
            warmup_seconds=args.warmup,
        )
        print()
        print(format_benchmark_summary(summary))
        if args.summary_json is not None:
            args.summary_json.parent.mkdir(parents=True, exist_ok=True)
            args.summary_json.write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(f"Aggregate JSON saved: {args.summary_json.resolve()}")
    except KeyboardInterrupt:
        print("Benchmark interrupted.")
        return 130
    finally:
        camera.release()
        scheduler.stop()
        if recorder is not None:
            recorder.close(summary=summary)
        runtime.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
