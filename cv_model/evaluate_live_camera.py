#!/usr/bin/env python3
"""Guided camera evaluation for hand signs, body movements, and attacks."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Optional

import cv2

from inference.combined_pipeline import (
    DEFAULT_HAND_LANDMARKER,
    DEFAULT_HAND_MODEL,
    DEFAULT_POSE_MODEL,
    CombinedNarutoPipeline,
)
from inference.diagnostics import SessionDiagnosticsRecorder
from inference.evaluation import (
    ATTACK_LABELS,
    ATTACK_SEALS,
    BODY_LABELS,
    HAND_LABELS,
    AttackAttemptEvaluator,
    AttackEvaluationSession,
    LabelAttemptEvaluator,
    LabelEvaluationSession,
    write_evaluation_reports,
)
from inference.runtime import PipelineRuntimeController
from inference.scheduler import LatestFrameScheduler, ScheduledResult


BODY_PROMPTS = {
    "idle": "Stand normally without making a movement",
    "jumping": "Perform a clear jump",
    "naruto_run": "Lean forward and hold both arms behind your torso",
    "bending_left": "Bend your torso clearly to your left",
    "bending_right": "Bend your torso clearly to your right",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure live trial-level accuracy without creating a camera dataset."
        ),
        epilog=(
            "Hand labels: "
            + ", ".join(HAND_LABELS)
            + "\nBody labels: "
            + ", ".join(BODY_LABELS)
            + "\nAttacks: "
            + ", ".join(ATTACK_LABELS)
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mode", choices=("hand", "body", "attack"), required=True)
    parser.add_argument(
        "--labels",
        nargs="+",
        default=("all",),
        help="Labels/attacks to test, or 'all' (default: all).",
    )
    parser.add_argument(
        "--attempts",
        type=int,
        default=3,
        help="Attempts per selected label (default: 3).",
    )
    parser.add_argument("--prepare-seconds", type=float, default=1.5)
    parser.add_argument(
        "--action-seconds",
        type=float,
        default=None,
        help="Override the action window (defaults: hand 3.5, body 4, attack 10).",
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--hand-confidence", type=float, default=None)
    parser.add_argument("--hand-model", type=Path, default=DEFAULT_HAND_MODEL)
    parser.add_argument("--hand-landmarker", type=Path, default=DEFAULT_HAND_LANDMARKER)
    parser.add_argument("--pose-model", type=Path, default=DEFAULT_POSE_MODEL)
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional aggregate metrics and detailed attempts JSON.",
    )
    parser.add_argument(
        "--attempts-csv",
        type=Path,
        default=None,
        help="Optional compact per-attempt CSV.",
    )
    parser.add_argument(
        "--report-jsonl",
        type=Path,
        default=None,
        help="Optional metadata-only per-frame JSONL diagnostics.",
    )
    parser.add_argument(
        "--report-csv",
        type=Path,
        default=None,
        help="Optional metadata-only per-frame CSV diagnostics.",
    )
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="Validate labels and load models without opening the camera.",
    )
    args = parser.parse_args()
    if args.attempts < 1:
        parser.error("--attempts must be positive")
    if args.prepare_seconds < 0:
        parser.error("--prepare-seconds cannot be negative")
    if args.action_seconds is not None and args.action_seconds <= 0:
        parser.error("--action-seconds must be positive")
    available = {
        "hand": HAND_LABELS,
        "body": BODY_LABELS,
        "attack": ATTACK_LABELS,
    }[args.mode]
    requested = tuple(label.lower() for label in args.labels)
    if "all" in requested:
        if len(requested) != 1:
            parser.error("'all' cannot be combined with explicit labels")
        args.labels = available
    else:
        unsupported = sorted(set(requested) - set(available))
        if unsupported:
            parser.error(f"unsupported {args.mode} labels: {', '.join(unsupported)}")
        if len(requested) != len(set(requested)):
            parser.error("labels cannot contain duplicates")
        args.labels = requested
    outputs = [
        path.resolve()
        for path in (
            args.summary_json,
            args.attempts_csv,
            args.report_jsonl,
            args.report_csv,
        )
        if path is not None
    ]
    if len(outputs) != len(set(outputs)):
        parser.error("all output paths must refer to different files")
    if args.action_seconds is None:
        args.action_seconds = {"hand": 3.5, "body": 4.0, "attack": 10.0}[
            args.mode
        ]
    return args


def _instruction(mode: str, label: str) -> str:
    if mode == "hand":
        if label == "zero":
            return "Keep both hands relaxed; do not form a hand seal"
        return f"Perform and hold the {label.upper()} hand seal"
    if mode == "body":
        return BODY_PROMPTS[label]
    seals = " > ".join(seal.upper() for seal in ATTACK_SEALS[label])
    return f"Perform {seals}; return to neutral between seals"


def _draw_status(
    frame: Any,
    *,
    mode: str,
    label: str,
    attempt_index: int,
    attempts_per_label: int,
    phase: str,
    remaining_seconds: float,
    last_output: Optional[dict[str, Any]],
    last_result: str,
) -> None:
    height, width = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (15, 15), (min(width - 15, 1050), 300), (8, 8, 8), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
    if phase == "ready":
        phase_text = "READY - press SPACE to start this attempt"
        phase_color = (80, 240, 255)
    elif phase == "prepare":
        phase_text = f"PREPARE / NEUTRAL: {remaining_seconds:.1f}s"
        phase_color = (70, 180, 255)
    else:
        phase_text = f"PERFORM NOW: {remaining_seconds:.1f}s"
        phase_color = (70, 255, 100)
    lines = [
        (f"MODE: {mode.upper()}   TARGET: {label.upper()}", (255, 255, 255)),
        (f"ATTEMPT: {attempt_index}/{attempts_per_label}", (220, 220, 220)),
        (phase_text, phase_color),
        (_instruction(mode, label), (200, 235, 255)),
    ]
    if last_output is not None:
        hand = last_output["hand"]
        body = last_output["body"]
        lines.extend(
            [
                (
                    f"HAND raw={hand['raw']['label']} accepted={hand['accepted_label']} "
                    f"stable={hand['stable_label']} emitted={hand['emitted_seal'] or '-'}",
                    (190, 220, 190),
                ),
                (
                    f"BODY raw={body['raw_label']} stable={body['stable_label']} "
                    f"emitted={body['emitted_movement'] or '-'}",
                    (210, 190, 255),
                ),
                (
                    "QUEUE: " + (" > ".join(last_output["queue"]["seals"]) or "-"),
                    (220, 220, 220),
                ),
            ]
        )
    y = 47
    for text, color in lines:
        cv2.putText(
            frame,
            text,
            (30, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            color,
            2,
            cv2.LINE_AA,
        )
        y += 35
    if last_result:
        cv2.putText(
            frame,
            last_result,
            (25, height - 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (80, 255, 120) if last_result.startswith("PASS") else (80, 120, 255),
            2,
            cv2.LINE_AA,
        )
    cv2.putText(
        frame,
        "SPACE start | Q or ESC finish and save partial results",
        (25, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (240, 240, 240),
        2,
        cv2.LINE_AA,
    )


def _record_scheduled(
    scheduled: Optional[ScheduledResult],
    *,
    evaluator: LabelAttemptEvaluator | AttackAttemptEvaluator,
    action_started_at_ms: int,
    action_ends_at_ms: int,
    scheduler: LatestFrameScheduler,
    recorder: Optional[SessionDiagnosticsRecorder],
) -> Optional[dict[str, Any]]:
    if scheduled is None:
        return None
    if scheduled.error is not None:
        raise RuntimeError(f"scheduled recognition failed: {scheduled.error}")
    runtime_frame = scheduled.runtime_frame
    assert runtime_frame is not None
    if recorder is not None:
        recorder.record(scheduled, scheduler.stats())
    phase = "prepare" if scheduled.captured_at_ms < action_started_at_ms else "action"
    if scheduled.captured_at_ms <= action_ends_at_ms:
        evaluator.record(runtime_frame.output, phase=phase)
    return runtime_frame.output


def _print_summary(summary: dict[str, Any]) -> None:
    print()
    print(f"Completed {summary['attempt_count']} {summary['mode']} attempts.")
    if summary["mode"] == "attack":
        print("attack       attempts  success  rate     failure reasons")
        for label, metric in summary["metrics"].items():
            if not metric["attempts"]:
                continue
            rate = f"{metric['success_rate']:.1%}"
            failures = ", ".join(
                f"{reason}={count}"
                for reason, count in metric["failure_reasons"].items()
            ) or "-"
            print(
                f"{label:12} {metric['attempts']:8d} {metric['successes']:8d} "
                f"{rate:8} {failures}"
            )
        return
    print("label          attempts  TP  FP  FN  precision  recall")
    for label, metric in summary["metrics"].items():
        if not metric["attempts"] and not metric["false_positives"]:
            continue
        precision = (
            f"{metric['precision']:.1%}" if metric["precision"] is not None else "-"
        )
        recall = f"{metric['recall']:.1%}" if metric["recall"] is not None else "-"
        print(
            f"{label:14} {metric['attempts']:8d} {metric['true_positives']:3d} "
            f"{metric['false_positives']:3d} {metric['false_negatives']:3d} "
            f"{precision:9} {recall:7}"
        )
    print("Confusion:")
    for expected, predictions in summary["confusion"].items():
        rendered = ", ".join(f"{label}={count}" for label, count in predictions.items())
        print(f"  {expected} -> {rendered}")


def main() -> int:
    args = parse_args()
    print("Loading locked hand-sign and pose models...")
    runtime = PipelineRuntimeController(
        pipeline_factory=lambda: CombinedNarutoPipeline(
            hand_model=args.hand_model,
            hand_landmarker=args.hand_landmarker,
            pose_model=args.pose_model,
            hand_confidence=args.hand_confidence,
        )
    ).start()
    if args.self_check:
        print(
            f"Evaluation self-check passed for {args.mode}: "
            + ", ".join(args.labels)
        )
        runtime.close()
        return 0

    camera = cv2.VideoCapture(args.camera)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not camera.isOpened():
        runtime.close()
        print(f"Error: camera {args.camera} could not be opened.")
        return 2
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
        camera.release()
        runtime.close()
        raise

    scheduler = LatestFrameScheduler(runtime).start()
    work = [
        (label, attempt_index)
        for label in args.labels
        for attempt_index in range(1, args.attempts + 1)
    ]
    session: LabelEvaluationSession | AttackEvaluationSession
    session = (
        AttackEvaluationSession()
        if args.mode == "attack"
        else LabelEvaluationSession(args.mode)
    )
    work_index = 0
    phase = "ready"
    evaluator: Optional[LabelAttemptEvaluator | AttackAttemptEvaluator] = None
    action_started_at_ms = 0
    action_ends_at_ms = 0
    last_output: Optional[dict[str, Any]] = None
    last_result = ""
    consecutive_failures = 0
    window_name = "NarutoCV - Guided Accuracy Evaluation"

    print("Guided evaluation started. Camera images will not be saved.")
    print("Press SPACE for each attempt; press Q to save partial results and quit.")
    try:
        while work_index < len(work):
            ok, frame = camera.read()
            if not ok:
                consecutive_failures += 1
                if consecutive_failures >= runtime.config.capture_failure_reset_threshold:
                    raise RuntimeError("camera frame read failed repeatedly")
                continue
            consecutive_failures = 0
            now = time.monotonic()
            now_ms = int(now * 1000.0)
            label, attempt_index = work[work_index]

            if phase != "ready":
                scheduler.submit(frame, captured_at_ms=now_ms)
                latest = _record_scheduled(
                    scheduler.poll_result(),
                    evaluator=evaluator,
                    action_started_at_ms=action_started_at_ms,
                    action_ends_at_ms=action_ends_at_ms,
                    scheduler=scheduler,
                    recorder=recorder,
                )
                if latest is not None:
                    last_output = latest
                phase = "prepare" if now_ms < action_started_at_ms else "action"
                if now_ms >= action_ends_at_ms:
                    if not scheduler.wait_until_idle(timeout=5.0):
                        raise TimeoutError("recognition worker did not finish the attempt")
                    latest = _record_scheduled(
                        scheduler.poll_result(),
                        evaluator=evaluator,
                        action_started_at_ms=action_started_at_ms,
                        action_ends_at_ms=action_ends_at_ms,
                        scheduler=scheduler,
                        recorder=recorder,
                    )
                    if latest is not None:
                        last_output = latest
                    result = evaluator.finish()
                    session.add(result)
                    if result["success"]:
                        last_result = "PASS: target recognized"
                    elif args.mode == "attack":
                        last_result = f"MISS: {result['failure_reason']}"
                    else:
                        last_result = (
                            "MISS: predicted "
                            + str(result["system_prediction"] or "no detection")
                        )
                    print(
                        f"{label} attempt {attempt_index}: "
                        + ("PASS" if result["success"] else last_result)
                    )
                    work_index += 1
                    phase = "ready"
                    evaluator = None
                    if work_index >= len(work):
                        break
                    label, attempt_index = work[work_index]

            remaining = 0.0
            if phase == "prepare":
                remaining = max(0.0, (action_started_at_ms - now_ms) / 1000.0)
            elif phase == "action":
                remaining = max(0.0, (action_ends_at_ms - now_ms) / 1000.0)
            display = cv2.flip(frame, 1)
            _draw_status(
                display,
                mode=args.mode,
                label=label,
                attempt_index=attempt_index,
                attempts_per_label=args.attempts,
                phase=phase,
                remaining_seconds=remaining,
                last_output=last_output,
                last_result=last_result,
            )
            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" ") and phase == "ready":
                if not scheduler.wait_until_idle(timeout=5.0):
                    raise TimeoutError("recognition worker did not reach trial boundary")
                scheduler.poll_result()
                runtime.reset(reason="evaluation_attempt")
                started_at_ms = int(time.monotonic() * 1000.0)
                action_started_at_ms = started_at_ms + int(
                    round(args.prepare_seconds * 1000.0)
                )
                action_ends_at_ms = action_started_at_ms + int(
                    round(args.action_seconds * 1000.0)
                )
                evaluator = (
                    AttackAttemptEvaluator(label, attempt_index)
                    if args.mode == "attack"
                    else LabelAttemptEvaluator(
                        args.mode,
                        label,
                        attempt_index,
                        action_started_at_ms,
                    )
                )
                phase = "prepare" if args.prepare_seconds else "action"
                last_output = None
    finally:
        camera.release()
        scheduler.stop()
        scheduler_stats = scheduler.stats()
        runtime.close()
        cv2.destroyAllWindows()
        if recorder is not None:
            recorder.close(
                summary={
                    "evaluation_mode": args.mode,
                    "completed_attempts": session.summary()["attempt_count"],
                    "scheduler": {
                        "submitted_frames": scheduler_stats.submitted_frames,
                        "processed_frames": scheduler_stats.processed_frames,
                        "dropped_frames": scheduler_stats.dropped_frames,
                        "failed_frames": scheduler_stats.failed_frames,
                    },
                }
            )

    summary = session.summary()
    write_evaluation_reports(
        summary,
        json_path=args.summary_json,
        csv_path=args.attempts_csv,
    )
    _print_summary(summary)
    if args.summary_json is not None:
        print(f"Summary JSON saved: {args.summary_json.resolve()}")
    if args.attempts_csv is not None:
        print(f"Attempt CSV saved: {args.attempts_csv.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
