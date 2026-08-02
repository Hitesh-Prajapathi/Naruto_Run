#!/usr/bin/env python3
"""Camera-only tester for combined NarutoCV recognition."""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

import cv2

from inference.combined_pipeline import (
    DEFAULT_HAND_LANDMARKER,
    DEFAULT_HAND_MODEL,
    DEFAULT_POSE_MODEL,
    CombinedNarutoPipeline,
    FrameResult,
)


POSE_CONNECTIONS = (
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (25, 27),
    (24, 26),
    (26, 28),
)
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)


def _mirror_bbox(
    bbox: tuple[int, int, int, int], width: int
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    return width - x2, y1, width - x1, y2


def _mirror_result_for_display(result: FrameResult, width: int) -> FrameResult:
    """Mirror overlay geometry after recognition has processed the raw frame."""
    hand = replace(
        result.hand,
        classification_bbox=_mirror_bbox(result.hand.classification_bbox, width),
        hand_bbox=(
            _mirror_bbox(result.hand.hand_bbox, width)
            if result.hand.hand_bbox is not None
            else None
        ),
        landmarks=tuple(
            tuple((1.0 - x, y) for x, y in landmarks)
            for landmarks in result.hand.landmarks
        ),
    )
    body = replace(
        result.body,
        landmarks=tuple((1.0 - x, y) for x, y in result.body.landmarks),
    )
    return replace(result, hand=hand, body=body)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test hand signs, body movement, and attacks through a camera."
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--hand-confidence",
        type=float,
        default=None,
        help="Optional global override; default uses calibrated per-class thresholds.",
    )
    parser.add_argument("--hand-model", type=Path, default=DEFAULT_HAND_MODEL)
    parser.add_argument("--hand-landmarker", type=Path, default=DEFAULT_HAND_LANDMARKER)
    parser.add_argument("--pose-model", type=Path, default=DEFAULT_POSE_MODEL)
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="Load all models and exit without opening the camera.",
    )
    return parser.parse_args()


def _draw_landmarks(frame, result: FrameResult) -> None:
    height, width = frame.shape[:2]
    for hand in result.hand.landmarks:
        for start, end in HAND_CONNECTIONS:
            p1, p2 = hand[start], hand[end]
            cv2.line(
                frame,
                (int(p1[0] * width), int(p1[1] * height)),
                (int(p2[0] * width), int(p2[1] * height)),
                (80, 220, 255),
                2,
            )
        for x, y in hand:
            cv2.circle(frame, (int(x * width), int(y * height)), 3, (0, 110, 255), -1)

    pose = result.body.landmarks
    if pose:
        for start, end in POSE_CONNECTIONS:
            p1, p2 = pose[start], pose[end]
            cv2.line(
                frame,
                (int(p1[0] * width), int(p1[1] * height)),
                (int(p2[0] * width), int(p2[1] * height)),
                (255, 120, 40),
                2,
            )
        for index in {point for connection in POSE_CONNECTIONS for point in connection}:
            x, y = pose[index]
            cv2.circle(frame, (int(x * width), int(y * height)), 4, (255, 60, 60), -1)


def _draw_panel(
    frame,
    result: FrameResult,
    fps: float,
    last_attack: str,
    calibration_remaining: float,
) -> None:
    height, width = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (12, 12), (min(860, width - 12), 292), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.68, frame, 0.32, 0, frame)

    hand = result.hand
    accepted = hand.accepted_label.upper()
    raw_status = (
        f"RAW: {hand.raw.label.upper()} {hand.raw.confidence * 100:.1f}%"
        f"  MARGIN {hand.raw.margin * 100:.1f}%"
    )
    if hand.rejection_reason and hand.rejection_reason != "model_zero":
        raw_status += f"  REJECTED ({hand.rejection_reason})"
    center = hand.center_prediction or hand.raw
    roi_status = (
        f"{hand.roi_prediction.label.upper()} {hand.roi_prediction.confidence * 100:.1f}%"
        if hand.roi_prediction is not None
        else "-"
    )
    lines = [
        (f"HAND: {hand.stable_label.upper()}  ACCEPTED: {accepted}", (50, 255, 90)),
        (raw_status, (175, 235, 255)),
        (
            f"VIEWS: CENTER {center.label.upper()} {center.confidence * 100:.1f}%"
            f"  ROI {roi_status}  [{hand.fusion_status}]",
            (120, 220, 240),
        ),
        (f"BODY: {result.body.stable_label.upper()}", (255, 190, 60)),
        ("SEALS: " + (" > ".join(result.seal_history).upper() or "-"), (230, 230, 230)),
        ("LAST ATTACK: " + (last_attack or "-"), (80, 120, 255)),
        (f"FPS: {fps:.1f}   PIPELINE: {result.processing_ms:.1f} ms", (190, 190, 190)),
    ]
    y = 42
    for text, color in lines:
        cv2.putText(frame, text, (28, y), cv2.FONT_HERSHEY_SIMPLEX, 0.68, color, 2, cv2.LINE_AA)
        y += 33

    x1, y1, x2, y2 = hand.classification_bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), (90, 220, 90), 2)
    cv2.putText(
        frame,
        "KEEP FACE, TORSO AND HAND SIGNS INSIDE THIS BOX",
        (x1 + 10, max(315, y1 + 26)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (90, 220, 90),
        2,
        cv2.LINE_AA,
    )

    if hand.hand_bbox:
        x1, y1, x2, y2 = hand.hand_bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 255), 2)

    if result.attack:
        banner_y1, banner_y2 = max(240, height // 2 - 55), max(340, height // 2 + 55)
        cv2.rectangle(frame, (0, banner_y1), (width, banner_y2), (20, 20, 170), -1)
        text = f"ATTACK: {result.attack.display_name}"
        size = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 1.35, 3)[0]
        cv2.putText(
            frame,
            text,
            ((width - size[0]) // 2, banner_y1 + 70),
            cv2.FONT_HERSHEY_DUPLEX,
            1.35,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )

    if calibration_remaining > 0:
        text = f"NEUTRAL CALIBRATION: {calibration_remaining:.1f}s"
        cv2.rectangle(frame, (0, height // 2 - 60), (width, height // 2 + 60), (30, 90, 160), -1)
        size = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 1.2, 3)[0]
        cv2.putText(
            frame,
            text,
            ((width - size[0]) // 2, height // 2 - 10),
            cv2.FONT_HERSHEY_DUPLEX,
            1.2,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        help_text = "Stand normally and move through NON-SIGN hand poses"
        help_size = cv2.getTextSize(help_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0]
        cv2.putText(
            frame,
            help_text,
            ((width - help_size[0]) // 2, height // 2 + 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        frame,
        "Q quit | R reset sequence | C neutral calibration | S screenshot",
        (18, height - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (235, 235, 235),
        2,
        cv2.LINE_AA,
    )


def main() -> int:
    args = parse_args()
    print("Loading hand-sign and pose models...")
    pipeline = CombinedNarutoPipeline(
        hand_model=args.hand_model,
        hand_landmarker=args.hand_landmarker,
        pose_model=args.pose_model,
        hand_confidence=args.hand_confidence,
    )
    if args.self_check:
        print("Self-check passed: hand classifier, hand detector, and pose detector loaded.")
        pipeline.close()
        return 0

    camera = cv2.VideoCapture(args.camera)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not camera.isOpened():
        pipeline.close()
        print(f"Error: camera {args.camera} could not be opened.")
        return 2

    print("Combined camera tester started.")
    print("Attacks use hand signs only:")
    print("  tiger>dragon>horse, bird>ram>rat, dog, monkey>boar>snake, ox>hare")
    print("Body movements are detected separately and never gate attacks.")
    print("Press Q in the camera window to quit.")

    previous = time.perf_counter()
    smoothed_fps = 0.0
    last_attack = ""
    last_attack_time = 0.0
    screenshot_index = 1
    calibration_end = 0.0
    calibration_samples = []
    window_name = "NarutoCV - Combined Camera Test"

    try:
        while camera.isOpened():
            ok, frame = camera.read()
            if not ok:
                print("Camera frame read failed.")
                break
            result = pipeline.process(frame)
            frame = cv2.flip(frame, 1)
            display_result = _mirror_result_for_display(result, frame.shape[1])
            now = time.perf_counter()
            calibrating = calibration_end > now
            if calibrating:
                calibration_samples.append(result.hand.raw)
                pipeline.hand.filter.reset()
                pipeline.attacks.reset()
            elif calibration_end > 0:
                adjustments = pipeline.hand.calibrate_neutral(calibration_samples)
                pipeline.reset()
                calibration_end = 0.0
                calibration_samples = []
                if adjustments:
                    formatted = ", ".join(
                        f"{label}(conf={values[0]:.3f},margin={values[1]:.3f})"
                        for label, values in sorted(adjustments.items())
                    )
                    print(f"Neutral calibration tightened: {formatted}")
                else:
                    print("Neutral calibration complete; no false labels needed adjustment.")
            instant_fps = 1.0 / max(now - previous, 1e-6)
            smoothed_fps = instant_fps if smoothed_fps == 0 else 0.9 * smoothed_fps + 0.1 * instant_fps
            previous = now

            if result.hand.emitted_seal and not calibrating:
                print(
                    f"Seal: {result.hand.emitted_seal} "
                    f"({result.hand.raw.confidence:.1%}, margin={result.hand.raw.margin:.1%})"
                )
            if result.body.emitted_movement and not calibrating:
                print(f"Body movement: {result.body.emitted_movement}")
            if result.queue_update.duplicate_ignored and not calibrating:
                print("Adjacent duplicate seal ignored.")
            if result.queue_update.timeout_cleared and not calibrating:
                print("Seal queue expired; starting a fresh sequence.")
            if result.queue_update.max_length_cleared and not calibrating:
                print("Three seals did not match an attack; queue cleared.")
            if result.queue_update.cooldown_suppressed and not calibrating:
                print("Repeated attack ignored during its cooldown.")
            if result.attack and not calibrating:
                last_attack = result.attack.display_name
                last_attack_time = now
                print(f"ATTACK RECOGNIZED: {last_attack}")
            if last_attack and now - last_attack_time > 4.0:
                last_attack = ""

            _draw_landmarks(frame, display_result)
            _draw_panel(
                frame,
                display_result,
                smoothed_fps,
                last_attack,
                max(0.0, calibration_end - now),
            )
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                pipeline.reset()
                last_attack = ""
                print("Recognition state reset.")
            if key == ord("c"):
                pipeline.reset()
                calibration_samples = []
                calibration_end = time.perf_counter() + 4.0
                print(
                    "Neutral calibration started: stand normally and move through "
                    "non-sign hand poses for four seconds."
                )
            if key == ord("s"):
                output = Path.cwd() / f"combined_tracker_{screenshot_index:03d}.png"
                cv2.imwrite(str(output), frame)
                screenshot_index += 1
                print(f"Screenshot saved: {output}")
    finally:
        camera.release()
        pipeline.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
