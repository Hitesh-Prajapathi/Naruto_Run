#!/usr/bin/env python3
"""Measure what your body actually produces -- Feature Brief 05 tuning aid.

The body thresholds are plain constants, not a trained model, so tuning them
is guesswork unless somebody measures a real person. This is that measurement.
It runs the *same* `BodyMovementRecognizer` the game runs, on the *same*
capture size the browser sends (640x360), and reports the numbers the game's
detectors actually compare against thresholds.

Everything here is measured from the two SHOULDER landmarks only, matching the
game. The hips are deliberately ignored: a webcam at desk height frames a
player from the chest up, so the hips fall outside the frame and MediaPipe
estimates rather than measures them.

Why it exists: thresholds were twice tuned against synthetic fixtures no real
body could produce, so the tests agreed with themselves while the game ignored
real movement. Run this before changing a threshold.

Usage (from the repo root):

    cv_model\\Nishit_Frontend\\venv\\Scripts\\python.exe ^
        cv_model\\Nishit_Frontend\\diagnose_body.py

Then, when it says so:
  1. stand still for a few seconds  (captures your baseline)
  2. lean left, back to centre, lean right, back to centre
  3. jump three times
  4. press q

Watch the TILT column: that is the number the game thresholds against.

It prints the peak values you produced and the thresholds those peaks imply.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections import deque
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from cv_model.inference.combined_pipeline import BodyMovementRecognizer  # noqa: E402

# Must match CV_CAPTURE_WIDTH / CV_CAPTURE_HEIGHT in `cvInputConfig.ts`, since
# the aspect correction below depends on the frame's shape.
CAPTURE_W = 640
CAPTURE_H = 360

LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--no-window", action="store_true", help="console output only")
    args = parser.parse_args()

    capture = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_W)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_H)
    if not capture.isOpened():
        print(f"Could not open camera {args.camera}.")
        return 1

    recognizer = BodyMovementRecognizer()

    hip_history: deque[float] = deque(maxlen=45)
    torso_history: deque[float] = deque(maxlen=45)
    previous_rise = 0.0
    previous_time: float | None = None

    peak_lean_left = 0.0
    peak_lean_right = 0.0
    peak_rise = 0.0
    peak_velocity = 0.0
    label_counts: dict[str, int] = {}
    emitted_counts: dict[str, int] = {}
    frames = 0
    started = time.time()

    print("Recording. Stand still, then lean left/right, then jump. 'q' to stop.\n")

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames += 1
        now = time.time()
        result = recognizer.process(frame, now)

        label_counts[result.stable_label] = label_counts.get(result.stable_label, 0) + 1
        if result.emitted_movement:
            emitted_counts[result.emitted_movement] = (
                emitted_counts.get(result.emitted_movement, 0) + 1
            )

        line = f"[{frames:5d}] {result.stable_label:<13}"
        if result.landmarks:
            pts = result.landmarks
            # Upper body only, exactly as poseAdapter.ts does it: both axes
            # converted to units of frame HEIGHT first, because MediaPipe
            # normalises x by width and y by height and mixing the two is a
            # 1.78x error on a 16:9 frame.
            aspect = CAPTURE_W / CAPTURE_H
            lx, ly = pts[LEFT_SHOULDER][0] * aspect, pts[LEFT_SHOULDER][1]
            rx, ry = pts[RIGHT_SHOULDER][0] * aspect, pts[RIGHT_SHOULDER][1]
            dx, dy = lx - rx, ly - ry
            # Full shoulder-to-shoulder length, so leaning does not make the
            # player appear to shrink.
            torso = float(np.hypot(dx, dy))
            shoulder_y = (ly + ry) / 2
            hip_history.append(shoulder_y)
            if torso > 1e-4:
                torso_history.append(torso)

            lean_x = result.metrics.get("lean_x", 0.0)

            # What the game now measures: the sine of the shoulder line's
            # roll. `naive` is the old hip-based metric, kept alongside so the
            # two can be compared in one run.
            naive = lean_x * aspect / torso if torso > 1e-4 else 0.0
            corrected = dy / torso if torso > 1e-4 else 0.0
            degrees = np.degrees(np.arcsin(np.clip(corrected, -1.0, 1.0)))

            baseline_torso = statistics.median(torso_history) if torso_history else torso
            rise = 0.0
            if len(hip_history) >= 15:
                rise = (statistics.median(hip_history) - shoulder_y) / max(baseline_torso, 1e-4)
            dt = (now - previous_time) if previous_time else 0.0
            velocity = (rise - previous_rise) / dt if dt > 1e-3 else 0.0
            previous_rise, previous_time = rise, now

            peak_lean_left = min(peak_lean_left, corrected)
            peak_lean_right = max(peak_lean_right, corrected)
            peak_rise = max(peak_rise, rise)
            peak_velocity = max(peak_velocity, velocity)

            line += (
                f" shoulders={torso:.3f} hipLean={naive:+.3f}"
                f" TILT={corrected:+.3f} ({degrees:+5.1f}deg)"
                f" rise={rise:+.3f} vel={velocity:+6.2f}"
            )
            if not args.no_window:
                for x, y in pts:
                    cv2.circle(frame, (int(x * frame.shape[1]), int(y * frame.shape[0])), 2, (0, 255, 0), -1)
        else:
            line += "  no pose"
        print(line)

        if not args.no_window:
            cv2.putText(frame, result.stable_label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("diagnose_body (q to quit)", cv2.flip(frame, 1))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    capture.release()
    cv2.destroyAllWindows()
    recognizer.close()

    elapsed = max(time.time() - started, 1e-6)
    print("\n" + "=" * 68)
    print(f"frames {frames} in {elapsed:.1f}s = {frames / elapsed:.1f} fps")
    print(f"stable labels seen : {label_counts}")
    print(f"label EDGES emitted: {emitted_counts or 'NONE -- the jump path cannot fire'}")
    print("-" * 68)
    print(f"peak shoulder tilt LEFT    : {peak_lean_left:+.3f}")
    print(f"peak shoulder tilt RIGHT   : {peak_lean_right:+.3f}")
    print(f"peak jump rise (shoulders) : {peak_rise:+.3f}")
    print(f"peak jump velocity         : {peak_velocity:+.2f} widths/s")
    lo, hi = abs(peak_lean_left), abs(peak_lean_right)
    if min(lo, hi) > 0 and max(lo, hi) / min(lo, hi) > 1.4:
        print("  NOTE: your two sides differ by more than 40%. Either you lean")
        print("  harder one way, or the camera is off to one side.")
    print("-" * 68)
    reachable = min(abs(peak_lean_left), abs(peak_lean_right))
    if reachable > 0:
        print(f"suggested leanEnter ~ {reachable * 0.55:.2f}   leanExit ~ {reachable * 0.30:.2f}")
    if peak_rise > 0:
        print(f"suggested jumpHeightFraction ~ {peak_rise * 0.45:.2f}"
              f"   jumpMinVelocity ~ {peak_velocity * 0.30:.2f}")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
