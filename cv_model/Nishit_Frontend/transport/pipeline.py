"""The pipeline the transport serves, with one behaviour added: a reset
genuinely forgets the previous person.

**The problem this exists to solve.** MediaPipe's pose landmarker runs in
VIDEO mode with `num_poses=1`. That combination tracks exactly one person and
is deliberately sticky: having found somebody, it follows them with a cheap
landmark pass and only re-runs the expensive detector once tracking fails.
Sticky is the right default for a single player -- it is faster and it stops
the pose flickering between frames.

It is the wrong behaviour when the person changes. With two people in shot the
model picks one and stays with them, so a second person stepping in front of
the camera is simply not seen, however long they wait. Observed directly:
detection worked for the first person and then reported "not detected" for the
next, in the same session.

`CombinedNarutoPipeline.reset()` does not help, because it resets the
consensus filter and the history buffers but never touches the landmarker --
the tracking state lives inside MediaPipe, not in any of the fields reset
clears. So the one operation whose whole purpose is "start again" left the
part that was stuck exactly as it was.

This subclass makes reset mean what it says by rebuilding the landmarker. The
next frame then goes through full detection and picks up whoever is actually
in front of the camera now.

Only the *body* recognizer is rebuilt. The hand recognizer's ONNX session is
expensive to construct and holds no per-person tracking state, so rebuilding
it would cost seconds and buy nothing.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

from cv_model.inference.combined_pipeline import (
    DEFAULT_HAND_LANDMARKER,
    DEFAULT_HAND_MODEL,
    DEFAULT_POSE_MODEL,
    BodyMovementRecognizer,
    CombinedNarutoPipeline,
    FrameResult,
    HandResult,
)
from cv_model.inference.pipeline_config import DEFAULT_PIPELINE_CONFIG, PipelineConfig

logger = logging.getLogger(__name__)


class SessionScopedPipeline(CombinedNarutoPipeline):
    """A `CombinedNarutoPipeline` whose `reset()` also re-detects the player."""

    def __init__(
        self,
        hand_model: Path = DEFAULT_HAND_MODEL,
        hand_landmarker: Path = DEFAULT_HAND_LANDMARKER,
        pose_model: Path = DEFAULT_POSE_MODEL,
        hand_confidence: Optional[float] = None,
        config: PipelineConfig = DEFAULT_PIPELINE_CONFIG,
        body_only: bool = False,
    ) -> None:
        super().__init__(
            hand_model=hand_model,
            hand_landmarker=hand_landmarker,
            pose_model=pose_model,
            hand_confidence=hand_confidence,
            config=config,
        )
        # The base class does not keep this, and the rebuild needs it.
        self._pose_model_path = Path(pose_model)
        self._body_only = body_only
        self._idle_hand: Optional[HandResult] = None
        if body_only:
            self._idle_hand = self._capture_idle_hand_result()

    def _capture_idle_hand_result(self) -> HandResult:
        """One real 'no hands here' result, to stand in for every frame.

        Produced by running the genuine recognizer once over a blank frame
        rather than by hand-building a `HandResult`. The serialiser validates
        this structure strictly, and a fabricated one would be a second
        definition of "no hands" to keep in step with the first.
        """
        blank = np.zeros((360, 640, 3), dtype=np.uint8)
        return self.hand.process(blank, 0.0)

    def process(self, frame_bgr: np.ndarray, timestamp: Optional[float] = None) -> FrameResult:
        """Body-only fast path, otherwise the base behaviour unchanged.

        Hand recognition costs about 54% of the per-frame budget -- MediaPipe
        Hands plus an ONNX classifier, measured at 26fps combined against
        64fps for the body alone. The browser game reads only body movement;
        hand signs are out of scope for that integration and `scene.html`
        never touches `output.hand`. Spending more than half the frame budget
        on output nothing consumes is latency the player feels directly, since
        every millisecond here is a millisecond the newest sample is older.

        Skipping it more than doubles the frame rate, which is the cheapest
        available defence against samples ever going stale.
        """
        if not self._body_only or self._idle_hand is None:
            return super().process(frame_bgr, timestamp)

        started = time.perf_counter()
        now = time.monotonic() if timestamp is None else timestamp
        body_result = self.body.process(frame_bgr, now)

        attack_started = time.perf_counter()
        # Fed None rather than skipped: the attack queue has timeouts of its
        # own, and starving it of updates would leave stale state behind if
        # hand input is ever turned back on mid-session.
        queue_update = self.attacks.update_detailed(None, now)
        attack_finished = time.perf_counter()

        processing_ms = (time.perf_counter() - started) * 1000.0
        attack_queue_ms = (attack_finished - attack_started) * 1000.0
        timings_ms = {
            **body_result.timings_ms,
            "hand_total_ms": 0.0,
            "attack_queue_ms": attack_queue_ms,
            "pipeline_overhead_ms": max(
                0.0,
                processing_ms
                - body_result.timings_ms.get("body_total_ms", 0.0)
                - attack_queue_ms,
            ),
            "recognition_total_ms": processing_ms,
        }
        return FrameResult(
            hand=self._idle_hand,
            body=body_result,
            attack=queue_update.attack,
            seal_history=queue_update.queue,
            queue_update=queue_update,
            processing_ms=processing_ms,
            timings_ms=timings_ms,
        )

    def reset(self) -> None:
        super().reset()
        self._rebuild_body()

    def _rebuild_body(self) -> None:
        """Swap in a landmarker with no memory of the last person.

        Costs roughly a tenth of a second (the model file is already on disk),
        and only happens on an explicit reset -- a client connecting or
        disconnecting, or the player asking to recalibrate -- so it never lands
        in the per-frame path.
        """
        previous = self.body
        try:
            self.body = BodyMovementRecognizer(self._pose_model_path, self.config.body)
        except Exception:
            # A failed rebuild must not take the session down: the old
            # recognizer still works, it is merely still tracking the previous
            # person, which is strictly better than no recognition at all.
            logger.exception("could not rebuild the pose landmarker; keeping the existing one")
            return
        try:
            previous.close()
        except Exception:  # pragma: no cover - close is best-effort
            logger.warning("failed to close the previous pose landmarker", exc_info=True)


def session_scoped_pipeline_factory(*, body_only: bool = False):
    """Factory in the shape `PipelineRuntimeController` expects.

    `body_only` skips hand recognition, which the browser game does not use
    and which costs more than half the frame budget. Off by default so that
    anything else pointed at this transport -- the diagnostics page, for one
    -- keeps working exactly as before.
    """

    def factory() -> SessionScopedPipeline:
        return SessionScopedPipeline(body_only=body_only)

    return factory
