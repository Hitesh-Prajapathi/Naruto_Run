from __future__ import annotations

import sys
import unittest
from pathlib import Path


CV_MODEL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CV_MODEL_ROOT))

from inference.combined_pipeline import (  # noqa: E402
    AttackEvent,
    BodyResult,
    Classification,
    FrameResult,
    HandResult,
    QueueUpdate,
)
from run_combined_camera import _mirror_result_for_display  # noqa: E402


class DisplayMirroringTests(unittest.TestCase):
    def test_only_display_geometry_is_reflected(self) -> None:
        attack = AttackEvent("test", "TEST", 1.0)
        queue = QueueUpdate(attack, "bird", False, False, False, False, ("bird",))
        hand = HandResult(
            raw=Classification("bird", 0.9, "zero", 0.1),
            accepted_label="bird",
            rejection_reason=None,
            stable_label="bird",
            emitted_seal="bird",
            classification_bbox=(10, 5, 30, 25),
            hand_bbox=(15, 8, 35, 28),
            landmarks=(((0.10, 0.20), (0.35, 0.40)),),
        )
        body = BodyResult(
            raw_label="bending_left",
            stable_label="bending_left",
            emitted_movement="bending_left",
            metrics={"lean_x": -0.1},
            landmarks=((0.25, 0.30),),
        )
        result = FrameResult(hand, body, attack, ("bird",), queue, 12.0)

        mirrored = _mirror_result_for_display(result, width=100)

        self.assertEqual(mirrored.hand.classification_bbox, (70, 5, 90, 25))
        self.assertEqual(mirrored.hand.hand_bbox, (65, 8, 85, 28))
        self.assertEqual(mirrored.hand.landmarks, (((0.90, 0.20), (0.65, 0.40)),))
        self.assertEqual(mirrored.body.landmarks, ((0.75, 0.30),))
        self.assertEqual(mirrored.hand.stable_label, result.hand.stable_label)
        self.assertEqual(mirrored.body.stable_label, result.body.stable_label)
        self.assertIs(mirrored.attack, result.attack)
        self.assertIs(mirrored.queue_update, result.queue_update)


if __name__ == "__main__":
    unittest.main()
