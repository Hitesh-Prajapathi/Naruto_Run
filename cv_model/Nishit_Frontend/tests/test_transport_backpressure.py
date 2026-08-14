from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cv_model.Nishit_Frontend.transport.connection import (
    BoundedEventQueue,
    ClientSession,
    EventQueueOverflow,
    FrameSequenceGuard,
    LatestSnapshotSlot,
    StaleFrameError,
)


class FrameSequenceGuardTests(unittest.TestCase):
    def test_accepts_strictly_increasing_sequences(self) -> None:
        guard = FrameSequenceGuard()
        guard.accept(1)
        guard.accept(2)
        guard.accept(10)  # gaps (dropped frames) are fine, only order matters
        self.assertEqual(guard.last_sequence, 10)

    def test_rejects_a_duplicate_sequence(self) -> None:
        guard = FrameSequenceGuard()
        guard.accept(5)
        with self.assertRaises(StaleFrameError):
            guard.accept(5)

    def test_rejects_an_out_of_order_sequence(self) -> None:
        guard = FrameSequenceGuard()
        guard.accept(5)
        with self.assertRaises(StaleFrameError):
            guard.accept(3)

    def test_reset_allows_any_sequence_to_start_the_stream_again(self) -> None:
        guard = FrameSequenceGuard()
        guard.accept(100)
        guard.reset()
        guard.accept(0)  # would have raised before reset()
        self.assertEqual(guard.last_sequence, 0)


class BoundedEventQueueTests(unittest.TestCase):
    def test_preserves_push_order_on_pop_all(self) -> None:
        queue = BoundedEventQueue(max_size=8)
        queue.push({"n": 1})
        queue.push({"n": 2})
        queue.push({"n": 3})

        self.assertEqual(queue.pop_all(), [{"n": 1}, {"n": 2}, {"n": 3}])

    def test_pop_all_drains_the_queue(self) -> None:
        queue = BoundedEventQueue(max_size=8)
        queue.push({"n": 1})
        queue.pop_all()

        self.assertEqual(len(queue), 0)
        self.assertEqual(queue.pop_all(), [])

    def test_raises_instead_of_dropping_or_reordering_when_full(self) -> None:
        queue = BoundedEventQueue(max_size=2)
        queue.push({"n": 1})
        queue.push({"n": 2})

        with self.assertRaises(EventQueueOverflow):
            queue.push({"n": 3})

        # The two buffered events must still be exactly what was pushed, in
        # order -- an overflowing client is disconnected (server.py), never
        # silently truncated or reordered.
        self.assertEqual(queue.pop_all(), [{"n": 1}, {"n": 2}])

    def test_rejects_a_non_positive_max_size(self) -> None:
        with self.assertRaises(ValueError):
            BoundedEventQueue(max_size=0)


class LatestSnapshotSlotTests(unittest.TestCase):
    def test_take_returns_none_when_nothing_is_pending(self) -> None:
        slot = LatestSnapshotSlot()
        self.assertIsNone(slot.take())
        self.assertFalse(slot.has_pending)

    def test_replace_keeps_only_the_newest_snapshot(self) -> None:
        slot = LatestSnapshotSlot()
        slot.replace({"frame_id": 1})
        slot.replace({"frame_id": 2})
        slot.replace({"frame_id": 3})

        self.assertEqual(slot.take(), {"frame_id": 3})

    def test_take_clears_the_pending_flag(self) -> None:
        slot = LatestSnapshotSlot()
        slot.replace({"frame_id": 1})
        slot.take()

        self.assertFalse(slot.has_pending)
        self.assertIsNone(slot.take())


class ClientSessionTests(unittest.TestCase):
    def test_starts_with_independent_fresh_guards(self) -> None:
        session = ClientSession(event_queue_size=4)
        self.assertIsNone(session.frame_guard.last_sequence)
        self.assertEqual(len(session.events), 0)
        self.assertFalse(session.snapshot.has_pending)
        self.assertFalse(session.is_controller)

    def test_reset_stream_state_only_clears_the_frame_guard(self) -> None:
        session = ClientSession(event_queue_size=4)
        session.frame_guard.accept(7)
        session.events.push({"n": 1})
        session.snapshot.replace({"frame_id": 1})

        session.reset_stream_state()

        self.assertIsNone(session.frame_guard.last_sequence)
        # Event/snapshot buffers are drained by the sender loop, not by a
        # stream reset -- they are unrelated to frame sequencing.
        self.assertEqual(len(session.events), 1)
        self.assertTrue(session.snapshot.has_pending)


if __name__ == "__main__":
    unittest.main()
