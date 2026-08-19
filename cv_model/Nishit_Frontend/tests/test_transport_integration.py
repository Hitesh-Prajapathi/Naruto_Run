"""Integration tests for the aiohttp transport server.

These drive the real HTTP + WebSocket boundary end to end (aiohttp's
TestClient against an in-process TestServer) using a fake recognition
pipeline, so no camera, model weights, MediaPipe, or ONNX Runtime inference
is required -- only the message contract and scheduling glue are under test.
The real CombinedNarutoPipeline dataclasses are still imported (see
combined_pipeline module docstring), which is why mediapipe/onnxruntime are
listed in requirements-transport.txt even though this file never runs them.
"""

from __future__ import annotations

import asyncio
import struct
import sys
import unittest
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2
import numpy as np
from aiohttp import WSMsgType
from aiohttp.test_utils import AioHTTPTestCase

from cv_model.inference.combined_pipeline import (
    AttackEvent,
    BodyResult,
    Classification,
    FrameResult,
    HandResult,
    QueueUpdate,
)
from cv_model.inference.runtime import RuntimeConfig

from cv_model.Nishit_Frontend.transport.protocol import (
    FRAME_HEADER,
    FRAME_MAGIC,
    MAX_FRAME_PAYLOAD_BYTES,
    FrameEnvelope,
    encode_frame,
)
from cv_model.Nishit_Frontend.transport.server import create_app

RECEIVE_TIMEOUT_SECONDS = 3.0


def _pack_raw_frame_bytes(
    *,
    version: int = 1,
    sequence: int = 0,
    captured_at_ms: int = 0,
    width: int = 32,
    height: int = 16,
    payload: bytes = b"",
    magic: bytes = FRAME_MAGIC,
    declared_payload_len: Optional[int] = None,
) -> bytes:
    """Hand-pack header bytes without FrameEnvelope.validate(), so tests can
    construct exactly the malformed/oversized wire data the server must
    reject -- encode_frame() would refuse to build these on purpose."""
    header = FRAME_HEADER.pack(
        magic,
        version,
        sequence,
        captured_at_ms,
        width,
        height,
        len(payload) if declared_payload_len is None else declared_payload_len,
    )
    return header + payload


def _encode_test_jpeg_frame(*, sequence: int, captured_at_ms: int = 1000) -> bytes:
    image = np.zeros((16, 32, 3), dtype=np.uint8)
    ok, buffer = cv2.imencode(".jpg", image)
    assert ok
    payload = buffer.tobytes()
    envelope = FrameEnvelope(
        version=1,
        sequence=sequence,
        captured_at_ms=captured_at_ms,
        width=32,
        height=16,
        payload=payload,
    )
    return encode_frame(envelope)


def _result(
    *, accepted_seal: Optional[str] = None, attack: Optional[AttackEvent] = None
) -> FrameResult:
    hand = HandResult(
        raw=Classification("dog", 0.90, "zero", 0.05),
        accepted_label=accepted_seal,
        rejection_reason=None,
        stable_label=accepted_seal or "zero",
        emitted_seal=accepted_seal,
        classification_bbox=(0, 0, 2, 2),
        hand_bbox=None,
    )
    body = BodyResult(
        raw_label="idle",
        stable_label="idle",
        emitted_movement=None,
        metrics={},
    )
    queue = QueueUpdate(
        attack=attack,
        accepted_seal=accepted_seal,
        duplicate_ignored=False,
        timeout_cleared=False,
        max_length_cleared=False,
        cooldown_suppressed=False,
        queue=(),
    )
    return FrameResult(
        hand=hand,
        body=body,
        attack=attack,
        seal_history=(),
        queue_update=queue,
        processing_ms=1.0,
    )


class _ResetCounter:
    def __init__(self) -> None:
        self.reset_count = 0

    def reset(self) -> None:
        self.reset_count += 1


class _FakeHandControl:
    def __init__(self) -> None:
        self.filter = _ResetCounter()

    def calibrate_neutral(self, samples):
        return {}


class _FakePipeline:
    """Minimal stand-in satisfying runtime.RecognitionPipeline without
    MediaPipe/ONNX. Tests configure `next_result` before submitting a frame
    so the transport layer's handling of a specific FrameResult is
    deterministic and observable."""

    def __init__(self) -> None:
        self.hand = _FakeHandControl()
        self.attacks = _ResetCounter()
        self.next_result: FrameResult = _result()
        self.process_calls = 0
        self.reset_count = 0
        self.close_count = 0

    def process(self, _frame: np.ndarray, _timestamp: float) -> FrameResult:
        self.process_calls += 1
        return self.next_result

    def reset(self) -> None:
        self.reset_count += 1

    def close(self) -> None:
        self.close_count += 1


class TransportServerTestCase(AioHTTPTestCase):
    async def get_application(self):
        self.fake_pipeline = _FakePipeline()
        return create_app(
            pipeline_factory=lambda: self.fake_pipeline,
            runtime_config=RuntimeConfig(default_calibration_seconds=1.0),
        )

    async def _connected_client(self, *, protocol_version: str = "1.0.0"):
        ws = await self.client.ws_connect("/ws")
        await ws.send_json({"type": "client_hello", "protocol_version": protocol_version})
        ready = await asyncio.wait_for(ws.receive_json(), RECEIVE_TIMEOUT_SECONDS)
        self.assertEqual(ready["type"], "server_ready")
        return ws

    async def _receive_typed(self, ws, expected_type: str, *, attempts: int = 5) -> dict:
        """Drain messages until one of `expected_type` arrives.

        state_snapshot/pipeline_event ordering vs. the ack for a control
        command is not part of the contract this test asserts, so tests that
        only care about one message type skip over the others.
        """
        for _ in range(attempts):
            message = await asyncio.wait_for(ws.receive_json(), RECEIVE_TIMEOUT_SECONDS)
            if message.get("type") == expected_type:
                return message
        self.fail(f"did not receive a {expected_type!r} message within {attempts} messages")

    # --- health --------------------------------------------------------

    async def test_health_reports_ready_after_startup(self) -> None:
        response = await self.client.get("/health")
        self.assertEqual(response.status, 200)
        body = await response.json()
        self.assertEqual(body["state"], "ready")
        self.assertTrue(body["runtime_started"])

    # --- handshake -------------------------------------------------------

    async def test_handshake_returns_server_ready_with_supported_controls(self) -> None:
        ws = await self._connected_client()
        await ws.close()

    async def test_handshake_rejects_a_non_hello_first_message(self) -> None:
        ws = await self.client.ws_connect("/ws")
        await ws.send_json({"type": "control", "command": "ping"})

        error = await asyncio.wait_for(ws.receive_json(), RECEIVE_TIMEOUT_SECONDS)
        self.assertEqual(error["type"], "error")
        self.assertEqual(error["code"], "handshake_failed")

        closing = await asyncio.wait_for(ws.receive(), RECEIVE_TIMEOUT_SECONDS)
        self.assertEqual(closing.type, WSMsgType.CLOSE)

    async def test_second_client_is_rejected_while_one_controls(self) -> None:
        first = await self._connected_client()
        second = await self.client.ws_connect("/ws")

        error = await asyncio.wait_for(second.receive_json(), RECEIVE_TIMEOUT_SECONDS)
        self.assertEqual(error["code"], "controller_already_connected")
        closing = await asyncio.wait_for(second.receive(), RECEIVE_TIMEOUT_SECONDS)
        self.assertEqual(closing.type, WSMsgType.CLOSE)

        await first.close()

    # --- frames ------------------------------------------------------------

    async def test_valid_frame_produces_a_state_snapshot(self) -> None:
        ws = await self._connected_client()
        self.fake_pipeline.next_result = _result()

        await ws.send_bytes(_encode_test_jpeg_frame(sequence=1))
        snapshot = await self._receive_typed(ws, "state_snapshot")

        self.assertEqual(snapshot["output"]["hand"]["stable_label"], "zero")
        await ws.close()

    async def test_accepted_seal_emits_a_hand_seal_event(self) -> None:
        ws = await self._connected_client()
        self.fake_pipeline.next_result = _result(accepted_seal="tiger")

        await ws.send_bytes(_encode_test_jpeg_frame(sequence=1))
        event_message = await self._receive_typed(ws, "pipeline_event")

        self.assertEqual(event_message["event"]["event_type"], "HAND_SEAL")
        self.assertEqual(event_message["event"]["payload"]["seal"], "tiger")
        await ws.close()

    async def test_attack_triggered_event_carries_the_display_name(self) -> None:
        ws = await self._connected_client()
        attack = AttackEvent("homura", "FIRE ATTACK", 2.0)
        self.fake_pipeline.next_result = _result(accepted_seal="horse", attack=attack)

        await ws.send_bytes(_encode_test_jpeg_frame(sequence=1))
        attack_event = None
        for _ in range(5):
            message = await asyncio.wait_for(ws.receive_json(), RECEIVE_TIMEOUT_SECONDS)
            if message.get("type") == "pipeline_event" and message["event"]["event_type"] == "ATTACK_TRIGGERED":
                attack_event = message
                break
        self.assertIsNotNone(attack_event, "expected an ATTACK_TRIGGERED event")
        self.assertEqual(attack_event["event"]["payload"]["name"], "homura")
        self.assertEqual(attack_event["event"]["payload"]["display_name"], "FIRE ATTACK")
        await ws.close()

    async def test_malformed_frame_receives_an_invalid_frame_error(self) -> None:
        ws = await self._connected_client()
        await ws.send_bytes(b"not a real frame envelope at all")

        error = await self._receive_typed(ws, "error")
        self.assertEqual(error["code"], "invalid_frame")
        await ws.close()

    async def test_oversized_payload_is_rejected_before_jpeg_decode(self) -> None:
        ws = await self._connected_client()
        oversized = _pack_raw_frame_bytes(
            sequence=1, payload=b"\x00" * (MAX_FRAME_PAYLOAD_BYTES + 1)
        )

        await ws.send_bytes(oversized)
        error = await self._receive_typed(ws, "error")
        self.assertEqual(error["code"], "invalid_frame")
        await ws.close()

    async def test_stale_frame_sequence_is_rejected(self) -> None:
        ws = await self._connected_client()

        await ws.send_bytes(_encode_test_jpeg_frame(sequence=5))
        await self._receive_typed(ws, "state_snapshot")

        await ws.send_bytes(_encode_test_jpeg_frame(sequence=3))
        error = await self._receive_typed(ws, "error")
        self.assertEqual(error["code"], "invalid_frame")
        await ws.close()

    # --- controls ------------------------------------------------------

    async def test_ping_control_is_acknowledged(self) -> None:
        ws = await self._connected_client()
        await ws.send_json({"type": "control", "command": "ping", "request_id": "p1"})

        ack = await self._receive_typed(ws, "ack")
        self.assertEqual(ack, {"type": "ack", "command": "ping", "ok": True, "request_id": "p1"})
        await ws.close()

    async def test_reset_control_resets_the_pipeline_and_is_acknowledged(self) -> None:
        ws = await self._connected_client()
        before = self.fake_pipeline.reset_count

        await ws.send_json({"type": "control", "command": "reset", "request_id": "r1"})
        ack = await self._receive_typed(ws, "ack")

        self.assertTrue(ack["ok"])
        self.assertGreater(self.fake_pipeline.reset_count, before)
        await ws.close()

    async def test_unknown_control_command_receives_an_error(self) -> None:
        ws = await self._connected_client()
        await ws.send_json({"type": "control", "command": "fly"})

        error = await self._receive_typed(ws, "error")
        self.assertEqual(error["code"], "invalid_control")
        await ws.close()


if __name__ == "__main__":
    unittest.main()
