"""Phase 6 hardening: origin list, limits, handshake, metrics, health, shutdown.

Drives the real aiohttp app through aiohttp's test utilities with a fake
recognition pipeline (the helpers are shared with test_transport_integration.py),
so no camera or model weights are needed.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import importlib.util
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import textwrap
import time
import unittest
import urllib.request
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
from aiohttp import ClientSession, WSMsgType
from aiohttp.client_exceptions import WSServerHandshakeError
from aiohttp.test_utils import AioHTTPTestCase

from narutocv.inference.runtime import RuntimeConfig, RuntimeState
from narutocv.transport import frame_decoder
from narutocv.transport.config import DEFAULT_ALLOWED_ORIGINS, TransportConfig, parse_origins
from narutocv.transport.connection import TokenBucket
from narutocv.transport.frame_decoder import jpeg_dimensions
from narutocv.transport.logging_config import JsonFormatter
from narutocv.transport.metrics import MetricsRegistry
from narutocv.transport.model_info import ModelBundle, bundle_from_entries, load_model_bundle
from narutocv.transport.protocol import (
    FRAME_HEADER_SIZE,
    MAX_FRAME_PAYLOAD_BYTES,
    PROTOCOL_VERSION,
    ProtocolError,
)
from narutocv.transport.server import (
    CLOSE_CODE_GOING_AWAY,
    CLOSE_CODE_IDLE_TIMEOUT,
    CLOSE_CODE_MODEL_MISMATCH,
    CLOSE_CODE_PROTOCOL_ERROR,
    CLOSE_CODE_VERSION_MISMATCH,
    TRANSPORT_SERVER_KEY,
    create_app,
)

# Reuse the fake pipeline and frame builders from the integration tests
# (pytest runs with --import-mode=importlib, so sibling modules are not
# importable by name).
_spec = importlib.util.spec_from_file_location(
    "_ncv_transport_integration", Path(__file__).with_name("test_transport_integration.py")
)
_integration = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_integration)
_FakePipeline = _integration._FakePipeline
_encode_test_jpeg_frame = _integration._encode_test_jpeg_frame
_pack_raw_frame_bytes = _integration._pack_raw_frame_bytes

TIMEOUT = 3.0
TEST_BUNDLE = ModelBundle(id="narutocv-testbundle", sha256="a" * 64, files=())


def _jpeg_bytes(width: int = 32, height: int = 16) -> bytes:
    ok, buffer = cv2.imencode(".jpg", np.zeros((height, width, 3), dtype=np.uint8))
    assert ok
    return buffer.tobytes()


class HardeningTestCase(AioHTTPTestCase):
    """Base: subclasses override CONFIG; each test gets a fresh fake pipeline."""

    CONFIG = TransportConfig()

    async def get_application(self):
        self.fake_pipeline = _FakePipeline()
        return create_app(
            pipeline_factory=lambda: self.fake_pipeline,
            runtime_config=RuntimeConfig(default_calibration_seconds=1.0),
            config=self.CONFIG,
            model_bundle=TEST_BUNDLE,
        )

    @property
    def transport(self):
        return self.app[TRANSPORT_SERVER_KEY]

    async def hello(self, ws, **fields):
        payload = {"type": "client_hello", "protocol_version": "1.0.0"}
        payload.update(fields)
        await ws.send_json(payload)

    async def connect(self, headers=None, **hello_fields):
        ws = await self.client.ws_connect("/ws", headers=headers)
        await self.hello(ws, **hello_fields)
        ready = await asyncio.wait_for(ws.receive_json(), TIMEOUT)
        self.assertEqual(ready["type"], "server_ready", ready)
        return ws, ready

    async def metrics_text(self) -> str:
        response = await self.client.get("/metrics")
        self.assertEqual(response.status, 200)
        return await response.text()

    async def metric(self, name: str) -> float:
        """Value of one sample line, e.g. 'narutocv_frames_dropped_total{reason="x"}'."""
        for line in (await self.metrics_text()).splitlines():
            if line.startswith(name + " "):
                return float(line.rsplit(" ", 1)[1])
        raise AssertionError(f"metric {name} not exported")

    async def wait_for_metric(self, name: str, at_least: float) -> float:
        deadline = time.monotonic() + TIMEOUT
        value = 0.0
        while time.monotonic() < deadline:
            value = await self.metric(name)
            if value >= at_least:
                return value
            await asyncio.sleep(0.02)
        self.fail(f"{name} is {value}, expected at least {at_least}")

    async def receive_close(self, ws):
        """Read until the socket reports a close; returns the close code."""
        for _ in range(20):
            msg = await asyncio.wait_for(ws.receive(), TIMEOUT)
            if msg.type == WSMsgType.CLOSE:
                return msg.data  # the code the server sent (ws.close_code may become 1006)
            if msg.type in (WSMsgType.CLOSING, WSMsgType.CLOSED, WSMsgType.ERROR):
                return ws.close_code
        self.fail("socket did not close")


# --- origin allow-list -------------------------------------------------------


class OriginPolicyTests(HardeningTestCase):
    async def test_listed_origin_is_accepted(self) -> None:
        ws, _ = await self.connect(headers={"Origin": "http://127.0.0.1:5173"})
        await ws.close()

    async def test_localhost_dev_origin_is_accepted(self) -> None:
        ws, _ = await self.connect(headers={"Origin": "http://localhost:5173"})
        await ws.close()

    async def test_foreign_origin_gets_403(self) -> None:
        with self.assertRaises(WSServerHandshakeError) as caught:
            await self.client.ws_connect("/ws", headers={"Origin": "https://evil.example"})
        self.assertEqual(caught.exception.status, 403)
        self.assertEqual(
            await self.metric('narutocv_connections_rejected_total{reason="origin"}'), 1
        )

    async def test_literal_null_origin_gets_403(self) -> None:
        # Sandboxed iframes and file:// pages send "Origin: null".
        with self.assertRaises(WSServerHandshakeError) as caught:
            await self.client.ws_connect("/ws", headers={"Origin": "null"})
        self.assertEqual(caught.exception.status, 403)

    async def test_trailing_slash_does_not_bypass_or_break_matching(self) -> None:
        ws, _ = await self.connect(headers={"Origin": "http://localhost:5173/"})
        await ws.close()

    async def test_missing_origin_is_allowed_by_default(self) -> None:
        # Only browsers send Origin; a missing header is not a hijacking vector.
        ws, _ = await self.connect()
        await ws.close()


class RequireOriginTests(HardeningTestCase):
    CONFIG = TransportConfig(require_origin=True)

    async def test_missing_origin_gets_403_when_required(self) -> None:
        with self.assertRaises(WSServerHandshakeError) as caught:
            await self.client.ws_connect("/ws")
        self.assertEqual(caught.exception.status, 403)


class AnyOriginTests(HardeningTestCase):
    CONFIG = TransportConfig(allowed_origins=None)

    async def test_wildcard_accepts_any_origin(self) -> None:
        ws, _ = await self.connect(headers={"Origin": "https://anything.example"})
        await ws.close()


class OriginConfigParsingTests(unittest.TestCase):
    def test_default_is_local_dev_only(self) -> None:
        self.assertEqual(parse_origins(None), DEFAULT_ALLOWED_ORIGINS)
        self.assertEqual(parse_origins("  "), DEFAULT_ALLOWED_ORIGINS)
        for origin in DEFAULT_ALLOWED_ORIGINS:
            self.assertRegex(origin, r"^http://(127\.0\.0\.1|localhost):\d+$")

    def test_env_list_is_parsed_and_normalised(self) -> None:
        self.assertEqual(
            parse_origins("https://a.example, https://b.example/"),
            frozenset({"https://a.example", "https://b.example"}),
        )

    def test_star_means_any(self) -> None:
        self.assertIsNone(parse_origins("https://a.example,*"))

    def test_from_env_reads_the_documented_variables(self) -> None:
        config = TransportConfig.from_env(
            {
                "NARUTOCV_ALLOWED_ORIGINS": "https://game.example",
                "NARUTOCV_MAX_SESSIONS": "3",
                "NARUTOCV_MAX_FPS": "45",
                "NARUTOCV_IDLE_TIMEOUT": "12",
                "NARUTOCV_REQUIRE_ORIGIN": "true",
            }
        )
        self.assertEqual(config.allowed_origins, frozenset({"https://game.example"}))
        self.assertEqual(config.max_sessions, 3)
        self.assertEqual(config.max_fps, 45)
        self.assertEqual(config.idle_timeout_s, 12)
        self.assertTrue(config.require_origin)

    def test_loopback_is_the_default_bind_address(self) -> None:
        from narutocv.transport.__main__ import parse_args

        args = parse_args([], env={})
        self.assertEqual(args.host, "127.0.0.1")


# --- size limits -------------------------------------------------------------


class SizeLimitTests(HardeningTestCase):
    async def test_oversize_message_is_closed_with_1009(self) -> None:
        ws, _ = await self.connect()
        too_big = b"\x00" * (FRAME_HEADER_SIZE + MAX_FRAME_PAYLOAD_BYTES + 128 * 1024)
        self.assertGreater(len(too_big), self.CONFIG.max_ws_message_bytes)

        with contextlib.suppress(ConnectionError):
            await ws.send_bytes(too_big)

        # The server cuts the socket mid-upload; the client may see the 1009 frame or
        # just the dropped connection (1006). The server-side metric below is the proof.
        self.assertIn(await self.receive_close(ws), (1009, 1006))
        await self.wait_for_metric('narutocv_connections_closed_total{reason="message_too_big"}', 1)

    async def test_largest_legitimate_frame_still_fits(self) -> None:
        # Exactly at the payload ceiling the header check, not the socket, answers.
        ws, _ = await self.connect()
        await ws.send_bytes(
            _pack_raw_frame_bytes(sequence=1, payload=b"\x00" * MAX_FRAME_PAYLOAD_BYTES)
        )
        reply = await asyncio.wait_for(ws.receive_json(), TIMEOUT)
        self.assertEqual(reply["code"], "invalid_frame")  # not a JPEG, but it was read
        await ws.close()

    async def test_oversize_text_message_is_rejected_not_parsed(self) -> None:
        ws, _ = await self.connect()
        await ws.send_str(json.dumps({"type": "control", "command": "ping", "pad": "x" * 20000}))
        reply = await asyncio.wait_for(ws.receive_json(), TIMEOUT)
        self.assertEqual(reply["code"], "invalid_control")
        await ws.close()

    async def test_set_mode_rejects_an_absurd_mode_string(self) -> None:
        ws, _ = await self.connect()
        await ws.send_json({"type": "control", "command": "set_mode", "mode": "x" * 500})
        reply = await asyncio.wait_for(ws.receive_json(), TIMEOUT)
        self.assertEqual(reply["code"], "invalid_control")
        await ws.close()


class JpegDimensionTests(HardeningTestCase):
    @staticmethod
    def _with_sof_dimensions(jpeg: bytes, width: int, height: int) -> bytes:
        data = bytearray(jpeg)
        index = bytes(data).find(b"\xff\xc0")
        assert index > 0
        data[index + 5 : index + 7] = height.to_bytes(2, "big")
        data[index + 7 : index + 9] = width.to_bytes(2, "big")
        return bytes(data)

    def test_reads_dimensions_without_decoding(self) -> None:
        self.assertEqual(jpeg_dimensions(_jpeg_bytes(48, 24)), (48, 24))

    def test_exif_thumbnail_sof_is_skipped(self) -> None:
        # An APP1 segment whose payload contains a thumbnail SOF (3x3) must not
        # be mistaken for the real frame header.
        thumb_sof = b"\xff\xc0\x00\x0b\x08\x00\x03\x00\x03\x01\x01\x11\x00"
        app1 = b"\xff\xe1" + (len(thumb_sof) + 2).to_bytes(2, "big") + thumb_sof
        real = _jpeg_bytes(64, 32)
        crafted = real[:2] + app1 + real[2:]
        self.assertEqual(jpeg_dimensions(crafted), (64, 32))

    def test_non_jpeg_payloads_are_rejected(self) -> None:
        ok, png = cv2.imencode(".png", np.zeros((16, 32, 3), dtype=np.uint8))
        self.assertTrue(ok)
        for payload in (png.tobytes(), b"", b"\xff", b"\xff\xd8", b"\xff\xd8\xff\xda\x00\x02"):
            with self.assertRaises(ProtocolError):
                jpeg_dimensions(payload)

    async def test_decompression_bomb_is_rejected_before_imdecode(self) -> None:
        bomb = self._with_sof_dimensions(_jpeg_bytes(32, 16), 65535, 65535)
        frame = _pack_raw_frame_bytes(sequence=1, width=32, height=16, payload=bomb)
        ws, _ = await self.connect()
        with mock.patch.object(frame_decoder.cv2, "imdecode") as imdecode:
            await ws.send_bytes(frame)
            reply = await asyncio.wait_for(ws.receive_json(), TIMEOUT)
        imdecode.assert_not_called()
        self.assertEqual(reply["code"], "invalid_frame")
        self.assertIn("exceed", reply["message"])
        await ws.close()

    async def test_header_that_lies_about_dimensions_is_rejected_before_imdecode(self) -> None:
        frame = _pack_raw_frame_bytes(
            sequence=1, width=640, height=480, payload=_jpeg_bytes(32, 16)
        )
        ws, _ = await self.connect()
        with mock.patch.object(frame_decoder.cv2, "imdecode") as imdecode:
            await ws.send_bytes(frame)
            reply = await asyncio.wait_for(ws.receive_json(), TIMEOUT)
        imdecode.assert_not_called()
        self.assertEqual(reply["code"], "invalid_frame")
        await ws.close()


# --- rate limiting -----------------------------------------------------------


class TokenBucketTests(unittest.TestCase):
    def test_allows_a_burst_then_refuses_then_refills(self) -> None:
        now = [0.0]
        bucket = TokenBucket(rate=10.0, burst=3, clock=lambda: now[0])
        self.assertEqual([bucket.try_consume() for _ in range(5)], [True, True, True, False, False])
        now[0] += 0.1  # one token at 10/s
        self.assertTrue(bucket.try_consume())
        self.assertFalse(bucket.try_consume())

    def test_never_holds_more_than_burst(self) -> None:
        now = [0.0]
        bucket = TokenBucket(rate=100.0, burst=2, clock=lambda: now[0])
        now[0] += 60.0
        self.assertEqual([bucket.try_consume() for _ in range(3)], [True, True, False])

    def test_rejects_nonsense_parameters(self) -> None:
        with self.assertRaises(ValueError):
            TokenBucket(rate=0, burst=1)
        with self.assertRaises(ValueError):
            TokenBucket(rate=1, burst=0)


class FrameRateCapTests(HardeningTestCase):
    CONFIG = TransportConfig(max_fps=0.5, frame_burst=2)  # refill is negligible in-test

    async def test_frames_over_the_cap_are_dropped_and_counted_without_disconnecting(self) -> None:
        ws, _ = await self.connect()
        for sequence in range(1, 9):
            await ws.send_bytes(_encode_test_jpeg_frame(sequence=sequence))

        await self.wait_for_metric("narutocv_frames_received_total", 8)
        dropped = await self.metric('narutocv_frames_dropped_total{reason="rate_limited"}')
        self.assertGreaterEqual(dropped, 5)
        self.assertLessEqual(dropped, 6)

        # Not disconnected, and no error was sent for the dropped frames: the
        # connection still answers a control message with its ack as the next
        # error/ack-typed message.
        await ws.send_json({"type": "control", "command": "ping", "request_id": "p"})
        for _ in range(10):
            message = await asyncio.wait_for(ws.receive_json(), TIMEOUT)
            self.assertNotEqual(message["type"], "error", message)
            if message["type"] == "ack":
                break
        else:
            self.fail("no ack after the burst")
        await ws.close()


class ControlRateCapTests(HardeningTestCase):
    CONFIG = TransportConfig(control_rate=0.5, control_burst=2)

    async def test_control_spam_is_limited(self) -> None:
        ws, _ = await self.connect()
        before = self.fake_pipeline.reset_count
        for _ in range(8):
            await ws.send_json({"type": "control", "command": "reset"})
        await self.wait_for_metric("narutocv_controls_dropped_total", 4)
        # Only the burst reached the (expensive) reset path.
        self.assertLessEqual(self.fake_pipeline.reset_count - before, 3)
        await ws.close()


class InvalidMessageCapTests(HardeningTestCase):
    CONFIG = TransportConfig(max_invalid_messages=5, error_reply_rate=0.5, error_reply_burst=1)

    async def test_error_replies_are_capped_and_the_abuser_is_disconnected(self) -> None:
        ws, _ = await self.connect()
        for _ in range(10):
            try:
                await ws.send_bytes(b"garbage")
            except ConnectionResetError:
                break
        self.assertEqual(await self.receive_close(ws), CLOSE_CODE_PROTOCOL_ERROR)
        self.assertGreater(await self.metric("narutocv_error_replies_suppressed_total"), 0)
        await self.wait_for_metric(
            'narutocv_connections_closed_total{reason="invalid_messages"}', 1
        )


# --- sessions, idle, heartbeat -----------------------------------------------


class MaxSessionsTests(HardeningTestCase):
    CONFIG = TransportConfig(max_sessions=2)

    async def test_connections_beyond_the_cap_are_refused_with_503(self) -> None:
        first, _ = await self.connect()
        second = await self.client.ws_connect("/ws")  # told "controller busy", still counts
        with self.assertRaises(WSServerHandshakeError) as caught:
            await self.client.ws_connect("/ws")
        self.assertEqual(caught.exception.status, 503)
        self.assertEqual(
            await self.metric('narutocv_connections_rejected_total{reason="max_sessions"}'), 1
        )
        await second.close()
        await first.close()

    async def test_capacity_is_released_on_every_exit_path(self) -> None:
        # Normal close, failed handshake (bad version), rejected-as-busy, and
        # client vanishing mid-handshake must all return their slot.
        for _ in range(4):
            ws, _ = await self.connect()
            await ws.close()
            bad = await self.client.ws_connect("/ws")
            await bad.send_json({"type": "client_hello", "protocol_version": "9.0.0"})
            await self.receive_close(bad)
            silent = await self.client.ws_connect("/ws")
            await silent.close()  # never sent a hello
        # After all that churn the cap must still admit two sockets.
        deadline = time.monotonic() + TIMEOUT
        while self.transport._open_connections and time.monotonic() < deadline:
            await asyncio.sleep(0.02)
        self.assertEqual(self.transport._open_connections, 0)
        a, _ = await self.connect()
        b = await self.client.ws_connect("/ws")
        await b.close()
        await a.close()
        self.assertEqual(
            await self.metric("narutocv_open_connections"), self.transport._open_connections
        )

    async def test_second_client_still_gets_controller_already_connected(self) -> None:
        first, _ = await self.connect()
        second = await self.client.ws_connect("/ws")
        error = await asyncio.wait_for(second.receive_json(), TIMEOUT)
        self.assertEqual(error["code"], "controller_already_connected")
        await first.close()


class SingleSessionCapTests(HardeningTestCase):
    CONFIG = TransportConfig(max_sessions=1)

    async def test_cap_of_one_refuses_the_second_socket_outright(self) -> None:
        first, _ = await self.connect()
        with self.assertRaises(WSServerHandshakeError) as caught:
            await self.client.ws_connect("/ws")
        self.assertEqual(caught.exception.status, 503)
        await first.close()


class IdleTimeoutTests(HardeningTestCase):
    CONFIG = TransportConfig(idle_timeout_s=0.3)

    async def test_silent_connection_is_closed_with_4408(self) -> None:
        ws, _ = await self.connect()
        self.assertEqual(await self.receive_close(ws), CLOSE_CODE_IDLE_TIMEOUT)
        await self.wait_for_metric('narutocv_connections_closed_total{reason="idle_timeout"}', 1)
        # The controller slot is free again.
        self.assertIsNone(self.transport._active_ws)

    async def test_activity_keeps_the_connection_alive(self) -> None:
        ws, _ = await self.connect()
        for _ in range(6):
            await ws.send_json({"type": "control", "command": "ping"})
            await asyncio.sleep(0.15)
        self.assertFalse(ws.closed)
        await ws.close()


class HeartbeatTests(HardeningTestCase):
    CONFIG = TransportConfig(heartbeat_s=0.2)

    async def test_server_sends_protocol_pings(self) -> None:
        ws = await self.client.ws_connect("/ws", autoping=False)
        await self.hello(ws)
        saw_ping = False
        deadline = time.monotonic() + TIMEOUT
        while time.monotonic() < deadline and not saw_ping:
            msg = await asyncio.wait_for(ws.receive(), TIMEOUT)
            saw_ping = msg.type == WSMsgType.PING
        self.assertTrue(saw_ping)
        await ws.close()


# --- handshake ----------------------------------------------------------------


class HandshakeTests(HardeningTestCase):
    async def test_server_ready_describes_the_server(self) -> None:
        ws, ready = await self.connect(protocol_version=PROTOCOL_VERSION)
        self.assertEqual(ready["protocol_version"], PROTOCOL_VERSION)
        self.assertEqual(ready["model"]["id"], "narutocv-testbundle")
        self.assertEqual(ready["model"]["sha256"], "a" * 64)
        self.assertIn("tiger", ready["classes"]["hand"])
        self.assertIn("idle", ready["classes"]["body"])
        self.assertIn("homura", ready["classes"]["attacks"])
        caps = ready["capabilities"]
        self.assertEqual(caps["max_fps"], self.CONFIG.max_fps)
        self.assertEqual(caps["max_frame_bytes"], MAX_FRAME_PAYLOAD_BYTES)
        self.assertIn("set_mode", ready["supported_controls"])
        await ws.close()

    async def test_old_1_0_0_client_without_model_id_still_works(self) -> None:
        ws, ready = await self.connect(protocol_version="1.0.0")
        self.assertEqual(ready["type"], "server_ready")
        await ws.close()

    async def test_matching_expected_model_id_is_accepted(self) -> None:
        ws, _ = await self.connect(
            protocol_version=PROTOCOL_VERSION, expected_model_id="narutocv-testbundle"
        )
        await ws.close()

    async def test_different_major_version_is_refused_with_a_clear_error_and_4426(self) -> None:
        ws = await self.client.ws_connect("/ws")
        await self.hello(ws, protocol_version="2.0.0")
        error = await asyncio.wait_for(ws.receive_json(), TIMEOUT)
        self.assertEqual(error["code"], "protocol_version_mismatch")
        self.assertIn("2.0.0", error["message"])
        self.assertIn(PROTOCOL_VERSION, error["message"])
        self.assertEqual(error["details"]["server_protocol_version"], PROTOCOL_VERSION)
        self.assertEqual(await self.receive_close(ws), CLOSE_CODE_VERSION_MISMATCH)
        self.assertEqual(
            await self.metric('narutocv_handshake_failures_total{reason="protocol_version"}'), 1
        )

    async def test_different_minor_version_either_way_is_accepted(self) -> None:
        for version in ("1.0.0", "1.1.0", "1.9.7"):
            ws, _ = await self.connect(protocol_version=version)
            await ws.close()
            await asyncio.sleep(0.05)

    async def test_wrong_model_id_is_refused_with_4412(self) -> None:
        ws = await self.client.ws_connect("/ws")
        await self.hello(ws, protocol_version=PROTOCOL_VERSION, expected_model_id="narutocv-other")
        error = await asyncio.wait_for(ws.receive_json(), TIMEOUT)
        self.assertEqual(error["code"], "model_mismatch")
        self.assertEqual(error["details"]["server_model_id"], "narutocv-testbundle")
        self.assertEqual(await self.receive_close(ws), CLOSE_CODE_MODEL_MISMATCH)
        self.assertEqual(
            await self.metric('narutocv_handshake_failures_total{reason="model_mismatch"}'), 1
        )

    async def test_malformed_version_string_is_a_handshake_failure(self) -> None:
        ws = await self.client.ws_connect("/ws")
        await self.hello(ws, protocol_version="banana")
        error = await asyncio.wait_for(ws.receive_json(), TIMEOUT)
        self.assertEqual(error["code"], "handshake_failed")
        self.assertEqual(await self.receive_close(ws), CLOSE_CODE_PROTOCOL_ERROR)
        self.assertEqual(
            await self.metric('narutocv_handshake_failures_total{reason="invalid"}'), 1
        )

    async def test_a_failed_handshake_frees_the_controller_slot(self) -> None:
        bad = await self.client.ws_connect("/ws")
        await self.hello(bad, protocol_version="3.0.0")
        await self.receive_close(bad)
        ws, _ = await self.connect()
        await ws.close()


class HandshakeTimeoutTests(HardeningTestCase):
    CONFIG = TransportConfig(handshake_timeout_s=0.2)

    async def test_client_that_never_says_hello_is_dropped_and_counted(self) -> None:
        ws = await self.client.ws_connect("/ws")
        error = await asyncio.wait_for(ws.receive_json(), TIMEOUT)
        self.assertEqual(error["code"], "handshake_failed")
        await self.receive_close(ws)
        self.assertEqual(
            await self.metric('narutocv_handshake_failures_total{reason="timeout"}'), 1
        )


class ModelBundleTests(unittest.TestCase):
    ENTRIES = [
        {"name": "b.task", "version": "v1", "sha256": "b" * 64},
        {"name": "a.onnx", "version": "v0", "sha256": "a" * 64},
    ]

    def test_bundle_id_is_order_independent_and_tracks_content(self) -> None:
        one = bundle_from_entries(self.ENTRIES)
        two = bundle_from_entries(list(reversed(self.ENTRIES)))
        changed = bundle_from_entries([self.ENTRIES[0], {**self.ENTRIES[1], "sha256": "c" * 64}])
        self.assertEqual(one, two)
        self.assertNotEqual(one.id, changed.id)
        self.assertEqual(one.id, "narutocv-" + one.sha256[:12])

    def test_repository_manifest_loads(self) -> None:
        bundle = load_model_bundle()
        self.assertTrue(bundle.id.startswith("narutocv-"))
        self.assertEqual(
            {f.name for f in bundle.files},
            {
                "best_model_A.onnx",
                "calibration_v0.json",
                "hand_landmarker.task",
                "pose_landmarker.task",
            },
        )

    def test_missing_manifest_reports_unversioned(self) -> None:
        with self.assertLogs("narutocv.transport", level="WARNING"):
            bundle = load_model_bundle(Path("/nonexistent/manifest.yaml"))
        self.assertEqual(bundle.id, "unversioned")
        self.assertIsNone(bundle.sha256)


class SchemaConformanceTests(HardeningTestCase):
    async def test_server_ready_and_error_conform_to_the_published_schema(self) -> None:
        try:
            import jsonschema
        except ImportError:
            self.skipTest("jsonschema is not installed")
        schema_path = Path(__file__).parents[2] / "schemas" / "transport_message_v1.schema.json"
        schema = json.loads(schema_path.read_text())
        validator = jsonschema.Draft202012Validator(schema)

        ws, ready = await self.connect(protocol_version=PROTOCOL_VERSION)
        validator.validate(ready)
        await ws.close()
        bad = await self.client.ws_connect("/ws")
        await self.hello(bad, protocol_version="7.0.0")
        validator.validate(await asyncio.wait_for(bad.receive_json(), TIMEOUT))
        validator.validate(
            {"type": "client_hello", "protocol_version": "1.1.0", "expected_model_id": "narutocv-x"}
        )
        validator.validate({"type": "client_hello", "protocol_version": "1.0.0"})
        validator.validate({"type": "control", "command": "set_mode", "mode": "full"})


# --- metrics, health, live -----------------------------------------------------


class MetricsEndpointTests(HardeningTestCase):
    async def test_exposition_format_and_content_type(self) -> None:
        response = await self.client.get("/metrics")
        self.assertEqual(response.status, 200)
        self.assertEqual(
            response.headers["Content-Type"], "text/plain; version=0.0.4; charset=utf-8"
        )
        text = await response.text()
        self.assertTrue(text.endswith("\n"))
        for line in text.splitlines():
            if line.startswith("#"):
                self.assertRegex(line, r"^# (HELP|TYPE) [a-z_]+ ")
            else:
                self.assertRegex(line, r"^[a-z_]+(\{[^}]*\})? [-+0-9.eEInf]+$")
        self.assertIn("# TYPE narutocv_stage_latency_ms histogram", text)
        self.assertIn(
            'narutocv_build_info{protocol_version="1.1.0",model_id="narutocv-testbundle"} 1', text
        )
        self.assertIn("narutocv_ready 1", text)

    async def test_counters_track_a_processed_frame_and_stage_histograms_fill(self) -> None:
        ws, _ = await self.connect()
        self.assertEqual(await self.metric("narutocv_active_sessions"), 1)
        await ws.send_bytes(_encode_test_jpeg_frame(sequence=1))
        await self.wait_for_metric("narutocv_frames_processed_total", 1)
        self.assertEqual(await self.metric("narutocv_frames_received_total"), 1)
        text = await self.metrics_text()
        self.assertRegex(text, r'narutocv_stage_latency_ms_count\{stage="runtime_total_ms"\} [1-9]')
        self.assertRegex(
            text, r'narutocv_stage_latency_ms_bucket\{stage="end_to_end_ms",le="\+Inf"\} [1-9]'
        )
        await ws.close()
        await self.wait_for_metric('narutocv_connections_closed_total{reason="client"}', 1)
        self.assertEqual(await self.metric("narutocv_active_sessions"), 0)

    async def test_invalid_and_stale_drops_are_labelled(self) -> None:
        ws, _ = await self.connect()
        await ws.send_bytes(b"junk")
        await ws.send_bytes(_encode_test_jpeg_frame(sequence=5))
        await ws.send_bytes(_encode_test_jpeg_frame(sequence=2))
        await self.wait_for_metric('narutocv_frames_dropped_total{reason="stale_sequence"}', 1)
        self.assertEqual(await self.metric('narutocv_frames_dropped_total{reason="invalid"}'), 1)
        await ws.close()


class MetricsRegistryTests(unittest.TestCase):
    def test_label_values_are_escaped_and_histogram_is_cumulative(self) -> None:
        registry = MetricsRegistry()
        counter = registry.counter("x_total", "help", ("r",))
        counter.inc(r='a"b\\c\nd')
        histogram = registry.histogram("h_ms", "help", ("stage",), buckets=(1, 10))
        for value in (0.5, 5, 50):
            histogram.observe(value, stage="s")
        text = registry.render()
        self.assertIn('x_total{r="a\\"b\\\\c\\nd"} 1', text)
        self.assertIn('h_ms_bucket{stage="s",le="1"} 1', text)
        self.assertIn('h_ms_bucket{stage="s",le="10"} 2', text)
        self.assertIn('h_ms_bucket{stage="s",le="+Inf"} 3', text)
        self.assertIn('h_ms_count{stage="s"} 3', text)

    def test_counters_cannot_decrease_and_label_sets_are_enforced(self) -> None:
        counter = MetricsRegistry().counter("c_total", "help", ("r",))
        with self.assertRaises(ValueError):
            counter.inc(-1, r="a")
        with self.assertRaises(ValueError):
            counter.inc(1)


class HealthAndLiveTests(HardeningTestCase):
    async def test_ready_server_reports_200_on_both(self) -> None:
        health = await self.client.get("/health")
        self.assertEqual(health.status, 200)
        body = await health.json()
        self.assertEqual(body["state"], "ready")
        self.assertEqual(body["model_id"], "narutocv-testbundle")
        self.assertEqual(body["protocol_version"], PROTOCOL_VERSION)
        for key in ("runtime_started", "active_controller", "connected_clients", "last_error"):
            self.assertIn(key, body)  # existing fields stay
        live = await self.client.get("/live")
        self.assertEqual(live.status, 200)
        self.assertEqual((await live.json())["status"], "alive")

    async def test_live_stays_200_while_health_is_503_when_draining(self) -> None:
        await self.app.shutdown()
        health = await self.client.get("/health")
        live = await self.client.get("/live")
        self.assertEqual(health.status, 503)
        self.assertEqual((await health.json())["state"], "shutting_down")
        self.assertEqual(live.status, 200)

    async def test_new_connections_are_refused_while_draining(self) -> None:
        await self.app.shutdown()
        with self.assertRaises(WSServerHandshakeError) as caught:
            await self.client.ws_connect("/ws")
        self.assertEqual(caught.exception.status, 503)


class DegradedStartupTests(AioHTTPTestCase):
    async def get_application(self):
        def broken_factory():
            raise RuntimeError("cannot open /secret/path/model.task")

        return create_app(pipeline_factory=broken_factory, model_bundle=TEST_BUNDLE)

    async def test_health_is_503_but_live_is_200_and_no_path_leaks(self) -> None:
        health = await self.client.get("/health")
        self.assertEqual(health.status, 503)
        body = await health.json()
        self.assertEqual(body["state"], "degraded")
        self.assertEqual(body["last_error"], "RuntimeError")
        self.assertNotIn("/secret", json.dumps(body))
        self.assertEqual((await self.client.get("/live")).status, 200)
        with self.assertRaises(WSServerHandshakeError) as caught:
            await self.client.ws_connect("/ws")
        self.assertEqual(caught.exception.status, 503)
        self.assertEqual(await self._reject_count("not_ready"), 1)

    async def _reject_count(self, reason: str) -> float:
        text = await (await self.client.get("/metrics")).text()
        for line in text.splitlines():
            if line.startswith(f'narutocv_connections_rejected_total{{reason="{reason}"}} '):
                return float(line.rsplit(" ", 1)[1])
        raise AssertionError("missing")


# --- graceful shutdown ----------------------------------------------------------


class GracefulShutdownTests(HardeningTestCase):
    CONFIG = TransportConfig(shutdown_grace_s=3.0)

    async def test_shutdown_closes_clients_going_away_and_releases_the_pipeline(self) -> None:
        ws, _ = await self.connect()
        await ws.send_bytes(_encode_test_jpeg_frame(sequence=1))
        await self.wait_for_metric("narutocv_frames_processed_total", 1)
        controller = self.transport.controller

        await self.app.shutdown()
        self.assertEqual(await self.receive_close(ws), CLOSE_CODE_GOING_AWAY)
        self.assertEqual(self.transport._open_connections, 0)
        self.assertEqual(self.fake_pipeline.close_count, 0)  # still open until cleanup

        await self.app.cleanup()
        self.assertEqual(self.fake_pipeline.close_count, 1)
        self.assertEqual(controller.state, RuntimeState.CLOSED)
        self.assertEqual(await self._closed_reason_count(), 1)

    async def _closed_reason_count(self) -> float:
        return self.transport.metrics.connections_closed.value(reason="shutdown")

    async def test_pipeline_is_closed_even_if_the_scheduler_will_not_stop(self) -> None:
        controller = self.transport.controller
        with (
            mock.patch.object(self.transport.scheduler, "stop", side_effect=TimeoutError("stuck")),
            self.assertLogs("narutocv.transport", level="ERROR"),
        ):
            await self.app.shutdown()
            await self.app.cleanup()
        self.assertEqual(self.fake_pipeline.close_count, 1)
        self.assertEqual(controller.state, RuntimeState.CLOSED)

    async def test_cleanup_tolerates_a_pipeline_whose_close_raises(self) -> None:
        self.fake_pipeline.close = mock.Mock(side_effect=RuntimeError("native crash"))
        with self.assertLogs("narutocv.transport", level="ERROR"):
            await self.app.shutdown()
            await self.app.cleanup()
        self.assertEqual(self.transport.controller.state, RuntimeState.CLOSED)


class SigtermTests(unittest.TestCase):
    """A real process receiving a real SIGTERM with a client still connected."""

    SCRIPT = textwrap.dedent(
        """
        import sys
        from aiohttp import web
        from narutocv.transport.config import TransportConfig
        from narutocv.transport.model_info import ModelBundle
        from narutocv.transport.server import create_app

        class Noop:
            def reset(self): pass
            def process(self, *a): raise AssertionError("no frames expected")
            def close(self):
                print("PIPELINE_CLOSED", flush=True)

        class Pipeline(Noop):
            hand = Noop(); attacks = Noop()

        app = create_app(
            pipeline_factory=Pipeline,
            config=TransportConfig(shutdown_grace_s=3.0),
            model_bundle=ModelBundle(id="narutocv-sigterm", sha256=None),
        )
        web.run_app(app, host="127.0.0.1", port=int(sys.argv[1]), shutdown_timeout=5.0,
                    access_log=None, print=None)
        print("EXITED_CLEANLY", flush=True)
        """
    )

    def test_sigterm_drains_the_client_closes_models_and_exits_promptly(self) -> None:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        env = {**os.environ, "NARUTOCV_LOG_FORMAT": "json"}
        process = subprocess.Popen(
            [sys.executable, "-I", "-c", self.SCRIPT, str(port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        try:
            deadline = time.monotonic() + 30
            while True:
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1):
                        break
                except Exception:
                    if process.poll() is not None or time.monotonic() > deadline:
                        self.fail(f"server did not start: {process.stderr.read()}")
                    time.sleep(0.1)

            async def scenario() -> int | None:
                async with ClientSession() as session:
                    ws = await session.ws_connect(f"http://127.0.0.1:{port}/ws")
                    await ws.send_json({"type": "client_hello", "protocol_version": "1.0.0"})
                    ready = await ws.receive_json(timeout=3)
                    self.assertEqual(ready["model"]["id"], "narutocv-sigterm")
                    process.send_signal(signal.SIGTERM)
                    for _ in range(10):
                        msg = await ws.receive(timeout=5)
                        if msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSED, WSMsgType.ERROR):
                            break
                    return ws.close_code

            started = time.monotonic()
            close_code = asyncio.run(scenario())
            code = process.wait(timeout=10)
            elapsed = time.monotonic() - started
            stdout = process.stdout.read()
        finally:
            if process.poll() is None:
                process.kill()
        self.assertEqual(close_code, CLOSE_CODE_GOING_AWAY)
        self.assertEqual(code, 0)
        self.assertIn("PIPELINE_CLOSED", stdout)
        self.assertLess(elapsed, 6.0)


# --- logging ------------------------------------------------------------------


class JsonLoggingTests(unittest.TestCase):
    def _format(self, **extra) -> dict:
        import logging

        record = logging.LogRecord(
            "narutocv.transport", logging.INFO, "f.py", 1, "hello %s", ("w",), None
        )
        for key, value in extra.items():
            setattr(record, key, value)
        return json.loads(JsonFormatter().format(record))

    def test_record_is_one_json_object_with_core_fields(self) -> None:
        payload = self._format(reason="x")
        self.assertEqual(payload["msg"], "hello w")
        self.assertEqual(payload["level"], "INFO")
        self.assertEqual(payload["reason"], "x")
        self.assertRegex(payload["ts"], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z$")

    def test_hostile_strings_cannot_forge_lines_or_flood(self) -> None:
        formatter = JsonFormatter()
        import logging

        record = logging.LogRecord(
            "n", logging.INFO, "f.py", 1, "%s", ('a\n{"level":"FAKE"}' + "x" * 5000,), None
        )
        line = formatter.format(record)
        self.assertNotIn("\n", line)
        self.assertLess(len(line), 1500)

    def test_connection_id_comes_from_the_context_var(self) -> None:
        from narutocv.transport import logging_config

        token = logging_config.connection_id.set("abc123")
        try:
            self.assertEqual(self._format()["conn"], "abc123")
        finally:
            logging_config.connection_id.reset(token)
        self.assertNotIn("conn", self._format())

    def test_explicit_conn_extra_wins_for_thread_contexts(self) -> None:
        self.assertEqual(self._format(conn="zzz")["conn"], "zzz")


class _CapturingHandler(logging.Handler):
    """Formats at emit time, like a real handler, so the connection-id
    context variable is still set."""

    def __init__(self) -> None:
        super().__init__(logging.INFO)
        self.lines: list[dict] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(json.loads(JsonFormatter().format(record)))


class ConnectionLogTests(HardeningTestCase):
    async def test_connection_lifecycle_is_logged_with_a_connection_id(self) -> None:
        handler = _CapturingHandler()
        logger = logging.getLogger("narutocv.transport")
        logger.addHandler(handler)
        previous_level = logger.level
        logger.setLevel(logging.INFO)
        try:
            ws, _ = await self.connect()
            await ws.close()
            await self.wait_for_metric('narutocv_connections_closed_total{reason="client"}', 1)
        finally:
            logger.removeHandler(handler)
            logger.setLevel(previous_level)
        lifecycle = [line for line in handler.lines if line["msg"].startswith("connection")]
        self.assertEqual(
            {line["msg"] for line in lifecycle}, {"connection opened", "connection closed"}
        )
        self.assertTrue(all(len(line["conn"]) == 12 for line in lifecycle))
        self.assertEqual(len({line["conn"] for line in lifecycle}), 1)


# keep dataclasses import used for config tweaks in future tests
_ = dataclasses

if __name__ == "__main__":
    unittest.main()
