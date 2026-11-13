"""Wire format for the browser<->Python transport bridge.

This module defines envelopes, versions, and validation only. It does not
open sockets, decode images, or touch the recognition pipeline -- see
frame_decoder.py and server.py for those layers. Keeping the protocol in one
dependency-free module lets it be unit tested without aiohttp, OpenCV, or a
running event loop.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Optional

PROTOCOL_VERSION = "1.0.0"
# Mirrors cv_model.inference.output_schema.SCHEMA_VERSION. Duplicated as a
# literal (rather than imported) so this module has zero dependency on the
# inference package and can be unit tested in isolation.
SCHEMA_VERSION = "1.0.0"

# --- Binary frame envelope -------------------------------------------------
#
# magic(4s) version(B) sequence(I) captured_at_ms(Q) width(H) height(H) payload_len(I)
FRAME_MAGIC = b"NCV1"
FRAME_HEADER = struct.Struct(">4sBIQHHI")
FRAME_HEADER_SIZE = FRAME_HEADER.size

MAX_FRAME_PAYLOAD_BYTES = 2 * 1024 * 1024  # 2 MiB JPEG ceiling
MAX_FRAME_WIDTH = 1920
MAX_FRAME_HEIGHT = 1080
MIN_FRAME_DIMENSION = 16


class ProtocolError(ValueError):
    """Raised when a client message violates the transport contract.

    Callers (server.py) reject the offending frame/message before it reaches
    recognition rather than letting a malformed payload propagate.
    """


class ControlCommand(str, Enum):
    RESET = "reset"
    BEGIN_CALIBRATION = "begin_calibration"
    FINISH_CALIBRATION = "finish_calibration"
    CANCEL_CALIBRATION = "cancel_calibration"
    PING = "ping"


SUPPORTED_CONTROLS = tuple(command.value for command in ControlCommand)


@dataclass(frozen=True)
class FrameEnvelope:
    """One decoded binary WebSocket frame message, pre-JPEG-decode."""

    version: int
    sequence: int
    captured_at_ms: int
    width: int
    height: int
    payload: bytes

    def validate(self) -> None:
        if self.version != 1:
            raise ProtocolError(f"unsupported frame protocol version {self.version}")
        if self.sequence < 0:
            raise ProtocolError("frame sequence cannot be negative")
        if self.captured_at_ms < 0:
            raise ProtocolError("captured_at_ms cannot be negative")
        if not (MIN_FRAME_DIMENSION <= self.width <= MAX_FRAME_WIDTH):
            raise ProtocolError(f"frame width {self.width} out of bounds")
        if not (MIN_FRAME_DIMENSION <= self.height <= MAX_FRAME_HEIGHT):
            raise ProtocolError(f"frame height {self.height} out of bounds")
        if not self.payload:
            raise ProtocolError("frame payload cannot be empty")
        if len(self.payload) > MAX_FRAME_PAYLOAD_BYTES:
            raise ProtocolError(
                f"frame payload {len(self.payload)} bytes exceeds the "
                f"{MAX_FRAME_PAYLOAD_BYTES} byte limit"
            )


def encode_frame(envelope: FrameEnvelope) -> bytes:
    """Pack a FrameEnvelope into wire bytes.

    Production frames are produced by the browser's framePublisher, not this
    backend. This exists for tests and the future minimal diagnostic client
    (context.md step-6 completion artifact).
    """
    envelope.validate()
    header = FRAME_HEADER.pack(
        FRAME_MAGIC,
        envelope.version,
        envelope.sequence,
        envelope.captured_at_ms,
        envelope.width,
        envelope.height,
        len(envelope.payload),
    )
    return header + envelope.payload


def decode_frame(data: bytes) -> FrameEnvelope:
    """Parse and validate the binary header of an inbound frame message."""
    if len(data) < FRAME_HEADER_SIZE:
        raise ProtocolError("frame shorter than the binary header")
    magic, version, sequence, captured_at_ms, width, height, payload_len = (
        FRAME_HEADER.unpack_from(data, 0)
    )
    if magic != FRAME_MAGIC:
        raise ProtocolError("frame magic bytes do not match the protocol")
    payload = data[FRAME_HEADER_SIZE:]
    if len(payload) != payload_len:
        raise ProtocolError(
            f"declared payload length {payload_len} does not match "
            f"{len(payload)} received bytes"
        )
    envelope = FrameEnvelope(version, sequence, captured_at_ms, width, height, payload)
    envelope.validate()
    return envelope


# --- JSON control-plane messages -------------------------------------------


def build_client_hello(*, protocol_version: str = PROTOCOL_VERSION) -> dict:
    return {"type": "client_hello", "protocol_version": protocol_version}


def build_server_ready(*, session_id: str) -> dict:
    return {
        "type": "server_ready",
        "protocol_version": PROTOCOL_VERSION,
        "schema_version": SCHEMA_VERSION,
        "session_id": session_id,
        "supported_controls": list(SUPPORTED_CONTROLS),
    }


def build_error(
    *, code: str, message: str, request_id: Optional[str] = None
) -> dict:
    payload: dict[str, Any] = {"type": "error", "code": code, "message": message}
    if request_id is not None:
        payload["request_id"] = request_id
    return payload


def build_ack(
    *,
    command: str,
    ok: bool,
    request_id: Optional[str] = None,
    error: Optional[str] = None,
) -> dict:
    payload: dict[str, Any] = {"type": "ack", "command": command, "ok": ok}
    if request_id is not None:
        payload["request_id"] = request_id
    if error is not None:
        payload["error"] = error
    return payload


def build_state_snapshot(
    *, session_id: str, frame_id: int, captured_at_ms: int, output: Mapping[str, Any]
) -> dict:
    return {
        "type": "state_snapshot",
        "session_id": session_id,
        "frame_id": frame_id,
        "captured_at_ms": captured_at_ms,
        "output": dict(output),
    }


def build_pipeline_event(event: Mapping[str, Any]) -> dict:
    return {"type": "pipeline_event", "event": dict(event)}


def parse_client_hello(raw: Mapping[str, Any]) -> str:
    """Validate the opening handshake message; returns the client's protocol version."""
    if raw.get("type") != "client_hello":
        raise ProtocolError(f"expected 'client_hello', received {raw.get('type')!r}")
    version = raw.get("protocol_version")
    if not isinstance(version, str) or not version:
        raise ProtocolError("client_hello requires a non-empty protocol_version")
    return version


def parse_control_message(raw: Mapping[str, Any]) -> tuple[ControlCommand, Optional[str]]:
    """Validate an inbound JSON control message.

    Returns (command, request_id). request_id is echoed back on the ack so the
    frontend can correlate pending UI state (game_implementation_plan.md
    invariant 3: connectionPanel actions are pending/success/failure driven
    only by correlated acknowledgements).
    """
    if raw.get("type") != "control":
        raise ProtocolError(f"expected a 'control' message, received {raw.get('type')!r}")
    command_value = raw.get("command")
    try:
        command = ControlCommand(command_value)
    except ValueError as error:
        raise ProtocolError(f"unsupported control command {command_value!r}") from error
    request_id = raw.get("request_id")
    if request_id is not None and not isinstance(request_id, str):
        raise ProtocolError("request_id must be a string when present")
    return command, request_id
