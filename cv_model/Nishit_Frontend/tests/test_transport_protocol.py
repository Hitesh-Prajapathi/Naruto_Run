from __future__ import annotations

import sys
import unittest
from pathlib import Path

# Bootstrap the repo root onto sys.path so `cv_model....` absolute imports
# resolve no matter how this module is invoked (plain `python -m unittest`,
# pytest, or an IDE runner) -- unlike a relative import, this doesn't depend
# on unittest discovery importing this file as part of the `tests` package.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cv_model.Nishit_Frontend.transport.protocol import (
    FRAME_HEADER_SIZE,
    FRAME_MAGIC,
    MAX_FRAME_PAYLOAD_BYTES,
    ControlCommand,
    FrameEnvelope,
    ProtocolError,
    build_ack,
    build_client_hello,
    build_error,
    build_pipeline_event,
    build_server_ready,
    build_state_snapshot,
    decode_frame,
    encode_frame,
    parse_client_hello,
    parse_control_message,
)


def _envelope(**overrides) -> FrameEnvelope:
    fields = dict(
        version=1,
        sequence=1,
        captured_at_ms=1000,
        width=32,
        height=16,
        payload=b"\xff\xd8fake-jpeg-bytes",
    )
    fields.update(overrides)
    return FrameEnvelope(**fields)


class FrameEnvelopeRoundTripTests(unittest.TestCase):
    def test_encode_then_decode_recovers_the_same_fields(self) -> None:
        original = _envelope(sequence=42, captured_at_ms=123456, width=640, height=360)
        wire = encode_frame(original)

        decoded = decode_frame(wire)

        self.assertEqual(decoded, original)

    def test_encoded_wire_bytes_start_with_the_magic_and_declared_header_size(self) -> None:
        wire = encode_frame(_envelope())

        self.assertTrue(wire.startswith(FRAME_MAGIC))
        self.assertEqual(len(wire), FRAME_HEADER_SIZE + len(_envelope().payload))


class FrameEnvelopeValidationTests(unittest.TestCase):
    def test_rejects_unsupported_version(self) -> None:
        with self.assertRaises(ProtocolError):
            _envelope(version=2).validate()

    def test_rejects_negative_sequence(self) -> None:
        with self.assertRaises(ProtocolError):
            _envelope(sequence=-1).validate()

    def test_rejects_width_below_minimum(self) -> None:
        with self.assertRaises(ProtocolError):
            _envelope(width=1).validate()

    def test_rejects_width_above_maximum(self) -> None:
        with self.assertRaises(ProtocolError):
            _envelope(width=99999).validate()

    def test_rejects_empty_payload(self) -> None:
        with self.assertRaises(ProtocolError):
            _envelope(payload=b"").validate()

    def test_rejects_oversized_payload(self) -> None:
        with self.assertRaises(ProtocolError):
            _envelope(payload=b"x" * (MAX_FRAME_PAYLOAD_BYTES + 1)).validate()

    def test_valid_envelope_does_not_raise(self) -> None:
        _envelope().validate()  # should not raise


class DecodeFrameWireErrorsTests(unittest.TestCase):
    def test_rejects_data_shorter_than_the_header(self) -> None:
        with self.assertRaises(ProtocolError):
            decode_frame(b"too short")

    def test_rejects_wrong_magic_bytes(self) -> None:
        wire = bytearray(encode_frame(_envelope()))
        wire[0:4] = b"XXXX"

        with self.assertRaises(ProtocolError):
            decode_frame(bytes(wire))

    def test_rejects_payload_length_mismatch(self) -> None:
        wire = encode_frame(_envelope(payload=b"0123456789"))
        # Truncate one trailing payload byte so the declared length in the
        # header no longer matches what actually follows it.
        truncated = wire[:-1]

        with self.assertRaises(ProtocolError):
            decode_frame(truncated)


class ControlPlaneMessageBuilderTests(unittest.TestCase):
    def test_client_hello_carries_the_current_protocol_version(self) -> None:
        message = build_client_hello()
        self.assertEqual(message["type"], "client_hello")
        self.assertTrue(message["protocol_version"])

    def test_server_ready_lists_every_supported_control(self) -> None:
        message = build_server_ready(session_id="abc123")
        self.assertEqual(message["type"], "server_ready")
        self.assertEqual(message["session_id"], "abc123")
        self.assertEqual(
            set(message["supported_controls"]),
            {command.value for command in ControlCommand},
        )

    def test_ack_includes_request_id_and_error_only_when_given(self) -> None:
        ok = build_ack(command="reset", ok=True, request_id="req-1")
        self.assertNotIn("error", ok)
        self.assertEqual(ok["request_id"], "req-1")

        failed = build_ack(command="reset", ok=False, error="boom")
        self.assertNotIn("request_id", failed)
        self.assertEqual(failed["error"], "boom")

    def test_error_message_shape(self) -> None:
        message = build_error(code="invalid_frame", message="bad header")
        self.assertEqual(message, {"type": "error", "code": "invalid_frame", "message": "bad header"})

    def test_state_snapshot_copies_the_output_mapping(self) -> None:
        output = {"schema_version": "1.0.0"}
        message = build_state_snapshot(
            session_id="s1", frame_id=7, captured_at_ms=999, output=output
        )
        self.assertEqual(message["output"], output)
        self.assertIsNot(message["output"], output)

    def test_pipeline_event_wraps_the_event_mapping(self) -> None:
        event = {"event_type": "HAND_SEAL"}
        message = build_pipeline_event(event)
        self.assertEqual(message, {"type": "pipeline_event", "event": event})


class ParseClientHelloTests(unittest.TestCase):
    def test_returns_the_protocol_version(self) -> None:
        self.assertEqual(parse_client_hello({"type": "client_hello", "protocol_version": "1.0.0"}), "1.0.0")

    def test_rejects_wrong_type(self) -> None:
        with self.assertRaises(ProtocolError):
            parse_client_hello({"type": "control", "protocol_version": "1.0.0"})

    def test_rejects_missing_protocol_version(self) -> None:
        with self.assertRaises(ProtocolError):
            parse_client_hello({"type": "client_hello"})


class ParseControlMessageTests(unittest.TestCase):
    def test_parses_a_valid_control_command(self) -> None:
        command, request_id = parse_control_message(
            {"type": "control", "command": "reset", "request_id": "r-1"}
        )
        self.assertIs(command, ControlCommand.RESET)
        self.assertEqual(request_id, "r-1")

    def test_request_id_is_optional(self) -> None:
        command, request_id = parse_control_message({"type": "control", "command": "ping"})
        self.assertIs(command, ControlCommand.PING)
        self.assertIsNone(request_id)

    def test_rejects_wrong_message_type(self) -> None:
        with self.assertRaises(ProtocolError):
            parse_control_message({"type": "client_hello", "command": "reset"})

    def test_rejects_unknown_command(self) -> None:
        with self.assertRaises(ProtocolError):
            parse_control_message({"type": "control", "command": "fly"})

    def test_rejects_non_string_request_id(self) -> None:
        with self.assertRaises(ProtocolError):
            parse_control_message({"type": "control", "command": "ping", "request_id": 5})


if __name__ == "__main__":
    unittest.main()
