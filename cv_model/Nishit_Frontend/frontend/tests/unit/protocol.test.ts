import { describe, expect, it } from "vitest";
import {
  FRAME_HEADER_SIZE,
  FRAME_MAGIC,
  MAX_FRAME_PAYLOAD_BYTES,
  ProtocolError,
  type FrameEnvelope,
  buildClientHello,
  buildControl,
  decodeFrame,
  encodeFrame,
  parseTransportMessage,
} from "../../src/transport/protocol";

function hexToBytes(hex: string): Uint8Array {
  const bytes = new Uint8Array(hex.length / 2);
  for (let i = 0; i < bytes.length; i += 1) {
    bytes[i] = parseInt(hex.slice(i * 2, i * 2 + 2), 16);
  }
  return bytes;
}

function bytesToHex(bytes: Uint8Array): string {
  return [...bytes].map((byte) => byte.toString(16).padStart(2, "0")).join("");
}

function textPayload(text: string): Uint8Array {
  return new TextEncoder().encode(text);
}

function envelope(overrides: Partial<FrameEnvelope> = {}): FrameEnvelope {
  return {
    version: 1,
    sequence: 1,
    capturedAtMs: 1000,
    width: 32,
    height: 16,
    payload: textPayload("fake-jpeg-bytes"),
    ...overrides,
  };
}

describe("cross-language wire format", () => {
  // Generated with the real Python encoder (protocol.py):
  //   env = FrameEnvelope(version=1, sequence=42, captured_at_ms=1731000000123,
  //                        width=640, height=360, payload=b"hello-jpeg-bytes")
  //   encode_frame(env).hex()
  // If this ever needs regenerating, run that from cv_model/Nishit_Frontend
  // with the transport venv active. A mismatch here means the TS and Python
  // encoders have drifted apart at the byte level.
  const GOLDEN_HEX =
    "4e435631010000002a0000019307a3de7b028001680000001068656c6c6f2d6a7065672d6279746573";

  it("matches the Python encode_frame() output byte-for-byte", () => {
    const wire = encodeFrame({
      version: 1,
      sequence: 42,
      capturedAtMs: 1731000000123,
      width: 640,
      height: 360,
      payload: textPayload("hello-jpeg-bytes"),
    });

    expect(bytesToHex(wire)).toBe(GOLDEN_HEX);
  });

  it("decodes the Python-produced golden bytes back to the same fields", () => {
    const decoded = decodeFrame(hexToBytes(GOLDEN_HEX));

    expect(decoded.version).toBe(1);
    expect(decoded.sequence).toBe(42);
    expect(decoded.capturedAtMs).toBe(1731000000123);
    expect(decoded.width).toBe(640);
    expect(decoded.height).toBe(360);
    expect(new TextDecoder().decode(decoded.payload)).toBe("hello-jpeg-bytes");
  });
});

describe("FrameEnvelope round trip", () => {
  it("recovers identical fields through encode then decode", () => {
    const original = envelope({ sequence: 7, capturedAtMs: 123456, width: 640, height: 360 });

    const decoded = decodeFrame(encodeFrame(original));

    expect(decoded.version).toBe(original.version);
    expect(decoded.sequence).toBe(original.sequence);
    expect(decoded.capturedAtMs).toBe(original.capturedAtMs);
    expect(decoded.width).toBe(original.width);
    expect(decoded.height).toBe(original.height);
    expect([...decoded.payload]).toEqual([...original.payload]);
  });

  it("starts the wire bytes with the magic and matches the declared header size", () => {
    const wire = encodeFrame(envelope());

    expect(bytesToHex(wire.subarray(0, 4))).toBe(bytesToHex(FRAME_MAGIC as Uint8Array));
    expect(wire.byteLength).toBe(FRAME_HEADER_SIZE + envelope().payload.byteLength);
  });
});

describe("FrameEnvelope validation", () => {
  it("rejects an unsupported version", () => {
    expect(() => encodeFrame(envelope({ version: 2 }))).toThrow(ProtocolError);
  });

  it("rejects a negative sequence", () => {
    expect(() => encodeFrame(envelope({ sequence: -1 }))).toThrow(ProtocolError);
  });

  it("rejects width below the minimum dimension", () => {
    expect(() => encodeFrame(envelope({ width: 1 }))).toThrow(ProtocolError);
  });

  it("rejects width above the maximum dimension", () => {
    expect(() => encodeFrame(envelope({ width: 99_999 }))).toThrow(ProtocolError);
  });

  it("rejects an empty payload", () => {
    expect(() => encodeFrame(envelope({ payload: new Uint8Array(0) }))).toThrow(ProtocolError);
  });

  it("rejects a payload over the byte limit", () => {
    expect(() =>
      encodeFrame(envelope({ payload: new Uint8Array(MAX_FRAME_PAYLOAD_BYTES + 1) })),
    ).toThrow(ProtocolError);
  });

  it("accepts a valid envelope without throwing", () => {
    expect(() => encodeFrame(envelope())).not.toThrow();
  });
});

describe("decodeFrame wire errors", () => {
  it("rejects data shorter than the header", () => {
    expect(() => decodeFrame(new Uint8Array(10))).toThrow(ProtocolError);
  });

  it("rejects the wrong magic bytes", () => {
    const wire = encodeFrame(envelope());
    wire.set(textPayload("XXXX"), 0);

    expect(() => decodeFrame(wire)).toThrow(ProtocolError);
  });

  it("rejects a payload-length mismatch", () => {
    const wire = encodeFrame(envelope({ payload: textPayload("0123456789") }));
    const truncated = wire.subarray(0, wire.byteLength - 1);

    expect(() => decodeFrame(truncated)).toThrow(ProtocolError);
  });
});

describe("JSON message builders", () => {
  it("builds a client_hello with the current protocol version", () => {
    const message = buildClientHello();
    expect(message.type).toBe("client_hello");
    expect(message.protocol_version).toBeTruthy();
  });

  it("omits request_id from control when not given", () => {
    const withId = buildControl("reset", "req-1");
    const withoutId = buildControl("ping");

    expect(withId.request_id).toBe("req-1");
    expect("request_id" in withoutId).toBe(false);
  });
});

describe("parseTransportMessage", () => {
  it("parses a well-formed server_ready message", () => {
    const parsed = parseTransportMessage({
      type: "server_ready",
      protocol_version: "1.0.0",
      schema_version: "1.0.0",
      session_id: "abc",
      supported_controls: ["ping"],
    });
    expect(parsed.type).toBe("server_ready");
  });

  it("rejects a message with no type field", () => {
    expect(() => parseTransportMessage({ foo: "bar" })).toThrow(ProtocolError);
  });

  it("rejects an unknown message type", () => {
    expect(() => parseTransportMessage({ type: "unknown_type" })).toThrow(ProtocolError);
  });

  it("rejects a malformed state_snapshot missing required fields", () => {
    expect(() => parseTransportMessage({ type: "state_snapshot", session_id: "s1" })).toThrow(
      ProtocolError,
    );
  });
});
