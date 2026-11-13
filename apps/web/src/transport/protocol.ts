/**
 * Wire format for the browser<->Python transport bridge.
 *
 * This is the TypeScript mirror of
 * cv_model/Nishit_Frontend/transport/protocol.py -- the binary frame header
 * layout, JSON message shapes, and constants here must stay byte-for-byte
 * and field-for-field identical to that module. See
 * tests/unit/protocol.test.ts for a golden fixture cross-checked against the
 * Python encoder's actual output.
 *
 * This module only defines shapes and pure encode/decode functions. It does
 * not open a socket -- see websocketClient.ts -- and it never re-interprets
 * a recognition decision, per game_implementation_plan.md invariant 2.
 */

export const PROTOCOL_VERSION = "1.0.0";
// Mirrors cv_model.inference.output_schema.SCHEMA_VERSION, duplicated as a
// literal for the same reason protocol.py duplicates it: zero cross-language
// import dependency.
export const SCHEMA_VERSION = "1.0.0";

// --- Binary frame envelope -------------------------------------------------
//
// Byte layout (big-endian, matches Python struct.Struct(">4sBIQHHI")):
//   offset  size  field
//   0       4     magic ("NCV1")
//   4       1     version (uint8)
//   5       4     sequence (uint32)
//   9       8     captured_at_ms (uint64)
//   17      2     width (uint16)
//   19      2     height (uint16)
//   21      4     payload_len (uint32)
//   25      ...   payload (JPEG bytes)
const FRAME_MAGIC_TEXT = "NCV1";
export const FRAME_MAGIC: Readonly<Uint8Array> = new Uint8Array(
  [...FRAME_MAGIC_TEXT].map((char) => char.charCodeAt(0)),
);
export const FRAME_HEADER_SIZE = 25;

export const MAX_FRAME_PAYLOAD_BYTES = 2 * 1024 * 1024; // 2 MiB JPEG ceiling
export const MAX_FRAME_WIDTH = 1920;
export const MAX_FRAME_HEIGHT = 1080;
export const MIN_FRAME_DIMENSION = 16;

export class ProtocolError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ProtocolError";
  }
}

export const CONTROL_COMMANDS = [
  "reset",
  "begin_calibration",
  "finish_calibration",
  "cancel_calibration",
  "ping",
] as const;
export type ControlCommand = (typeof CONTROL_COMMANDS)[number];

export interface FrameEnvelope {
  version: number;
  sequence: number;
  capturedAtMs: number;
  width: number;
  height: number;
  payload: Uint8Array;
}

function validateFrameEnvelope(envelope: FrameEnvelope): void {
  if (envelope.version !== 1) {
    throw new ProtocolError(`unsupported frame protocol version ${envelope.version}`);
  }
  if (envelope.sequence < 0) {
    throw new ProtocolError("frame sequence cannot be negative");
  }
  if (envelope.capturedAtMs < 0) {
    throw new ProtocolError("capturedAtMs cannot be negative");
  }
  if (envelope.width < MIN_FRAME_DIMENSION || envelope.width > MAX_FRAME_WIDTH) {
    throw new ProtocolError(`frame width ${envelope.width} out of bounds`);
  }
  if (envelope.height < MIN_FRAME_DIMENSION || envelope.height > MAX_FRAME_HEIGHT) {
    throw new ProtocolError(`frame height ${envelope.height} out of bounds`);
  }
  if (envelope.payload.byteLength === 0) {
    throw new ProtocolError("frame payload cannot be empty");
  }
  if (envelope.payload.byteLength > MAX_FRAME_PAYLOAD_BYTES) {
    throw new ProtocolError(
      `frame payload ${envelope.payload.byteLength} bytes exceeds the ` +
        `${MAX_FRAME_PAYLOAD_BYTES} byte limit`,
    );
  }
}

/** Pack a FrameEnvelope into wire bytes for sending over the binary WebSocket. */
export function encodeFrame(envelope: FrameEnvelope): Uint8Array {
  validateFrameEnvelope(envelope);
  const buffer = new ArrayBuffer(FRAME_HEADER_SIZE + envelope.payload.byteLength);
  const view = new DataView(buffer);
  const bytes = new Uint8Array(buffer);

  bytes.set(FRAME_MAGIC, 0);
  view.setUint8(4, envelope.version);
  view.setUint32(5, envelope.sequence, false);
  view.setBigUint64(9, BigInt(Math.trunc(envelope.capturedAtMs)), false);
  view.setUint16(17, envelope.width, false);
  view.setUint16(19, envelope.height, false);
  view.setUint32(21, envelope.payload.byteLength, false);
  bytes.set(envelope.payload, FRAME_HEADER_SIZE);

  return bytes;
}

/** Parse and validate the binary header of an inbound/outbound frame message. */
export function decodeFrame(data: ArrayBuffer | Uint8Array): FrameEnvelope {
  const bytes = data instanceof Uint8Array ? data : new Uint8Array(data);
  if (bytes.byteLength < FRAME_HEADER_SIZE) {
    throw new ProtocolError("frame shorter than the binary header");
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);

  for (let index = 0; index < FRAME_MAGIC.length; index += 1) {
    if (bytes[index] !== FRAME_MAGIC[index]) {
      throw new ProtocolError("frame magic bytes do not match the protocol");
    }
  }

  const version = view.getUint8(4);
  const sequence = view.getUint32(5, false);
  const capturedAtMs = Number(view.getBigUint64(9, false));
  const width = view.getUint16(17, false);
  const height = view.getUint16(19, false);
  const payloadLen = view.getUint32(21, false);
  const payload = bytes.subarray(FRAME_HEADER_SIZE);

  if (payload.byteLength !== payloadLen) {
    throw new ProtocolError(
      `declared payload length ${payloadLen} does not match ` +
        `${payload.byteLength} received bytes`,
    );
  }

  const envelope: FrameEnvelope = { version, sequence, capturedAtMs, width, height, payload };
  validateFrameEnvelope(envelope);
  return envelope;
}

// --- V1 recognition contract (mirrors cv_model/schemas/pipeline_output_v1
// and pipeline_event_v1). Kept intentionally structural/permissive on
// strings (label names, event types) rather than re-encoding the recognition
// catalog here -- the backend is the single source of truth for what labels
// and attacks exist. -------------------------------------------------------

export interface Prediction {
  label: string;
  confidence: number;
  second_label: string;
  second_confidence: number;
  margin: number;
}

export type BBox = readonly [number, number, number, number];
export type Point = readonly [number, number];

export interface HandGeometry {
  classification_bbox: BBox;
  hand_bbox: BBox | null;
  landmarks: Point[][];
}

export interface BodyGeometry {
  landmarks: Point[];
}

export interface HandOutput {
  raw: Prediction;
  center: Prediction;
  roi: Prediction | null;
  accepted_label: string;
  rejection_reason: string | null;
  stable_label: string;
  emitted_seal: string | null;
  fusion_status: string;
  detected_hand_count: number;
  geometry: HandGeometry | null;
}

export interface BodyOutput {
  raw_label: string;
  stable_label: string;
  emitted_movement: string | null;
  metrics: Record<string, number>;
  geometry: BodyGeometry | null;
}

export interface QueueOutput {
  seals: string[];
  accepted_seal: string | null;
  duplicate_ignored: boolean;
  timeout_cleared: boolean;
  max_length_cleared: boolean;
  cooldown_suppressed: boolean;
}

export interface AttackOutput {
  name: string;
  display_name: string;
  recognized_at_ms: number;
}

export interface PipelineOutputV1 {
  schema_version: "1.0.0";
  session_id: string;
  frame_id: number;
  captured_at_ms: number;
  processing_ms: number;
  hand: HandOutput;
  body: BodyOutput;
  queue: QueueOutput;
  attack: AttackOutput | null;
}

export const PIPELINE_EVENT_TYPES = [
  "HAND_SEAL",
  "BODY_MOVEMENT",
  "ATTACK_TRIGGERED",
  "QUEUE_CLEARED",
  "PIPELINE_RESET",
] as const;
export type PipelineEventType = (typeof PIPELINE_EVENT_TYPES)[number];

export interface PipelineEventV1 {
  schema_version: "1.0.0";
  event_id: string;
  event_sequence: number;
  event_type: PipelineEventType;
  session_id: string;
  frame_id: number;
  captured_at_ms: number;
  payload: Record<string, unknown>;
}

// --- JSON control-plane messages -------------------------------------------

export interface ClientHelloMessage {
  type: "client_hello";
  protocol_version: string;
}

export interface ServerReadyMessage {
  type: "server_ready";
  protocol_version: string;
  schema_version: string;
  session_id: string;
  supported_controls: ControlCommand[];
}

export interface ControlMessage {
  type: "control";
  command: ControlCommand;
  request_id?: string;
}

export interface AckMessage {
  type: "ack";
  command: ControlCommand;
  ok: boolean;
  request_id?: string;
  error?: string;
}

export interface ErrorMessage {
  type: "error";
  code: string;
  message: string;
  request_id?: string;
}

export interface StateSnapshotMessage {
  type: "state_snapshot";
  session_id: string;
  frame_id: number;
  captured_at_ms: number;
  output: PipelineOutputV1;
}

export interface PipelineEventMessage {
  type: "pipeline_event";
  event: PipelineEventV1;
}

export type TransportMessage =
  | ClientHelloMessage
  | ServerReadyMessage
  | ControlMessage
  | AckMessage
  | ErrorMessage
  | StateSnapshotMessage
  | PipelineEventMessage;

export function buildClientHello(): ClientHelloMessage {
  return { type: "client_hello", protocol_version: PROTOCOL_VERSION };
}

export function buildControl(command: ControlCommand, requestId?: string): ControlMessage {
  return requestId === undefined
    ? { type: "control", command }
    : { type: "control", command, request_id: requestId };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

/**
 * Parse and narrow an inbound JSON WebSocket message.
 *
 * This checks the discriminant `type` field and required keys only -- it
 * deliberately does not deep-validate `output`/`event` payload shape at
 * every field (that duplicate-of-the-schema validation lives server-side;
 * see transport_message_v1.schema.json's description fields). A malformed
 * top-level message is still rejected rather than silently passed through.
 */
export function parseTransportMessage(raw: unknown): TransportMessage {
  if (!isRecord(raw) || typeof raw.type !== "string") {
    throw new ProtocolError("message is not a tagged transport JSON object");
  }
  switch (raw.type) {
    case "server_ready":
      if (
        typeof raw.protocol_version !== "string" ||
        typeof raw.schema_version !== "string" ||
        typeof raw.session_id !== "string" ||
        !Array.isArray(raw.supported_controls)
      ) {
        throw new ProtocolError("malformed server_ready message");
      }
      return raw as unknown as ServerReadyMessage;
    case "ack":
      if (typeof raw.command !== "string" || typeof raw.ok !== "boolean") {
        throw new ProtocolError("malformed ack message");
      }
      return raw as unknown as AckMessage;
    case "error":
      if (typeof raw.code !== "string" || typeof raw.message !== "string") {
        throw new ProtocolError("malformed error message");
      }
      return raw as unknown as ErrorMessage;
    case "state_snapshot":
      if (
        typeof raw.session_id !== "string" ||
        typeof raw.frame_id !== "number" ||
        typeof raw.captured_at_ms !== "number" ||
        !isRecord(raw.output)
      ) {
        throw new ProtocolError("malformed state_snapshot message");
      }
      return raw as unknown as StateSnapshotMessage;
    case "pipeline_event":
      if (!isRecord(raw.event)) {
        throw new ProtocolError("malformed pipeline_event message");
      }
      return raw as unknown as PipelineEventMessage;
    default:
      throw new ProtocolError(`unsupported transport message type ${String(raw.type)}`);
  }
}
