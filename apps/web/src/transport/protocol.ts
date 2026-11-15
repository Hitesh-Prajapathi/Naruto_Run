/**
 * Wire format for the browser<->Python transport bridge.
 *
 * This is the TypeScript mirror of
 * src/narutocv/transport/protocol.py -- the binary frame header
 * layout, JSON message shapes, and constants here must stay byte-for-byte
 * and field-for-field identical to that module. See
 * tests/unit/protocol.test.ts for a golden fixture cross-checked against the
 * Python encoder's actual output.
 *
 * This module only defines shapes and pure encode/decode functions. It does
 * not open a socket -- see websocketClient.ts -- and it never re-interprets
 * a recognition decision, per game_implementation_plan.md invariant 2.
 */

// 1.1.0 is additive over 1.0.0 (see schemas/transport_message_v1.schema.json):
// the hello may carry `expected_model_id`; server_ready adds `model`,
// `classes` and `capabilities`. 1.2.0 is additive over 1.1.0:
// `capabilities.hand_decoder`, and the HAND_SEAL payload gains `decoder`,
// `t_start_ms`, `t_emit_ms` and `decoder_confidence` (see HandSealPayload).
// Peers are compatible when MAJOR matches.
export const PROTOCOL_VERSION = "1.2.0";
export const PROTOCOL_MAJOR = 1;

/** WebSocket close codes (mirror src/narutocv/transport/server.py). */
export const CLOSE_CODE_IDLE_TIMEOUT = 4408;
export const CLOSE_CODE_CONTROLLER_TAKEN = 4409;
export const CLOSE_CODE_MODEL_MISMATCH = 4412;
export const CLOSE_CODE_VERSION_MISMATCH = 4426;
export const CLOSE_CODE_BACKPRESSURE = 4429;
/** Closes after which reconnecting cannot succeed until something is upgraded. */
export const FATAL_CLOSE_CODES: readonly number[] = [
  CLOSE_CODE_MODEL_MISMATCH,
  CLOSE_CODE_VERSION_MISMATCH,
];

/** Parse the MAJOR of "MAJOR.MINOR.PATCH"; throws ProtocolError otherwise. */
export function parseProtocolMajor(version: string): number {
  const match = /^(\d+)\.(\d+)\.(\d+)$/.exec(version);
  if (!match) {
    throw new ProtocolError(`protocol_version ${JSON.stringify(version)} is not MAJOR.MINOR.PATCH`);
  }
  return Number(match[1]);
}
// Mirrors narutocv.inference.output_schema.SCHEMA_VERSION, duplicated as a
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
  "set_mode",
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

// --- V1 recognition contract (mirrors schemas/pipeline_output_v1
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

/** The hand-seal decoder behind HAND_SEAL events (protocol 1.2.0). */
export type HandDecoderMode = "legacy" | "v1";

/** `output.hand.decoder` (1.2.0): present only when the v1 decoder is active. */
export interface HandDecoderOutput {
  mode: "v1";
  event: {
    label: string;
    t_start_ms: number;
    t_emit_ms: number;
    confidence: number;
  } | null;
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
  /** 1.2.0, optional: absent on older servers and for the legacy decoder. */
  decoder?: HandDecoderOutput;
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

/**
 * HAND_SEAL event payload. `seal`, `stable_label` and `confidence` (the raw frame
 * confidence) exist since 1.0.0; everything else is 1.2.0 and optional. `t_start_ms` /
 * `t_emit_ms` are on the pipeline clock, i.e. the same basis as the frame's
 * `captured_at_ms` (the browser's `performance.now()` for frames this client sent), and are
 * only present for the v1 decoder. `t_start_ms` is the smoothed tau_on crossing, so it lags
 * the true onset by the smoother's rise time.
 */
export interface HandSealPayload {
  seal: string;
  stable_label?: string;
  confidence?: number;
  decoder?: HandDecoderMode;
  t_start_ms?: number;
  t_emit_ms?: number;
  decoder_confidence?: number;
}

function finiteNumber(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

/**
 * Narrow an untyped event payload to a HandSealPayload. Returns null when there is no
 * string `seal`; ignores optional fields of the wrong type instead of throwing, so a
 * 1.0/1.1 server (no new fields) and a future one (extra fields) both parse.
 */
export function parseHandSealPayload(payload: unknown): HandSealPayload | null {
  if (!isRecord(payload) || typeof payload.seal !== "string") return null;
  const out: HandSealPayload = { seal: payload.seal };
  if (typeof payload.stable_label === "string") out.stable_label = payload.stable_label;
  const confidence = finiteNumber(payload.confidence);
  if (confidence !== undefined) out.confidence = confidence;
  if (payload.decoder === "legacy" || payload.decoder === "v1") out.decoder = payload.decoder;
  const tStart = finiteNumber(payload.t_start_ms);
  if (tStart !== undefined) out.t_start_ms = tStart;
  const tEmit = finiteNumber(payload.t_emit_ms);
  if (tEmit !== undefined) out.t_emit_ms = tEmit;
  const decoderConfidence = finiteNumber(payload.decoder_confidence);
  if (decoderConfidence !== undefined) out.decoder_confidence = decoderConfidence;
  return out;
}

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
  /** 1.1.0: refuse to talk to a server running a different model bundle. */
  expected_model_id?: string;
}

export interface ModelFileInfo {
  name: string;
  version: string;
  sha256: string;
}

export interface ModelBundleInfo {
  id: string;
  sha256: string | null;
  files: ModelFileInfo[];
}

export interface ServerClasses {
  hand: string[];
  body: string[];
  attacks: string[];
}

export interface ServerCapabilities {
  initial_mode?: string;
  recognition_modes?: string[];
  geometry?: boolean;
  calibration?: boolean;
  max_fps?: number;
  max_frame_bytes?: number;
  max_frame_width?: number;
  max_frame_height?: number;
  idle_timeout_s?: number;
  heartbeat_s?: number;
  /** 1.2.0: decoder behind HAND_SEAL events. Absent on older servers, which are legacy. */
  hand_decoder?: HandDecoderMode;
  [key: string]: unknown;
}

export interface ServerReadyMessage {
  type: "server_ready";
  protocol_version: string;
  schema_version: string;
  session_id: string;
  supported_controls: ControlCommand[];
  /** 1.1.0 additions; absent when talking to a 1.0.0 server. */
  model?: ModelBundleInfo;
  classes?: ServerClasses;
  capabilities?: ServerCapabilities;
}

export interface ControlMessage {
  type: "control";
  command: ControlCommand;
  request_id?: string;
  mode?: string;
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
  details?: Record<string, unknown>;
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

export function buildClientHello(expectedModelId?: string): ClientHelloMessage {
  const hello: ClientHelloMessage = { type: "client_hello", protocol_version: PROTOCOL_VERSION };
  if (expectedModelId !== undefined) hello.expected_model_id = expectedModelId;
  return hello;
}

export function buildControl(
  command: ControlCommand,
  requestId?: string,
  mode?: string,
): ControlMessage {
  const msg: ControlMessage = { type: "control", command };
  if (requestId !== undefined) msg.request_id = requestId;
  if (mode !== undefined) msg.mode = mode;
  return msg;
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
