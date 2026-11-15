/** Session model + session.json builder (output validated against schemas/collection_session_v1.schema.json in tests). */
import { PROTOCOL_VERSION, type ClipPlan, type ProtocolConfig, type Seal } from "./protocol";
import { spanOffsets, type LabeledSegment, type SegmentOffsets } from "./timing";

export type CameraType = "laptop" | "external" | "phone";
export type Lighting = "low" | "normal" | "backlit";
export type Background = "plain" | "cluttered";

export interface SubjectInfo {
  subject_id: string;
  camera_type: CameraType;
  lighting: Lighting;
  background: Background;
  distance_m: 0.5 | 1 | 2;
  dominant_hand: "L" | "R";
  monk_skin_tone: number | null;
  consent: { accepted: true; accepted_at: string; text_hash: string };
}

export interface DeviceInfo {
  user_agent: string;
  platform: string;
  screen: { width: number; height: number; device_pixel_ratio: number };
  camera_label?: string;
  requested: { width: number; height: number; fps: number };
  achieved: { width: number; height: number; fps_measured: number };
  mirrored_display: true;
}

/** Persisted in IndexedDB; everything needed to resume. */
export interface SessionRecord {
  session_id: string;
  subject: SubjectInfo;
  device: DeviceInfo;
  protocol: { version: string; seed: string; config: ProtocolConfig };
  plan: ClipPlan[];
  started_at: string;
  ended_at: string | null;
  status: "in_progress" | "complete";
}

/** Per-clip metadata stored alongside (not inside) the blob. */
export interface ClipMeta {
  clip_id: string;
  block: ClipPlan["block"];
  label?: Seal;
  sequence?: Seal[];
  take: number;
  attempt: number;
  mime: string;
  ext: "webm" | "mp4";
  size_bytes: number;
  duration_ms: number;
  width: number;
  height: number;
  fps_measured: number;
  /** seal/combo: filled; free: undefined. */
  segments?: LabeledSegment[];
  free?: { free_start: number; free_end: number };
}

export function extForMime(mime: string): "webm" | "mp4" {
  return mime.includes("mp4") ? "mp4" : "webm";
}

export function clipToJson(m: ClipMeta) {
  const base: Record<string, unknown> = {
    clip_id: m.clip_id,
    block: m.block,
    take: m.take,
    attempt: m.attempt,
    mime: m.mime,
    file: `clips/${m.clip_id}.${m.ext}`,
    size_bytes: m.size_bytes,
    duration_ms: m.duration_ms,
    width: m.width,
    height: m.height,
    fps_measured: m.fps_measured,
  };
  if (m.block === "seal") {
    base.label = m.label;
    const { label: _l, ...offs } = m.segments![0];
    base.phase_offsets_ms = offs satisfies SegmentOffsets;
  } else if (m.block === "combo") {
    base.sequence = m.sequence;
    base.segments = m.segments;
    base.phase_offsets_ms = spanOffsets(m.segments!);
  } else {
    base.phase_offsets_ms = m.free;
  }
  return base;
}

export function buildSessionJson(rec: SessionRecord, clips: ClipMeta[]) {
  const ordered = [...clips].sort((a, b) => planIndex(rec, a) - planIndex(rec, b));
  return {
    schema_version: "collection_session_v1" as const,
    subject: rec.subject,
    session: {
      session_id: rec.session_id,
      started_at: rec.started_at,
      ended_at: rec.ended_at,
      status: rec.status,
    },
    device: rec.device,
    protocol: rec.protocol,
    clips: ordered.map(clipToJson),
  };
}

function planIndex(rec: SessionRecord, c: ClipMeta): number {
  const i = rec.plan.findIndex((p) => p.clip_id === c.clip_id);
  return i < 0 ? Number.MAX_SAFE_INTEGER : i;
}

export function newProtocolInfo(seed: string, config: ProtocolConfig) {
  return { version: PROTOCOL_VERSION, seed, config };
}

// ---- ids -------------------------------------------------------------------

const CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ";

export function randomCode(len: number, getRandom: (n: number) => Uint8Array = defaultRandom): string {
  const bytes = getRandom(len);
  let s = "";
  for (let i = 0; i < len; i++) s += CROCKFORD[bytes[i] % 32];
  return s;
}
function defaultRandom(n: number): Uint8Array {
  return crypto.getRandomValues(new Uint8Array(n));
}
export const newSubjectId = () => `S-${randomCode(6)}`;
export function newSessionId(now: Date = new Date()): string {
  const ts = now.toISOString().replace(/[-:]/g, "").replace(/\..*/, "").replace("T", "-");
  return `${ts}-${randomCode(4).toLowerCase()}`;
}
export const newSeed = () => randomCode(10);

/** Next unrecorded plan item, or null when finished. */
export function nextPlanIndex(plan: ClipPlan[], done: Set<string>): number | null {
  const i = plan.findIndex((p) => !done.has(p.clip_id));
  return i < 0 ? null : i;
}
