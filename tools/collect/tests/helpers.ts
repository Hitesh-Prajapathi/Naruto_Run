import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import Ajv2020 from "ajv/dist/2020.js";
import addFormats from "ajv-formats";
import { buildTimeline, clipEndMs, DEFAULT_CONFIG, generatePlan, type ClipPlan } from "../src/protocol";
import { PhaseClock } from "../src/timing";
import { newProtocolInfo, type ClipMeta, type DeviceInfo, type SessionRecord, type SubjectInfo } from "../src/session";

export const schemaPath = fileURLToPath(new URL("../../../schemas/collection_session_v1.schema.json", import.meta.url));

export function makeValidator() {
  const ajv = new Ajv2020({ allErrors: true, strict: true, strictRequired: false });
  addFormats(ajv);
  return ajv.compile(JSON.parse(readFileSync(schemaPath, "utf8")));
}

export const subject: SubjectInfo = {
  subject_id: "S-7K3QMD",
  camera_type: "laptop",
  lighting: "normal",
  background: "plain",
  distance_m: 1,
  dominant_hand: "R",
  monk_skin_tone: null,
  consent: { accepted: true, accepted_at: "2026-11-13T10:00:00.000Z", text_hash: "deadbeef" },
};

export const device: DeviceInfo = {
  user_agent: "vitest",
  platform: "test",
  screen: { width: 1920, height: 1080, device_pixel_ratio: 2 },
  requested: { width: 1280, height: 720, fps: 30 },
  achieved: { width: 1280, height: 720, fps_measured: 29.8 },
  mirrored_display: true,
};

/** Runs a plan item's timeline against a simulated clock (jitter in ms added to every mark). */
export function simulateClip(p: ClipPlan, jitter = 0, attempt = 1): ClipMeta {
  const steps = buildTimeline(p, DEFAULT_CONFIG);
  const clock = new PhaseClock();
  clock.anchor(10_000);
  for (const s of steps) {
    for (const m of s.marks) clock.mark(m.segment, m.name, 10_000 + s.at + jitter);
    if (p.block === "free" && s.phase === "end") clock.markFree("end", 10_000 + s.at + jitter);
  }
  if (p.block === "free") clock.markFree("start", 10_000 + steps.find((x) => x.phase === "free")!.at + jitter);
  const seals = p.block === "seal" ? [p.label!] : p.sequence;
  return {
    clip_id: p.clip_id,
    block: p.block,
    label: p.label,
    sequence: p.sequence,
    take: p.take,
    attempt,
    mime: "video/webm;codecs=vp9",
    ext: "webm",
    size_bytes: 1234,
    duration_ms: clipEndMs(steps, DEFAULT_CONFIG),
    width: 1280,
    height: 720,
    fps_measured: 29.8,
    segments: seals ? clock.segments(seals) : undefined,
    free: p.block === "free" ? { free_start: clock.freeStart!, free_end: clock.freeEnd! } : undefined,
  };
}

export function makeRecord(seed = "TESTSEED01", status: SessionRecord["status"] = "in_progress"): SessionRecord {
  return {
    session_id: "20261113-100000-abcd",
    subject,
    device,
    protocol: newProtocolInfo(seed, DEFAULT_CONFIG),
    plan: generatePlan(seed, DEFAULT_CONFIG),
    started_at: "2026-11-13T10:00:00.000Z",
    ended_at: status === "complete" ? "2026-11-13T10:20:00.000Z" : null,
    status,
  };
}
