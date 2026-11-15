import { describe, expect, it } from "vitest";
import { buildSessionJson, newSessionId, newSubjectId, randomCode } from "../src/session";
import { makeRecord, makeValidator, simulateClip } from "./helpers";

const validate = makeValidator();
const fullSession = () => {
  const rec = makeRecord("S1", "complete");
  // Record in reverse to prove the builder re-orders by plan.
  const clips = [...rec.plan].reverse().map((p) => simulateClip(p, 3));
  return buildSessionJson(rec, clips);
};
const errs = () => JSON.stringify(validate.errors);

describe("session.json builder + schema", () => {
  it("a complete session validates", () => {
    const s = fullSession();
    expect(validate(s), errs()).toBe(true);
    expect(s.clips).toHaveLength(43);
    expect(s.clips.map((c) => c.clip_id)).toEqual(makeRecord("S1").plan.map((p) => p.clip_id));
  });

  it("an in-progress partial session validates", () => {
    const rec = makeRecord("S1");
    const s = buildSessionJson(rec, rec.plan.slice(0, 5).map((p) => simulateClip(p)));
    expect(validate(s), errs()).toBe(true);
    expect(s.session.ended_at).toBeNull();
  });

  it("serialises through JSON unchanged (no undefined leaks)", () => {
    const s = JSON.parse(JSON.stringify(fullSession()));
    expect(validate(s), errs()).toBe(true);
  });

  it("clip shapes by block", () => {
    const s = fullSession();
    const seal = s.clips[0] as any, combo = s.clips[39] as any, free = s.clips[42] as any;
    expect(seal.label).toBeTruthy();
    expect(seal.phase_offsets_ms).toMatchObject({ form_start: 3003, hold_start: 5003, hold_end: 8003, release_end: 9503 });
    expect(combo.sequence).toEqual(["tiger", "horse"]);
    expect(combo.segments).toHaveLength(2);
    expect(free.phase_offsets_ms).toEqual({ free_start: 3003, free_end: 123003 });
    expect(free.label).toBeUndefined();
  });

  it("rejects malformed sessions", () => {
    const bad = (mutate: (s: any) => void) => {
      const s = JSON.parse(JSON.stringify(fullSession()));
      mutate(s);
      return validate(s);
    };
    expect(bad((s) => delete s.clips[0].label)).toBe(false);
    expect(bad((s) => (s.clips[0].label = "pigeon"))).toBe(false);
    expect(bad((s) => delete s.clips[0].phase_offsets_ms.hold_end)).toBe(false);
    expect(bad((s) => (s.clips[0].phase_offsets_ms.hold_end = -1))).toBe(false);
    expect(bad((s) => (s.clips[42].label = "rat"))).toBe(false);
    expect(bad((s) => delete s.clips[39].segments)).toBe(false);
    expect(bad((s) => (s.subject.monk_skin_tone = 11))).toBe(false);
    expect(bad((s) => (s.subject.monk_skin_tone = 4))).toBe(true);
    expect(bad((s) => (s.subject.distance_m = 3))).toBe(false);
    expect(bad((s) => (s.subject.email = "a@b.c"))).toBe(false); // no PII fields allowed
    expect(bad((s) => (s.device.mirrored_display = false))).toBe(false);
    expect(bad((s) => (s.clips[0].file = "../evil.webm"))).toBe(false);
    expect(bad((s) => (s.subject.subject_id = "../x"))).toBe(false);
    expect(bad((s) => (s.schema_version = "v2"))).toBe(false);
  });
});

describe("ids", () => {
  it("subject and session ids match the schema pattern", () => {
    expect(newSubjectId()).toMatch(/^S-[0-9A-Z]{6}$/);
    expect(newSessionId(new Date("2026-11-13T10:20:30Z"))).toMatch(/^20261113-102030-[0-9a-z]{4}$/);
    expect(randomCode(5, () => new Uint8Array([0, 1, 31, 32, 33]))).toBe("01Z01");
  });
});
