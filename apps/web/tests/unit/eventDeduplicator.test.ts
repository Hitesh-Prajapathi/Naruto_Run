import { describe, expect, it } from "vitest";
import { EventDeduplicator } from "../../src/transport/eventDeduplicator";
import type { PipelineEventV1 } from "../../src/transport/protocol";

function event(eventId: string): PipelineEventV1 {
  return {
    schema_version: "1.0.0",
    event_id: eventId,
    event_sequence: 0,
    event_type: "HAND_SEAL",
    session_id: "s1",
    frame_id: 1,
    captured_at_ms: 0,
    payload: {},
  };
}

describe("EventDeduplicator", () => {
  it("admits the first occurrence of an event_id", () => {
    const dedup = new EventDeduplicator();
    expect(dedup.admit(event("a"))).toBe(true);
  });

  it("rejects a repeated event_id", () => {
    const dedup = new EventDeduplicator();
    dedup.admit(event("a"));

    expect(dedup.admit(event("a"))).toBe(false);
  });

  it("treats different event_ids independently", () => {
    const dedup = new EventDeduplicator();
    expect(dedup.admit(event("a"))).toBe(true);
    expect(dedup.admit(event("b"))).toBe(true);
    expect(dedup.admit(event("a"))).toBe(false);
  });

  it("evicts the oldest tracked id once maxTracked is exceeded", () => {
    const dedup = new EventDeduplicator(2);
    dedup.admit(event("a"));
    dedup.admit(event("b"));
    dedup.admit(event("c")); // evicts "a"

    expect(dedup.admit(event("a"))).toBe(true); // re-admitted: no longer tracked
    expect(dedup.admit(event("c"))).toBe(false); // still tracked
  });

  it("forgets everything after reset", () => {
    const dedup = new EventDeduplicator();
    dedup.admit(event("a"));
    dedup.reset();

    expect(dedup.admit(event("a"))).toBe(true);
    expect(dedup.trackedCount).toBe(1);
  });

  it("rejects a non-positive maxTracked", () => {
    expect(() => new EventDeduplicator(0)).toThrow(RangeError);
  });
});
