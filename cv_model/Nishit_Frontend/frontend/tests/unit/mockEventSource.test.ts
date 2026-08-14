import { beforeEach, afterEach, describe, expect, it, vi } from "vitest";
import { MockEventSource, type MockTapeEntry } from "../../src/transport/mockEventSource";
import type { PipelineEventV1, PipelineOutputV1 } from "../../src/transport/protocol";

function output(overrides: Partial<PipelineOutputV1> = {}, capturedAtMs = 0): PipelineOutputV1 {
  return {
    schema_version: "1.0.0",
    session_id: "mock-session",
    frame_id: 1,
    captured_at_ms: capturedAtMs,
    processing_ms: 10,
    hand: {
      raw: { label: "zero", confidence: 0.9, second_label: "tiger", second_confidence: 0.05, margin: 0.85 },
      center: { label: "zero", confidence: 0.9, second_label: "tiger", second_confidence: 0.05, margin: 0.85 },
      roi: null,
      accepted_label: "zero",
      rejection_reason: null,
      stable_label: "zero",
      emitted_seal: null,
      fusion_status: "center_only",
      detected_hand_count: 0,
      geometry: null,
    },
    body: { raw_label: "idle", stable_label: "idle", emitted_movement: null, metrics: {}, geometry: null },
    queue: { seals: [], accepted_seal: null, duplicate_ignored: false, timeout_cleared: false, max_length_cleared: false, cooldown_suppressed: false },
    attack: null,
    ...overrides,
  };
}

function handSealEvent(id: string, atMs: number): PipelineEventV1 {
  return {
    schema_version: "1.0.0",
    event_id: id,
    event_sequence: 1,
    event_type: "HAND_SEAL",
    session_id: "mock-session",
    frame_id: 1,
    captured_at_ms: atMs,
    payload: { seal: "tiger" },
  };
}

describe("MockEventSource", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("transitions idle -> connecting -> open on connect()", () => {
    const source = new MockEventSource([], { latencyMs: 150 });
    const states: string[] = [];
    source.onConnectionStateChange((state) => states.push(state));

    source.connect();
    expect(source.connectionState).toBe("connecting");

    vi.advanceTimersByTime(150);
    expect(source.connectionState).toBe("open");
    expect(states).toEqual(["connecting", "open"]);
  });

  it("delivers a tape entry after its scheduled time plus latency", () => {
    const tape: MockTapeEntry[] = [{ atMs: 1000, event: handSealEvent("e1", 1000) }];
    const source = new MockEventSource(tape, { latencyMs: 150 });
    const received: PipelineEventV1[] = [];
    source.onEvent((event) => received.push(event));

    source.connect();
    vi.advanceTimersByTime(150); // handshake completes, tape scheduled

    vi.advanceTimersByTime(999);
    expect(received).toHaveLength(0);

    vi.advanceTimersByTime(151); // crosses atMs(1000) + latency(150)
    expect(received).toHaveLength(1);
    expect(received[0]?.event_id).toBe("e1");
  });

  it("delivers the output as a state_snapshot with a matching session id", () => {
    const tape: MockTapeEntry[] = [{ atMs: 0, output: output({}, 500) }];
    const source = new MockEventSource(tape, { latencyMs: 0, sessionId: "test-session" });
    let snapshot: unknown = null;
    source.onStateSnapshot((message) => {
      snapshot = message;
    });

    source.connect();
    // 1ms rather than 0: with zero configured latency both the handshake and
    // the tape entry are scheduled as 0ms timers, and a same-tick 0ms->0ms
    // chain needs a non-zero tick ceiling to guarantee both hops fire.
    vi.advanceTimersByTime(1);

    expect(snapshot).toMatchObject({ type: "state_snapshot", session_id: "test-session" });
  });

  it("does not deliver anything after disconnect() cancels pending timers", () => {
    const tape: MockTapeEntry[] = [{ atMs: 100, event: handSealEvent("e1", 100) }];
    const source = new MockEventSource(tape, { latencyMs: 0 });
    const received: PipelineEventV1[] = [];
    source.onEvent((event) => received.push(event));

    source.connect();
    vi.advanceTimersByTime(1); // open; the tape's 100ms entry is now scheduled but not due
    expect(source.connectionState).toBe("open");
    source.disconnect();

    vi.advanceTimersByTime(1000);
    expect(received).toHaveLength(0);
    expect(source.connectionState).toBe("closed");
  });

  it("acknowledges a control command", () => {
    const source = new MockEventSource([], { latencyMs: 10 });
    let ack: unknown = null;
    source.onAck((message) => {
      ack = message;
    });

    source.sendControl("reset", "req-1");
    vi.advanceTimersByTime(10);

    expect(ack).toEqual({ type: "ack", command: "reset", ok: true, request_id: "req-1" });
  });

  it("discards sendFrame calls without throwing", () => {
    const source = new MockEventSource([]);
    expect(() => source.sendFrame(new Uint8Array([1, 2, 3]))).not.toThrow();
  });
});
