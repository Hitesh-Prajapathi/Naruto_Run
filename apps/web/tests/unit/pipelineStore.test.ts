import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { PipelineStore } from "../../src/state/pipelineStore";
import { MockEventSource, type MockTapeEntry } from "../../src/transport/mockEventSource";
import { Signal, type ConnectionState, type PipelineEventSource, type Unsubscribe } from "../../src/transport/eventSource";
import type {
  AckMessage,
  ControlCommand,
  ErrorMessage,
  PipelineEventV1,
  PipelineOutputV1,
  StateSnapshotMessage,
} from "../../src/transport/protocol";

/** A hand-driven double implementing the exact same PipelineEventSource
 * contract as WebSocketClient, so the pipelineStore boundary can be exercised
 * without either real networking or the mock's timer-based replay. */
class FakeSource implements PipelineEventSource {
  connectionState: ConnectionState = "idle";
  private readonly connectionSignal = new Signal<ConnectionState>();
  private readonly snapshotSignal = new Signal<StateSnapshotMessage>();
  private readonly eventSignal = new Signal<PipelineEventV1>();
  private readonly ackSignal = new Signal<AckMessage>();
  private readonly errorSignal = new Signal<ErrorMessage>();

  connect(): void {
    this.connectionState = "open";
    this.connectionSignal.emit("open");
  }

  disconnect(): void {
    this.connectionState = "closed";
    this.connectionSignal.emit("closed");
  }

  sendFrame(): void {}
  sendControl(_command: ControlCommand): void {}

  onConnectionStateChange(handler: (state: ConnectionState) => void): Unsubscribe {
    return this.connectionSignal.subscribe(handler);
  }
  onStateSnapshot(handler: (snapshot: StateSnapshotMessage) => void): Unsubscribe {
    return this.snapshotSignal.subscribe(handler);
  }
  onEvent(handler: (event: PipelineEventV1) => void): Unsubscribe {
    return this.eventSignal.subscribe(handler);
  }
  onAck(handler: (ack: AckMessage) => void): Unsubscribe {
    return this.ackSignal.subscribe(handler);
  }
  onError(handler: (error: ErrorMessage) => void): Unsubscribe {
    return this.errorSignal.subscribe(handler);
  }

  emitSnapshot(snapshot: StateSnapshotMessage): void {
    this.snapshotSignal.emit(snapshot);
  }
  emitEvent(event: PipelineEventV1): void {
    this.eventSignal.emit(event);
  }
  emitError(error: ErrorMessage): void {
    this.errorSignal.emit(error);
  }
}

function output(sessionId: string, frameId: number): PipelineOutputV1 {
  return {
    schema_version: "1.0.0",
    session_id: sessionId,
    frame_id: frameId,
    captured_at_ms: frameId * 33,
    processing_ms: 10,
    hand: {
      raw: { label: "tiger", confidence: 0.9, second_label: "horse", second_confidence: 0.03, margin: 0.87 },
      center: { label: "tiger", confidence: 0.9, second_label: "horse", second_confidence: 0.03, margin: 0.87 },
      roi: null,
      accepted_label: "tiger",
      rejection_reason: null,
      stable_label: "tiger",
      emitted_seal: "tiger",
      fusion_status: "center_only",
      detected_hand_count: 1,
      geometry: null,
    },
    body: { raw_label: "idle", stable_label: "idle", emitted_movement: null, metrics: {}, geometry: null },
    queue: { seals: ["tiger"], accepted_seal: "tiger", duplicate_ignored: false, timeout_cleared: false, max_length_cleared: false, cooldown_suppressed: false },
    attack: null,
  };
}

function handSealEvent(sessionId: string, eventId: string): PipelineEventV1 {
  return {
    schema_version: "1.0.0",
    event_id: eventId,
    event_sequence: 1,
    event_type: "HAND_SEAL",
    session_id: sessionId,
    frame_id: 1,
    captured_at_ms: 100,
    payload: { seal: "tiger" },
  };
}

describe("PipelineStore against a hand-driven source", () => {
  it("starts idle with no output", () => {
    const store = new PipelineStore(new FakeSource());
    const state = store.getState();
    expect(state.connectionState).toBe("idle");
    expect(state.latestOutput).toBeNull();
  });

  it("tracks connection state changes", () => {
    const source = new FakeSource();
    const store = new PipelineStore(source);
    store.attach();

    source.connect();

    expect(store.getState().connectionState).toBe("open");
  });

  it("keeps only the newest snapshot", () => {
    const source = new FakeSource();
    const store = new PipelineStore(source);
    store.attach();
    source.connect();

    source.emitSnapshot({ type: "state_snapshot", session_id: "s1", frame_id: 1, captured_at_ms: 33, output: output("s1", 1) });
    source.emitSnapshot({ type: "state_snapshot", session_id: "s1", frame_id: 2, captured_at_ms: 66, output: output("s1", 2) });

    expect(store.getState().latestFrameId).toBe(2);
  });

  it("drops a duplicate event_id", () => {
    const source = new FakeSource();
    const store = new PipelineStore(source);
    store.attach();
    source.connect();

    const event = handSealEvent("s1", "s1:0001");
    source.emitEvent(event);
    source.emitEvent(event); // repeat -- must have no second effect

    expect(store.getState().recentEvents).toHaveLength(1);
  });

  it("resets dedup and event history across a session boundary", () => {
    const source = new FakeSource();
    const store = new PipelineStore(source);
    store.attach();
    source.connect();

    source.emitEvent(handSealEvent("s1", "s1:0001"));
    expect(store.getState().recentEvents).toHaveLength(1);

    // A reconnect landed on a new backend session (different session_id).
    source.emitEvent(handSealEvent("s2", "s1:0001")); // same event_id, new session
    const state = store.getState();
    expect(state.sessionId).toBe("s2");
    expect(state.recentEvents).toHaveLength(1); // old history cleared, this one re-admitted
  });

  it("records the last error", () => {
    const source = new FakeSource();
    const store = new PipelineStore(source);
    store.attach();

    source.emitError({ type: "error", code: "invalid_frame", message: "bad header" });

    expect(store.getState().lastError).toBe("invalid_frame: bad header");
  });

  it("stops updating after detach()", () => {
    const source = new FakeSource();
    const store = new PipelineStore(source);
    store.attach();
    store.detach();

    source.emitEvent(handSealEvent("s1", "s1:0001"));

    expect(store.getState().recentEvents).toHaveLength(0);
  });
});

describe("PipelineStore interchangeability (Phase B gate)", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  it("reaches equivalent state whether driven by a hand-built source or MockEventSource", () => {
    const sessionId = "gate-session";

    const fake = new FakeSource();
    const fakeStore = new PipelineStore(fake);
    fakeStore.attach();
    fake.connect();
    fake.emitSnapshot({ type: "state_snapshot", session_id: sessionId, frame_id: 1, captured_at_ms: 100, output: output(sessionId, 1) });
    fake.emitEvent(handSealEvent(sessionId, `${sessionId}:0001`));

    const tape: MockTapeEntry[] = [
      { atMs: 0, output: output(sessionId, 1) },
      { atMs: 0, event: handSealEvent(sessionId, `${sessionId}:0001`) },
    ];
    const mock = new MockEventSource(tape, { latencyMs: 0, sessionId });
    const mockStore = new PipelineStore(mock);
    mockStore.attach();
    mock.connect();
    // 1ms rather than 0: with zero configured latency, the handshake and tape
    // delivery are both scheduled as 0ms timers chained within one callback;
    // a non-zero tick ceiling is needed to guarantee both hops fire.
    vi.advanceTimersByTime(1);

    const fakeState = fakeStore.getState();
    const mockState = mockStore.getState();

    expect(mockState.connectionState).toBe("open");
    expect(fakeState.connectionState).toBe("open");
    expect(mockState.sessionId).toBe(fakeState.sessionId);
    expect(mockState.latestOutput?.hand.stable_label).toBe(fakeState.latestOutput?.hand.stable_label);
    expect(mockState.recentEvents.map((event) => event.event_id)).toEqual(
      fakeState.recentEvents.map((event) => event.event_id),
    );
  });
});
