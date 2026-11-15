import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { WebSocketClient } from "../../src/transport/websocketClient";
import { PROTOCOL_VERSION } from "../../src/transport/protocol";

/** Minimal WebSocket double: enough surface for websocketClient.ts, plus
 * test-only `simulate*` helpers to drive it from the outside. readyState
 * values (0-3) follow the WebSocket spec, so `WebSocket.OPEN` comparisons in
 * the client under test line up with this fake regardless of environment. */
class FakeWebSocket {
  static readonly CONNECTING = 0;
  static readonly OPEN = 1;
  static readonly CLOSING = 2;
  static readonly CLOSED = 3;

  readyState = FakeWebSocket.CONNECTING;
  binaryType = "blob";
  readonly sent: unknown[] = [];
  onopen: (() => void) | null = null;
  onmessage: ((event: { data: string | ArrayBuffer }) => void) | null = null;
  onerror: (() => void) | null = null;
  onclose: ((event?: CloseEvent) => void) | null = null;

  constructor(public readonly url: string) {}

  send(data: unknown): void {
    this.sent.push(data);
  }

  close(): void {
    if (this.readyState === FakeWebSocket.CLOSED) {
      return;
    }
    this.readyState = FakeWebSocket.CLOSED;
    this.onclose?.();
  }

  simulateOpen(): void {
    this.readyState = FakeWebSocket.OPEN;
    this.onopen?.();
  }

  simulateMessage(data: string): void {
    this.onmessage?.({ data });
  }

  simulateUnexpectedClose(): void {
    this.readyState = FakeWebSocket.CLOSED;
    this.onclose?.();
  }

  lastSentJson(): unknown {
    const last = this.sent.at(-1);
    return typeof last === "string" ? JSON.parse(last) : last;
  }
}

function serverReady(
  sessionId = "session-1",
  extra: Record<string, unknown> = {},
  protocolVersion = "1.0.0",
): string {
  return JSON.stringify({
    type: "server_ready",
    protocol_version: protocolVersion,
    schema_version: "1.0.0",
    session_id: sessionId,
    supported_controls: ["ping", "reset"],
    ...extra,
  });
}

const MODEL = { id: "narutocv-abc123", sha256: "0".repeat(64), files: [] };

describe("WebSocketClient", () => {
  let sockets: FakeWebSocket[];
  let factory: (url: string) => WebSocket;

  beforeEach(() => {
    vi.useFakeTimers();
    sockets = [];
    factory = (url: string) => {
      const socket = new FakeWebSocket(url);
      sockets.push(socket);
      return socket as unknown as WebSocket;
    };
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  function makeClient(overrides: Partial<ConstructorParameters<typeof WebSocketClient>[0]> = {}) {
    return new WebSocketClient({
      url: "ws://localhost:8765/ws",
      webSocketFactory: factory,
      heartbeatIntervalMs: 1000,
      initialBackoffMs: 100,
      maxBackoffMs: 400,
      handshakeTimeoutMs: 500,
      ...overrides,
    });
  }

  it("sends client_hello as soon as the socket opens", () => {
    const client = makeClient();
    client.connect();

    sockets[0]!.simulateOpen();

    expect(sockets[0]!.lastSentJson()).toMatchObject({ type: "client_hello", protocol_version: PROTOCOL_VERSION });
    // No expected model configured: the optional field must be absent so a
    // 1.0.0-era server sees exactly the hello it always did.
    expect(sockets[0]!.lastSentJson()).not.toHaveProperty("expected_model_id");
  });

  it("sends expected_model_id in the hello when configured", () => {
    const client = makeClient({ expectedModelId: "narutocv-abc123" });
    client.connect();
    sockets[0]!.simulateOpen();

    expect(sockets[0]!.lastSentJson()).toMatchObject({ expected_model_id: "narutocv-abc123" });
  });

  it("exposes the 1.1.0 model, classes and capabilities from server_ready", () => {
    const client = makeClient();
    client.connect();
    sockets[0]!.simulateOpen();
    sockets[0]!.simulateMessage(
      serverReady("s", { model: MODEL, classes: { hand: ["dog"], body: [], attacks: [] } }, "1.1.0"),
    );

    expect(client.connectionState).toBe("open");
    expect(client.serverReady?.model?.id).toBe("narutocv-abc123");
    expect(client.serverReady?.classes?.hand).toEqual(["dog"]);
  });

  it("advertises the hand decoder from a 1.2.0 server_ready, defaulting to legacy", () => {
    const client = makeClient();
    expect(client.handDecoder).toBe("legacy");
    client.connect();
    sockets[0]!.simulateOpen();
    sockets[0]!.simulateMessage(
      serverReady("s", { model: MODEL, capabilities: { hand_decoder: "v1" } }, "1.2.0"),
    );

    expect(client.connectionState).toBe("open");
    expect(client.serverReady?.capabilities?.hand_decoder).toBe("v1");
    expect(client.handDecoder).toBe("v1");
  });

  it("treats a 1.1.0 server (no hand_decoder capability) as legacy and still opens", () => {
    const client = makeClient();
    client.connect();
    sockets[0]!.simulateOpen();
    sockets[0]!.simulateMessage(
      serverReady("s", { model: MODEL, capabilities: { max_fps: 60 } }, "1.1.0"),
    );

    expect(client.connectionState).toBe("open");
    expect(client.handDecoder).toBe("legacy");
  });

  it("still opens against a 1.0.0 server that sends no model info", () => {
    const client = makeClient();
    client.connect();
    sockets[0]!.simulateOpen();
    sockets[0]!.simulateMessage(serverReady());

    expect(client.connectionState).toBe("open");
    expect(client.serverReady?.model).toBeUndefined();
  });

  it("treats a different server major version as fatal and does not reconnect", () => {
    const client = makeClient();
    const errors: string[] = [];
    client.onError((error) => errors.push(error.code));
    client.connect();
    sockets[0]!.simulateOpen();
    sockets[0]!.simulateMessage(serverReady("s", {}, "2.0.0"));
    vi.advanceTimersByTime(60_000);

    expect(errors).toContain("protocol_version_mismatch");
    expect(client.connectionState).toBe("error");
    expect(sockets).toHaveLength(1);
  });

  it("treats a model id that differs from the expected one as fatal", () => {
    const client = makeClient({ expectedModelId: "narutocv-other" });
    const errors: string[] = [];
    client.onError((error) => errors.push(error.code));
    client.connect();
    sockets[0]!.simulateOpen();
    sockets[0]!.simulateMessage(serverReady("s", { model: MODEL }, "1.1.0"));
    vi.advanceTimersByTime(60_000);

    expect(errors).toContain("model_mismatch");
    expect(client.connectionState).toBe("error");
    expect(sockets).toHaveLength(1);
  });

  it("does not reconnect after the server rejects the version (error + close 4426)", () => {
    const client = makeClient();
    const errors: string[] = [];
    client.onError((error) => errors.push(error.code));
    client.connect();
    sockets[0]!.simulateOpen();
    sockets[0]!.simulateMessage(
      JSON.stringify({
        type: "error",
        code: "protocol_version_mismatch",
        message: "client speaks protocol 0.9.0",
        details: { server_protocol_version: "1.1.0" },
      }),
    );
    sockets[0]!.readyState = FakeWebSocket.CLOSED;
    sockets[0]!.onclose?.({ code: 4426 } as unknown as CloseEvent);
    vi.advanceTimersByTime(60_000);

    expect(errors).toEqual(["protocol_version_mismatch"]);
    expect(client.connectionState).toBe("error");
    expect(sockets).toHaveLength(1);
  });

  it("does not reconnect on a bare fatal close code", () => {
    const client = makeClient();
    client.connect();
    sockets[0]!.simulateOpen();
    sockets[0]!.simulateMessage(serverReady());
    sockets[0]!.readyState = FakeWebSocket.CLOSED;
    sockets[0]!.onclose?.({ code: 4412 } as unknown as CloseEvent);
    vi.advanceTimersByTime(60_000);

    expect(client.connectionState).toBe("error");
    expect(sockets).toHaveLength(1);
  });

  it("still reconnects after a non-fatal close such as the idle timeout (4408)", () => {
    const client = makeClient({ initialBackoffMs: 100 });
    client.connect();
    sockets[0]!.simulateOpen();
    sockets[0]!.simulateMessage(serverReady());
    sockets[0]!.readyState = FakeWebSocket.CLOSED;
    sockets[0]!.onclose?.({ code: 4408 } as unknown as CloseEvent);
    vi.advanceTimersByTime(100);

    expect(sockets).toHaveLength(2);
  });

  it("becomes open and records the session id after server_ready", () => {
    const client = makeClient();
    const states: string[] = [];
    client.onConnectionStateChange((state) => states.push(state));

    client.connect();
    sockets[0]!.simulateOpen();
    sockets[0]!.simulateMessage(serverReady("abc"));

    expect(client.connectionState).toBe("open");
    expect(client.activeSessionId).toBe("abc");
    expect(states).toEqual(["connecting", "open"]);
  });

  it("forwards a state_snapshot message to subscribers", () => {
    const client = makeClient();
    client.connect();
    sockets[0]!.simulateOpen();
    sockets[0]!.simulateMessage(serverReady());

    let received: unknown = null;
    client.onStateSnapshot((snapshot) => {
      received = snapshot;
    });
    sockets[0]!.simulateMessage(
      JSON.stringify({ type: "state_snapshot", session_id: "session-1", frame_id: 1, captured_at_ms: 10, output: {} }),
    );

    expect(received).toMatchObject({ type: "state_snapshot", frame_id: 1 });
  });

  it("sends a ping on the heartbeat interval once open", () => {
    const client = makeClient({ heartbeatIntervalMs: 1000 });
    client.connect();
    sockets[0]!.simulateOpen();
    sockets[0]!.simulateMessage(serverReady());

    vi.advanceTimersByTime(1000);

    const pings = sockets[0]!.sent.filter((raw) => typeof raw === "string" && JSON.parse(raw).command === "ping");
    expect(pings).toHaveLength(1);
  });

  it("does not send frames before the socket is open", () => {
    const client = makeClient();
    client.connect(); // socket exists but not yet OPEN

    client.sendFrame(new Uint8Array([1, 2, 3]));

    expect(sockets[0]!.sent).toHaveLength(0);
  });

  it("sends frames once open", () => {
    const client = makeClient();
    client.connect();
    sockets[0]!.simulateOpen();
    sockets[0]!.simulateMessage(serverReady());

    const frame = new Uint8Array([1, 2, 3]);
    client.sendFrame(frame);

    expect(sockets[0]!.sent.at(-1)).toBe(frame);
  });

  it("emits a handshake_timeout error and closes the socket if server_ready never arrives", () => {
    const client = makeClient({ handshakeTimeoutMs: 500 });
    const errors: string[] = [];
    client.onError((error) => errors.push(error.code));

    client.connect();
    sockets[0]!.simulateOpen();
    vi.advanceTimersByTime(500);

    expect(errors).toContain("handshake_timeout");
    expect(sockets[0]!.readyState).toBe(FakeWebSocket.CLOSED);
  });

  it("reconnects with doubling backoff after an unexpected close", () => {
    const client = makeClient({ initialBackoffMs: 100, maxBackoffMs: 1000 });
    client.connect();
    sockets[0]!.simulateOpen();
    sockets[0]!.simulateMessage(serverReady());

    sockets[0]!.simulateUnexpectedClose();
    expect(client.connectionState).toBe("reconnecting");
    expect(sockets).toHaveLength(1);

    vi.advanceTimersByTime(100); // first backoff
    expect(sockets).toHaveLength(2);

    sockets[1]!.simulateUnexpectedClose();
    vi.advanceTimersByTime(199);
    expect(sockets).toHaveLength(2); // 200ms not yet elapsed
    vi.advanceTimersByTime(1);
    expect(sockets).toHaveLength(3); // second backoff = 200ms
  });

  it("does not reconnect after a manual disconnect()", () => {
    const client = makeClient();
    client.connect();
    sockets[0]!.simulateOpen();
    sockets[0]!.simulateMessage(serverReady());

    client.disconnect();
    vi.advanceTimersByTime(10_000);

    expect(client.connectionState).toBe("closed");
    expect(sockets).toHaveLength(1);
  });

  it("emits invalid_message on unparseable JSON instead of throwing", () => {
    const client = makeClient();
    const errors: string[] = [];
    client.onError((error) => errors.push(error.code));

    client.connect();
    sockets[0]!.simulateOpen();

    expect(() => sockets[0]!.simulateMessage("not json{{{")).not.toThrow();
    expect(errors).toContain("invalid_message");
  });
});
