import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { WebSocketClient } from "../../src/transport/websocketClient";

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
  onclose: (() => void) | null = null;

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

function serverReady(sessionId = "session-1"): string {
  return JSON.stringify({
    type: "server_ready",
    protocol_version: "1.0.0",
    schema_version: "1.0.0",
    session_id: sessionId,
    supported_controls: ["ping", "reset"],
  });
}

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

    expect(sockets[0]!.lastSentJson()).toMatchObject({ type: "client_hello", protocol_version: "1.0.0" });
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
