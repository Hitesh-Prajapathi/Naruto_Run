/**
 * Real transport implementation of PipelineEventSource: connects to
 * cv_model/Nishit_Frontend/serve_transport.py's GET /ws, performs the
 * client_hello/server_ready handshake, heartbeats with `ping`, and
 * reconnects with bounded exponential backoff on an unexpected close.
 *
 * Interchangeable with mockEventSource.ts at the pipelineStore boundary --
 * this class carries bytes and parses messages; it never decides what a
 * gesture or attack is.
 */

import {
  type AckMessage,
  type ControlCommand,
  type ErrorMessage,
  type PipelineEventV1,
  type StateSnapshotMessage,
  type TransportMessage,
  ProtocolError,
  buildClientHello,
  buildControl,
  parseTransportMessage,
} from "./protocol";
import {
  type ConnectionState,
  type PipelineEventSource,
  type Unsubscribe,
  Signal,
} from "./eventSource";

export interface WebSocketClientOptions {
  url: string;
  /** How often to send a `ping` control once the session is open. */
  heartbeatIntervalMs?: number;
  /** First reconnect delay; doubles each consecutive failure up to maxBackoffMs. */
  initialBackoffMs?: number;
  maxBackoffMs?: number;
  /** How long to wait for server_ready after the socket opens. */
  handshakeTimeoutMs?: number;
  /** Injectable for tests -- must return a WebSocket-shaped object. */
  webSocketFactory?: (url: string) => WebSocket;
}

const DEFAULT_HEARTBEAT_INTERVAL_MS = 15_000;
const DEFAULT_INITIAL_BACKOFF_MS = 500;
const DEFAULT_MAX_BACKOFF_MS = 8_000;
const DEFAULT_HANDSHAKE_TIMEOUT_MS = 5_000;

export class WebSocketClient implements PipelineEventSource {
  private state: ConnectionState = "idle";
  private socket: WebSocket | null = null;
  private manuallyDisconnected = false;
  private reconnectAttempts = 0;
  private sessionId: string | null = null;

  private heartbeatTimer: ReturnType<typeof setInterval> | null = null;
  private handshakeTimer: ReturnType<typeof setTimeout> | null = null;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;

  private readonly connectionSignal = new Signal<ConnectionState>();
  private readonly snapshotSignal = new Signal<StateSnapshotMessage>();
  private readonly eventSignal = new Signal<PipelineEventV1>();
  private readonly ackSignal = new Signal<AckMessage>();
  private readonly errorSignal = new Signal<ErrorMessage>();

  constructor(private readonly options: WebSocketClientOptions) {}

  get connectionState(): ConnectionState {
    return this.state;
  }

  get activeSessionId(): string | null {
    return this.sessionId;
  }

  connect(): void {
    this.manuallyDisconnected = false;
    if (this.state === "open" || this.state === "connecting") {
      return;
    }
    this.openSocket();
  }

  disconnect(): void {
    this.manuallyDisconnected = true;
    this.clearTimers();
    if (this.socket) {
      // Avoid a duplicate close-driven reconnect attempt for a socket we are
      // intentionally tearing down.
      this.socket.onclose = null;
      this.socket.close();
      this.socket = null;
    }
    this.setState("closed");
  }

  sendFrame(bytes: Uint8Array): void {
    if (this.socket && this.socket.readyState === WebSocket.OPEN) {
      this.socket.send(bytes);
    }
  }

  sendControl(command: ControlCommand, requestId?: string): void {
    if (this.socket && this.socket.readyState === WebSocket.OPEN) {
      this.socket.send(JSON.stringify(buildControl(command, requestId)));
    }
  }

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

  private openSocket(): void {
    this.setState(this.reconnectAttempts > 0 ? "reconnecting" : "connecting");
    const factory = this.options.webSocketFactory ?? ((url: string) => new WebSocket(url));
    const socket = factory(this.options.url);
    socket.binaryType = "arraybuffer";
    this.socket = socket;

    this.handshakeTimer = setTimeout(() => {
      this.errorSignal.emit({
        type: "error",
        code: "handshake_timeout",
        message: "server_ready was not received within the handshake window",
      });
      socket.close();
    }, this.options.handshakeTimeoutMs ?? DEFAULT_HANDSHAKE_TIMEOUT_MS);

    socket.onopen = () => {
      socket.send(JSON.stringify(buildClientHello()));
    };
    socket.onmessage = (event: MessageEvent) => {
      this.handleMessage(event);
    };
    socket.onerror = () => {
      this.errorSignal.emit({ type: "error", code: "socket_error", message: "WebSocket transport error" });
    };
    socket.onclose = () => {
      this.handleClose();
    };
  }

  private handleMessage(event: MessageEvent): void {
    if (typeof event.data !== "string") {
      // Binary messages only ever flow browser -> server on this contract;
      // the server never streams video back (game_implementation_plan.md).
      return;
    }
    let parsed: TransportMessage;
    try {
      parsed = parseTransportMessage(JSON.parse(event.data));
    } catch (error) {
      const message = error instanceof ProtocolError ? error.message : String(error);
      this.errorSignal.emit({ type: "error", code: "invalid_message", message });
      return;
    }

    switch (parsed.type) {
      case "server_ready":
        this.sessionId = parsed.session_id;
        this.clearHandshakeTimer();
        this.reconnectAttempts = 0;
        this.setState("open");
        this.startHeartbeat();
        break;
      case "ack":
        this.ackSignal.emit(parsed);
        break;
      case "error":
        this.errorSignal.emit(parsed);
        break;
      case "state_snapshot":
        this.snapshotSignal.emit(parsed);
        break;
      case "pipeline_event":
        this.eventSignal.emit(parsed.event);
        break;
      case "client_hello":
      case "control":
        // Never sent server -> client; ignore rather than error the session
        // over a message shape that simply doesn't apply to this direction.
        break;
    }
  }

  private startHeartbeat(): void {
    this.stopHeartbeat();
    const interval = this.options.heartbeatIntervalMs ?? DEFAULT_HEARTBEAT_INTERVAL_MS;
    this.heartbeatTimer = setInterval(() => this.sendControl("ping"), interval);
  }

  private stopHeartbeat(): void {
    if (this.heartbeatTimer !== null) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }

  private clearHandshakeTimer(): void {
    if (this.handshakeTimer !== null) {
      clearTimeout(this.handshakeTimer);
      this.handshakeTimer = null;
    }
  }

  private handleClose(): void {
    this.stopHeartbeat();
    this.clearHandshakeTimer();
    this.socket = null;
    this.sessionId = null;
    if (this.manuallyDisconnected) {
      this.setState("closed");
      return;
    }
    this.scheduleReconnect();
  }

  private scheduleReconnect(): void {
    this.setState("reconnecting");
    const initial = this.options.initialBackoffMs ?? DEFAULT_INITIAL_BACKOFF_MS;
    const max = this.options.maxBackoffMs ?? DEFAULT_MAX_BACKOFF_MS;
    const delay = Math.min(max, initial * 2 ** this.reconnectAttempts);
    this.reconnectAttempts += 1;
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      if (!this.manuallyDisconnected) {
        this.openSocket();
      }
    }, delay);
  }

  private clearTimers(): void {
    this.stopHeartbeat();
    this.clearHandshakeTimer();
    if (this.reconnectTimer !== null) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
  }

  private setState(next: ConnectionState): void {
    this.state = next;
    this.connectionSignal.emit(next);
  }
}
