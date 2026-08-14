/**
 * The shared contract `websocketClient.ts` and `mockEventSource.ts` both
 * implement. `pipelineStore.ts` (and everything built on it in later phases)
 * depends only on this interface, never on which implementation is wired up
 * -- that's the Phase-B gate: "mockEventSource and websocketClient are
 * interchangeable at the pipelineStore boundary."
 */

import type {
  AckMessage,
  ControlCommand,
  ErrorMessage,
  PipelineEventV1,
  StateSnapshotMessage,
} from "./protocol";

export type ConnectionState =
  | "idle"
  | "connecting"
  | "open"
  | "reconnecting"
  | "closed"
  | "error";

export type Unsubscribe = () => void;

export interface PipelineEventSource {
  readonly connectionState: ConnectionState;

  connect(): void;
  disconnect(): void;

  /** Unmirrored JPEG frame bytes, already wrapped by encodeFrame(). A mock
   * source accepts and discards these; only websocketClient actually sends. */
  sendFrame(bytes: Uint8Array): void;
  sendControl(command: ControlCommand, requestId?: string): void;

  onConnectionStateChange(handler: (state: ConnectionState) => void): Unsubscribe;
  onStateSnapshot(handler: (snapshot: StateSnapshotMessage) => void): Unsubscribe;
  onEvent(handler: (event: PipelineEventV1) => void): Unsubscribe;
  onAck(handler: (ack: AckMessage) => void): Unsubscribe;
  onError(handler: (error: ErrorMessage) => void): Unsubscribe;
}

/** Minimal typed pub/sub used by both implementations to avoid duplicating
 * the same subscribe/unsubscribe/emit boilerplate five times each. */
export class Signal<T> {
  private readonly handlers = new Set<(value: T) => void>();

  subscribe(handler: (value: T) => void): Unsubscribe {
    this.handlers.add(handler);
    return () => {
      this.handlers.delete(handler);
    };
  }

  emit(value: T): void {
    // Copy before iterating: a handler unsubscribing itself (or another
    // handler) mid-emit must not skip or double-call a sibling.
    for (const handler of [...this.handlers]) {
      handler(value);
    }
  }

  get subscriberCount(): number {
    return this.handlers.size;
  }
}
