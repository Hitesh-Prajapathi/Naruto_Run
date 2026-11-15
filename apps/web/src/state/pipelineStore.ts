/**
 * The single source of truth for "what did the backend just say", built
 * only against the PipelineEventSource interface -- never against
 * WebSocketClient or MockEventSource concretely. That's what makes the two
 * interchangeable (Phase B gate): swap the constructor argument and every
 * subscriber here (today: diagnosticsHud; later: game/scene consumers)
 * behaves identically.
 *
 * Two invariants this store owns:
 *  - Only the newest state_snapshot is kept (matches the backend's own
 *    "latest snapshot only" contract -- this store does not add staleness).
 *  - Every event passes through EventDeduplicator first: a repeated
 *    event_id (retry, reconnect overlap, a mock tape replayed) has no
 *    second effect downstream.
 */

import { EventDeduplicator } from "../transport/eventDeduplicator";
import type { ConnectionState, PipelineEventSource, Unsubscribe } from "../transport/eventSource";
import type { ErrorMessage, PipelineEventV1, PipelineOutputV1 } from "../transport/protocol";

export interface PipelineStoreState {
  connectionState: ConnectionState;
  sessionId: string | null;
  latestOutput: PipelineOutputV1 | null;
  latestFrameId: number | null;
  latestCapturedAtMs: number | null;
  /** Small ring buffer of the most recent de-duplicated events, newest last.
   * Sized for HUD display, not as an authoritative event log. */
  recentEvents: PipelineEventV1[];
  lastError: string | null;
}

export type PipelineStoreListener = (state: Readonly<PipelineStoreState>) => void;

const INITIAL_STATE: PipelineStoreState = {
  connectionState: "idle",
  sessionId: null,
  latestOutput: null,
  latestFrameId: null,
  latestCapturedAtMs: null,
  recentEvents: [],
  lastError: null,
};

export interface PipelineStoreOptions {
  maxRecentEvents?: number;
}

export class PipelineStore {
  private state: PipelineStoreState = { ...INITIAL_STATE, recentEvents: [] };
  private readonly listeners = new Set<PipelineStoreListener>();
  private readonly deduplicator = new EventDeduplicator();
  private readonly maxRecentEvents: number;
  private unsubscribers: Unsubscribe[] = [];

  constructor(
    private readonly source: PipelineEventSource,
    options: PipelineStoreOptions = {},
  ) {
    this.maxRecentEvents = options.maxRecentEvents ?? 20;
  }

  /** Start listening to `source`. Safe to call again after detach(). */
  attach(): void {
    this.detach();
    this.unsubscribers = [
      this.source.onConnectionStateChange((connectionState) => this.patch({ connectionState })),
      this.source.onStateSnapshot((snapshot) => this.handleSnapshot(snapshot)),
      this.source.onEvent((event) => this.handleEvent(event)),
      this.source.onError((error) => this.handleError(error)),
    ];
  }

  detach(): void {
    for (const unsubscribe of this.unsubscribers.splice(0)) {
      unsubscribe();
    }
  }

  subscribe(listener: PipelineStoreListener): Unsubscribe {
    this.listeners.add(listener);
    listener(this.state);
    return () => {
      this.listeners.delete(listener);
    };
  }

  getState(): Readonly<PipelineStoreState> {
    return this.state;
  }

  private handleSnapshot(snapshot: {
    session_id: string;
    frame_id: number;
    captured_at_ms: number;
    output: PipelineOutputV1;
  }): void {
    this.resetOnNewSession(snapshot.session_id);
    this.patch({
      sessionId: snapshot.session_id,
      latestOutput: snapshot.output,
      latestFrameId: snapshot.frame_id,
      latestCapturedAtMs: snapshot.captured_at_ms,
    });
  }

  private handleEvent(event: PipelineEventV1): void {
    this.resetOnNewSession(event.session_id);
    if (!this.deduplicator.admit(event)) {
      return;
    }
    const recentEvents = [...this.state.recentEvents, event].slice(-this.maxRecentEvents);
    // Events carry session_id independently of snapshots -- an event-only
    // stream (or one where the event arrives before its snapshot) must still
    // update the tracked session, or resetOnNewSession() never fires.
    this.patch({ sessionId: event.session_id, recentEvents });
  }

  private handleError(error: ErrorMessage): void {
    this.patch({ lastError: `${error.code}: ${error.message}` });
  }

  private resetOnNewSession(sessionId: string): void {
    if (this.state.sessionId !== null && this.state.sessionId !== sessionId) {
      // A reconnect (new backend runtime session) or a mock tape restarting
      // under a different session id: old dedup history and displayed
      // events must never bleed across that boundary.
      this.deduplicator.reset();
      this.patch({ recentEvents: [] });
    }
  }

  private patch(update: Partial<PipelineStoreState>): void {
    this.state = { ...this.state, ...update };
    for (const listener of [...this.listeners]) {
      listener(this.state);
    }
  }
}
