/**
 * Replays a scripted tape of PipelineOutputV1/PipelineEventV1 entries behind
 * the exact same PipelineEventSource contract websocketClient.ts implements.
 *
 * This is what unblocks Phases C-G from needing a live backend: point
 * pipelineStore at a MockEventSource instead of a WebSocketClient and every
 * downstream consumer (HUD now; scene/game/effects in later phases) behaves
 * identically. See game_implementation_plan.md Phase B gate.
 */

import {
  type AckMessage,
  type ControlCommand,
  buildControl,
} from "./protocol";
import {
  type ConnectionState,
  type PipelineEventSource,
  type Unsubscribe,
  Signal,
} from "./eventSource";
import type { PipelineEventV1, PipelineOutputV1, StateSnapshotMessage, ErrorMessage } from "./protocol";

export interface MockTapeEntry {
  /** Milliseconds from tape start (i.e. from connect() completing) at which
   * this entry is delivered, before latency/jitter is added. */
  atMs: number;
  output?: PipelineOutputV1;
  event?: PipelineEventV1;
}

export interface MockEventSourceOptions {
  /** Base one-way delivery latency. Defaults to the plan's 150 ms figure. */
  latencyMs?: number;
  /** +/- uniform jitter added to latencyMs per delivered entry. */
  jitterMs?: number;
  /** Replay the tape again after it finishes. Off by default: a scripted
   * level tape finishing is meaningful and should not silently repeat. */
  loop?: boolean;
  sessionId?: string;
  setTimeoutFn?: typeof setTimeout;
  clearTimeoutFn?: typeof clearTimeout;
  random?: () => number;
}

export class MockEventSource implements PipelineEventSource {
  private state: ConnectionState = "idle";
  private readonly connectionSignal = new Signal<ConnectionState>();
  private readonly snapshotSignal = new Signal<StateSnapshotMessage>();
  private readonly eventSignal = new Signal<PipelineEventV1>();
  private readonly ackSignal = new Signal<AckMessage>();
  private readonly errorSignal = new Signal<ErrorMessage>();
  private readonly timers = new Set<ReturnType<typeof setTimeout>>();
  private readonly sessionId: string;
  private frameCounter = 0;

  constructor(
    private readonly tape: readonly MockTapeEntry[],
    private readonly options: MockEventSourceOptions = {},
  ) {
    this.sessionId = options.sessionId ?? "mock-session";
  }

  get connectionState(): ConnectionState {
    return this.state;
  }

  connect(): void {
    if (this.state === "open" || this.state === "connecting") {
      return;
    }
    this.setState("connecting");
    this.after(this.jitteredLatency(), () => {
      this.setState("open");
      this.scheduleTape();
    });
  }

  disconnect(): void {
    this.clearTimers();
    this.setState("closed");
  }

  sendFrame(_bytes: Uint8Array): void {
    // Discarded: the mock replays a fixed tape and never reacts to input,
    // matching the real backend's contract shape without a live pipeline.
  }

  sendControl(command: ControlCommand, requestId?: string): void {
    const control = buildControl(command, requestId);
    this.after(this.jitteredLatency(), () => {
      this.ackSignal.emit({
        type: "ack",
        command: control.command,
        ok: true,
        ...(control.request_id !== undefined ? { request_id: control.request_id } : {}),
      });
    });
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

  private setState(next: ConnectionState): void {
    this.state = next;
    this.connectionSignal.emit(next);
  }

  private jitteredLatency(): number {
    const base = this.options.latencyMs ?? 150;
    const jitter = this.options.jitterMs ?? 0;
    if (jitter <= 0) {
      return base;
    }
    const random = this.options.random ?? Math.random;
    const offset = (random() * 2 - 1) * jitter;
    return Math.max(0, Math.round(base + offset));
  }

  private scheduleTape(): void {
    for (const entry of this.tape) {
      this.after(entry.atMs + this.jitteredLatency(), () => this.deliver(entry));
    }
    if (this.options.loop && this.tape.length > 0) {
      const tapeEnd = Math.max(...this.tape.map((entry) => entry.atMs));
      this.after(tapeEnd + this.jitteredLatency() + 50, () => {
        if (this.state === "open") {
          this.scheduleTape();
        }
      });
    }
  }

  private deliver(entry: MockTapeEntry): void {
    if (this.state !== "open") {
      return;
    }
    if (entry.output) {
      this.frameCounter += 1;
      this.snapshotSignal.emit({
        type: "state_snapshot",
        session_id: this.sessionId,
        frame_id: this.frameCounter,
        captured_at_ms: entry.output.captured_at_ms,
        output: entry.output,
      });
    }
    if (entry.event) {
      this.eventSignal.emit(entry.event);
    }
  }

  private after(delayMs: number, callback: () => void): void {
    const setTimeoutFn = this.options.setTimeoutFn ?? setTimeout;
    let timer: ReturnType<typeof setTimeout>;
    timer = setTimeoutFn(() => {
      this.timers.delete(timer);
      callback();
    }, delayMs);
    this.timers.add(timer);
  }

  private clearTimers(): void {
    const clearTimeoutFn = this.options.clearTimeoutFn ?? clearTimeout;
    for (const timer of this.timers) {
      clearTimeoutFn(timer);
    }
    this.timers.clear();
  }
}
