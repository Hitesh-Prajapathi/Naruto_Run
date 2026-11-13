/**
 * Renders whatever pipelineStore currently holds -- nothing here decides a
 * gesture, an attack, or a threshold. This is Phase B's proof that
 * mockEventSource and websocketClient are interchangeable: point the same
 * DiagnosticsHud at a store built on either and it renders identically.
 */

import type { PipelineStore, PipelineStoreState } from "../state/pipelineStore";
import type { CameraState } from "../camera/cameraController";
import type { Unsubscribe } from "../transport/eventSource";

export interface PublisherStats {
  sentFrames: number;
  droppedTicks: number;
}

const CONNECTION_STATUS_CLASS: Record<PipelineStoreState["connectionState"], string> = {
  idle: "status-warn",
  connecting: "status-warn",
  open: "status-ok",
  reconnecting: "status-warn",
  closed: "status-err",
  error: "status-err",
};

export class DiagnosticsHud {
  private cameraState: CameraState = "idle";
  private publisherStats: PublisherStats = { sentFrames: 0, droppedTicks: 0 };
  private storeState: PipelineStoreState | null = null;

  constructor(private readonly container: HTMLElement) {
    this.render();
  }

  bind(store: PipelineStore): Unsubscribe {
    return store.subscribe((state) => {
      this.storeState = state;
      this.render();
    });
  }

  setCameraState(state: CameraState): void {
    this.cameraState = state;
    this.render();
  }

  setPublisherStats(stats: PublisherStats): void {
    this.publisherStats = stats;
    this.render();
  }

  private render(): void {
    const state = this.storeState;
    const connectionClass = state ? CONNECTION_STATUS_CLASS[state.connectionState] : "status-warn";
    const output = state?.latestOutput ?? null;
    const lastEvent = state?.recentEvents.at(-1) ?? null;

    // Third element is "" (not undefined) for "no extra class" -- keeps the
    // tuple type a plain string under exactOptionalPropertyTypes.
    const rows: Array<[string, string, string]> = [
      ["camera", this.cameraState, ""],
      ["connection", state?.connectionState ?? "idle", connectionClass],
      ["session", state?.sessionId ?? "-", ""],
      ["frame", state?.latestFrameId?.toString() ?? "-", ""],
      ["hand (stable)", output?.hand.stable_label ?? "-", ""],
      ["hand (emitted)", output?.hand.emitted_seal ?? "-", ""],
      ["body (stable)", output?.body.stable_label ?? "-", ""],
      ["body (emitted)", output?.body.emitted_movement ?? "-", ""],
      ["queue", output ? `[${output.queue.seals.join(", ")}]` : "-", ""],
      ["attack", output?.attack ? output.attack.display_name : "-", ""],
      ["last event", lastEvent ? `${lastEvent.event_type} @${lastEvent.captured_at_ms}ms` : "-", ""],
      ["events seen", (state?.recentEvents.length ?? 0).toString(), ""],
      ["frames sent", this.publisherStats.sentFrames.toString(), ""],
      ["ticks dropped", this.publisherStats.droppedTicks.toString(), ""],
      ["last error", state?.lastError ?? "-", state?.lastError ? "status-err" : ""],
    ];

    const items = rows
      .map(([label, value, statusClass]) => {
        const valueClass = statusClass ? ` class="${statusClass}"` : "";
        return `<dt>${label}</dt><dd${valueClass}>${escapeHtml(value)}</dd>`;
      })
      .join("");

    this.container.innerHTML = `<dl>${items}</dl>`;
  }
}

function escapeHtml(value: string): string {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}
