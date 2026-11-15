/**
 * Phase B entry point: wires camera + transport (real or mock) + store + HUD
 * together. No game/scene/effects code lives here yet -- that starts in
 * Phase C. This file's only job is to prove the plumbing works end to end.
 */

import { loadConfig } from "./config";
import { CameraController, type CameraState } from "./camera/cameraController";
import { FramePublisher } from "./camera/framePublisher";
import { createDemoTape } from "./transport/demoTape";
import { MockEventSource } from "./transport/mockEventSource";
import { WebSocketClient } from "./transport/websocketClient";
import type { PipelineEventSource } from "./transport/eventSource";
import { PipelineStore } from "./state/pipelineStore";
import { CameraPreview } from "./ui/cameraPreview";
import { DiagnosticsHud } from "./ui/diagnosticsHud";

function requireElement<T extends Element>(selector: string): T {
  const element = document.querySelector<T>(selector);
  if (!element) {
    throw new Error(`missing required element: ${selector}`);
  }
  return element;
}

function describeCamera(state: CameraState, error: string | null): string {
  switch (state) {
    case "idle":
      return "camera: idle";
    case "requesting":
      return "camera: requesting permission…";
    case "ready":
      return "camera: ready";
    case "denied":
      return `camera: permission denied${error ? ` (${error})` : ""}`;
    case "ended":
      return "camera: stopped";
    case "error":
      return `camera: error${error ? ` (${error})` : ""}`;
  }
}

function main(): void {
  const config = loadConfig();

  const video = requireElement<HTMLVideoElement>("#camera-source");
  const previewCanvas = requireElement<HTMLCanvasElement>("#camera-preview");
  const statusEl = requireElement<HTMLDivElement>("#camera-status");
  const startButton = requireElement<HTMLButtonElement>("#camera-start");
  const hudContainer = requireElement<HTMLDivElement>("#diagnostics-hud");

  const source: PipelineEventSource = config.useMock
    ? new MockEventSource(createDemoTape(), {
        latencyMs: config.mockLatencyMs,
        jitterMs: config.mockJitterMs,
        loop: true,
      })
    : new WebSocketClient({ url: config.transportUrl });

  const store = new PipelineStore(source);
  store.attach();

  const hud = new DiagnosticsHud(hudContainer);
  hud.bind(store);

  const camera = new CameraController({ video });
  const preview = new CameraPreview(video, previewCanvas);

  let publisher: FramePublisher | null = null;

  camera.onStateChange((state) => {
    statusEl.textContent = describeCamera(state, camera.lastError);
    hud.setCameraState(state);
    startButton.disabled = state === "requesting" || state === "ready";

    if (state === "ready") {
      preview.start();
      publisher = new FramePublisher({
        video,
        source,
        targetFps: config.targetFps,
        jpegQuality: config.jpegQuality,
        captureWidth: config.captureWidth,
        captureHeight: config.captureHeight,
      });
      publisher.start();
      startPublisherStatsLoop(publisher, hud);
    } else {
      preview.stop();
      publisher?.stop();
      publisher = null;
    }
  });

  startButton.addEventListener("click", () => {
    void camera.start();
  });

  window.addEventListener("beforeunload", () => {
    publisher?.stop();
    preview.stop();
    camera.stop();
    source.disconnect();
  });

  source.connect();
}

function startPublisherStatsLoop(publisher: FramePublisher, hud: DiagnosticsHud): void {
  const interval = window.setInterval(() => {
    if (!publisher.isRunning) {
      window.clearInterval(interval);
      return;
    }
    hud.setPublisherStats(publisher.stats);
  }, 500);
}

main();
