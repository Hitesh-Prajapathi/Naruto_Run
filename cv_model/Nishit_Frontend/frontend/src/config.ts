/**
 * Central frontend settings for Phase B. Kept as one small file rather than
 * a config/ folder -- gameConfig.ts (lanes, speeds, damage) and
 * levelDefinition.ts don't exist yet because there is no level or game loop
 * yet (Phases C/D/F); adding them now would be building ahead of review.
 */

const DEFAULT_TRANSPORT_URL = "ws://127.0.0.1:8765/ws";

export interface AppConfig {
  transportUrl: string;
  /** Force MockEventSource even if a real backend is reachable. Useful for
   * demoing the UI/HUD without cv_model/Nishit_Frontend/serve_transport.py
   * running. Toggle with ?mock=1 in the page URL. */
  useMock: boolean;
  captureWidth: number;
  captureHeight: number;
  targetFps: number;
  jpegQuality: number;
  mockLatencyMs: number;
  mockJitterMs: number;
}

function readQueryFlag(name: string): boolean {
  if (typeof window === "undefined") {
    return false;
  }
  return new URLSearchParams(window.location.search).get(name) === "1";
}

export function loadConfig(): AppConfig {
  return {
    transportUrl: (import.meta.env.VITE_TRANSPORT_URL as string | undefined) ?? DEFAULT_TRANSPORT_URL,
    useMock: readQueryFlag("mock"),
    captureWidth: 640,
    captureHeight: 360,
    targetFps: 30,
    jpegQuality: 0.75,
    mockLatencyMs: 150,
    mockJitterMs: 40,
  };
}
