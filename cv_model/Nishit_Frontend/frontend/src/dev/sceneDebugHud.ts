/**
 * On-screen readout for the Phase-C scene harness: FPS, pool construction
 * counts (should go flat after warmup -- the "no allocation spikes" gate),
 * and collision feedback. Debug-only; not part of the real game HUD.
 */

import { TRACKING_REASON_TEXT, type BodyInputDebug } from "../input/bodyInputSource";
import type { BodyTuning } from "../input/bodyTuning";

export interface SceneDebugStats {
  fps: number;
  obstacleActive: number;
  obstacleConstructed: number;
  lane: number;
  airborne: boolean;
  totalHits: number;
  segmentElapsedSeconds: number;
  segmentDurationSeconds: number;
  /**
   * Feature Brief 05 §7.3. Extends this panel rather than adding a second
   * one, behind the same toggle. Absent when CV input is disabled, in which
   * case the panel renders exactly as it did before.
   */
  body?: BodyInputDebug | undefined;
  bodyTuning?: BodyTuning | undefined;
  /** End-to-end latency, capture → intent applied. */
  latencyMs?: number | undefined;
}

export class SceneDebugHud {
  constructor(private readonly container: HTMLElement) {}

  render(stats: SceneDebugStats): void {
    const rows: Array<[string, string]> = [
      ["fps", stats.fps.toFixed(0)],
      ["lane", String(stats.lane)],
      ["airborne", stats.airborne ? "yes" : "no"],
      ["obstacles active", String(stats.obstacleActive)],
      ["obstacles constructed", String(stats.obstacleConstructed)],
      ["collisions", String(stats.totalHits)],
      [
        "segment",
        `${stats.segmentElapsedSeconds.toFixed(1)}s / ${stats.segmentDurationSeconds.toFixed(1)}s`,
      ],
    ];

    // --- Feature Brief 05 §7.3 -----------------------------------------
    // "Build this first, before tuning anything. You cannot tune thresholds
    // you cannot see." Every value the brief lists is here, each shown next
    // to the threshold it is being compared against -- a bare number tells
    // you nothing about whether it is about to fire.
    if (stats.body) {
      const body = stats.body;
      const tuning = stats.bodyTuning;
      const lean = body.lean;
      const jump = body.jump;

      rows.push(["— body input —", ""]);
      // The reason, not just the verdict: "lost" alone sent us chasing the
      // wrong cause three separate times.
      rows.push([
        "tracking",
        body.reason === "ok"
          ? body.status
          : `${body.status} — ${TRACKING_REASON_TEXT[body.reason]}`,
      ]);
      if (body.geometrySource !== "shoulders" && body.geometrySource !== "none") {
        rows.push(["geometry from", `${body.geometrySource} (shoulders unreadable)`]);
      }
      rows.push([
        "lean",
        tuning
          ? `${signed(lean.value)}  (enter ${tuning.leanEnter} / exit ${tuning.leanExit})`
          : signed(lean.value),
      ]);
      rows.push(["lean bar", meter(lean.value, tuning?.leanEnter ?? 1)]);
      rows.push(["lean state", `${lean.state}  (held ${lean.heldFrames}f)`]);
      rows.push(["returned to neutral", lean.hasReturnedToNeutral ? "yes" : "NO — blocked"]);
      // Peaks, not instantaneous values: the peaks are what the thresholds
      // are actually compared against, so showing the live number alone made
      // the overlay disagree with the detector's own decision.
      rows.push([
        "jump rise",
        tuning
          ? `${jump.rise.toFixed(3)}  peak ${jump.peakRise.toFixed(3)}  (fires ≥ ${tuning.jumpHeightFraction})`
          : jump.rise.toFixed(3),
      ]);
      rows.push([
        "jump velocity",
        tuning
          ? `${signed(jump.velocity)}  peak ${signed(jump.peakVelocity)}  (needs ≥ ${tuning.jumpMinVelocity})`
          : signed(jump.velocity),
      ]);
      rows.push([
        "jump refractory",
        jump.refractoryMs > 0 ? `${jump.refractoryMs.toFixed(0)}ms — blocked` : "ready",
      ]);
      rows.push(["pipeline label", jump.labelActive ? "jumping" : "—"]);
      rows.push([
        "baseline",
        `tilt ${signed(body.baseline.leanOffset)}  shoulders ${body.baseline.bodyScale.toFixed(3)}  (${body.baseline.sampleCount} samples)`,
      ]);
      // A large drift is the signature of "one side works, the other
      // doesn't", so it is worth reading at a glance rather than inferring
      // from a lean value that refuses to sit at zero.
      rows.push([
        "neutral drift",
        tuning && Math.abs(body.neutralBias) > tuning.leanEnter * 0.5
          ? `${signed(body.neutralBias)} — large, press R to recalibrate`
          : signed(body.neutralBias),
      ]);
      // Both lean and jump are measured between the two shoulders, so how
      // wide apart they sit is the single number that says whether the
      // framing can support either.
      rows.push([
        "framing",
        body.baseline.bodyScale > 0.75
          ? `shoulders ${body.baseline.bodyScale.toFixed(2)} — too close, step back`
          : body.baseline.bodyScale < 0.15
            ? `shoulders ${body.baseline.bodyScale.toFixed(2)} — too far, step closer`
            : "ok",
      ]);
      // A sample age far above one frame means the pipeline is behind, and
      // the tracking status will read "lost" no matter how clearly the player
      // is standing there -- worth calling out rather than leaving as a
      // number nobody thinks to compare against staleEventMs.
      const stale = tuning !== undefined && body.sampleAgeMs > tuning.staleEventMs;
      rows.push([
        "event rate",
        `${body.sampleRate}/s  (age ${body.sampleAgeMs.toFixed(0)}ms${stale ? " — BEHIND" : ""})`,
      ]);
      rows.push(["intents", `${body.laneEvents} lane / ${body.jumpEvents} jump`]);
      if (stats.latencyMs !== undefined) {
        rows.push(["end-to-end", `${stats.latencyMs.toFixed(0)}ms`]);
      }
    }

    const items = rows
      .map(([label, value]) =>
        value === ""
          ? `<dt class="dbg-sep">${escapeHtml(label)}</dt><dd></dd>`
          : `<dt>${escapeHtml(label)}</dt><dd>${escapeHtml(value)}</dd>`,
      )
      .join("");
    this.container.innerHTML = `<dl>${items}</dl>`;
  }
}

function signed(value: number): string {
  return `${value >= 0 ? "+" : ""}${value.toFixed(3)}`;
}

/** A crude text meter, so a glance shows how close a value is to firing. */
function meter(value: number, threshold: number): string {
  const width = 11; // odd, so there is a true centre
  const centre = Math.floor(width / 2);
  const clamped = Math.max(-1, Math.min(1, threshold > 0 ? value / threshold : 0));
  const position = Math.round(centre + clamped * centre);
  const cells: string[] = [];
  for (let i = 0; i < width; i += 1) {
    cells.push(i === position ? "#" : i === centre ? "|" : "-");
  }
  return `L ${cells.join("")} R`;
}

function escapeHtml(value: string): string {
  return value.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
