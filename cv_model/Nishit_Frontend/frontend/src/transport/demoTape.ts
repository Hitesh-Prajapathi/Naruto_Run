/**
 * A small illustrative MockEventSource tape used only for local dev/demo
 * (main.ts, when no backend is running). It exercises one hand seal, one
 * body movement, and one attack so the diagnostics HUD has something real to
 * render without a live camera or Python process.
 *
 * This is deliberately NOT the "Segment-1 + Battle-1" scripted tape the plan
 * mentions for later phases -- that tape is level/battle content and belongs
 * with Phase C (environment) and Phase F (battle), once there is a level and
 * a battle to script against. Building it now would be getting ahead of the
 * phase currently in review.
 */

import type { MockTapeEntry } from "./mockEventSource";
import type { PipelineOutputV1 } from "./protocol";

const SESSION_ID = "demo-session";

function baseOutput(overrides: Partial<PipelineOutputV1>, capturedAtMs: number): PipelineOutputV1 {
  return {
    schema_version: "1.0.0",
    session_id: SESSION_ID,
    frame_id: Math.round(capturedAtMs / 33),
    captured_at_ms: capturedAtMs,
    processing_ms: 31.2,
    hand: {
      raw: { label: "zero", confidence: 0.98, second_label: "tiger", second_confidence: 0.01, margin: 0.97 },
      center: { label: "zero", confidence: 0.98, second_label: "tiger", second_confidence: 0.01, margin: 0.97 },
      roi: null,
      accepted_label: "zero",
      rejection_reason: null,
      stable_label: "zero",
      emitted_seal: null,
      fusion_status: "center_only",
      detected_hand_count: 0,
      geometry: null,
    },
    body: {
      raw_label: "idle",
      stable_label: "idle",
      emitted_movement: null,
      metrics: {},
      geometry: null,
    },
    queue: {
      seals: [],
      accepted_seal: null,
      duplicate_ignored: false,
      timeout_cleared: false,
      max_length_cleared: false,
      cooldown_suppressed: false,
    },
    attack: null,
    ...overrides,
  };
}

export function createDemoTape(): MockTapeEntry[] {
  const tape: MockTapeEntry[] = [];

  tape.push({ atMs: 0, output: baseOutput({}, 0) });

  const tigerAtMs = 1500;
  tape.push({
    atMs: tigerAtMs,
    output: baseOutput(
      {
        hand: {
          raw: { label: "tiger", confidence: 0.94, second_label: "horse", second_confidence: 0.03, margin: 0.91 },
          center: { label: "tiger", confidence: 0.94, second_label: "horse", second_confidence: 0.03, margin: 0.91 },
          roi: null,
          accepted_label: "tiger",
          rejection_reason: null,
          stable_label: "tiger",
          emitted_seal: "tiger",
          fusion_status: "center_only",
          detected_hand_count: 1,
          geometry: null,
        },
        queue: {
          seals: ["tiger"],
          accepted_seal: "tiger",
          duplicate_ignored: false,
          timeout_cleared: false,
          max_length_cleared: false,
          cooldown_suppressed: false,
        },
      },
      tigerAtMs,
    ),
    event: {
      schema_version: "1.0.0",
      event_id: `${SESSION_ID}:00000001`,
      event_sequence: 1,
      event_type: "HAND_SEAL",
      session_id: SESSION_ID,
      frame_id: Math.round(tigerAtMs / 33),
      captured_at_ms: tigerAtMs,
      payload: { seal: "tiger", stable_label: "tiger", confidence: 0.94 },
    },
  });

  const horseAtMs = 2600;
  tape.push({
    atMs: horseAtMs,
    output: baseOutput(
      {
        hand: {
          raw: { label: "horse", confidence: 0.91, second_label: "tiger", second_confidence: 0.02, margin: 0.89 },
          center: { label: "horse", confidence: 0.91, second_label: "tiger", second_confidence: 0.02, margin: 0.89 },
          roi: null,
          accepted_label: "horse",
          rejection_reason: null,
          stable_label: "horse",
          emitted_seal: "horse",
          fusion_status: "center_only",
          detected_hand_count: 1,
          geometry: null,
        },
        queue: {
          seals: [],
          accepted_seal: "horse",
          duplicate_ignored: false,
          timeout_cleared: false,
          max_length_cleared: false,
          cooldown_suppressed: false,
        },
        attack: { name: "homura", display_name: "FIRE ATTACK", recognized_at_ms: horseAtMs },
      },
      horseAtMs,
    ),
    event: {
      schema_version: "1.0.0",
      event_id: `${SESSION_ID}:00000002`,
      event_sequence: 2,
      event_type: "ATTACK_TRIGGERED",
      session_id: SESSION_ID,
      frame_id: Math.round(horseAtMs / 33),
      captured_at_ms: horseAtMs,
      payload: { name: "homura", display_name: "FIRE ATTACK", recognized_at_ms: horseAtMs },
    },
  });

  const jumpAtMs = 4000;
  tape.push({
    atMs: jumpAtMs,
    output: baseOutput(
      {
        body: {
          raw_label: "jumping",
          stable_label: "jumping",
          emitted_movement: "jumping",
          metrics: { jump_height: 0.06 },
          geometry: null,
        },
      },
      jumpAtMs,
    ),
    event: {
      schema_version: "1.0.0",
      event_id: `${SESSION_ID}:00000003`,
      event_sequence: 3,
      event_type: "BODY_MOVEMENT",
      session_id: SESSION_ID,
      frame_id: Math.round(jumpAtMs / 33),
      captured_at_ms: jumpAtMs,
      payload: { movement: "jumping", stable_label: "jumping", metrics: { jump_height: 0.06 } },
    },
  });

  return tape;
}
