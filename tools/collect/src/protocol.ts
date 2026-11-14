/**
 * Prompted-recording protocol. EVERYTHING the owner might want to tune lives here.
 * Bump PROTOCOL_VERSION whenever the defaults or semantics change.
 */
import { hashSeed, mulberry32, shuffle } from "./rng";

export const PROTOCOL_VERSION = "1.0.0";

export const SEALS = [
  "bird", "boar", "dog", "dragon", "hare", "horse", "monkey", "ox", "ram", "rat", "snake", "tiger", "zero",
] as const;
export type Seal = (typeof SEALS)[number];

export interface ProtocolConfig {
  seals: Seal[];
  /** Takes per seal; the order of the 13 seals is reshuffled for every take. */
  sealTakes: number;
  countdownSec: number;
  /** Seal block phase durations. */
  formMs: number;
  holdMs: number;
  releaseMs: number;
  /** Combo block: prompted sequences (one clip each). Owner: edit this list. */
  combos: Seal[][];
  /** Combo per-seal phase durations (shorter than the single-seal block). */
  comboFormMs: number;
  comboHoldMs: number;
  comboReleaseMs: number;
  /** Free-motion block. */
  freeMotionMs: number;
  freeSuggestions: string[];
  freeSuggestionEveryMs: number;
  /** Extra recording after the last phase ends, so the release is not clipped. */
  tailMs: number;
}

export const DEFAULT_CONFIG: ProtocolConfig = {
  seals: [...SEALS],
  sealTakes: 3,
  countdownSec: 3,
  formMs: 2000,
  holdMs: 3000,
  releaseMs: 1500,
  combos: [
    ["tiger", "horse"],
    ["snake", "dragon"],
    ["dog", "rat"],
  ],
  comboFormMs: 1500,
  comboHoldMs: 1500,
  comboReleaseMs: 500,
  freeMotionMs: 120_000,
  freeSuggestions: [
    "Talk with your hands as if explaining something",
    "Touch your face and hair",
    "Use your phone (or pretend to)",
    "Clap",
    "Prayer hands, then relax",
    "Stretch your arms and fingers",
  ],
  freeSuggestionEveryMs: 12_000,
  tailMs: 300,
};

/** Short protocol for automated smoke tests (enabled with ?fast in the URL). Never use for real data. */
export const FAST_CONFIG: ProtocolConfig = {
  ...DEFAULT_CONFIG,
  seals: ["tiger", "rat"],
  sealTakes: 2,
  countdownSec: 1,
  formMs: 400,
  holdMs: 500,
  releaseMs: 300,
  combos: [["tiger", "horse"]],
  comboFormMs: 300,
  comboHoldMs: 300,
  comboReleaseMs: 200,
  freeMotionMs: 2500,
  freeSuggestionEveryMs: 1000,
};

export type Block = "seal" | "combo" | "free";

export interface ClipPlan {
  /** Unique within the session, filesystem-safe, e.g. c003_seal_ram_t1 */
  clip_id: string;
  /** 0-based position in the whole plan. */
  index: number;
  block: Block;
  take: number;
  label?: Seal;
  sequence?: Seal[];
}

const pad = (n: number) => String(n + 1).padStart(3, "0");

/** Deterministic plan from (seed, config). Same inputs -> identical plan. */
export function generatePlan(seed: string, cfg: ProtocolConfig = DEFAULT_CONFIG): ClipPlan[] {
  const plan: ClipPlan[] = [];
  const push = (p: Omit<ClipPlan, "index" | "clip_id">, tag: string) => {
    const index = plan.length;
    plan.push({ ...p, index, clip_id: `c${pad(index)}_${tag}` });
  };
  for (let take = 1; take <= cfg.sealTakes; take++) {
    const rng = mulberry32(hashSeed(`${seed}|seal|take${take}`));
    for (const label of shuffle(cfg.seals, rng)) {
      push({ block: "seal", take, label }, `seal_${label}_t${take}`);
    }
  }
  cfg.combos.forEach((sequence, i) => {
    push({ block: "combo", take: 1, sequence: [...sequence] }, `combo${i + 1}_${sequence.join("-")}`);
  });
  push({ block: "free", take: 1 }, "free");
  return plan;
}

export function planCounts(plan: ClipPlan[]): Record<Block, number> {
  const c: Record<Block, number> = { seal: 0, combo: 0, free: 0 };
  for (const p of plan) c[p.block]++;
  return c;
}

// ---------------------------------------------------------------------------
// Timeline: nominal schedule of UI steps relative to clip start (recording start).
// ---------------------------------------------------------------------------

export type MarkName = "form_start" | "hold_start" | "hold_end" | "release_end";

export interface Mark {
  segment: number;
  name: MarkName;
}

export interface TimelineStep {
  /** Nominal time (ms from clip start) at which the step should be shown. */
  at: number;
  phase: "countdown" | "form" | "hold" | "release" | "free" | "end";
  /** Seal being prompted (form/hold/release), segment index for combos. */
  seal?: Seal;
  segment?: number;
  /** Countdown number (3,2,1). */
  n?: number;
  /** Free-motion suggestion text. */
  text?: string;
  /** Ground-truth marks recorded when this step is displayed. */
  marks: Mark[];
}

export function buildTimeline(plan: ClipPlan, cfg: ProtocolConfig = DEFAULT_CONFIG): TimelineStep[] {
  const steps: TimelineStep[] = [];
  let t = 0;
  for (let n = cfg.countdownSec; n >= 1; n--) {
    steps.push({ at: t, phase: "countdown", n, marks: [] });
    t += 1000;
  }
  if (plan.block === "free") {
    steps.push({ at: t, phase: "free", marks: [], text: cfg.freeSuggestions[0] });
    const freeStart = t;
    for (let k = 1; k * cfg.freeSuggestionEveryMs < cfg.freeMotionMs; k++) {
      steps.push({
        at: freeStart + k * cfg.freeSuggestionEveryMs,
        phase: "free",
        marks: [],
        text: cfg.freeSuggestions[k % cfg.freeSuggestions.length],
      });
    }
    // free_start / free_end are handled by the caller from the first 'free' and the 'end' step.
    steps.push({ at: freeStart + cfg.freeMotionMs, phase: "end", marks: [] });
    return steps;
  }
  const seals = plan.block === "seal" ? [plan.label!] : plan.sequence!;
  const [fMs, hMs, rMs] =
    plan.block === "seal"
      ? [cfg.formMs, cfg.holdMs, cfg.releaseMs]
      : [cfg.comboFormMs, cfg.comboHoldMs, cfg.comboReleaseMs];
  seals.forEach((seal, segment) => {
    const formMarks: Mark[] = [{ segment, name: "form_start" }];
    // release_end of the previous segment coincides with this form start.
    if (segment > 0) formMarks.unshift({ segment: segment - 1, name: "release_end" });
    steps.push({ at: t, phase: "form", seal, segment, marks: formMarks });
    t += fMs;
    steps.push({ at: t, phase: "hold", seal, segment, marks: [{ segment, name: "hold_start" }] });
    t += hMs;
    steps.push({ at: t, phase: "release", seal, segment, marks: [{ segment, name: "hold_end" }] });
    t += rMs;
  });
  steps.push({ at: t, phase: "end", marks: [{ segment: seals.length - 1, name: "release_end" }] });
  return steps;
}

/** When to stop the MediaRecorder (ms from clip start). */
export function clipEndMs(steps: TimelineStep[], cfg: ProtocolConfig = DEFAULT_CONFIG): number {
  return steps[steps.length - 1].at + cfg.tailMs;
}
