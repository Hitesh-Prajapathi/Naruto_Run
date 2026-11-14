/**
 * Online seal decoder -- TypeScript port of `src/narutocv/temporal/decoder.py`
 * (spec: `docs/specs/seal_decoder_v1.md`).
 *
 * Input is one calibrated probability vector per frame plus a timestamp in milliseconds;
 * output is at most one {@link SealEvent} per seal entry. All timing is in milliseconds,
 * never frames. The state machine, comparison operators and operand order are identical to
 * the Python reference; parity is enforced by `tests/golden/decoder_v1.json`
 * (see `docs/specs/parity.md`).
 *
 * Wire-format names (`t_start_ms`, `half_life_ms`, ...) are kept in snake_case so a
 * `DecoderConfig` can be loaded straight from `calibration.json`.
 */

export const DEFAULT_LABELS: readonly string[] = [
  "bird",
  "boar",
  "dog",
  "dragon",
  "hare",
  "horse",
  "monkey",
  "ox",
  "ram",
  "rat",
  "snake",
  "tiger",
  "zero",
];

export interface SealEvent {
  label: string;
  /** First update of the continuous run with smoothed p >= tau_on. */
  t_start_ms: number;
  /** Timestamp of the update that emitted the event. */
  t_emit_ms: number;
  /** Smoothed probability of `label` at emission. */
  confidence: number;
}

export interface DecoderConfig {
  labels: string[];
  idle_label: string | null;
  smoother: "ema" | "hmm";
  half_life_ms: number;
  tau_on: number;
  tau_off: number;
  tau_reject: number;
  min_dwell_ms: number;
  refractory_ms: number;
  max_gap_ms: number;
  prior_boost: number;
  use_prior: boolean;
  hmm_p_stay: number;
  hmm_ref_ms: number;
}

export interface DecoderStats {
  frames: number;
  nonmonotonic: number;
  gaps: number;
  events: number;
}

const CONFIG_KEYS: readonly string[] = [
  "labels",
  "idle_label",
  "smoother",
  "half_life_ms",
  "tau_on",
  "tau_off",
  "tau_reject",
  "min_dwell_ms",
  "refractory_ms",
  "max_gap_ms",
  "prior_boost",
  "use_prior",
  "hmm_p_stay",
  "hmm_ref_ms",
];

/** Defaults identical to Python's `DecoderConfig()`. */
export function defaultDecoderConfig(): DecoderConfig {
  return {
    labels: [...DEFAULT_LABELS],
    idle_label: "zero",
    smoother: "ema",
    half_life_ms: 80.0,
    tau_on: 0.6,
    tau_off: 0.4,
    tau_reject: 0.3,
    min_dwell_ms: 120.0,
    refractory_ms: 250.0,
    max_gap_ms: 200.0,
    prior_boost: 3.0,
    use_prior: true,
    hmm_p_stay: 0.95,
    hmm_ref_ms: 33.3,
  };
}

/** Validate a full config (same rules and order as Python's `DecoderConfig.__post_init__`). */
export function validateDecoderConfig(cfg: DecoderConfig): DecoderConfig {
  const k = cfg.labels.length;
  if (k < 2 || new Set(cfg.labels).size !== k) {
    throw new Error("labels must be at least two unique names");
  }
  if (cfg.idle_label !== null && !cfg.labels.includes(cfg.idle_label)) {
    throw new Error("idle_label must be one of labels");
  }
  if (cfg.smoother !== "ema" && cfg.smoother !== "hmm") {
    throw new Error("smoother must be 'ema' or 'hmm'");
  }
  if (!(0.0 < cfg.tau_reject && cfg.tau_reject <= cfg.tau_off && cfg.tau_off < cfg.tau_on && cfg.tau_on <= 1.0)) {
    throw new Error("require 0 < tau_reject <= tau_off < tau_on <= 1");
  }
  // Written as `x <= 0` (not `!(x > 0)`) to match Python, which lets NaN through here.
  if (cfg.half_life_ms <= 0.0 || cfg.hmm_ref_ms <= 0.0) {
    throw new Error("half_life_ms and hmm_ref_ms must be positive");
  }
  if (cfg.min_dwell_ms < 0.0 || cfg.refractory_ms < 0.0) {
    throw new Error("min_dwell_ms and refractory_ms must be >= 0");
  }
  if (cfg.max_gap_ms <= 0.0) throw new Error("max_gap_ms must be positive");
  if (cfg.prior_boost < 1.0) throw new Error("prior_boost must be >= 1");
  if (!(1.0 / k < cfg.hmm_p_stay && cfg.hmm_p_stay < 1.0)) {
    throw new Error("hmm_p_stay must be in (1/K, 1)");
  }
  return cfg;
}

/** Defaults overlaid with `overrides`, then validated. Unknown keys are an error. */
export function makeDecoderConfig(overrides: Partial<DecoderConfig> = {}): DecoderConfig {
  const unknown = Object.keys(overrides).filter((key) => !CONFIG_KEYS.includes(key));
  if (unknown.length > 0) {
    throw new Error(`unknown DecoderConfig keys: ${JSON.stringify(unknown.sort())}`);
  }
  const merged: DecoderConfig = { ...defaultDecoderConfig(), ...overrides };
  merged.labels = [...merged.labels];
  return validateDecoderConfig(merged);
}

/** Load a config from the JSON object stored in `calibration.json` (`DecoderConfig.to_dict`). */
export function decoderConfigFromDict(data: Record<string, unknown>): DecoderConfig {
  return makeDecoderConfig(data as Partial<DecoderConfig>);
}

/** Online filter over class probabilities. */
export interface Smoother {
  reset(): void;
  /** Fold in one observation `dtMs` after the previous one (ignored on the first). */
  update(probs: Float64Array, dtMs: number): Float64Array;
}

/** Time-aware exponential moving average: `alpha = 1 - 2**(-dt / half_life)`. */
export class EmaSmoother implements Smoother {
  state: Float64Array | null = null;

  constructor(readonly halfLifeMs: number) {}

  reset(): void {
    this.state = null;
  }

  update(probs: Float64Array, dtMs: number): Float64Array {
    if (this.state === null) {
      this.state = Float64Array.from(probs);
    } else {
      const alpha = 1.0 - Math.pow(2.0, -dtMs / this.halfLifeMs);
      const next = new Float64Array(probs.length);
      for (let i = 0; i < probs.length; i += 1) {
        const s = this.state[i] as number;
        next[i] = s + alpha * ((probs[i] as number) - s);
      }
      this.state = next;
    }
    return this.state;
  }
}

/**
 * Online forward filter with a sticky transition matrix (see the Python docstring for the
 * closed form): `decay = lam ** (dt / ref)`, `predicted = decay * b + (1 - decay) / K`,
 * `b = predicted * p ** (dt / ref)`, normalised (uniform if the sum underflows).
 */
export class HMMSmoother implements Smoother {
  readonly lam: number;
  state: Float64Array | null = null;

  constructor(
    readonly nClasses: number,
    readonly pStay: number = 0.95,
    readonly refMs: number = 33.3,
  ) {
    if (!(1.0 / nClasses < pStay && pStay < 1.0)) throw new Error("p_stay must be in (1/K, 1)");
    this.lam = (nClasses * pStay - 1.0) / (nClasses - 1.0);
  }

  reset(): void {
    this.state = null;
  }

  update(probs: Float64Array, dtMs: number): Float64Array {
    const k = this.nClasses;
    let belief = new Float64Array(k);
    if (this.state === null) {
      belief.set(probs);
    } else {
      const decay = Math.pow(this.lam, dtMs / this.refMs);
      const power = dtMs / this.refMs;
      for (let i = 0; i < k; i += 1) {
        const predicted = decay * (this.state[i] as number) + (1.0 - decay) / k;
        belief[i] = predicted * Math.pow(probs[i] as number, power);
      }
    }
    let total = 0;
    for (let i = 0; i < k; i += 1) total += belief[i] as number;
    if (total <= 1e-300) {
      belief = new Float64Array(k).fill(1.0 / k);
    } else {
      for (let i = 0; i < k; i += 1) belief[i] = (belief[i] as number) / total;
    }
    this.state = belief;
    return belief;
  }
}

export interface UpdateOptions {
  /** Grammar prior: classes expected next (ignored unless `use_prior`). */
  expectedNext?: readonly string[] | undefined;
}

/** Streaming decoder; call {@link SealDecoder.update} once per frame. */
export class SealDecoder {
  readonly config: DecoderConfig;
  stats: DecoderStats = { frames: 0, nonmonotonic: 0, gaps: 0, events: 0 };

  private readonly index = new Map<string, number>();
  private readonly idle: number | null;
  private readonly smoother: Smoother;
  private lastT: number | null = null;
  private active: number | null = null;
  private pending: number | null = null;
  private pendingStart = 0.0;
  private lastEmitT: number | null = null;
  private smoothedState: Float64Array = new Float64Array(0);

  /** `config` entries override the defaults; the result is validated. */
  constructor(config: Partial<DecoderConfig> = {}) {
    this.config = makeDecoderConfig(config);
    this.config.labels.forEach((name, i) => this.index.set(name, i));
    this.idle = this.config.idle_label !== null ? (this.index.get(this.config.idle_label) as number) : null;
    this.smoother =
      this.config.smoother === "hmm"
        ? new HMMSmoother(this.config.labels.length, this.config.hmm_p_stay, this.config.hmm_ref_ms)
        : new EmaSmoother(this.config.half_life_ms);
  }

  /** Copy of the last smoothed vector (empty before the first accepted frame). */
  get smoothed(): Float64Array {
    return Float64Array.from(this.smoothedState);
  }

  get activeLabel(): string | null {
    return this.active === null ? null : (this.config.labels[this.active] as string);
  }

  /** Forget all state (smoothing, hysteresis, refractory) and counters. */
  reset(): void {
    this.smoother.reset();
    this.stats = { frames: 0, nonmonotonic: 0, gaps: 0, events: 0 };
    this.lastT = null;
    this.active = null;
    this.pending = null;
    this.pendingStart = 0.0;
    this.lastEmitT = null;
    this.smoothedState = new Float64Array(0);
  }

  /** Clip negatives to 0 and renormalise (all-zero becomes uniform); throws on bad input. */
  private prepare(probs: ArrayLike<number>): Float64Array {
    const k = this.config.labels.length;
    if (probs.length !== k) {
      throw new Error(`probs must have shape (${k},), got (${probs.length},)`);
    }
    const p = new Float64Array(k);
    for (let i = 0; i < k; i += 1) {
      const v = Number(probs[i]);
      if (!Number.isFinite(v)) throw new Error("probs must be finite");
      p[i] = v;
    }
    for (let i = 0; i < k; i += 1) if ((p[i] as number) < 0) p[i] = 0;
    let total = 0;
    for (let i = 0; i < k; i += 1) total += p[i] as number;
    if (total <= 0.0) return new Float64Array(k).fill(1.0 / k);
    for (let i = 0; i < k; i += 1) p[i] = (p[i] as number) / total;
    return p;
  }

  private applyPrior(p: Float64Array, expectedNext: readonly string[]): Float64Array {
    const weights = new Float64Array(p.length).fill(1.0);
    for (const name of expectedNext) {
      const idx = this.index.get(name);
      if (idx === undefined) throw new Error(`unknown label in expected_next: ${JSON.stringify(name)}`);
      if (idx !== this.idle) weights[idx] = this.config.prior_boost;
    }
    const boosted = new Float64Array(p.length);
    let total = 0;
    for (let i = 0; i < p.length; i += 1) {
      boosted[i] = (p[i] as number) * (weights[i] as number);
      total += boosted[i] as number;
    }
    if (!(total > 0.0)) return p;
    for (let i = 0; i < p.length; i += 1) boosted[i] = (boosted[i] as number) / total;
    return boosted;
  }

  /** Process one frame; returns the events emitted by it (zero or one). */
  update(probs: ArrayLike<number>, tMs: number, options: UpdateOptions = {}): SealEvent[] {
    const cfg = this.config;
    let p = this.prepare(probs);
    if (!Number.isFinite(tMs)) throw new Error("t_ms must be finite");
    if (this.lastT !== null && tMs <= this.lastT) {
      this.stats.nonmonotonic += 1;
      return [];
    }
    const expectedNext = options.expectedNext;
    if (cfg.use_prior && expectedNext !== undefined && expectedNext.length > 0) {
      p = this.applyPrior(p, expectedNext);
    }

    let dt: number;
    if (this.lastT === null) {
      dt = 0.0;
    } else {
      dt = tMs - this.lastT;
      if (dt > cfg.max_gap_ms) {
        // Continuity broken: restart smoothing, drop any dwell in progress.
        this.stats.gaps += 1;
        this.smoother.reset();
        this.pending = null;
      }
    }
    this.lastT = tMs;
    this.stats.frames += 1;
    const s = this.smoother.update(p, dt);
    this.smoothedState = s;

    let top = 0;
    for (let i = 1; i < s.length; i += 1) if ((s[i] as number) > (s[top] as number)) top = i;
    const sTop = s[top] as number;
    // An active seal ends only when its own smoothed probability decays below tau_off
    // (pure hysteresis: an idle-dominated frame above tau_off does not end it).
    if (this.active !== null && (s[this.active] as number) < cfg.tau_off) this.active = null;
    // Open-set reject: nothing confident, or idle is the arg-max -> no candidate.
    if (sTop < cfg.tau_reject || top === this.idle) {
      this.pending = null;
      return [];
    }
    // Candidate: the top class, if non-idle and above tau_on.
    if (!(sTop >= cfg.tau_on)) {
      this.pending = null;
      return [];
    }
    const candidate = top;
    if (this.active === candidate) return [];
    if (this.active !== null) this.active = null; // direct switch to a different, confident seal
    if (this.pending !== candidate) {
      this.pending = candidate;
      this.pendingStart = tMs;
    }
    if (tMs - this.pendingStart < cfg.min_dwell_ms) return [];
    if (this.lastEmitT !== null && tMs - this.lastEmitT < cfg.refractory_ms) {
      return []; // deferred: stays pending, emits after the refractory period
    }
    const event: SealEvent = {
      label: cfg.labels[candidate] as string,
      t_start_ms: this.pendingStart,
      t_emit_ms: tMs,
      confidence: s[candidate] as number,
    };
    this.active = candidate;
    this.pending = null;
    this.lastEmitT = tMs;
    this.stats.events += 1;
    return [event];
  }
}
