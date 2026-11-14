/**
 * Probability calibration and stacked fusion (inference only) -- TypeScript port of
 * `src/narutocv/models/calibration.py`. Fitting stays in Python; weights arrive as the
 * JSON dicts produced by `to_dict()` (`calibration.json` in the model bundle).
 *
 * Parity with Python is enforced by `tests/golden/calibration_v1.json`
 * (see `docs/specs/parity.md`).
 */

export const T_MIN = 0.05;
export const T_MAX = 20.0;

/** Numerically stable log-softmax of one row (same operation order as the numpy version). */
export function logSoftmax(logits: ArrayLike<number>): Float64Array {
  const n = logits.length;
  let max = -Infinity;
  for (let i = 0; i < n; i += 1) if ((logits[i] as number) > max) max = logits[i] as number;
  const shifted = new Float64Array(n);
  let sum = 0;
  for (let i = 0; i < n; i += 1) {
    shifted[i] = (logits[i] as number) - max;
    sum += Math.exp(shifted[i] as number);
  }
  const logSum = Math.log(sum);
  for (let i = 0; i < n; i += 1) shifted[i] = (shifted[i] as number) - logSum;
  return shifted;
}

/** Numerically stable softmax of one row. */
export function softmax(logits: ArrayLike<number>): Float64Array {
  const ls = logSoftmax(logits);
  for (let i = 0; i < ls.length; i += 1) ls[i] = Math.exp(ls[i] as number);
  return ls;
}

export interface TemperatureScalerDict {
  type: "temperature";
  temperature: number;
  fitted?: boolean;
}

/** `p = softmax(logits / T)`, `T in [0.05, 20]`. */
export class TemperatureScaler {
  readonly temperature: number;
  readonly fitted: boolean;

  constructor(temperature = 1.0, fitted = false) {
    if (!(T_MIN <= temperature && temperature <= T_MAX)) {
      throw new Error(`temperature must be within [${T_MIN}, ${T_MAX}]`);
    }
    this.temperature = temperature;
    this.fitted = fitted;
  }

  static fromDict(data: { type?: unknown; temperature?: unknown; fitted?: unknown }): TemperatureScaler {
    if (data.type !== "temperature") throw new Error("not a temperature scaler payload");
    return new TemperatureScaler(Number(data.temperature), data.fitted === undefined ? true : Boolean(data.fitted));
  }

  toDict(): TemperatureScalerDict {
    return { type: "temperature", temperature: this.temperature, fitted: this.fitted };
  }

  /** Scaled logits `logits / T`. */
  transformLogits(logits: ArrayLike<number>): Float64Array {
    const out = new Float64Array(logits.length);
    for (let i = 0; i < logits.length; i += 1) out[i] = (logits[i] as number) / this.temperature;
    return out;
  }

  /** Calibrated probabilities for one row of logits. */
  transform(logits: ArrayLike<number>): Float64Array {
    return softmax(this.transformLogits(logits));
  }

  predictProba(logits: ArrayLike<number>): Float64Array {
    return this.transform(logits);
  }
}

export interface VectorScalerDict {
  type: "vector";
  n_classes: number;
  l2: number;
  scale: number[];
  bias: number[];
}

/** `p = softmax(a * logits + b)` with per-class scale and bias. */
export class VectorScaler {
  constructor(
    readonly scale: Float64Array,
    readonly bias: Float64Array,
  ) {
    if (scale.length !== bias.length) throw new Error("scale and bias must have the same length");
  }

  static fromDict(data: VectorScalerDict | { type?: unknown; scale?: unknown; bias?: unknown }): VectorScaler {
    if (data.type !== "vector") throw new Error("not a vector scaler payload");
    return new VectorScaler(Float64Array.from(data.scale as number[]), Float64Array.from(data.bias as number[]));
  }

  transform(logits: ArrayLike<number>): Float64Array {
    if (logits.length !== this.scale.length) throw new Error("logits have the wrong number of classes");
    const z = new Float64Array(logits.length);
    for (let i = 0; i < z.length; i += 1) {
      z[i] = (logits[i] as number) * (this.scale[i] as number) + (this.bias[i] as number);
    }
    return softmax(z);
  }

  predictProba(logits: ArrayLike<number>): Float64Array {
    return this.transform(logits);
  }
}

/** `log(clip(p, eps, 1))` -- the per-branch input representation for fusion. */
export function logProbFeatures(probs: ArrayLike<number>, eps = 1e-6): Float64Array {
  const out = new Float64Array(probs.length);
  for (let i = 0; i < probs.length; i += 1) {
    const v = probs[i] as number;
    // numpy clip(v, eps, 1.0): min(max(v, eps), 1)
    out[i] = Math.log(Math.min(Math.max(v, eps), 1.0));
  }
  return out;
}

/**
 * Concatenate per-branch log-probabilities (each of length `K_b`) and meta features for ONE
 * sample (the batch version of the Python helper applied to a single row).
 */
export function buildFusionFeatures(
  branchProbs: readonly ArrayLike<number>[],
  meta: ArrayLike<number> | null = null,
  eps = 1e-6,
): Float64Array {
  const parts: ArrayLike<number>[] = branchProbs.map((p) => logProbFeatures(p, eps));
  if (meta !== null) parts.push(meta);
  const total = parts.reduce((n, p) => n + p.length, 0);
  const out = new Float64Array(total);
  let offset = 0;
  for (const part of parts) {
    for (let i = 0; i < part.length; i += 1) out[offset + i] = part[i] as number;
    offset += part.length;
  }
  return out;
}

export interface StackedFusionDict {
  type: "stacked_fusion";
  l2: number;
  standardize: boolean;
  mean: number[];
  std: number[];
  weights: number[][];
  bias: number[];
}

/** Multinomial logistic regression; inference is `softmax(((x - mean) / std) @ W + b)`. */
export class StackedFusion {
  private constructor(
    readonly l2: number,
    readonly standardize: boolean,
    readonly mean: Float64Array,
    readonly std: Float64Array,
    /** `weights[d]` is the row of D x K weights belonging to input feature d. */
    readonly weights: readonly Float64Array[],
    readonly bias: Float64Array,
  ) {}

  static fromDict(data: StackedFusionDict): StackedFusion {
    if ((data as { type?: unknown }).type !== "stacked_fusion") {
      throw new Error("not a stacked fusion payload");
    }
    return new StackedFusion(
      Number(data.l2),
      Boolean(data.standardize),
      Float64Array.from(data.mean),
      Float64Array.from(data.std),
      data.weights.map((row) => Float64Array.from(row)),
      Float64Array.from(data.bias),
    );
  }

  get nClasses(): number {
    return this.bias.length;
  }

  get nFeatures(): number {
    return this.weights.length;
  }

  /** Fused logits for one feature row (feed to {@link TemperatureScaler}). */
  predictLogits(x: ArrayLike<number>): Float64Array {
    const d = this.weights.length;
    if (x.length !== d) throw new Error("feature matrix has the wrong shape");
    const k = this.bias.length;
    const out = new Float64Array(k);
    for (let j = 0; j < k; j += 1) out[j] = 0;
    for (let i = 0; i < d; i += 1) {
      const xs = ((x[i] as number) - (this.mean[i] as number)) / (this.std[i] as number);
      const row = this.weights[i] as Float64Array;
      for (let j = 0; j < k; j += 1) out[j] = (out[j] as number) + xs * (row[j] as number);
    }
    for (let j = 0; j < k; j += 1) out[j] = (out[j] as number) + (this.bias[j] as number);
    return out;
  }

  predictProba(x: ArrayLike<number>): Float64Array {
    return softmax(this.predictLogits(x));
  }
}
