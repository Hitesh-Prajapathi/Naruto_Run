/** Shared helpers for the golden-fixture parity tests (see docs/specs/parity.md). */

/** Revive the string-encoded non-finite floats the Python generator writes. */
export function revive<T>(value: T): T {
  if (typeof value === "string") {
    if (value === "NaN") return Number.NaN as unknown as T;
    if (value === "Infinity") return Number.POSITIVE_INFINITY as unknown as T;
    if (value === "-Infinity") return Number.NEGATIVE_INFINITY as unknown as T;
    return value;
  }
  if (Array.isArray(value)) return value.map((v) => revive(v)) as unknown as T;
  if (value !== null && typeof value === "object") {
    const out: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(value)) out[k] = revive(v);
    return out as T;
  }
  return value;
}

/** Largest absolute element-wise difference (Infinity on a length mismatch). */
export function maxAbsDiff(a: ArrayLike<number>, b: ArrayLike<number>): number {
  if (a.length !== b.length) return Number.POSITIVE_INFINITY;
  let m = 0;
  for (let i = 0; i < a.length; i += 1) {
    const d = Math.abs((a[i] as number) - (b[i] as number));
    if (!(d <= m)) m = d;
  }
  return m;
}

/** Worst deviation per fixture family; printed when PARITY_VERBOSE is set. */
export const deviations: Record<string, number> = {};

export function track(family: string, value: number): void {
  deviations[family] = Math.max(deviations[family] ?? 0, value);
}

export function report(): void {
  const env = (globalThis as { process?: { env?: Record<string, string | undefined> } }).process?.env;
  if (env?.["PARITY_VERBOSE"]) {
    console.info("PARITY max deviation", JSON.stringify(deviations));
  }
}
