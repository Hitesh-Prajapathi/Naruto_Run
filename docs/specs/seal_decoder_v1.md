# Seal decoder spec v1

Reference implementation: `src/narutocv/temporal/decoder.py`
(`SealDecoder`, `DecoderConfig`, `EmaSmoother`, `HMMSmoother`, `SealEvent`).
Tuning / evaluation: `src/narutocv/temporal/tuning.py`.

The decoder consumes one **calibrated** probability vector per frame plus a
timestamp in milliseconds and emits at most one event per seal entry. All time
logic uses timestamps, never frame counts. The TypeScript port must produce the
same events for the same input stream (compare on recorded streams; floating
point `Math.pow` vs Python `**` differences are far below any threshold).

## 1. Configuration (`DecoderConfig`, stored in `calibration.json`)

| field | default | meaning |
|---|---|---|
| `labels` | 13 classes, alphabetical, `zero` last | class order of the probability vector |
| `idle_label` | `"zero"` | class meaning "no seal" (may be `null`) |
| `smoother` | `"ema"` | `"ema"` or `"hmm"` |
| `half_life_ms` | 80 | EMA half-life |
| `tau_on` | 0.6 | smoothed p required to enter |
| `tau_off` | 0.4 | active seal ends when its smoothed p drops below |
| `tau_reject` | 0.3 | open-set reject floor on the max smoothed p |
| `min_dwell_ms` | 120 | continuous time at or above `tau_on` before emitting |
| `refractory_ms` | 250 | minimum time between two emissions (global) |
| `max_gap_ms` | 200 | a larger inter-frame gap breaks continuity |
| `prior_boost` | 3.0 | multiplier for grammar-expected classes (>= 1) |
| `use_prior` | true | master switch for the grammar prior (evaluation: turn off) |
| `hmm_p_stay`, `hmm_ref_ms` | 0.95, 33.3 | HMM smoother parameters |

Validation: `0 < tau_reject <= tau_off < tau_on <= 1`, `labels` unique with at
least two entries, `idle_label in labels`, `half_life_ms > 0`, `max_gap_ms > 0`,
`min_dwell_ms, refractory_ms >= 0`, `1/K < hmm_p_stay < 1`. JSON round-trip via
`to_dict` / `from_dict` (unknown keys are an error; `labels` is a JSON list).

## 2. State

```
last_t          : number | null
smoother state  : vector K | null      (EMA value or HMM belief)
active          : class index | null   (seal currently "held", already emitted)
pending         : class index | null   (candidate accumulating dwell)
pending_start   : number
last_emit_t     : number | null
stats           : frames, nonmonotonic, gaps, events
```

`reset()` clears everything, including `last_emit_t` (refractory) and stats, so
timestamps may restart from 0 afterwards.

## 3. Input handling

`update(probs, t_ms, expected_next?)`

1. `probs` must have length K and be finite (else throw). Clip negatives to 0 and
   renormalise to sum 1 (an all-zero vector becomes uniform).
2. `t_ms` must be finite. If `last_t != null and t_ms <= last_t`: increment
   `stats.nonmonotonic`, return `[]`, **change nothing else** (the frame is
   ignored). Equal timestamps count as non-monotonic.
3. Grammar prior (only if `use_prior` and `expected_next` is non-empty): see 4.
4. `dt = 0` on the first frame, else `t_ms - last_t`. If `dt > max_gap_ms`
   (and not first frame): increment `stats.gaps`, reset the smoother (it
   re-initialises from the current frame) and set `pending = null`. The *active*
   seal is kept, so a camera stall in the middle of a held seal does not produce a
   duplicate event; the active seal still ends through the normal `tau_off` rule.
   `last_t = t_ms`.

## 4. Grammar prior

If `use_prior` is true and `expected_next` is non-empty: every name must be in
`labels` (else throw). `w[i] = prior_boost` for each expected class that is not
the idle class, else 1; `p = p * w; p = p / sum(p)`. This happens **before**
smoothing, so the prior influences the smoothed state. If `use_prior` is false
the argument is not even validated, and the decoder is bit-identical to calling
without it. Reports must give metrics with and without the prior.

## 5. Smoothers

**EMA** (time aware): first frame `s = p`; afterwards
`alpha = 1 - 2^(-dt / half_life_ms)`, `s = s + alpha * (p - s)`.
One half-life closes half the gap regardless of frame rate.

**HMM** (online forward filter, K states, sticky transitions: stay `p_stay`,
uniform otherwise, per `hmm_ref_ms`): with `lam = (K * p_stay - 1) / (K - 1)`
and first frame `b = p`, afterwards

```
decay     = lam ** (dt / hmm_ref_ms)             # fractional power of the transition matrix
predicted = decay * b + (1 - decay) / K          # valid because sum(b) = 1
b         = predicted * p ** (dt / hmm_ref_ms)   # tempered emission (calibrated posterior as likelihood)
b         = b / sum(b)                           # if sum(b) < 1e-300, b = uniform
```

The predict step is exactly frame-rate independent; the tempered emission makes the
accumulated evidence per unit time approximately independent of fps (not exact,
because normalisation and prediction do not commute). Prefer EMA unless the HMM
shows a measured gain.

## 6. Decision logic (after smoothing, `s` = smoothed vector)

```
top = argmax(s)                                   # lowest index on ties

# (a) hysteresis exit: only the active seal's own probability matters
if active != null and s[active] < tau_off: active = null

# (b) open-set reject: no candidate
if s[top] < tau_reject or top == idle: pending = null; return []

# (c) candidate = top class if s[top] >= tau_on, else none
if s[top] < tau_on: pending = null; return []
c = top
if active == c: return []                         # still held, no re-entry
active = null                                     # direct switch to another confident seal
if pending != c: pending = c; pending_start = t_ms
if t_ms - pending_start < min_dwell_ms: return []
if last_emit_t != null and t_ms - last_emit_t < refractory_ms: return []   # deferred
emit SealEvent(label[c], t_start_ms = pending_start, t_emit_ms = t_ms, confidence = s[c])
active = c; pending = null; last_emit_t = t_ms
```

Consequences:

- **One event per entry.** After emission the seal stays `active` until its smoothed
  probability falls below `tau_off`. Re-entering the same seal therefore requires
  dropping below `tau_off` first, then climbing back to `tau_on` for `min_dwell_ms`.
- **Flicker immunity.** A signal hovering around `tau_on` can restart `pending`
  repeatedly but only emits after an uninterrupted `min_dwell_ms`; once emitted it
  cannot chatter until it really releases below `tau_off`.
- **Refractory is global and defers.** If the dwell completes inside the
  refractory window the candidate stays `pending` and is emitted by the first
  update after the window if it is still at or above `tau_on` (`t_start_ms` is
  preserved). It is not dropped.
- `t_start_ms` is the first update of the uninterrupted run with `s >= tau_on`
  (smoothed, so it lags the true onset by the EMA rise time). `t_emit_ms` is the
  timestamp of the emitting update. `confidence` is the smoothed probability of the
  label at emission.
- Idle-dominated frames never end an active seal on their own while its smoothed
  probability is at or above `tau_off`; they do block new entries.

## 7. Latency bound (clean seal)

For a seal whose calibrated probability steps from near 0 to `p` and holds, the
smoothed value reaches `tau_on` after
`half_life_ms * log2((p - p0) / (p - tau_on))` (p0 = starting smoothed value). With
defaults, p0 = 0.01, p = 0.9: about 125 ms. Emission is then at most
`min_dwell_ms` plus one frame interval later (about 250 ms after onset at 30 fps).
The tests assert this bound.

Frame-rate independence (tested for EMA): the same underlying signal sampled at 30
and 60 fps yields the same events, with `t_start_ms` within one coarse frame and
`t_emit_ms` within two coarse frames (dwell is checked on sample timestamps, so it
quantises on top of the crossing time).

## 8. Matching ground truth (for tuning/evaluation)

A predicted event matches a ground-truth interval `[start, end]` of the same label
if `start <= t_emit_ms <= end + tolerance_ms` (default 300 ms). One-to-one, greedy by
time: predictions in `t_emit_ms` order each take the earliest-starting unmatched
eligible interval. Unmatched predictions are false activations (a double emission
yields one match and one false activation); unmatched intervals are misses.
Metrics: false activations per minute (over total stream minutes), event recall,
median time-to-detect (`t_emit - gt.start` of matched events; `inf` when nothing
matched). `grid_search` sweeps `(tau_on, tau_off, min_dwell_ms, half_life_ms)`
(`tau_reject` clamped to `min(base, tau_off)`) and `pareto_frontier` returns the
non-dominated set (FA/min down, recall up, TTD down; duplicates removed).
`select_operating_point(points, max_fa_per_min=0.5)` implements the plan's target.

## 9. TypeScript port checklist

- Same state machine, same operand order (the comparisons `>=` for `tau_on`,
  `<` for `tau_off`, `<=` for non-monotonic).
- `argmax` returns the lowest index on ties.
- `2 ** x` -> `Math.pow(2, x)`; keep all numbers as float64 `number`.
- Golden test: export a few recorded probability streams with expected events from
  the Python decoder and assert event equality (labels exact, times to 1e-6 ms).
