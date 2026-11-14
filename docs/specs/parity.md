# Python / TypeScript parity (golden tests)

The browser recognition path (`apps/web/src/recognition/`) is a port of three Python
reference implementations. The two must produce identical results, so both are tested
against the same recorded fixtures.

| Python reference | TypeScript port | Fixture | Spec |
|---|---|---|---|
| `src/narutocv/features/landmarks.py` | `landmarkFeatures.ts` | `tests/golden/features_v1.json` | `landmark_features_v1.md` |
| `src/narutocv/temporal/decoder.py` | `sealDecoder.ts` | `tests/golden/decoder_v1.json` | `seal_decoder_v1.md` |
| `src/narutocv/models/calibration.py` (`TemperatureScaler`, `VectorScaler`, `StackedFusion` inference) | `calibration.ts` | `tests/golden/calibration_v1.json` | docstrings |

Fitting (`fit`) exists only in Python; TypeScript loads the `to_dict()` JSON.

## How it works

1. `scripts/golden/generate_golden.py` runs the **Python** implementations on deterministic
   inputs (seeded `numpy.random.default_rng`, inputs rounded to a fixed number of decimals)
   and writes input + expected output into `tests/golden/*.json`.
2. `tests/golden/test_golden_up_to_date.py` (pytest) regenerates everything in memory and
   asserts it equals the committed files. If a Python implementation or the generator
   changes, this test fails until the fixtures are regenerated and committed.
3. The Vitest suites in `apps/web/tests/unit/recognition/*.parity.test.ts` import the same
   JSON files and run the **TypeScript** port on the recorded inputs. They fail until the
   port reproduces the new expected values.

**Rule: change the Python spec -> regenerate golden -> the TS test fails until ported.**
Never edit a fixture by hand and never loosen a tolerance to make a port pass.

## Regenerating

```
uv run python scripts/golden/generate_golden.py     # rewrites tests/golden/*.json
uv run pytest tests/golden                          # drift check
cd apps/web && npx vitest run tests/unit/recognition
```

Review the fixture diff in the PR: it is the behavioural diff of the change. Run
`PARITY_VERBOSE=1 npx vitest run tests/unit/recognition` to print the worst deviation per
fixture family.

## What the fixtures contain

* **features_v1.json** (53 cases): single right/left hands over a curl sweep and rotations,
  non-square and extreme image sizes (including 1x1, 100x1000), Left+Right pairs, input
  order swaps, palms-facing poses (flag = 1) and co-planar poses (flag = 0), same-label
  tie-break (ordered, swapped, identical hands, equal mean x), three hands (cut by score,
  tied scores, one degenerate), zero hands, NaN / infinite / all-zero / sub-epsilon palms,
  a tiny but valid hand, the lateral-axis fallback (zero, parallel, and the lowest-index
  `argmin` ties for `y = (1,0,0)` and `y = (0,0,1)`), non-finite and missing scores,
  lower-case and whitespace-padded handedness, an invalid label (expects an error), planar
  hands and hands outside the image.
* **decoder_v1.json** (27 streams): clean seals at 15/30/60 fps, jitter, non-zero start,
  direct switches, refractory deferral, flicker around `tau_on`, hold/release/re-entry, slow
  decay, gaps above `max_gap` (held seal and pending dwell), dropped frames below it,
  equal/backwards timestamps, `reset()` mid-stream, grammar prior on/off, idle never
  boosted, zero-dominant and open-set-reject streams, zero / negative / unnormalised
  probability vectors, `idle_label = null`, five HMM streams (30/60 fps, noisy, gap, exact
  zeros) and a custom 4-class configuration. Each records events, final stats, the per-frame
  `(argmax, max smoothed)` trace, the final smoothed vector and the final active label.
  The streams were checked to be insensitive to 1e-8 relative perturbations of the inputs
  (no knife-edge decisions).
* **calibration_v1.json**: temperature scalers (T = 0.05, 0.7, 1, 1.7, 20) on logits that
  include uniform rows and +/-60..80 extremes, a vector scaler, and a stacked-fusion pipeline
  (`build_fusion_features` -> `predict_logits` -> `predict_proba` -> temperature) plus a
  non-standardised fusion.

Non-finite floats cannot be written in JSON; the generator writes them as the strings
`"NaN"`, `"Infinity"`, `"-Infinity"` and the TS tests revive them (`goldenTypes.ts`).

## Tolerances and why

| Family | TS vs fixture | Why | Observed max deviation |
|---|---|---|---|
| Features (float32 output) | <= 1e-5 abs per element; discrete entries (facing flag, hand count, presence flags) exactly equal | Both sides compute in float64 and round to float32 at the end, so results are normally bit-identical; the slack covers 1 float32 ulp when `Math.*` / BLAS differ in the last double bit. The spec's own bound is 1e-4. | 0 (bit-identical on all 52 cases) |
| Decoder events | `label`, `t_start_ms`, `t_emit_ms` **exact**; `confidence` <= 1e-9 | Event times are copies of input timestamps, never derived by arithmetic, so they must match exactly. Confidence is a smoothed probability (`Math.pow` vs numpy `**`, pairwise vs sequential sums). The spec suggests 1e-6 ms for times; exact is stricter and achievable. | confidence 1.1e-16 |
| Decoder stats, active label, per-frame argmax | exact | Integer / discrete state. | exact |
| Decoder smoothed values (trace max, final vector) | <= 1e-9 | Float64 filter state. | 2.2e-16 |
| Calibration (probabilities, logits, features) | <= 1e-9 | Float64 end to end; differences come only from `exp`/`log`/matmul ordering. | 1.1e-14 (fusion logits) |
| Drift test (Python regeneration vs committed JSON) | 1e-6 features, 1e-9 others (relative and absolute) | Regenerating on another platform (different BLAS/libm) may change the last bits of Python's own result; the drift test must not fail on that. Inputs are rounded and seeded so they are identical everywhere. | |

Decisions that depend on thresholds (`tau_on`, `tau_off`, `min_dwell`, facing cosine,
degenerate-palm epsilons) are the reason the fixtures avoid knife-edge inputs: a 1-ulp
difference there would legitimately flip the result and is not a port bug.

## Spec issues found while porting (for the supervisor to decide)

The TS port matches Python in every case below; none was silently changed.

1. **NaN scores with more than two hands.** `assign_slots` sorts by `(-score, index)`. With a
   NaN score the Python key comparison is not a total order, so which two hands survive
   depends on CPython's sort internals; the TS comparator cannot reproduce that in general.
   Feature spec 4.3 says non-finite scores count as 0 for `mean_score`, but section 3 does not
   say how they rank. Suggest: rank non-finite scores as 0 (or -inf) in both languages. Not
   covered by the fixtures (NaN score is only tested with at most two hands).
2. **Same-label tie-break uses a float sum.** The "smaller mean x" comparison uses
   `numpy.mean` (pairwise summation for 21 values) in Python and a sequential sum in TS. For
   hands whose means differ by about 1 ulp the slot could differ. Equal means (identical
   x columns) agree exactly (tested). Suggest documenting that mean-x ties within a tolerance
   (e.g. 1e-9) resolve by input order, in both languages.
3. **Config validation accepts NaN.** `DecoderConfig.__post_init__` uses `x <= 0`, `x < 0`,
   `x < 1` for `half_life_ms`, `hmm_ref_ms`, `min_dwell_ms`, `refractory_ms`, `max_gap_ms`
   and `prior_boost`, so a NaN passes in Python (the `tau_*` chain and `hmm_p_stay` reject
   it). TS was written with the same comparison forms to match. Suggest `not (x > 0)` style
   checks in Python, then regenerate and update TS.
4. **`str.strip().lower()` vs `trim().toLowerCase()`.** Equivalent for ASCII labels; exotic
   Unicode whitespace or case folding could differ. Irrelevant for MediaPipe labels.
5. **Decoder spec 9** asks for event times "to 1e-6 ms"; the tests enforce exact equality,
   which holds because times are input timestamps. Spec text could be tightened.
