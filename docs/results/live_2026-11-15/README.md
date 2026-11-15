# Live webcam test: legacy vs v1 hand decoder (2026-11-15)

Same image model (`models/best_model_A.onnx`, the contaminated v0 model), two hand-seal decoders:

- **legacy**: 13 hand-tuned per-class confidence floors + 5-frame vote filter (`EvidenceHandFilter`)
- **v1**: temperature scaling (identity placeholder, T = 1.0, `models/calibration_v0.json`) + hysteresis `SealDecoder` with untuned default thresholds (`tau_on` 0.6)

Protocol: `scripts/evaluate_live_camera.py --mode hand`, 13 labels x 3 attempts, 1.5 s neutral preparation then a 3.5 s action window with the target hidden until the action starts. One subject (the owner), one session per decoder, the same room, back to back. Raw summaries: [`legacy.json`](legacy.json), [`v1.json`](v1.json) (metadata only, no images).

## Results

| | Attempts correct | 95% Wilson CI | Median time-to-detect (successful attempts) | False events in neutral phase |
|---|---|---|---|---|
| legacy | 24 / 39 (61.5%) | [0.46, 0.75] | 1,103 ms | 1 (1,405 frames) |
| v1 | 23 / 39 (59.0%) | [0.43, 0.73] | 1,395 ms | 1 (1,348 frames) |

The difference is not meaningful: the intervals overlap almost entirely.

| Seal | legacy | v1 |
|---|---|---|
| bird | 3/3 | 3/3 |
| boar | 3/3 | 2/3 (none) |
| dog | 3/3 | 0/3 (ox, ox, ox) |
| dragon | 2/3 (none) | 0/3 (none, none, ox) |
| hare | 1/3 (ox, ox) | 0/3 (ox, ox, ox) |
| horse | 0/3 (none, none, none) | 3/3 |
| monkey | 0/3 (ox, ox, ox) | 0/3 (ox, ox, ox) |
| ox | 3/3 | 2/3 (monkey) |
| ram | 0/3 (ox, none, none) | 3/3 |
| rat | 3/3 | 3/3 |
| snake | 0/3 (tiger, none, none) | 3/3 |
| tiger | 3/3 | 1/3 (ox, none) |
| zero | 3/3 | 3/3 |

## Reading

- **The bottleneck is the image model, not the decoder.** Per-frame raw predictions (before any decoder) show the same failure modes in both runs: `ox` absorbs hare, monkey, dog and dragon (for example hare was raw-predicted `ox` in 77–82% of action frames), and `tiger` is the fallback whenever the hands are unclear. Both decoders faithfully pass these errors on.
- **Seal-level results swing between sessions** (horse, ram, snake 0/3 → 3/3; dog 3/3 → 0/3), which points to an unstable model rather than to one decoder being better.
- v1 emitted far fewer duplicate events (2 vs 12), but more unexpected events during the action window (14 vs 11), mostly brief `ox` events while the hands move into position. Its thresholds are untuned defaults and were deliberately **not** tuned on this test (39 attempts from one person would overfit).
- `zero` passed 3/3 in both runs only because the hands were mostly out of view. The training data's `zero` class is "two fists together" while the evaluation prompt says "relaxed hands"; this is unresolved.
- This agrees with the source-disjoint baseline ([v0 baseline](../v0-baseline.md): macro-F1 0.537): the fix is multi-subject data, not decoder tuning.

Limitations: one subject, n = 3 per seal, one session per decoder; not a subject-disjoint evaluation.
