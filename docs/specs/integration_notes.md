# Integration notes: replacing `EvidenceHandFilter` with calibration + `SealDecoder`

Bridge document, written before the integration. **Status:** the integration now exists behind
a feature flag (`hand_decoder: "legacy" | "v1"`, default `legacy`); sections 2-5 describe the
*target* design and section 6 records what was actually wired and where it deviates. It describes how the new
model-agnostic core (`narutocv.features`, `narutocv.models.calibration`,
`narutocv.temporal`) plugs into `cv_model/inference/combined_pipeline.py` and the
transport layer, and what moves into the model bundle.

## 1. What exists today (`combined_pipeline.py`)

`HandSignRecognizer.process(frame_bgr, timestamp)` does, per frame:

1. classify a 0.6 centre crop and the hand-ROI crop with the same ONNX model, fuse
   with a fixed weight (`_fuse_predictions`);
2. `_accept()`: per-class confidence floor (`HAND_CONFIDENCE_THRESHOLDS`), per-class
   margin floor (`HAND_MARGIN_THRESHOLDS`), plus special cases (ram -> rat when
   rat >= 0.12, dog pair guard, no-hands gating) -> an accepted label or `"zero"`;
3. `EvidenceHandFilter.update(label)`: 5-frame window, 3 votes, 2 neutral frames
   to reset; returns `(stable, emitted)`;
4. `HandResult.emitted_seal` flows to `AttackRecognizer.update_detailed(...)`,
   then `output_schema` -> `PipelineEventType.HAND_SEAL`.

`calibrate_neutral()` additionally tightens the per-class floors at runtime.

## 2. Replacement data flow

```
frame -> HandLandmarker (2 hands) ----> features.two_hand_features(...)   [A: 84-d]
      \-> hand-union crop ------------> image branch ONNX                 [B: logits]
 [A] logits, [B] logits, hand_count, landmark_quality
      -> StackedFusion.predict_logits(build_fusion_features(...))
      -> TemperatureScaler.transform  -> p(seal | frame)  (K = 13, includes "zero")
      -> SealDecoder.update(p, t_ms, expected_next=...)   -> list[SealEvent]
      -> HandResult.emitted_seal / HAND_SEAL event
```

Concrete edits when the model is integrated (not done here):

- **Delete** from `HandSignRecognizer`: `_accept`, `calibrate_neutral`,
  `_should_gate_without_hands`, the ram/rat and dog/tiger guards, the
  `confidence_thresholds` / `margin_thresholds` dicts, and
  `hand_config.py` (`HAND_CONFIDENCE_THRESHOLDS`, `HAND_MARGIN_THRESHOLDS`,
  `DEFAULT_HAND_REQUIRED_VOTES`). Roles move as follows:

  | old mechanism | new mechanism |
  |---|---|
  | per-class confidence floors | calibrated probabilities + global `tau_on` / `tau_off` |
  | per-class margin floors | calibration (temperature) + `tau_reject`; the stacked fusion sees both branches |
  | ram/rat and dog/tiger guards | the landmark branch separating them; no special cases |
  | no-hands gating | `meta.hand_count` / presence mask feeds fusion, and the image branch falls back to pose-wrist ROI; the decoder's `tau_reject` handles the rest |
  | `EvidenceHandFilter` 5-frame / 3-vote | `SealDecoder` (EMA half-life, hysteresis, `min_dwell_ms`) |
  | `calibrate_neutral` session tightening | per-user threshold changes, if wanted, edit `DecoderConfig.tau_on` only (explicit, logged) |

- **Replace** `self.filter = EvidenceHandFilter(...)` with
  `self.decoder = SealDecoder(bundle.decoder_config)`. `reset()` keeps its meaning
  (`self.decoder.reset()`).
- **Per frame:** build `HandObservation`s from `detected.hand_landmarks` +
  `detected.handedness` (label and score) and call
  `two_hand_features(hands, image_size=(width, height))`. Feed the calibrated
  vector to `decoder.update(p, t_ms)`. `HandResult.stable_label` becomes
  `decoder.active_label or "zero"`; `HandResult.emitted_seal` becomes the label of
  the returned event (at most one per call); `HandResult.raw` stays the calibrated
  frame-level top-2 for diagnostics.
- **Combo grammar:** the attack/combo FSM (`AttackRecognizer`) already knows which
  seal is expected next; pass its candidate labels as `expected_next` (only while a
  combo is in progress). Gate with `DecoderConfig.use_prior`; evaluation runs with
  the prior off and on.

### Timestamps (important)

`HandSignRecognizer._video_timestamp_ms()` bumps timestamps to be strictly
increasing (`max(candidate, last + 1)`) because MediaPipe VIDEO mode requires it.
Do **not** feed that bumped value to the decoder. Feed it the raw capture time in
milliseconds (`timestamp * 1000.0`, a float). The decoder itself ignores and counts
non-monotonic stamps. Use one clock throughout: the pipeline's `timestamp`
(`time.monotonic()` seconds) is the same clock used for `captured_at_ms` in
`PipelineEventV1`; `SealEvent.t_start_ms` / `t_emit_ms` must be converted from
the same `* 1000` basis, so a consumer can subtract `captured_at_ms` of a frame to
get a latency. On the browser side use `performance.now()` for both.

## 3. What goes into the model bundle (`calibration.json`)

`hand_config.py` is deleted; everything below is read from the bundle
(`{onnx files, labels.json, calibration.json, feature_spec.json, model_card.md, sha256}`):

```json
{
  "bundle_version": "1.0.0",
  "feature_spec_version": "1.0.0",
  "labels": ["bird", "...", "zero"],
  "idle_label": "zero",
  "fusion": { "type": "stacked_fusion", "...": "StackedFusion.to_dict()" },
  "calibration": { "type": "temperature", "temperature": 1.37, "fitted": true },
  "decoder": { "...": "DecoderConfig.to_dict()" }
}
```

- `fusion` / `calibration`: produced by `StackedFusion.to_dict()` and
  `TemperatureScaler.to_dict()` (a `VectorScaler.to_dict()` payload is also
  possible; each payload carries a `type` tag, but a bundle loader dispatching on it
  is not written yet). Fit calibration on held-out *subjects* (LOSO), never on the
  training frames.
- `decoder`: `DecoderConfig.to_dict()`, i.e. `half_life_ms`, `tau_on`, `tau_off`,
  `tau_reject`, `min_dwell_ms`, `refractory_ms`, `max_gap_ms`, `prior_boost`,
  `use_prior`, `smoother`, HMM parameters. The plan's "(T, tau_on, tau_off, dwell)"
  are these fields. Values come from `grid_search` + `select_operating_point`
  on validation subjects (target <= 0.5 false activations/min).
- `feature_spec.json` = `export_feature_spec()` output; the loader should refuse a
  bundle whose `feature_spec_version` differs from the code's `FEATURE_SPEC_VERSION`.
- `labels` order must equal the model output order and `DecoderConfig.labels`.

## 4. Event contract for the game / transport

The decoder emits `SealEvent`:

| field | type | meaning |
|---|---|---|
| `label` | string | seal name (never the idle class) |
| `t_start_ms` | float, ms | first update of the uninterrupted run with smoothed p >= `tau_on` |
| `t_emit_ms` | float, ms | timestamp of the update that emitted it; the moment the game may react |
| `confidence` | float in [0,1] | smoothed calibrated probability at emission |

Needed by the game / transport:

- The existing `HAND_SEAL` event payload (`events.py`, `dispatch_frame`) currently
  carries `seal`, `stable_label`, and `confidence` (the raw frame confidence). Extend
  it, additively and keeping `schema_version` compatible, with `t_start_ms`,
  `t_emit_ms` and the decoder `confidence`; the TypeScript protocol (`protocol.ts` /
  `websocketClient.ts`) needs the same optional fields.
- `captured_at_ms` of the frame that produced the event should equal `t_emit_ms`
  rounded, provided the decoder is fed the same capture clock (to be asserted in the
  integration test); `t_emit_ms - t_start_ms` is the dwell the game can use for UI feedback
  (e.g. a charging ring), and `t_emit_ms - gt_start` is the time-to-detect metric.
- Exactly one event per seal entry; the game must not de-duplicate on its own. Held
  seals do not repeat. A seal re-fires only after release and the refractory period.
- The game's sequence logic (`sealSequence.ts`, `AttackRecognizer`) should treat the
  event as the seal accepted. The legacy `stable_label`/`emitted_seal` pair maps to
  `decoder.active_label` / the event label.
- For the in-browser port, `SealDecoder` runs client-side with the same JSON config;
  no server round trip, so `t_emit_ms` is local `performance.now()`.

## 5. Open items

- Decide whether `StackedFusion` consumes calibrated or raw branch outputs (the
  helper `build_fusion_features` takes probabilities and applies `log(clip(p, 1e-6))`;
  pass per-branch softmax of the ONNX logits).
- The grammar prior needs the combo FSM to expose "expected next seal(s)" per frame
  (today `AttackRecognizer` keeps a rolling queue only).
- Golden streams (recorded probability sequences + expected events) should be exported
  from the Python decoder to test the TypeScript port.

## 6. What was actually wired (flag `hand_decoder`, default `legacy`)

The integration exists behind `hand_decoder: "legacy" | "v1"` (`PipelineConfig`, `--hand-decoder`,
`NARUTOCV_HAND_DECODER`); see `docs/transport.md` for the protocol 1.2.0 contract and
`src/narutocv/inference/hand_decoder.py` for the code. Deviations from sections 2-5 above:

- **Nothing was deleted.** `hand_config.py`, `_accept`, `EvidenceHandFilter` and
  `calibrate_neutral` stay for the legacy path; v1 bypasses the thresholds. `calibrate_neutral`
  returns `{}` in v1.
- **Single branch, no `StackedFusion`.** The input to the decoder is the existing centre/ROI fused
  probability vector: `TemperatureScaler.transform(log(clip(p, 1e-12)))` -> `SealDecoder.update`.
  At the shipped `T = 1.0` that is an identity. The unused hook is `STACKED_FUSION_HOOK` in
  `hand_decoder.py`; a fitted `T != 1` should be refit on *fused probabilities*, not on logits.
- **Calibration source** is `models/calibration_v0.json` (`temperature`, `decoder`, `provenance`),
  tracked in `models/manifest.yaml`. Its decoder block is `DecoderConfig` defaults except
  `use_prior: false`.
- **No-hands gate kept** as a structural gate (idle vector into the decoder), because the
  landmarker's "no hand" is evidence, not a per-class threshold. The ram/rat and dog/tiger guards are
  not applied in v1.
- **Grammar prior**: `AttackRecognizer.expected_next(now)` returns the sorted union of the next
  seals of every attack whose sequence has the current queue as a strict prefix (empty queue,
  expired queue, or no matching attack: `()`). It is passed to `SealDecoder.update(expected_next=)`
  by both `CombinedNarutoPipeline.process` and the `hand_only` branch of
  `SessionScopedPipeline.process`, and only when `use_prior` is on (default off).
- **Clock**: the decoder gets `timestamp * 1000.0` (not MediaPipe's bumped integer timestamp). In
  the transport that equals the frame's `captured_at_ms`.
- **`t_start_ms`** stays at the smoothed `tau_on` crossing with no lag correction (documented in
  `hand_decoder.py` and `docs/transport.md`).
- **Event path**: the decoder event travels in `output.hand.decoder` (v1 only) because
  `dispatch_frame` builds events purely from `PipelineOutputV1`.
- Section 5 is resolved: the combo FSM exposes `expected_next`, golden streams exist
  (`tests/golden/decoder_v1.json`), and `StackedFusion` is deferred until there is a second branch.

**Untested on a real camera.** The v1 path is covered by unit tests on synthetic probability
streams, a stubbed-classifier run of `HandSignRecognizer.process`, and a real-model transport smoke
run on synthetic frames. Its thresholds are a placeholder; do not read accuracy from it.
