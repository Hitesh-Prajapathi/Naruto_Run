# Model card template

Copy to `docs/models/<name>-<version>.md` and fill in every field. Write `TBD` for anything not measured; never estimate. Ship a copy inside the model bundle.

## Model details

- **Name / version (semver):**
- **Bundle contents:** onnx file(s), `labels.json`, `calibration.json` (temperature T, tau_on, tau_off, dwell ms), `feature_spec.json`, this card, sha256 of each file
- **Architecture:** (landmark MLP / image branch backbone and input size / stacked fusion)
- **Parameters / file size / quantisation (FP32, FP16, INT8):**
- **Training code:** git SHA, config file path, seed
- **Training data:** dataset version, DVC hash, `splits.yaml` hash (link the [dataset card](dataset_card.md))
- **Training date / hardware:**
- **License:**

## Intended use

- Real-time recognition of the 13 classes in [docs/seals.md](seals.md) from a webcam, for the game demo.
- **Out of scope:** identity, emotion, or any biometric use; surveillance; sign languages other than the defined seals.

## Classes and canonical definitions

Seal spec version/commit: . Dragon variant used: .

## Evaluation data

- Protocol: leave-one-subject-out on N = TBD subjects; vault subjects (TBD) evaluated once at the end.
- Subjects, sessions, cameras, lighting: see dataset card.
- Legacy v0 data used as: (training source only / not used).

## Metrics (all with 95% bootstrap CI, resampled by subject)

| Metric | Value |
|---|---|
| Macro-F1 (LOSO) | TBD |
| Accuracy (LOSO) | TBD |
| Per-class recall (min across seals, and which seal) | TBD |
| ECE (after calibration) | TBD |
| Open-set reject rate on non-seal motion | TBD |

Per-class precision/recall table and confusion matrix: link to the file in `docs/results/`.

## Event-level and streaming metrics

| Metric | Value |
|---|---|
| False activations per minute (idle/free motion), N minutes = TBD | TBD |
| Combo precision / recall | TBD |
| Time-to-detect P50 / P95 | TBD |
| Glass-to-glass latency P50 / P95 / P99 | TBD |
| Metrics with and without the grammar prior | TBD |

## Robustness slices

Macro-F1 per slice (lighting, background, distance, camera, Monk Skin Tone bucket, handedness). Flag any slice more than 10 points below overall.

| Slice | Bucket | n subjects | Macro-F1 |
|---|---|---:|---:|
| | | TBD | TBD |

## Performance

| Hardware | Runtime (WebGPU / WASM / CPU) | FPS | P95 per-frame latency | Bundle size | First-load time |
|---|---|---:|---:|---:|---:|
| | | TBD | TBD | TBD | TBD |

Quantisation comparison (FP32 vs FP16 vs INT8): macro-F1 delta, latency, size: TBD.

## Failure modes and limitations

- Top confident errors: link to the failure gallery.
- Seals with known weakness, hand-visibility assumptions (see [landmark spike](results/landmark_spike_v0.md)), lighting/camera limits.
- Population coverage limits (skin tone, hand size, handedness, age).

## Ethical considerations and privacy

- Data collected with explicit consent, pseudonymous IDs, deletion supported.
- Inference location: (browser only: no frames leave the device / server mode: frames sent to a local server). Status: TBD.

## Changelog

| Version | Date | Change | Macro-F1 (LOSO) |
|---|---|---|---|
