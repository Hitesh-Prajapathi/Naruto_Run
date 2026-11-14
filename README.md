# Real-Time Hand-Seal Recognition Engine

<!-- TODO(owner): record a 10-15 s screen capture of the game being driven by hand seals and save it as docs/media/demo.gif, then replace the placeholder below with: ![demo](docs/media/demo.gif) -->
> **Demo GIF placeholder** (to be recorded: `docs/media/demo.gif`).

A real-time engine that recognises two-handed "hand seals" and upper-body movement from a webcam and turns them into typed events with a streaming temporal decoder and a combo state machine. The demo theme is an anime-style endless runner (Three.js) where seals cast attacks and body lean/jump steers the character.

This repository is also a worked example of auditing your own ML results: the first version reported 99% accuracy, which turned out to measure memorisation. The honest numbers and the plan to fix them are below.

## What's inside

```
capture -> scheduler -> landmark / image recognition -> fusion + calibration -> temporal decoder -> combo FSM -> game
```

| Stage | Today (v0, implemented) | Target (planned, see [architecture](docs/architecture.md)) |
|---|---|---|
| Capture | Browser `getUserMedia`, JPEG frames over a binary WebSocket protocol (NCV1) | In-browser Web Worker, `ImageBitmap` transfer |
| Scheduler | 1-slot "latest frame wins" mailbox (`LatestFrameScheduler`) | same idea in the worker |
| Recognition | YOLOv8n-derived ONNX image classifier (13 classes) + MediaPipe hand ROI; MediaPipe Pose for body | Landmark branch + image branch on a two-hand crop |
| Fusion / calibration | Centre-crop / hand-ROI weighted fusion, 13 hand-tuned confidence floors | Stacked fusion + temperature scaling, one global threshold |
| Temporal | 5-frame vote filter (`EvidenceHandFilter`) | Hysteresis decoder with dwell in milliseconds |
| Combo | `AttackRecognizer` FSM (seal sequences to attacks) | same, ported to TS from one spec, parity-tested |
| Game | Vite + TypeScript + Three.js client in `apps/web` | same, browser-only mode |

Also here: a multi-subject, prompted-video **data collection tool** (`tools/collect`) for the subject-disjoint dataset that the next model needs.

## Results (honest)

The v0 numbers below were found to be misleading. This table keeps the original claim next to what is actually known. Only numbers from files in this repository are used; anything not yet measured is marked TBD.

| Claim (v0, July) | Why it was misleading | Honest measurement |
|---|---|---|
| **99.64% val / 99.11% test accuracy** for the 13-class image model | Random, stratified, frame-level 80/10/10 split. The original train and test folders were merged first, so near-identical consecutive frames sit in train, val and test. This measures memorisation. | Image-model leakage delta and group-disjoint (cross-session) accuracy: `TBD — see docs/results/v0-baseline.md` |
| **"Trained on 2,860 images"** | About 1,795 are real train images; the rest are augmented copies made to balance each class to 220. The raw dataset is 2,245 images. | 2,245 raw images (2,159 train + 86 test folders), see [dataset card](docs/dataset_card.md) |
| *(implicit)* "diverse real-world data" | 1,797 of 2,245 images (80%) come from one machine, captured within 21 minutes on one day in 2020; the other 437 come from one other source over 4 days. Effectively one subject, two sessions. | Single-subject training data; no subject-disjoint evaluation exists yet |
| **Works live** | Live webcam test, 3 attempts per seal: dragon was never detected in either run (0/3 and 0/3); ram went 0/3 to 1/3 and was predicted as rat 5 of 6 times; dog went 0/3 to 1/3 (scattered: monkey, hare). Time-to-detect median about 1.0 to 2.7 s for most seals. | Source: `03_refinement_execution_plan.md` section 0.1 (v1.0 and v1.1 live runs) |
| *(spike)* Landmark features generalise better | Not tested on the image model. Landmark-only classifier, random (leaky) split vs cross-session split on the same data (all images): | S0 random/leaky 0.886 (logreg) / 0.904 (MLP) accuracy; cross-session S1a (train burst, test other source) 0.487 / 0.487; S1b (reverse) 0.761 / 0.829. CIs for S1a do not overlap S0. Details: [landmark spike](docs/results/landmark_spike_v0.md) |
| *(spike)* MediaPipe sees both hands | Only 52.7% of images get 2 hands, 38.8% get 1, 8.6% get none. Interlocked seals fail: ram 3.3% two-hand, tiger 5.2% two-hand (42.4% no hand), ox 12.5%, boar 16.5%, dog 37.4%. | Same spike. Proxy run on legacy data, not the final 3-subject recording |

Event-level metrics (combo precision/recall, false activations per minute, time-to-detect), per-slice robustness, and glass-to-glass latency are not measured yet. See [results index](docs/results/README.md).

## Quickstart in 5 commands

Requires [uv](https://docs.astral.sh/uv/) and Node.js.

```bash
make setup         # uv sync --all-extras + npm ci
make fetch-models  # download + sha256-verify the models in models/manifest.yaml
make test          # pytest + vitest
make serve         # Python transport server for the browser game (127.0.0.1:8765)
make web           # Vite dev server (http://127.0.0.1:5173), in a second terminal
```

`make help` lists the other targets (`lint`, `fmt`, `mypy`, `camera`, `build-web`).

## Repository layout

```
apps/web/          Three.js game (Vite + TypeScript); consumes pipeline events
src/narutocv/
  inference/       recognisers, scheduler, runtime controller, output schema, events
  transport/       aiohttp WebSocket bridge (NCV1 binary frames in, JSON events out)
tools/collect/     browser tool for prompted multi-subject video collection (+ optional receiver)
training/          legacy v0 data preparation script and exploration notebook
models/            manifest.yaml (+ the legacy ONNX); other models fetched by `make fetch-models`
schemas/           JSON schemas for pipeline output/events, transport, collection sessions
scripts/           entry points and the landmark spike scripts
tests/             Python tests
docs/              architecture, ADRs, seal spec, dataset card, results
```

## Documentation

- [Architecture](docs/architecture.md) (current and target)
- [Architecture decision records](docs/adr/)
- [Seal specification](docs/seals.md)
- [Dataset card](docs/dataset_card.md) and [model card template](docs/model_card_template.md)
- [Results index](docs/results/README.md)
- [Refinement execution plan](03_refinement_execution_plan.md) and [project plan](02_narutocv.md)
- [Contributing](CONTRIBUTING.md)
- `docs/archive/` is historical. Its accuracy claims are superseded and must not be cited.

## Limitations

- **Single-subject training data.** The legacy dataset is effectively one person and two sessions.
- **No subject-disjoint evaluation yet.** Until the multi-subject collection exists, no honest generalisation number for the image model is available; the group-disjoint re-evaluation is tracked in `docs/results/v0-baseline.md` (TBD).
- **MediaPipe fails on interlocked seals.** Per the landmark spike, ram, tiger, ox, boar and dog often yield one or zero detected hands, so a landmark branch cannot carry the model alone.
- **Dragon is unresolved.** It was never detected live, and the canonical variant has not been chosen by the owner ([seal spec](docs/seals.md)).
- **Latency is slow for a game** (about 1 to 3 s time-to-detect in v0 live tests), and true glass-to-glass latency is unmeasured.
- Collection tool and browser behaviour on Safari, Android and real cameras are untested.

## Roadmap

Phased in [03_refinement_execution_plan.md](03_refinement_execution_plan.md): honest baseline, multi-subject data, hand-centric hybrid model with calibration and a temporal decoder, event-level and latency evaluation, in-browser inference, production hardening. The decisions behind it are in [docs/adr](docs/adr/).

## Privacy

- **Planned, not implemented:** in browser-only inference mode, no camera frames will leave the device.
- **Today:** the game streams JPEG frames from the browser to a Python server. The server is intended to run on loopback (`127.0.0.1`) on your own machine, but frames do cross a process boundary.
- Data collection (`tools/collect`) records video only after an explicit consent screen, uses pseudonymous subject IDs, and supports deleting a subject. See `tools/collect/README.md`.

## License

MIT, see [LICENSE](LICENSE). Third-party assets (3D models and textures in `apps/web/public`) may carry their own licenses; see the note in [CONTRIBUTING.md](CONTRIBUTING.md).
