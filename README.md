# Naruto Run: real-time hand-seal recognition engine

<!-- TODO(owner): record a 10-15 s screen capture of the game driven by hand seals, save it as docs/media/demo.gif and embed it here. -->

A webcam-controlled browser game and the computer-vision engine behind it. The engine recognises 13 two-handed "hand seals" and 5 upper-body movements in real time, turns them into typed events, and feeds a Three.js endless runner: lean and jump to steer, chain seals to cast attacks, fight a boss.

The project is also a worked example of **auditing your own ML results**. Version 0 reported 99.6% accuracy. That number turned out to measure memorisation of near-duplicate frames; the honest out-of-source number is macro-F1 0.54. The repository keeps both, explains the gap, and rebuilds the system around evaluation that can be trusted.

## How it works

```
browser camera ─► NCV1 binary WebSocket ─► latest-frame scheduler ─► recognition ─► decoder ─► combo FSM ─► game events
 (apps/web)          (JPEG frames)          (1-slot mailbox)         (ONNX + MediaPipe)  (legacy | v1)
```

| Stage | Implementation |
|---|---|
| **Capture & transport** | The browser owns the camera and streams JPEG frames over a versioned binary protocol (NCV1, protocol 1.2.0) to an aiohttp server on loopback. Origin allow-list, frame-size and rate limits, handshake with capability negotiation, one controlling client, health/metrics endpoints, graceful shutdown, Docker image. |
| **Scheduling** | `LatestFrameScheduler`: a single-slot "latest frame wins" mailbox, so latency cannot build up when inference is slower than capture. Dropped/superseded frames are counted. |
| **Hand-seal recognition** | 13-class YOLOv8n-derived image classifier (ONNX) on the centre crop and on a MediaPipe HandLandmarker region, fused. |
| **Body movement** | MediaPipe Pose → idle, jump, "Naruto run" lean, bend left, bend right. |
| **Decoding** (selectable) | `legacy`: 13 hand-tuned per-class confidence floors + 5-frame vote. `v1` (opt-in, `--hand-decoder v1`): temperature scaling + a streaming hysteresis decoder (`tau_on`/`tau_off`, dwell and refractory time in milliseconds, open-set reject, optional grammar prior from the combo state). Specified once and ported to TypeScript with golden-fixture parity tests. |
| **Combos** | `AttackRecognizer` state machine: tiger → horse = fire, snake → dragon = water, monkey → ox = sand, dog → rat = wind, hare = lightning dodge. |
| **Game** | Vite + TypeScript + Three.js: endless forest runner, lane/jump controller driven by body movement, seal-sequence HUD, boss encounters with a special attack and a hare-seal counter, game-over flow. Keyboard fallback when no camera is used. |

Also in the repo: a consent-first **multi-subject data collection tool** (`tools/collect`) that records prompted seal videos with pseudonymous subject IDs, for the subject-disjoint dataset the next model needs.

Details: [architecture](docs/architecture.md) · [decision records](docs/adr/) · [transport protocol](docs/transport.md) · [seal spec](docs/seals.md) · [decoder spec](docs/specs/seal_decoder_v1.md)

## Results (honest)

Every number below comes from a committed file; see the [results index](docs/results/README.md).

**The 99.6% was leakage.** The original pipeline merged the train and test folders and split frames at random. 86% of test images have a same-class near-duplicate in train (perceptual hash distance ≤ 6), 70% an exact twin. Retraining the same recipe on group-disjoint splits ([v0 baseline](docs/results/v0-baseline.md)):

| Split (yolov8n-cls, same recipe) | Macro-F1 [95% CI] |
|---|---|
| Random frame-level split (the original protocol, leaky) | 0.998 [0.994, 1.000] |
| **Train on the main capture source, test on the other source** | **0.537 [0.494, 0.578]** |
| Reverse direction | 0.867 |
| Leave-one-session-out | 0.635 – 0.942 |

These are source/session-disjoint, not subject-disjoint: the 2,245-image legacy dataset comes from two capture sources and five sessions (80% of it from one 21-minute burst), and the number of distinct people is unknown and likely very small ([dataset card](docs/dataset_card.md)).

**Live webcam, legacy vs v1 decoder** (13 seals × 3 attempts, one subject, [details](docs/results/live_2026-11-15/README.md)):

| Decoder | Correct | 95% CI | Median time-to-detect |
|---|---|---|---|
| legacy | 24 / 39 | [0.46, 0.75] | 1.10 s |
| v1 | 23 / 39 | [0.43, 0.73] | 1.40 s |

The decoders tie. Raw per-frame predictions show the image model itself is the bottleneck (`ox` absorbs hare, monkey and dog; `tiger` is the fallback), which matches the 0.54 out-of-source number. The fix is multi-subject data, not threshold tuning.

**Landmark spike** ([details](docs/results/landmark_spike_v0.md)): MediaPipe detects both hands in only 52.7% of images, and almost never for interlocked seals (ram 3.3%, tiger 5.2%). A landmark-only classifier drops from 0.89 accuracy (leaky split) to 0.49 across sources. So landmarks are planned as one gated branch of a hybrid model, not a replacement.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/), Node.js (current LTS) and a webcam.

```bash
make setup          # uv sync --all-extras + npm ci
make fetch-models   # download + sha256-verify the MediaPipe models (models/manifest.yaml)
make test           # pytest + vitest
make serve          # recognition server on 127.0.0.1:8765  (add v1: uv run python -m narutocv.transport --hand-decoder v1)
make web            # game on http://127.0.0.1:5173, in a second terminal
```

Open **http://127.0.0.1:5173/?cv=1** to drive the game with the camera (without `?cv=1` it uses the keyboard).

Other entry points:

```bash
make camera                                                       # OpenCV window with live recognition overlay
uv run python scripts/evaluate_live_camera.py --mode hand         # guided accuracy test, reference photos beside the feed
uv run python scripts/evaluate_live_camera.py --mode hand --hand-decoder v1 --summary-json out.json
```

`make help` lists everything else (`lint`, `fmt`, `mypy`, `build-web`, `ci`).

## Repository layout

```
apps/web/            Three.js game (Vite + TypeScript); src/recognition/ = TS port of features, decoder, calibration
src/narutocv/
  inference/         recognisers, scheduler, runtime controller, decoders, output schema, events
  transport/         aiohttp WebSocket bridge (NCV1 frames in, JSON events out)
  features/          canonical two-hand landmark features (spec v1)
  temporal/          SealDecoder (hysteresis, dwell, open-set reject) + tuning
  models/            temperature / vector scaling, stacked fusion
  data/, eval/       dataset manifest + leakage audit, metrics with bootstrap CIs
tools/collect/       consent-first multi-subject video collection app + receiver
scripts/             camera/eval entry points, baseline/ (honest retraining), spikes/, golden/ fixtures
models/              manifest.yaml (sha256-pinned), legacy ONNX, calibration_v0.json
schemas/             JSON schemas: pipeline output/events, transport, collection sessions
docs/                architecture, ADRs, specs, dataset card, results, plan/
tests/               Python tests (unit, property-based, parity, transport)
```

## Engineering

- **Tests:** about 350 Python tests (pytest + Hypothesis), about 520 web tests (Vitest), 35 for the collection tool. Golden fixtures keep the Python reference and the TypeScript port numerically identical.
- **Tooling:** uv-managed Python 3.12, ruff, mypy, pre-commit, Makefile, GitHub Actions for Python, web and a model-quality gate.
- **Reproducibility:** models are pinned by sha256 in `models/manifest.yaml`; every published number links to the file and command that produced it.

## Limitations

- Training data is effectively one person; there is no subject-disjoint evaluation yet.
- The deployed ONNX model saw most of the old test set during training, so it must not be used to report accuracy.
- Live accuracy is about 60% on 13 seals, and time-to-detect is 1 to 1.5 s, which is slow for a game.
- MediaPipe often misses one or both hands on interlocked seals.
- The v1 decoder's thresholds and calibration are untuned placeholders until multi-subject validation data exists.
- Browser-only inference (no frames leaving the device) is designed but not implemented; today frames go to a local Python process.

## Roadmap

From the [execution plan](docs/plan/03_refinement_execution_plan.md):

1. **Data:** record 15–25 consenting subjects with `tools/collect`, hold 3–5 out as an untouched final test set, freeze leave-one-subject-out splits.
2. **Model:** hand-centric hybrid (image branch + gated landmark branch), real calibration, decoder tuned on validation subjects (target ≤ 0.5 false activations per minute).
3. **Evaluation:** event-level metrics on continuous video, glass-to-glass latency rig, robustness slices.
4. **Browser:** in-browser worker (MediaPipe JS + onnxruntime-web) using the TS port, deploy.

## Credits

Contributors: [Hitesh Prajapathi](https://github.com/Hitesh-Prajapathi) (recognition engine, training, evaluation, transport), Nishit Daruwala (initial browser game and body-movement input) and Ram Kapadia. Hand-seal names and the theme come from the *Naruto* franchise; this is a non-commercial fan and learning project with no affiliation. The legacy training images are a third-party 2020 dataset whose license and consent status are unconfirmed, so they are not redistributed here.

## License

Code: MIT, see [LICENSE](LICENSE). The 3D models and textures in `apps/web/public/` are third-party assets and are **not** covered by the MIT license; see [CONTRIBUTING.md](CONTRIBUTING.md).
