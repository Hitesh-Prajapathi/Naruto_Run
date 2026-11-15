# NarutoCV: Refinement Execution Plan

> Builds on [02_narutocv.md](02_narutocv.md). That file says *what* must be true at the end. This file says *how*, in order, with exit criteria per phase.
> Every number below that describes the current state comes from the repo as of 2026-11-13 (reports/, `prepare_dataset.py`, dataset filenames, test runs). Nothing is estimated unless marked.

---

## 0. Diagnosis: why hand signs are poor (read this first)

### 0.1 What the live evaluations show

From `reports/hand-accuracy.json` (v1.0) and `reports/hand-v1.1-rerun-accuracy.json` (v1.1). These are live webcam tests, 3 attempts per seal:

| Seal | v1.0 | v1.1 | Failure mode |
|---|---|---|---|
| dragon | 0/3 | **0/3** | never detected |
| ram | 0/3 | **1/3** | predicted as **rat** 5 of 6 times |
| dog | 0/3 | **1/3** | scattered: monkey, hare |
| tiger | 3/3 | 2/3 | → rat |
| horse | 1/3 | 2/3 | missed |
| bird, boar, hare, monkey, ox, rat, snake, zero | mixed | 3/3 | ok |

- Time-to-detect after the seal is formed: **median ~1.0–2.7 s** for most seals. That's far too slow for a game.
- `cv_model/inference/hand_config.py` holds **13 hand-tuned confidence floors (0.32 → 0.92) and 13 margin floors**, plus special-case guards ("Rat and Ram are the model's dominant live-camera confusion pair", "Tiger → Dog paired guard"). This is what a model that doesn't generalize looks like when it gets patched: each threshold fixes one session and breaks another.

### 0.2 Root causes (verified)

1. **Effectively one subject and one session.** The dataset filenames are UUID1s, which encode the capturing machine and a timestamp:
   - **1,797 of 2,245 images (80%)** come from one machine, captured **within 21 minutes on 2020-02-21**. These are burst frames.
   - The other 437 come from one other source over 4 days. All 86 of the original "test" images are from a single day of that source.
2. **Frame-level leakage.** `cv_model/data/prepare_dataset.py` merges the original train and test folders, then does a **random stratified 80/10/10 split per frame**. Near-identical consecutive burst frames land in train, val *and* test. So **99.64% val / 99.11% test measures memorisation, not recognition.**
3. **"2,860 training images" is inflated.** About 1,795 are real; the rest are augmented copies made to balance classes to 220 each.
4. **Whole-frame classifier, not a hand classifier.** Training images are centre-cropped full frames (background, face, clothing included). At runtime a 0.6 centre-crop + 0.4 hand-ROI fusion is applied, but the ROI branch was never trained on ROI crops. The model learned the 2020 room, not the hands.
5. **No canonical seal definition.** Dragon at 0/3 in both runs, with zero confusion mass, suggests the way it's performed live doesn't match the dataset's version. Ram↔rat is the same issue: the pixel difference is small, the finger-topology difference is large.
6. **Temporal filter cost.** A 5-frame vote with 3 votes needed, on top of strict per-class floors, means correct but under-confident frames get thrown away. Hence the 1–3 s latency.

**Conclusion:** more threshold tuning won't fix this. It needs new data from many people, a hand-centric model, calibrated probabilities, and a proper temporal decoder. Phases 2 to 4 below do that.

---

## Phase 0: Repo cleanup (do this before any refinement)

Goal: a clean, single-environment, professionally structured repo that a stranger can clone and run in 5 commands. No ML changes yet.

### 0.A Protect in-flight work (Day 1, 30 min)
- [ ] 15 modified files (+710/−82 lines: boss system, seal sequence, combat HUD, transport pipeline) are **uncommitted on `main`**. Branch them first:
  `git switch -c wip/boss-seal-sequence && git add -A cv_model/Nishit_Frontend && git commit`
- [ ] Note: `test_session_pipeline.py` fails to import in the root `.venv` (that venv is missing transport deps). Fix it as part of 0.C, not now.
- [ ] Tag the current state: `git tag v0-baseline` so the old numbers can always be reproduced (needed for the "before vs after" story).

### 0.B Shrink and de-junk (Day 1)
| Item | Size | Action |
|---|---|---|
| Unreachable git objects | **2.02 GiB loose, 2,385 unreachable** | Something large (probably the dataset) was `git add`ed and never committed. Check with `git fsck --unreachable`, then `git gc --prune=now`. |
| `cv_model/data/venv/` | 1.4 GB | Second venv. Delete after 0.C. |
| `.venv/` | 446 MB | Rebuild with `uv` in 0.C. |
| `cv_model/data/prepared_dataset{,.zip}` | 551 MB | Leaky split. Delete once Phase 2 produces the new split (keep the zip off-repo for the baseline repro). |
| `cv_model/data/hand_landmarker.task` | 7.8 MB | Duplicate of `cv_model/models/hand_landmarker.task`. Delete. |
| `cv_model/models/pretrained/*.pt` (tracked) | 11 MB | Only the notebook uses them. Remove from git; download at train time. |
| `pose_landmarker.task` at repo root (tracked) | 5.5 MB | Move into `models/`, fetched by script, verified by sha256. |
| `reports/` (untracked) | 62 MB | Keep only `*accuracy.json` + benchmark JSON as `docs/results/v0/`. Frame dumps (`*-frames.{csv,jsonl}`) go to artifact storage, not git. |
| `run_combined_tracker.command`, `test_combined_pipeline.command`, `start_game.ps1` | — | Replace with a `Makefile` / `justfile` (`make camera`, `make serve`, `make web`, `make test`). |
| `.DS_Store`, `__pycache__/` on disk | — | Already ignored. Run `git clean -X -n` to review, then remove. |

### 0.C One toolchain (Day 2)
- [ ] Single `pyproject.toml` managed by **uv**, with extras: `runtime` (opencv, mediapipe, onnxruntime), `transport` (aiohttp), `train` (torch, ultralytics, timm), `eval` (sklearn, pandas, matplotlib), `dev` (pytest, ruff, mypy, pre-commit). Commit `uv.lock`.
- [ ] Switch Python tests to **pytest** (they currently run only via `unittest discover`, and pytest isn't installed). Baseline today: **63 pass (cv_model), 52 pass + 1 import error (transport)**.
- [ ] Web baseline today: **370 Vitest tests pass** (02_narutocv.md says 288, so update it). Add `eslint`, `prettier`, `tsc --noEmit` scripts.
- [ ] `pre-commit`: ruff (lint + format), mypy (strict on `inference/`), eslint, prettier, a large-file guard (>2 MB), and `nbstripout` for notebooks.

### 0.D Target layout (Day 2–3)
Rename away from a person's name in module paths (`Nishit_Frontend`) and separate concerns:

```
narutocv/
├── apps/web/                    # was cv_model/Nishit_Frontend/frontend (Three.js game)
├── services/transport/          # was cv_model/Nishit_Frontend/transport + serve_transport.py
├── src/narutocv/                # Python library (importable, typed)
│   ├── data/                    # collection, manifests, splits, dedup
│   ├── features/                # landmark canonicalisation + features
│   ├── models/                  # image branch, landmark branch, fusion, calibration
│   ├── temporal/                # HMM / hysteresis decoder, combo FSM
│   ├── inference/               # runtime, scheduler, pipeline (was cv_model/inference)
│   └── eval/                    # LOSO, slices, event metrics, latency
├── tools/collect/               # data-collection app (Phase 2)
├── tools/label-review/          # clip review/trim UI
├── training/                    # configs + scripts (notebook becomes a script; notebook kept as exploration/)
├── models/manifest.yaml         # name, version, url, sha256, size per artifact
├── schemas/                     # protocol + event JSON schemas (merge the two schemas/ dirs)
├── docs/                        # architecture, dataset card, model card, results/, adr/
│   └── archive/                 # context.md, implementation_plan.md (historical; contains old claims)
├── tests/                       # python tests (unit / integration / golden)
├── Makefile  pyproject.toml  uv.lock  .pre-commit-config.yaml
└── .github/workflows/
```
- [ ] Use `git mv` throughout to keep history. Fix imports and paths in one PR, with all tests green.
- [ ] `INTERVIEW_STUDY_GUIDE.md` and `presentation_guide.md` are personal prep. Keep them **outside the public repo** (or gitignored under `notes/`).
- [ ] `context.md` makes claims ("state-of-the-art", "≥99% Top-1") that Phase 1 will disprove. Move it to `docs/archive/` with a header saying it's superseded.

**Exit criteria:** fresh clone → `uv sync && make fetch-models && make test && make web` works. Repo checkout < 60 MB (excluding node_modules). All tests green under pytest/vitest. `v0-baseline` tag exists.

---

## Phase 1: Honest baseline on existing data (Week 1, 2–3 days)

Before building anything new, measure how bad the current model really is. This becomes the "before" column in the README and is a strong interview story ("I found my own leakage").

- [ ] `narutocv.data.manifest`: build a parquet manifest of every raw image: `path, class, source_node (UUID MAC), capture_ts, session_id (source+date), phash, width, height`.
- [ ] **Near-duplicate audit:** count pHash clusters (Hamming ≤ 6) that cross the old train/val/test split. Report the leakage percentage.
- [ ] **Group-disjoint re-evaluation** of the *existing* `best_model_A.onnx`:
  - Train on the 2020-02-21 burst source, test on the other source (and the reverse).
  - `GroupKFold` by `session_id`.
- [ ] Eval package v1 (reused in every later phase): macro-F1, per-class P/R, confusion matrix, **ECE + reliability diagram**, bootstrap 95% CIs (with ~86 test images per source, CIs matter).
- [ ] Commit `docs/results/v0-baseline.md`: old claimed numbers next to honest numbers.

**Exit:** a group-disjoint macro-F1 for v0 with a CI. Expect it to be far below 99%. That's the point.

---

## Phase 2: Data, the real fix (Weeks 1–3)

### 2.1 Canonical seal spec (Day 1, before collecting)
- [ ] `docs/seals.md`: one reference photo per seal from front and side, a finger-by-finger description, and the common confusions (ram vs rat, dog vs hare, tiger vs rat). Decide on **one** dragon variant. Collection subjects are shown this card.
- [ ] Pick the left/right convention and record handedness, so mirror augmentation can be validated per class.

### 2.2 De-risking spike: can MediaPipe see interlocked hands? (Day 1–2)
Run this before committing to an architecture. It decides how much weight the landmark branch can carry.
- [ ] Record 3 subjects × 13 seals × 5 s. For each seal, measure the % of frames where HandLandmarker returns **2 hands**, 1 hand, or 0, and landmark jitter (std of normalised fingertip positions while the seal is held).
- [ ] Expected: open seals (tiger, ram, rat, snake) will be fine. Tightly interlocked ones (boar, dog, dragon?) may drop to 1 hand. That table decides the fusion design in Phase 3.

### 2.3 Collection tool `tools/collect/` (Days 2–4)
Browser-based, so friends can run it on their own laptops and phones:
- [ ] Consent screen (purpose, retention, right to delete) → pseudonymous `subject_id` → metadata form: camera type, lighting (low/normal/backlit), background (plain/cluttered), distance (0.5/1/2 m), dominant hand, optional self-reported **Monk Skin Tone** bucket.
- [ ] Prompted protocol per session: for each seal (shuffled order), "form → hold 3 s → release", × 3 takes. Then **3 prompted combos**. Then **2 min of free natural motion** (talking with hands, phone, face touching, clapping, prayer hands). These are hard negatives and feed the false-activations-per-minute metric.
- [ ] Records **video clips** (WebM, 720p, 30 fps) plus prompt timestamps, not single frames. Clips give frame-level and event-level ground truth from the same capture.
- [ ] Uploads to private storage (S3/R2/GCS bucket or HF private dataset) with a per-subject folder, so deleting one subject is one command.

### 2.4 Collection targets
| | Minimum | Target |
|---|---|---|
| Subjects | 15 | 25 |
| Sessions/subject | 1 | 2 (different room/light) |
| Takes/seal/session | 3 | 3 |
| Continuous labeled video | 30 min | 60 min |
| Cameras | laptop webcam | + phone + external webcam |
| Held-out "vault" subjects (never touched until the end) | 3 | 5 |

### 2.5 Labelling and versioning
- [ ] Auto-label from prompts, then **auto-trim transitions** using landmark stability (keep frames where hand pose velocity is low). Review in `tools/label-review` (gallery per clip, accept/trim/reject).
- [ ] Data versioning with **DVC** (remote = the bucket). Each dataset version has a frozen `splits.yaml`: LOSO folds + a vault subject list.
- [ ] Legacy 2020 data stays in as one extra source, `legacy_*`, grouped by session. Never let it decide test results alone.
- [ ] `docs/dataset_card.md`: subjects, sessions, demographics buckets (aggregated only), class balance, protocol, consent, known gaps.

**Exit:** ≥15 subjects collected, DVC-versioned, splits frozen, dataset card written.

---

## Phase 3: Model redesign (Weeks 3–5)

### 3.1 Architecture: hand-centric hybrid
```
camera ─► HandLandmarker (2 hands) ─┬─► [A] Landmark branch: canonicalised 2×21×3 + handedness
        └► Pose wrists (fallback ROI)│        → features (joint angles, fingertip–fingertip and
                                     │          inter-hand distances, palm normals) → MLP (~50k params)
                                     └─► [B] Image branch: two-hand union crop (+margin, pose-wrist
                                              fallback when <2 hands) → MobileNetV3-S / EffNet-Lite0 /
                                              YOLOv8n-cls @ 160–224
                 [A] + [B] + hand_count + landmark_quality ─► stacked fusion (logistic reg.)
                 ─► temperature scaling ─► calibrated p(seal | frame)
                 ─► temporal decoder (3.3) ─► seal events ─► combo FSM ─► game
```
Why this addresses each root cause:
- Landmarks don't depend on background, lighting or skin tone, so they fix root cause 4.
- Finger topology separates **ram vs rat** and **dog vs hare** directly, which fixes root cause 5.
- The image branch covers frames where interlocking breaks the landmarker (from the 2.2 spike).
- Calibrated probabilities let **one global threshold plus an open-set reject** replace the 26 per-class constants in `hand_config.py`.

### 3.2 Training recipe
- [ ] Training becomes **scripts + YAML configs** (`training/configs/*.yaml`). The notebook is kept only for exploration. Deterministic seeds. Every run logs to **MLflow** (local file store is fine) or W&B: config, git SHA, dataset DVC hash, metrics, artifacts.
- [ ] Image-branch augmentation: **background replacement** (MediaPipe selfie segmentation → composite random backgrounds), colour/white-balance and skin-tone-safe jitter, motion blur, JPEG artefacts, low-light noise, scale/position jitter of the crop (simulates landmarker ROI error). Mirror flip only for classes that 2.1 confirms are mirror-safe.
- [ ] Landmark-branch augmentation: small 3D rotations, per-joint Gaussian noise matched to the jitter measured in 2.2, random hand dropout (forces robustness to one hand going missing).
- [ ] **Hard-negative mining** from the free-motion clips into `zero`/`unknown`.
- [ ] Model selection on LOSO macro-F1. **Vault subjects only touched once, at the end.**

### 3.3 Temporal decoder (fixes the 1–3 s latency)
Replace the 5-frame/3-vote filter:
- [ ] Exponential smoothing of calibrated probabilities with **hysteresis** (enter at τ_on, exit at τ_off < τ_on) and a **minimum dwell in milliseconds**, not frames, so it behaves the same at 20 and 60 fps.
- [ ] Optional HMM/Viterbi over {seal states + transition + idle}, with the transition matrix learned from combo clips.
- [ ] **Grammar-aware decoding** (keyword-spotting style): when the game is mid-combo, add a configurable prior to the expected next seals. Always report metrics **with and without** this prior, so the gain is shown honestly.
- [ ] Tune τ_on, τ_off and dwell on validation subjects to hit a target **false activations/min ≤ 0.5** while maximising combo recall. Show the trade-off curve (FA/min vs recall), DET-curve style.

### 3.4 Experiment matrix (each row = one MLflow run group, LOSO)
| ID | Model | Purpose |
|---|---|---|
| E0 | v0 `best_model_A.onnx` as-is | honest baseline on the new subjects |
| E1 | v0 architecture retrained on new data, full frame | how much of the gain is data alone |
| E2 | image branch on hand-union crops | does cropping help |
| E3 | landmark MLP only | invariance ceiling, and failure on interlocked seals |
| E4 | E2+E3 stacked fusion | expected winner |
| E5 | E4 + temporal decoder | event-level metrics |
| E6 | E5 + per-user few-shot adaptation (5 s per seal at onboarding, prototype/kNN head on landmark embeddings) | personalisation as a stretch, reported separately |

### 3.5 Quantisation and export
- [ ] ONNX export for each branch, then compare FP32 vs FP16 vs INT8 (static quantisation, calibrated on a train subset): macro-F1 delta vs latency vs size.
- [ ] **Model bundle** = `{onnx files, labels.json, calibration.json (T, τ_on, τ_off, dwell), feature_spec.json, model_card.md, sha256}`, versioned with semver. `hand_config.py` is deleted. Thresholds come from the bundle.

**Exit:** E4/E5 LOSO macro-F1 with CIs. Per-class recall ≥ 0.85 for **every** seal, explicitly including dragon, ram and dog. ECE < 0.05. Bundle published.

---

## Phase 4: Event-level, streaming and latency evaluation (Week 5)

- [ ] `narutocv.eval.events`: on continuous labeled video, report **combo precision/recall**, **false activations/min** on idle/free motion, and **time-to-detect** (P50/P95) from the start of a stable seal to the event. Event matching uses an IoU window (configurable tolerance).
- [ ] **Glass-to-glass latency rig:** the screen shows a ms timer plus a flash on event, and a phone at 240 fps films both the hand and the screen. A script reads frame indices. Report P50/P95/P99 **separately** from the per-stage timings the pipeline already logs (`final-step6-benchmark.json` today: body ~10 ms, capture 30 fps, 0.2% drops).
- [ ] **Robustness slice table** on vault + LOSO subjects: lighting × background × distance × camera × MST bucket × handedness. Flag any slice where macro-F1 is more than 10 points below overall.
- [ ] **Failure-mode gallery:** the top 20 confident errors with crops and landmarks, in `docs/results/`.
- [ ] Final vault evaluation, run once, recorded with the git SHA and bundle version.

**Exit:** every Definition-of-Done metric in 02_narutocv.md has a measured number.

---

## Phase 5: In-browser deployment (Weeks 5–6)

- [ ] Run the whole recogniser in a **Web Worker**: MediaPipe Tasks Vision (JS) for hand + pose landmarks, and **onnxruntime-web** for the image branch and fusion (WebGPU EP, falling back to WASM SIMD+threads). Frames come in via `requestVideoFrameCallback` → `OffscreenCanvas` → `ImageBitmap` transfer (zero-copy).
- [ ] Port the temporal decoder and combo FSM to TS **from the same spec**. Add a **golden parity test**: recorded clip → Python events vs browser events must match within ±1 event / ±50 ms. Run it in CI with Playwright and a fake camera (`--use-file-for-fake-video-capture`).
- [ ] Keep the Python service as the **reference implementation and eval engine**. The game picks `source = browser | websocket` (the `eventSource` abstraction already exists).
- [ ] Model assets served from GitHub Releases / HF Hub with sha256 checks and Cache-Storage caching. Lazy-load the image branch.
- [ ] Hardware table: M-series Mac, mid Windows laptop (CPU-only, WASM), Android Chrome (WebGPU if available). Columns: FPS, P95 per-frame latency, glass-to-glass P95, bundle size, first-load time.
- [ ] Privacy statement: in browser mode **no frames leave the device**.

---

## Phase 6: Production hardening (runs in parallel from Week 2)

### Product / UX
- [ ] **Onboarding and calibration flow:** camera permission → lighting/distance check (warn if face brightness or hand size is out of range) → neutral-pose calibration → optional few-shot personalisation (E6).
- [ ] **Seal trainer mode:** reference card next to the live camera, a per-class confidence bar, and landmark overlay showing "what the model sees". This doubles as a debugging tool and demo material.
- [ ] Graceful degradation: low FPS → drop the image branch and use landmarks only; no WebGPU → WASM; camera lost → pause and resume.

### Reliability and code quality
- [ ] mypy strict on `narutocv.inference` / `features` / `temporal`, and TS `strict` throughout.
- [ ] Property-based tests (Hypothesis) for feature canonicalisation: invariance to translation/scale/rotation, and the mirror mapping.
- [ ] Transport server: origin allow-list, max frame size, per-connection rate limit, protocol version + model-bundle version handshake (reject mismatches with a clear error). Backpressure tests already exist, so extend them.
- [ ] Structured JSON logging plus an opt-in local diagnostics overlay. Per-stage timing histograms already exist; expose them at `/metrics` (Prometheus format) in server mode.
- [ ] Dockerfile for the transport service (slim, non-root, healthcheck at the existing `health.py`).

### CI/CD (GitHub Actions)
| Workflow | Contents |
|---|---|
| `python.yml` | uv sync → ruff → mypy → pytest (unit + integration) |
| `web.yml` | npm ci → eslint → tsc → vitest → build → Playwright (fake camera) |
| `model-gate.yml` | on changes to `src/narutocv/models`, `training/`, or `models/manifest.yaml`: run eval on a **frozen mini test set (~2 subjects, small, DVC-pulled)**. Fail if macro-F1 drops >2 pts, ECE rises >0.02, bundle exceeds the size budget, or the CPU latency benchmark regresses >15%. |
| `parity.yml` | golden Python↔browser event-parity test |
| `deploy.yml` | on tag: build web → GitHub Pages/Vercel, attach model bundle to the Release |

### Documentation
- [ ] README: GIF first, then one-line pitch, **results table (old vs honest vs new)**, architecture diagram, "run in 5 commands", Limitations, dataset and model cards.
- [ ] ADRs in `docs/adr/`: (1) why the landmark+image hybrid, (2) why calibration instead of per-class thresholds, (3) browser-first inference, (4) the leakage finding.
- [ ] Neutral public name, e.g. **"Real-Time Hand-Seal Recognition Engine"**, with the anime game as the demo theme.

---

## Timeline (solo, ~6–7 weeks)

| Week | Focus | Gate |
|---|---|---|
| 1 | Phase 0 cleanup → Phase 1 honest baseline → 2.1 seal spec + 2.2 MediaPipe spike → start collection tool | clean repo, v0 honest numbers, architecture confirmed |
| 2 | Collection tool done, collect ≥8 subjects; CI python/web live | data v0.1 in DVC |
| 3 | Collect to ≥15 subjects; E0–E3 | first LOSO numbers |
| 4 | E4–E5, calibration, temporal decoder, quantisation | per-class recall ≥0.85 incl. dragon/ram/dog |
| 5 | Event metrics, glass-to-glass rig, slices, vault eval; start browser port | all DoD metrics measured |
| 6 | Browser inference + parity test, hardware table, deploy | live public demo |
| 7 | README, cards, ADRs, demo video, polish, E6 stretch | public repo + resume bullets filled |

## Risks and mitigations
| Risk | Mitigation |
|---|---|
| MediaPipe fails on interlocked seals | Found early by the 2.2 spike. The image branch plus a hand-count feature covers it. Worst case: a small custom two-hand keypoint fine-tune (stretch). |
| Can't recruit 15 people | Browser collection tool lets remote friends contribute in 10 min. Minimum viable is 10 subjects + LOSO, reported with honest CIs. |
| Results look "worse" than 99.6% | That's expected and intended. The story is leakage found → fixed → measured on unseen people. |
| Browser/Python divergence | Single spec + golden parity test in CI. |
| Scope creep (E6, custom keypoints) | Marked as stretch. Not on the critical path. |

## Definition of done (extends 02_narutocv.md)
- [ ] Phase 0 exit criteria met; `v0-baseline` tag; repo < 60 MB
- [ ] v0 honest (group-disjoint) numbers published alongside the old claims
- [ ] ≥15 subjects, dataset card, DVC-versioned, vault untouched until final eval
- [ ] LOSO macro-F1 + CI; **every seal recall ≥ 0.85**; ECE < 0.05
- [ ] False activations/min ≤ 0.5, combo P/R, time-to-detect P95 < 400 ms (target)
- [ ] Glass-to-glass P50/P95/P99 measured
- [ ] Robustness slice table + failure gallery
- [ ] Browser demo live, parity test green, hardware table
- [ ] CI with model-regression gate; README/model card/ADRs; demo video
