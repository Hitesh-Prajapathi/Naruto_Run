# Refinement Plan — NarutoCV (Real-Time Hand-Seal & Gesture Recognition)

> **Context:** This repo is one of several portfolio projects I'm hardening for AI/ML Engineer job applications. The goal is to be interview-proof: every claim has to be reproducible, measured under real-world conditions, and backed by a public demo. This file is the working plan for this repo.

## Current state
- YOLOv8n (1.45M params, 5.83 MB ONNX, 3.2 ms inference), 12 hand seals (13 classes), trained with PyTorch DDP on 2,860 images
- Reported 99.64% val / 99.11% test accuracy
- Dual-view Bayesian fusion (0.6 centre crop + 0.4 EMA hand ROI), MediaPipe Pose body tracking, FSM combo parser
- 1-slot latest-frame mailbox scheduler (Condition-guarded, not lock-free), custom 21-byte binary WebSocket protocol (NCV1), 31.2 ms pipeline
- Three.js 60 FPS client, 288 Vitest + 30 Playwright tests
- **No public repo or demo.** This is the biggest problem.

## Gaps an interviewer will hit
- **"99.6% on 2,860 images: are train and test frames from the same people and sessions?"** If so, the accuracy is inflated by leakage. This is the first thing a CV interviewer asks.
- "How does it work on someone it has never seen, in a different room and light?"
- "How many false triggers per minute while someone just moves their hands?"
- "Is 31 ms the model time or the true camera-to-screen time?"

## Plan

### Phase 1: Honest evaluation (highest priority)
- **Subject-disjoint splits:** re-split by person and session, then report **Leave-One-Subject-Out (LOSO)** accuracy. Collect data from **15–25 different people** (classmates, dance crew) so the test set has unseen subjects.
- Metrics: **macro-F1**, per-class precision/recall, confusion matrix, **Expected Calibration Error (ECE)**.
- **Robustness test set** with slices: lighting (low/normal/backlit), background clutter, distance (0.5/1/2 m), camera (laptop webcam vs phone), skin tone (Monk Skin Tone scale buckets), left vs right hand dominance. Report accuracy per slice in a table.
- Publish a **dataset card**: subjects, sessions, consent, class balance, collection protocol.

### Phase 2: Event-level and streaming metrics (what users actually feel)
- Record 30+ minutes of labeled continuous video: seal sequences mixed with **natural non-seal hand motion**.
- Metrics: **combo detection precision/recall**, **false activations per minute** of idle motion, **time-to-detect** after a seal is formed.
- Measure **true glass-to-glass latency** (e.g. film a screen showing a ms timer alongside the hand, or LED flash method). Report P50/P95/P99, separate from model inference time.

### Phase 3: Deployment and performance
- Run fully **in the browser with ONNX Runtime Web (WebGPU + WASM fallback)** so the demo needs no server. Publish on GitHub Pages / Vercel.
- Performance table across hardware: M-series Mac, mid-range Windows laptop CPU, Android phone. Columns: FPS, P95 latency, model size.
- Quantization study: FP32 vs FP16 vs INT8 → accuracy delta vs latency.

### Phase 4: Publish
- Public repo, README with GIF at top, architecture diagram (capture → scheduler → model → fusion → FSM → render), all results tables, Limitations.
- CI: lint, Vitest, Playwright, and a model eval on a small frozen test set.
- Consider a neutral public name and title, e.g. "Real-Time Hand-Seal Recognition Engine". The anime branding is fine as a theme but shouldn't be the headline.

## Definition of done
- [ ] Subject-disjoint / LOSO accuracy + macro-F1 reported (expect it to drop from 99.6%; that's fine and credible)
- [ ] Robustness slice table
- [ ] False activations/min + combo precision/recall on continuous video
- [ ] Measured glass-to-glass latency P95
- [ ] Browser demo live, public repo, video

## Resume bullet templates
- "Achieved **[X]% macro-F1 on unseen subjects (leave-one-subject-out, N=[N] people)** for 13-class hand-seal recognition with a 5.8 MB YOLOv8n."
- "Held **[X] false activations/min** with [Y]% combo recall on [Z] min of continuous real-world video."
- "Delivered **P95 [X] ms glass-to-glass latency** at [Y] FPS fully in-browser (ONNX Runtime Web/WebGPU)."
