# 🌀 NarutoCV — Master Architecture, History & Technical Context

> **Project:** NarutoCV — Real-Time Computer Vision Jutsu & Gesture Recognition Engine  
> **Version:** 1.5.0 (Latest-Frame Runtime + Performance Instrumentation)
> **Repository Root:** `/Users/hiteshprajapathi/Desktop/Naruto_Run/`
> **Status:** Locked recognition with versioned events, latest-frame scheduling, and phase timings

---

## 📜 Table of Contents
1. [Executive Summary & Core Objectives](#1-executive-summary--core-objectives)
2. [Master Architecture — Dual-Pipeline System](#2-master-architecture--dual-pipeline-system)
3. [Biomechanical Landmark & Vector Geometry Specifications](#3-biomechanical-landmark--vector-geometry-specifications)
4. [Dataset Engineering & Preprocessing Pipeline (Component 1)](#4-dataset-engineering--preprocessing-pipeline-component-1)
5. [Model Exploration, Training Trajectory & Benchmarks (Component 2)](#5-model-exploration-training-trajectory--benchmarks-component-2)
6. [ONNX Web Runtime & Browser Engine Architecture](#6-onnx-web-runtime--browser-engine-architecture)
7. [Jutsu Catalog, State Machine & Debounce Filter Specifications](#7-jutsu-catalog-state-machine--debounce-filter-specifications)
8. [Repository Asset & File Directory Map](#8-repository-asset--file-directory-map)
9. [Detailed Technical Blueprint for Remaining Components (3 & 4)](#9-detailed-technical-blueprint-for-remaining-components-3--4)

---

## 1. Executive Summary & Core Objectives

**NarutoCV** is a state-of-the-art, web-native Computer Vision application that enables users to execute iconic anime Jutsu spells by physically performing traditional **Naruto Hand Signs** and **Body Stances** in front of a webcam.

### Technical Performance Targets:
- **Inference Speed:** Real-time 30–60 FPS on standard consumer laptops and web browsers.
- **Classification Accuracy:** $\ge 99\%$ Top-1 accuracy across 12 traditional Naruto hand seals + 1 "zero" (no sign) baseline class.
- **Inference Latency Budget:** Total end-to-end processing pipeline latency $\le 25\text{ ms}$ per video frame.
- **Zero Heavy Backend Infrastructure:** All computer vision inference, 3D landmark extraction, debouncing, and state machine transitions run **100% client-side** in the user's web browser using WebGL / WebGPU and ONNX Runtime Web.

---

## 2. Master Architecture — Dual-Pipeline System

To achieve real-time 60 FPS performance without frame dropping or lag, NarutoCV uses a **Decoupled Dual-Pipeline System**. Hand classification and body gesture tracking are handled by specialized parallel pipelines.

```
                                📹 Full Webcam Video Stream (1280×720 @ 60 FPS)
                                                    │
              ┌─────────────────────────────────────┴─────────────────────────────────────┐
              ▼                                                                           ▼
   📷 PIPELINE 1: Hand Sign Classification                                     🤸 PIPELINE 2: Body Movement Tracking
 (Center frame + stabilized color hand ROI)                                    (Focus: Full Camera Field of View)
              │                                                                           │
   1. Center-square RGB view + MediaPipe Hands                                 1. MediaPipe Pose (33 3D Joint Landmarks)
      Extract combined two-hand box and smooth it with EMA                         Calculates joint angles & spatial vectors
              │                                                                           │
   2. Trained Classifier on both RGB views                                     2. Geometric Heuristic Engine
      Fuse class probabilities: 60% center + 40% hand ROI                         - Naruto Run Stance (Lean >25° + Arms back)
              │                                                                   - Jumping (Hip height above baseline)
   3. Delayed no-hand guard + 5-frame evidence filter                              - Bending Left / Right (Torso offset)
      Emits one validated event per held hand seal                                 │
              │                                                                           │
              └─────────────────────────────────────┬─────────────────────────────────────┘
                                                    ▼
                                    ⚙️ JUTSU COMBO STATE MACHINE
                                - Sequence Matcher (e.g. Tiger → Dragon → Horse)
                                - 3.0s Inter-Seal Timeout, Duplicate Suppression
                                - Trigger Signals → Anime Canvas FX + Audio
```

### Why a Dual Pipeline Architecture?
1. **Resolution & Spatial Focus:** Hand sign classification keeps the training-matched center-square view and supplements it with a tight combined-hand ROI. Body tracking requires the wide camera field to track shoulders, hips, and knees.
2. **Computational Heterogeneity:** 
   - Hand signs use a custom-trained **YOLOv8 Nano Classifier (`best_model_A.onnx`)** executed via WebGL ONNX Runtime.
   - Body movements use **MediaPipe Pose** with deterministic vector geometry (dot products, coordinate heuristics), consuming only $\sim 0.1\text{ ms}$ CPU time per frame without requiring neural network training.

---

## 3. Biomechanical Landmark & Vector Geometry Specifications

### 3.1 MediaPipe 21 Hand Landmark Structure
For landmark-based approaches, MediaPipe Hands returns 21 3D keypoints per hand $(x, y, z)$, where $x, y \in [0, 1]$ are normalized image coordinates and $z$ represents depth relative to the wrist.

```
        [4] Thumb Tip
         │
        [3] IP         [8] Index Tip  [12] Middle Tip  [16] Ring Tip  [20] Pinky Tip
         │                │               │                │               │
        [2] MCP        [7] PIP         [11] PIP         [15] PIP        [19] PIP
         │                │               │                │               │
        [1] CMC        [6] MCP         [10] MCP         [14] MCP        [18] MCP
         └────────────────┴───────────────┴────────────────┴───────────────┘
                                         │
                                      [5] Index MCP ... [17] Pinky MCP
                                         │
                                      [0] WRIST (Origin Reference)
```

#### Landmark Vector Normalization Formula
To ensure invariance against camera distance (scale) and hand position in frame (translation):

$$\mathbf{p}_i = \begin{bmatrix} x_i \\ y_i \\ z_i \end{bmatrix}, \quad i \in \{0, 1, \dots, 20\}$$

1. **Translation Invariance (Center at Wrist):**
   $$\mathbf{p}'_i = \mathbf{p}_i - \mathbf{p}_0$$

2. **Scale Invariance (Unit Max Distance):**
   $$S = \max_{i} \|\mathbf{p}'_i\|_\infty, \quad \mathbf{\hat{p}}_i = \frac{\mathbf{p}'_i}{S}$$

3. **Flat Vector Representation (63-D Vector):**
   $$\mathbf{V}_{\text{hand}} = \Big[ \mathbf{\hat{p}}_{0,x}, \mathbf{\hat{p}}_{0,y}, \mathbf{\hat{p}}_{0,z}, \dots, \mathbf{\hat{p}}_{20,x}, \mathbf{\hat{p}}_{20,y}, \mathbf{\hat{p}}_{20,z} \Big]^T \in \mathbb{R}^{63}$$

---

### 3.2 Body Movement Geometry Formulas (MediaPipe Pose 33 Landmarks)

```
        [11] L_Shoulder ────── [12] R_Shoulder
              │                     │
              │     [23] L_Hip ─────┼───── [24] R_Hip
              │            │        │        │
        [13] L_Elbow       │  [14] R_Elbow   │
              │            │        │        │
        [15] L_Wrist       │  [16] R_Wrist   │
                     [25] L_Knee        [26] R_Knee
```

#### 1. Naruto Run Stance Detection
- **Torso Forward Lean Angle ($\theta_{\text{torso}}$):**
  Calculated using the vector connecting hip midpoint to shoulder midpoint relative to vertical vector $\mathbf{\hat{k}} = [0, -1, 0]^T$:
  $$\mathbf{M}_{\text{shoulder}} = \frac{\mathbf{P}_{11} + \mathbf{P}_{12}}{2}, \quad \mathbf{M}_{\text{hip}} = \frac{\mathbf{P}_{23} + \mathbf{P}_{24}}{2}$$
  $$\mathbf{v}_{\text{torso}} = \mathbf{M}_{\text{shoulder}} - \mathbf{M}_{\text{hip}}$$
  $$\theta_{\text{torso}} = \arccos\left( \frac{\mathbf{v}_{\text{torso}} \cdot \mathbf{\hat{k}}}{\|\mathbf{v}_{\text{torso}}\|} \right) > 25^\circ$$

- **Arms Extended Backward Condition:**
  Both wrists must be positioned behind hips in the $Z$-depth axis:
  $$Z_{15} > Z_{23} + \delta_z \quad \text{and} \quad Z_{16} > Z_{24} + \delta_z \quad (\delta_z \approx 0.05)$$

#### 2. Jump Detection
- **Hip Height Delta ($\Delta y_{\text{hip}}$):**
  Compares the current hip midpoint against the rolling median standing baseline:
  $$\Delta y_{\text{hip}} = \operatorname{median}(y_{\text{hip history}}) - y_{\text{hip current}} > 0.045$$

#### 3. Left / Right Bend Detection
- **Torso Horizontal Offset ($\Delta x_{\text{torso}}$):**
  Uses the normalized horizontal offset between shoulder and hip midpoints:
  $$\Delta x_{\text{torso}} = x_{\text{shoulder}} - x_{\text{hip}}$$
  Values above $0.06$ emit `bending_right`; values below $-0.06$ emit `bending_left`.

Lightning and stone remain hand-seal attacks. They are not body movement labels and body movement events never gate or trigger attacks.

---

## 4. Dataset Engineering & Preprocessing Pipeline (Component 1)

### 4.1 Raw Dataset Audit
- **Source Directory:** `Pure Naruto Hand Sign Data/`
- **Total Images:** 2,245 original images across 13 classes (`bird`, `boar`, `dog`, `dragon`, `hare`, `horse`, `monkey`, `ox`, `ram`, `rat`, `snake`, `tiger`, `zero`).
- **Data Anomalies Resolved:**
  - **435 RGBA (4-channel) images** (640×480 PNGs) force-converted to standard 3-channel RGB.
  - **Resolution Heterogeneity:** 3 distinct resolution groups (1280×720, 640×480, 3264×1840) standardized via center-cropping to 1:1 square aspect ratio followed by Lanczos downsampling to **224 × 224 pixels**.
  - **Class Imbalance:** Original train split ranged from 122 images (`ram`) to 273 images (`dog`).

### 4.2 Preprocessing Script Implementation (`cv_model/data/prepare_dataset.py`)

1. **Stratified Split (80% / 10% / 10%):**
   - **Train:** 1,796 images (80%)
   - **Validation:** 224 images (10%)
   - **Test:** 225 images (10%)
   - Fixed seed (`random_state=42`) for 100% reproducible splits across runs.

2. **Target Class Balancing (Augmented Oversampling):**
   - Every class in the training split was augmented to exactly **220 images**.
   - Total augmented training set size: **2,860 images** (1,796 original + 1,064 synthetic).

3. **Augmentation Rules & Strict Constraints:**
   - **CRITICAL RULE — NO Flips Allowed:** Horizontal and vertical flips were strictly **disabled**. Naruto hand seals are asymmetrical (left vs right hand placement is distinct per sign). Flipping an image creates invalid hand sign ground truth.
   - **Permitted Augmentations:**
     - Random rotation: $\pm 15^\circ$
     - Brightness jitter: $\pm 20\%$
     - Contrast & Color jitter: $\pm 20\%$
     - Gaussian blur: radius $0.1 - 1.0$ ($p=0.3$)
     - Random resized crop / zoom: scale $0.85 - 1.0$
     - Additive Gaussian noise: $\sigma = 0.01 - 0.03$ ($p=0.2$)

4. **Output Artifact:** `cv_model/data/prepared_dataset.zip` (285 MB).

---

## 5. Model Exploration, Training Trajectory & Benchmarks (Component 2)

We evaluated three competing model architectures on Kaggle with GPU acceleration (Tesla T4 ×2) to determine the optimal model for real-time web inference.

### 5.1 Model Comparison Matrix

| Metric | Approach A: YOLOv8n-cls on 224×224 RGB Crops | Approach B: YOLOv8n-cls on 64×64 Landmark Grids | Approach C: PyTorch MLP on 63-D Landmark Vectors |
|:---|:---:|:---:|:---:|
| **Model Type** | Convolutional Neural Network | Convolutional Neural Network | Fully Connected 3-Layer MLP |
| **Input Feature** | 224×224×3 RGB Hand Crop | 64×64×3 Reshaped Landmark Grid | 63-D Normalized Coordinate Vector |
| **Top-1 Validation Acc** | **99.64% 🏆** | 82.50% | 76.40% |
| **Top-5 Validation Acc** | **100.0%** | 96.10% | 94.20% |
| **Inference Time (CPU)** | **3.2 ms** | 4.5 ms (including MediaPipe) | 0.8 ms |
| **Model Size** | **5.8 MB (ONNX)** | 3.0 MB | 78 KB (`.pth`) |
| **MediaPipe Dependency** | **Optional** (Box crop only) | Required | Required |
| **Overall Verdict** | **WINNER (Selected for Prod)** | Alternative | Baseline |

---

### 5.2 Approach A Training Trajectory (YOLOv8n-cls)
- **Base Architecture:** `yolov8n-cls.pt` (56 layers, 1,454,941 parameters)
- **Hyperparameters:**
  - Optimizer: `AdamW` ($\text{lr}_0 = 0.01$, $\text{lr}_f = 0.0001$)
  - Warmup: 3 epochs ($\text{momentum} = 0.8$)
  - Weight Decay: $0.0005$
  - Label Smoothing: $0.1$
  - Batch Size: 64 (Distributed Data Parallel across 2× T4 GPUs)

```
Epoch 1/50  [==========] Loss: 2.3210  |  Top-1 Acc: 79.50%  |  Top-5 Acc: 94.20%
Epoch 2/50  [==========] Loss: 0.8784  |  Top-1 Acc: 90.20%  |  Top-5 Acc: 98.70%
Epoch 3/50  [==========] Loss: 0.2295  |  Top-1 Acc: 96.90%  |  Top-5 Acc: 99.60%
Epoch 4/50  [==========] Loss: 0.1228  |  Top-1 Acc: 98.70%  |  Top-5 Acc: 100.0%
Epoch 5/50  [==========] Loss: 0.0656  |  Top-1 Acc: 99.64% 🏆 |  Top-5 Acc: 100.0%
Epoch 15/50 [==========] Early Stopping Triggered (Patience=10 reached, best model @ Epoch 5)
```

---

## 6. ONNX Web Runtime & Browser Engine Architecture

The web application runs `best_model_A.onnx` directly inside the browser using **ONNX Runtime Web (`onnxruntime-web`)**.

The live camera contract keeps inference and presentation separate: raw,
unmirrored camera pixels are passed to the recognition pipelines, while only
the displayed preview is mirrored for natural interaction. Hand and pose
landmarks plus pixel bounding boxes are reflected horizontally before drawing
so overlays remain aligned with the mirrored preview. This preserves the
dataset rule that model inputs are never horizontally flipped.

The current Python prototype uses two **color** classifier views. The
training-matched center square is always evaluated. When MediaPipe finds one
or two hands, their combined square bounding box is stabilized using an
exponential moving average (`alpha=0.45`) and evaluated by the same ONNX
classifier. Class probabilities are fused as `0.60 × center + 0.40 × ROI`
before the existing per-label confidence/margin and temporal evidence filters.
This is deliberately a fallback-assisted design rather than an ROI-only
design: MediaPipe misses some valid poses, especially Hare and Boar.

After three consecutive frames without hand landmarks, non-hand predictions
are rejected as `no_hand_landmarks`. The first two missing frames are tolerated
to avoid flicker from a brief detector dropout. Boar and Hare remain eligible
for center-view recognition during absence because their detector recall is
weak; `zero` remains the normal neutral class. The smoothed box is discarded
after the third missing frame, so an old crop is never reused.

```
                  Raw Webcam Frame
                 /                 \
        Center Square        MediaPipe Hands
            RGB              Combined Box + EMA
             │                      │
             └── ONNX RGB 224² ─────┘
                        │
         60/40 Class-Probability Fusion
                        │
       Per-Class Thresholds + No-Hand Guard
                        │
             5-Frame Evidence Filter
                        │
              Debounced Seal Event
```

### Hybrid preprocessing validation (225-image held-out split)

| Variant | Top-1 accuracy across all images | Decision |
|:---|---:|:---|
| Center-square RGB only | 98.67% | Preserve as the reliable base view |
| Center RGB + color ROI fusion | **99.11%** | Adopt (`center_weight=0.60`) |
| Grayscale ROI | 82.67% | Reject; discards useful model input information |
| Immediate hard no-hand gate | 97.33% | Reject; detector misses valid hand signs |

MediaPipe found hands in 88.0% of the split overall, but only 36.8% of Hare
images and 66.7% of Boar images. Conditional on detection, the color ROI was
highly accurate; the main ROI failure mode was detector absence, not the ONNX
classifier. This evidence is why the implementation uses fusion and a delayed,
class-aware absence guard instead of replacing the current classifier input.

The implemented video-mode path was then run end to end on the same 225
images: the fused raw prediction and the post-threshold accepted output both
scored **223/225 (99.11%)**. Dog, Rat, Ram, Hare, and Boar were all 100% on
this split. The previous center-only accepted output also reached 223/225
because the Rat/Ram geometry resolver repaired its extra raw error; therefore,
the claimed benefit of fusion is a less heuristic-dependent raw prediction and
a stabilized live crop, not an inflated post-processing accuracy claim.

On an Apple M1 Max, 75 timed Python frames through the combined hand and pose
pipeline averaged 41.89 ms (51.51 ms p95); one additional ONNX crop inference
averaged 2.39 ms. The current Python prototype therefore does not yet satisfy
the browser blueprint's 25 ms target. Camera-mode profiling and scheduling are
still required before making a production real-time performance claim.

### Locked camera-tested baseline (2026-08-02)

The user completed a live camera check and approved this recognition behavior
as the baseline to preserve. The locked configuration includes:

- raw, unmirrored frames for all inference and a mirrored display only;
- center-square plus stabilized color hand-ROI probability fusion (`0.60/0.40`);
- EMA hand-box smoothing (`alpha=0.45`);
- a three-frame delayed no-hand guard with Boar and Hare fallbacks;
- per-label confidence and margin thresholds plus the Rat/Ram resolver;
- the five-frame hand evidence filter and one-event-per-held-sign behavior;
- adjacent duplicate suppression, a maximum three-seal queue, timeout clearing,
  one-noise-event combo recovery, and attack cooldowns;
- body movement recognition running independently from hand-seal attacks.

Treat these settings and behaviors as a regression baseline. Future work may
integrate downstream consumers or add explicitly requested features, but must
not retune or replace this recognition path unless the user explicitly reopens
recognition changes. Run `./run_combined_tracker.command` for the approved live
camera test and `./test_combined_pipeline.command` for regression validation.

### Versioned backend output contract (V1)

`cv_model/inference/output_schema.py` converts an internal `FrameResult` into
the stable, JSON-safe `PipelineOutputV1` contract. The adapter does not modify
recognition state or decisions. It only reads the completed frame result.

```json
{
  "schema_version": "1.0.0",
  "session_id": "camera-test-001",
  "frame_id": 42,
  "captured_at_ms": 123456,
  "processing_ms": 41.25,
  "hand": {
    "accepted_label": "rat",
    "stable_label": "rat",
    "emitted_seal": "rat",
    "fusion_status": "center_roi_fused"
  },
  "body": {
    "stable_label": "jumping",
    "emitted_movement": "jumping"
  },
  "queue": {"seals": [], "accepted_seal": "rat"},
  "attack": {
    "name": "shippu",
    "display_name": "SHIPPU / WIND",
    "recognized_at_ms": 123456
  }
}
```

The abbreviated example omits the required prediction, metric, queue-state,
and optional geometry fields for readability. The authoritative machine
contract is `cv_model/schemas/pipeline_output_v1.schema.json`.

- `PipelineOutputSerializer.to_dict()` returns plain JSON-compatible values.
- `PipelineOutputSerializer.to_json()` produces compact, deterministic JSON.
- `include_geometry=False` is the compact default. When enabled, normalized
  hand/pose landmarks and pixel bounding boxes are included.
- Missing ROI predictions, emitted events, geometry, and attacks are explicit
  `null` values rather than omitted fields.
- Non-finite numeric values and unexpected object fields are rejected.
- A major version change is required for breaking field changes. Minor versions
  may add backward-compatible fields; patch versions may correct validation or
  documentation without changing the contract.

The V1 frame serializer defines serialization only. It does not print, save,
or transmit outputs. Sparse in-process event dispatch is handled by the next
layer; external transport remains a future pipeline stage.

### Unified event dispatcher

`cv_model/inference/events.py` converts each validated `PipelineOutputV1` frame
into sparse external events. Event production is edge-triggered by the locked
pipeline result; it does not perform a second recognition decision.

| Event | Emission condition |
|:---|:---|
| `HAND_SEAL` | The attack queue accepts a newly emitted stable hand seal |
| `BODY_MOVEMENT` | The body evidence filter emits a movement transition |
| `ATTACK_TRIGGERED` | The hand-seal sequence matcher recognizes an attack |
| `QUEUE_CLEARED` | Timeout, maximum length, cooldown, or attack completion clears the queue |
| `PIPELINE_RESET` | Manual reset, calibration transition, or camera recovery resets state |

Frame events have deterministic order: hand seal, body movement, attack, then
queue clearing. Every event carries schema version `1.0.0`, a session-scoped
event ID and sequence, frame ID, capture timestamp, event type, and typed
payload. The machine contract is
`cv_model/schemas/pipeline_event_v1.schema.json`.

Subscribers may listen to all events or selected event types. Delivery is
synchronous to preserve order, but subscriber failures are isolated: one
consumer cannot stop recognition or prevent other consumers from receiving the
event. Failures are returned in a `DispatchReport` for explicit handling.

### Runtime pipeline controller

`cv_model/inference/runtime.py` owns the operational lifecycle around the
locked `CombinedNarutoPipeline`:

- lazy model creation and explicit `CREATED → RUNNING → CLOSED` lifecycle;
- session IDs plus monotonic frame IDs and capture timestamps;
- V1 serialization and sparse event dispatch for every processed frame;
- manual reset events and clean, idempotent shutdown;
- timed or explicit neutral calibration with event suppression while samples
  are collected, followed by a `calibration_complete` reset event;
- consecutive camera-failure counting and recognition-state recovery after the
  configured threshold (three failures by default);
- callback subscription and unsubscription through the event dispatcher.

The camera tester now runs frames and reset/calibration commands through this
controller. It still supplies the original raw frame to the same locked
recognizer and mirrors only the display. The runtime adds orchestration and
external outputs; it does not change thresholds, fusion, evidence, body rules,
attack sequences, or queue behavior.

No network, file, WebSocket, or frontend transport is implemented at this
stage. Those consumers can now attach to the stable event callback contract.

The controller/event integration was verified with the full automated suite
and a real model-backed smoke test. Three consecutive held-out Dog frames
produced the locked evidence transition followed by `HAND_SEAL`,
`ATTACK_TRIGGERED`, and `QUEUE_CLEARED` in the documented order.

### Latest-frame scheduler

`cv_model/inference/scheduler.py` separates camera capture from recognition
using one processing worker and a one-slot pending-frame mailbox. While a frame
is being processed, each newly captured frame replaces the older pending frame.
The worker therefore receives the freshest available input instead of working
through a stale FIFO backlog.

- MediaPipe and ONNX still run sequentially on one worker, preserving their
  state and the locked recognition order.
- Reset and calibration commands are serialized against frame processing.
- Reset/calibration commands quiesce the worker, then discard pending and
  completed pre-command data so an old image cannot cross the state boundary.
- Worker exceptions are returned as failed scheduled results without silently
  terminating the scheduler.
- Shutdown may discard or drain the final pending frame and waits for the
  active recognition call to finish safely.
- Scheduler statistics expose submitted, processed, dropped, failed, and
  superseded-result counts plus pending/processing state.

The camera tester now displays scheduled results using the exact raw frame that
produced each recognition output. Display mirroring remains presentation-only.
It prints final scheduler counts when the camera window closes.

### Detailed performance instrumentation

Timing is observational only and uses `time.perf_counter()`. It does not alter
confidence, thresholds, evidence, queueing, or attack recognition.

| Layer | Available measurements |
|:---|:---|
| Hand classifier | Center/ROI preprocessing, ONNX execution, and postprocessing |
| Hand detector | MediaPipe input preparation and Hand Landmarker execution |
| Body pipeline | Pose preparation, Pose Landmarker, geometry postprocessing, total |
| Recognition | Hand total, body total, attack queue, pipeline overhead, total |
| Runtime | Recognition wrapper, V1 serialization, event derivation, callbacks, overhead, total |
| Scheduler | Queue wait, worker elapsed time, capture-to-result end-to-end time |

Core phase values are stored in `FrameResult.timings_ms`; serialization and
callback values are added to `RuntimeFrame.timings_ms`; scheduler latency is
carried by `ScheduledResult`. The live panel shows hand, pose, runtime,
end-to-end latency, and cumulative dropped frames. The reproducible aggregate
benchmark command remains the next separate implementation step.

A real-model scheduler smoke check rapidly submitted ten held-out Dog frames.
The one-slot mailbox processed the newest frame and counted nine replacements
as dropped instead of building a backlog. That single processed frame measured
3.72 ms center ONNX, 2.50 ms ROI ONNX, 31.56 ms Hand Landmarker, 10.28 ms Pose
Landmarker, 0.06 ms serialization, and 49.25 ms total runtime. These are a
single-run wiring check, not an aggregate performance benchmark.

### ONNX Model Metadata
- **File Name:** `best_model_A.onnx`
- **File Size:** $5.83\text{ MB}$
- **Input Node Name:** `images`
- **Input Tensor Shape:** `[1, 3, 224, 224]` (Float32)
- **Output Node Name:** `output0`
- **Output Tensor Shape:** `[1, 13]` (Logits for 13 classes)

---

## 7. Jutsu Catalog, State Machine & Debounce Filter Specifications

### 7.1 Master Jutsu Catalog

| Jutsu Name | Element | Required Hand Seal Sequence | Special Timing & Behavior | Visual FX Animation |
|:---|:---:|:---|:---|:---|
| **Homura (火炎)** | Fire 🔥 | `tiger` $\rightarrow$ `dragon` $\rightarrow$ `horse` | Standard 3-seal combo (Max 3.0s window between seals) | Fireball Eruption & Radial Flame Burst |
| **Shippū (疾風)** | Wind 🌪️ | `bird` $\rightarrow$ `ram` $\rightarrow$ `rat` | Standard 3-seal combo | Swirling Tornado & Wind Blade Cutting Particles |
| **Ikazuchi (雷光)** | Lightning ⚡ | `dog` *(Single Seal)* | **Instant Action:** Single-seal attack trigger | Chidori Lightning Discharge & Electric Sparks |
| **Daichi (大地)** | Stone 🗿 | `monkey` $\rightarrow$ `boar` $\rightarrow$ `snake` | Standard 3-seal attack combo | Earth Wall Shatter & Ground Crag Barriers |
| **Ryūsui (流水)** | Water 🌊 | `ox` $\rightarrow$ `hare` | Fast 2-seal tactical combo | Water Vortex Ring & Expanding Wave Particles |

---

### 7.2 5-Frame Evidence Filter
To eliminate transient noise when a user transitions between hand positions, a **5-frame evidence window** buffers accepted predictions:

```
Frame t-4: [ TIGER ]
Frame t-3: [ TIGER ]
Frame t-2: [ TIGER ]  ──►  Evidence reaches 3 accepted observations
Frame t-1: [ TIGER ]        Stable label: TIGER
Frame t:   [ DRAGON ]       Held TIGER does not emit a duplicate event
```

- **Buffer Length:** 5 frames ($\approx 83\text{ ms}$ at 60 FPS)
- **Evidence Threshold:** 3 accepted observations by default; neutral calibration can raise a noisy class to 4.
- **Neutral Reset:** Two consecutive `zero` observations clear old positive evidence, so the next sign must earn fresh votes.
- **Confidence Cutoff:** Per-class confidence and margin thresholds are used instead of one global cutoff.

---

### 7.3 State Machine Transition Diagram

```mermaid
stateDiagram-v2
    [*] --> IDLE

    state "🔥 Homura (Fire Jutsu)" as Fire {
        TIGER --> DRAGON: Valid Seal & T < 3.0s
        DRAGON --> HORSE: Valid Seal & T < 3.0s
        HORSE --> CAST_FIRE: Trigger FX & Reset
    }

    state "🌪️ Shippū (Wind Jutsu)" as Wind {
        BIRD --> RAM: Valid Seal & T < 3.0s
        RAM --> RAT: Valid Seal & T < 3.0s
        RAT --> CAST_WIND: Trigger FX & Reset
    }

    state "⚡ Ikazuchi (Lightning Attack)" as Lightning {
        DOG --> CAST_LIGHTNING: Instant Trigger & Reset
    }

    state "🗿 Daichi (Stone Attack)" as Stone {
        MONKEY --> BOAR: Valid Seal & T < 3.0s
        BOAR --> SNAKE: Valid Seal & T < 3.0s
        SNAKE --> CAST_STONE: Trigger FX & Reset
    }

    state "🌊 Ryūsui (Water Jutsu)" as Water {
        OX --> HARE: Valid Seal & T < 3.0s
        HARE --> CAST_WATER: Trigger FX & Reset
    }

    IDLE --> TIGER: Detect Tiger
    IDLE --> BIRD: Detect Bird
    IDLE --> DOG: Detect Dog
    IDLE --> MONKEY: Detect Monkey
    IDLE --> OX: Detect Ox

    Fire --> IDLE: Timeout (T > 3.0s) / Three unmatched seals
    Wind --> IDLE: Timeout (T > 3.0s) / Three unmatched seals
    Stone --> IDLE: Timeout (T > 3.0s) / Three unmatched seals
    Water --> IDLE: Timeout (T > 3.0s) / Three unmatched seals

    CAST_FIRE --> IDLE
    CAST_WIND --> IDLE
    CAST_LIGHTNING --> IDLE
    CAST_STONE --> IDLE
    CAST_WATER --> IDLE
```

---

## 8. Repository Asset & File Directory Map

All trained weights, notebooks, scripts, and datasets are organized under `cv_model/`:

```
Naruto_Run/
├── context.md                             # 👈 THIS DOCUMENT (Master Architecture & Context)
├── implementation_plan.md                 # Original architecture breakdown & execution strategy
├── Pure Naruto Hand Sign Data/            # Original raw dataset (2,245 images)
│
└── cv_model/
    ├── inference/
    │   ├── combined_pipeline.py            # Locked hand, body, queue, and attack pipeline
    │   ├── pipeline_config.py              # Validated recognition configuration
    │   ├── output_schema.py                # Versioned JSON-safe PipelineOutputV1 adapter
    │   ├── events.py                       # Sparse typed events and isolated callbacks
    │   ├── runtime.py                      # Lifecycle, calibration, recovery, and dispatch
    │   └── scheduler.py                    # Latest-frame mailbox and processing worker
    ├── schemas/
    │   ├── pipeline_output_v1.schema.json  # Authoritative V1 frame contract
    │   └── pipeline_event_v1.schema.json   # Authoritative V1 event contract
    ├── tests/
    │   ├── test_combined_pipeline.py       # Recognition and state-machine regressions
    │   ├── test_display_mirroring.py       # Display-only reflection contract
    │   ├── test_output_schema.py           # V1 golden and validation tests
    │   ├── test_events_runtime.py          # Dispatcher and controller regressions
    │   └── test_scheduler_instrumentation.py # Frame dropping and timing regressions
    ├── data/
    │   ├── prepare_dataset.py             # Local dataset prep & augmentation script
    │   ├── prepared_dataset/              # 80/10/10 split dataset folder (train/val/test)
    │   └── prepared_dataset.zip          # 285 MB Kaggle-ready dataset archive
    │
    ├── models/
    │   ├── best_model_A.onnx              # ⚡ PRODUCTION MODEL: Trained YOLOv8n-cls (5.8 MB)
    │   ├── best_mlp.pth                   # PyTorch MLP weights (78 KB)
    │   ├── hand_landmarker.task           # MediaPipe 3D Hand Landmarker task file (7.8 MB)
    │   └── pretrained/
    │       ├── yolov8n-cls.pt             # PyTorch base YOLOv8 classification weights
    │       └── yolo26n.pt                 # PyTorch base YOLO weights
    │
    └── training/
        ├── V1_Model_training.ipynb        # 📓 Kaggle notebook with full execution logs (99.6% Acc)
        ├── build_notebook.py              # Notebook generator script
        ├── naruto_handsign_training.ipynb # Working training notebook template
        └── test_pipeline_local.py         # Local dry-run verification script
```

---

## 9. Detailed Technical Blueprint for Remaining Components (3 & 4)

### 9.1 Component 3: Frontend Architecture (`frontend/src/`)

```
frontend/
├── index.html                           # Main entry point & HUD layout
├── css/
│   └── styles.css                       # Anime dark mode UI design system
└── src/
    ├── config/
    │   ├── gestures.js                  # 13 class label definitions & mappings
    │   ├── jutsuCombos.js               # 5 Jutsu sequence definitions & timing limits
    │   └── constants.js                 # FPS targets, canvas dimensions, debounce rules
    │
    ├── utils/
    │   ├── imageProcessor.js            # Image crop, resize, and Float32 tensor conversion
    │   └── mathHelpers.js               # 3D vector angle & distance calculations
    │
    ├── pipelines/
    │   ├── handSignPipeline.js          # Pipeline 1: MediaPipe Hands + ONNX Runner
    │   ├── bodyGesturePipeline.js       # Pipeline 2: MediaPipe Pose Vector Math
    │   └── pipelineManager.js           # Multi-pipeline orchestrator
    │
    ├── engine/
    │   ├── debounceFilter.js            # 5-frame sliding window consensus filter
    │   ├── comboMatcher.js              # Jutsu sequence state machine
    │   └── soundEngine.js               # Web Audio API sound FX player
    │
    └── gfx/
        ├── particleSystem.js            # Particle emitter base class
        ├── fireFX.js                    # Flame & Fireball particle renderer
        ├── windFX.js                    # Tornado & Wind Blade particle renderer
        ├── lightningFX.js               # Chidori lightning spark renderer
        ├── earthFX.js                   # Earth wall & Crag fracture renderer
        └── waterFX.js                   # Water vortex & Splash wave renderer
```

---

### 9.2 Particle Rendering Specifications for Jutsu Visual FX

1. **Fire Jutsu (Homura):**
   - 300 active radial particles initialized at hand position.
   - Per-particle parameters: lifetime $0.8\text{ s}$, expansion velocity $v \sim \mathcal{N}(5, 2)\text{ px/frame}$, color gradient $\text{Yellow} \rightarrow \text{Orange} \rightarrow \text{Deep Red} \rightarrow \text{Smoke Grey}$.

2. **Wind Jutsu (Shippū):**
   - Swirling logarithmic spiral particle paths:
     $$r(\theta) = a e^{b \theta}, \quad \theta(t) = \theta_0 + \omega t$$
   - Particle shape: Tapered semi-transparent cyan/white wind blades.

3. **Lightning Dodge (Ikazuchi):**
   - Jagged polyline electrical discharge generator using Midpoint Displacement algorithm ($N=4$ recursions).
   - Random blue/violet glowing stroke with additive blending (`globalCompositeOperation = 'lighter'`).

4. **Stone Defense (Daichi):**
   - Polygonal rock shard particles rising from screen bottom with gravity acceleration $g = 0.5\text{ px/frame}^2$.
   - Screen shake effect (random camera offset $\Delta x, \Delta y \in [-10, 10]\text{ px}$ decaying over 0.5s).

5. **Water Jutsu (Ryūsui):**
   - Concentric expanding ring ripples with sine-wave alpha decay:
     $$\alpha(t) = \sin\left(\frac{\pi t}{T}\right) \cdot (1 - \frac{r}{R_{\max}})$$
   - Blue/teal fluid particles with metaball blending.

---

### 9.3 Verification & Quality Assurance Plan
- [ ] **Unit Testing:** Validate `comboMatcher.js` state machine transitions against synthetic hand seal sequence arrays.
- [ ] **Latency Benchmark:** Profile `ort.InferenceSession.run()` execution time across WebGL and WASM backends.
- [ ] **Cross-Browser Testing:** Verify webcam stream capture & WebGL particle rendering in Chrome, Edge, and Safari.
