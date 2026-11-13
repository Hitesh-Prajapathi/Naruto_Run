# Nishit_Frontend

Nishit's workspace for the game frontend build (`game_implementation_plan.md`
at the repo root). Everything for the browser game — the Python transport
bridge and, from Phase B onward, the Vite/TS/Three.js client — lives inside
this folder so it never collides with the rest of the team's work directly
under `cv_model/`.

**Hard rule this folder follows:** the locked recognition pipeline in
`cv_model/inference/` is never modified from here. This folder only carries
its output to a browser and, eventually, renders a game around it.

## Status

| Phase | What | Status |
|:---|:---|:---|
| **A** | Local Python↔browser WebSocket transport | ✅ Done, tested |
| **B** | Frontend scaffold + mock event source | ✅ Done, tested |
| **C** | Environment & map (Three.js) | ✅ Done, tested |
| **D** | Character & movement (real Naruto model + lane/jump controller) | ✅ Done, tested |
| **D+** | Visual & animation overhaul (`naruto_run_agent_brief.md`) | ✅ Done, tested, awaiting review |
| **E+** | Obito boss encounters + Game Over screen (`naruto_run_brief_02_...`) | ✅ Done, tested, awaiting review |
| **E++** | Obito's special attack + Hare counter (`naruto_run_brief_04_...`) | ✅ Done, tested, awaiting review |
| — | Brief 03 (combat stance, jutsu catalog, finish line) | ⚠️ Brief not in the repo — see below |
| — | Hand-seal CV wiring, audio, polish | Not started |

## Phase A — Transport bridge

`transport/` is a loopback HTTP + WebSocket server that wraps the existing
`PipelineRuntimeController` / `LatestFrameScheduler` from `cv_model.inference`
without changing anything about how they work:

- `GET /health` — process/model/runtime readiness.
- `GET /ws` — one binary-frame-in, JSON-events-and-snapshots-out WebSocket
  session. Only one controlling client at a time (see
  `game_implementation_plan.md` §9.1 "Keep one recognition producer").
- Frames come in as a small fixed binary header (magic bytes, version,
  sequence, capture timestamp, width, height, length) followed by a JPEG
  payload. See `transport/protocol.py`.
- State comes out as capped `state_snapshot` messages (15 Hz ceiling) and
  `pipeline_event` messages delivered immediately, in dispatcher order.
- Supported controls: `reset`, `begin_calibration`, `finish_calibration`,
  `cancel_calibration`, `ping` — every one gets a correlated `ack`.
- A malformed frame, oversized payload, stale/duplicate sequence, or unknown
  control is rejected with a typed `error` message; it never reaches
  recognition.
- A second client connecting while one already controls the session is
  rejected immediately (`controller_already_connected`) rather than mixed in.
- On disconnect the runtime is reset, so temporal/attack evidence can never
  cross into whatever client reconnects next.

Machine-readable message contracts: `schemas/transport_message_v1.schema.json`
(every JSON message) and `schemas/transport_control_v1.schema.json` (the
control-command subset).

### Gameplay-design review (done before writing any code)

Per your instruction, I re-reviewed the two CV-driven gameplay risks flagged
in `game_implementation_plan.md` §1.1 before starting:

1. **`naruto_run` masks bending.** The body classifier is a strict priority
   chain (`jumping > naruto_run > bending_right > bending_left > idle`), so
   while the pose reads as naruto-run, a bend is never reported — no lane
   steering is possible during it. **No design change needed**: the plan
   already treats `naruto_run` as an optional, non-mandatory sprint state
   rather than the default running pose, so lanes/jump stay driven purely by
   `bending_left` / `bending_right` / `jumping` the whole time. This just
   needs to be respected in Phase E, not solved architecturally.
2. **Cooldown suppression / latency honesty.** Already resolved by design
   (live seal-queue HUD + cooldown ring, ≥1.5 s obstacle/telegraph windows).
   No change needed here either.

Conclusion: the existing plan already accounts for both, so nothing was
changed. This transport layer does not alter or reinterpret any recognition
output — it forwards `PipelineOutputV1`/`PipelineEventV1` unchanged.

## Setup

A dedicated virtual environment lives at `venv/` inside this folder (already
created and populated; `venv/` is covered by the repo's root `.gitignore`, so
it is never committed). To recreate it from scratch:

```powershell
# from cv_model/Nishit_Frontend
"C:\Program Files\Python313\python.exe" -m venv venv
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install -r requirements-transport.txt
```

## Running the tests

From the repository root:

```powershell
.\cv_model\Nishit_Frontend\venv\Scripts\python.exe -m unittest discover -s cv_model\Nishit_Frontend\tests -v
```

52/52 passing as of this Phase-A delivery:

- `test_transport_protocol.py` — binary envelope encode/decode, every
  validation edge case, JSON control-plane message builders/parsers.
- `test_transport_backpressure.py` — frame sequence guard, bounded event
  queue overflow behavior, latest-snapshot-only slot.
- `test_transport_integration.py` — the real aiohttp server end to end
  (handshake, second-client rejection, valid/malformed/oversized/stale
  frames, control acks, `HAND_SEAL`/`ATTACK_TRIGGERED` event delivery) using
  a fake in-memory pipeline — no camera or model weights required.

## Running the server

```powershell
.\cv_model\Nishit_Frontend\venv\Scripts\python.exe serve_transport.py
```

Binds to `127.0.0.1:8765` by default (loopback only, per the step-6 contract).
`--port`, `--host`, `--allow-origin` (repeatable), and `--log-level` are
available — run with `--help` for details. This starts the **real**
recognition pipeline (MediaPipe + ONNX Runtime), so first startup takes a
few seconds while the models load.

There is no frontend yet to connect to it (that's Phase B). A minimal
diagnostic client that connects, sends synthetic frames, and prints received
events is a natural first Phase-B smoke test.

## What's deliberately not here yet (Phase A)

- No real end-to-end browser test at the time Phase A shipped — needed Phase
  B's `websocketClient`, which now exists (see below).
- No CORS/origin allowlist enforced by default (`--allow-origin` is opt-in);
  fine for loopback-only local dev, revisit before anything binds off
  `127.0.0.1`.

---

## Phase B — Frontend scaffold + mock event source

`frontend/` is a Vite + TypeScript app (no framework, per
`game_implementation_plan.md`'s Phase 7 stack decision — DOM/CSS/Canvas only;
Three.js arrives with Phase C). It proves the full plumbing works end to end
*before* Phase C starts building the actual game on top of it.

```
frontend/src/
├── main.ts                  # wires everything below together
├── config.ts                # transport URL, mock toggle, capture settings
├── transport/
│   ├── protocol.ts          # TS mirror of transport/protocol.py, byte-for-byte
│   ├── eventSource.ts        # the shared contract both sources implement
│   ├── websocketClient.ts   # real client: handshake, heartbeat, backoff
│   ├── mockEventSource.ts    # replays a scripted tape at the same interface
│   ├── demoTape.ts          # small illustrative tape for local dev/demo
│   └── eventDeduplicator.ts  # at-most-once event_id consumption
├── state/pipelineStore.ts    # latest snapshot + de-duplicated recent events
├── camera/
│   ├── cameraController.ts  # idle/requesting/ready/denied/ended/error
│   └── framePublisher.ts    # unmirrored capture -> JPEG -> binary frame
└── ui/
    ├── cameraPreview.ts      # display-only CSS-mirrored preview
    └── diagnosticsHud.ts     # renders whatever pipelineStore currently holds
```

### The Phase-B gate

> "mockEventSource and websocketClient are interchangeable at the
> pipelineStore boundary. Diagnostics HUD shows live state from either."

Both implement the exact same `PipelineEventSource` interface
(`transport/eventSource.ts`). `PipelineStore` and `DiagnosticsHud` are built
only against that interface — neither has any idea which one is wired up.
`tests/unit/pipelineStore.test.ts` has a test that drives the *same* tape
through a hand-built fake source and through the real `MockEventSource`, and
asserts the resulting store state is equivalent. `main.ts` picks the source
with one `?mock=1` query-flag check; nothing else changes.

### Verified

- **62/62 Vitest unit tests pass** across protocol, dedup, mock source,
  WebSocket client (fake-`WebSocket`-driven: handshake, heartbeat, doubling
  backoff, handshake timeout, malformed-JSON handling), pipeline store
  (including the interchangeability gate), camera controller (all six
  states, fake `getUserMedia`), and frame publisher (drop-stale-tick
  backpressure, verified with a gated/deferred encode promise).
- **Cross-language wire compatibility**: `tests/unit/protocol.test.ts`
  encodes a known `FrameEnvelope` in TypeScript and asserts the output is
  byte-identical to the real Python `encode_frame()`'s output for the same
  envelope (golden hex fixture, generated once from the actual transport
  venv). Decoding the Python-produced bytes back in TS is also checked.
- `npm run typecheck` (strict, `exactOptionalPropertyTypes`) and
  `npm run build` both pass clean.
- Dev server boots (`npm run dev`, `127.0.0.1:5173`) and serves the page with
  `?mock=1` active.
- `npm audit`: 0 vulnerabilities (forced Vite 8 / Vitest 4 to clear a dev
  -server-only advisory chain — noted here since it was a version bump beyond
  what was originally pinned).

### Bugs this phase's tests caught (fixed, not just noted)

- `pipelineStore.ts` only updated `sessionId` from `state_snapshot` messages,
  never from `pipeline_event` messages. An event-only stream (or an event
  arriving before its first snapshot) left `sessionId` stuck at `null`
  forever, which silently broke the reconnect/session-boundary reset logic.
  Fixed: both handlers now update it.
- `framePublisher.ts` called `Blob.arrayBuffer()` directly, hard-coupling it
  to a fully spec-compliant `Blob` (jsdom's doesn't implement that method).
  Fixed: `CaptureSurface.encodeJpeg()` now returns a minimal `BlobLike`
  (`{ arrayBuffer(): Promise<ArrayBuffer> }`), which real browser `Blob`s
  satisfy structurally and tests can fake trivially.

### Running it

```powershell
cd cv_model\Nishit_Frontend\frontend
npm install
npm run typecheck   # tsc --noEmit
npm run test        # vitest run (62 tests)
npm run build        # typecheck + production build
npm run dev          # http://127.0.0.1:5173 -- add ?mock=1 to skip the backend
```

With `?mock=1`, the page runs entirely off `demoTape.ts` (one hand seal, one
Fire Attack, one jump) with 150 ms±40 ms injected latency — no camera, no
Python process, no network required. Without it, `main.ts` connects a real
`WebSocketClient` to `ws://127.0.0.1:8765/ws`, i.e. Phase A's
`serve_transport.py` needs to be running.

### What's deliberately not here yet (Phase B)

- No game/scene/effects code — that's Phases C (environment/Three.js), D
  (character), F (battle), G (effects). `src/game/`, `src/scene/`,
  `src/effects/` don't exist yet on purpose.
- No `config/gameConfig.ts` / `config/levelDefinition.ts` — nothing to
  configure without a level yet; `src/config.ts` covers Phase B's own
  settings only.
- No Playwright browsers installed/run yet. `@playwright/test` is a
  devDependency and `npm run e2e` is wired up, but `npx playwright install`
  hasn't been run in this environment — the 62 Vitest unit tests (including
  a fake-`getUserMedia`-driven camera-permission suite) are this phase's
  verification. A `tests/e2e/` Playwright suite with a real fake-media-stream
  browser run is reasonable to add either now on request or as part of a
  later phase's gate.

  **Update (Phase C):** this Playwright suite now exists — see below. It was
  added because Phase C needed real WebGL verification jsdom can't provide.

---

## Phase C — Environment & map (Three.js)

`frontend/scene.html` is a second, independent dev-harness page (separate
from Phase B's `index.html`) that proves the environment/map systems work
together: a scrolling jungle track, pooled obstacle spawning driven by
`levelDefinition.ts`, AABB collision detection, and a keyboard-driven
placeholder cube standing in for the not-yet-built real player character.

```
frontend/
├── scene.html                        # Phase C dev harness page
└── src/
    ├── sceneMain.ts                  # wires it all together
    ├── config/
    │   ├── gameConfig.ts             # lanes, speed, camera, obstacle catalog
    │   └── levelDefinition.ts        # Segment 1's authored obstacle script
    ├── scene/
    │   ├── sceneRoot.ts              # renderer/camera/render loop
    │   ├── skyAndLighting.ts         # sun + hemisphere + fog
    │   ├── jungleTrack.ts            # recycled ground tiles
    │   ├── environmentProps.ts       # trees/rocks, parented to track tiles
    │   ├── objectPool.ts             # generic acquire/release pool
    │   └── trackRecycling.ts         # pure tile-recycling math
    ├── game/
    │   ├── obstacleSpawner.ts        # pooled, level-driven obstacle meshes
    │   ├── obstacleSchedule.ts       # pure spawn-timeline bookkeeping
    │   ├── collisionSystem.ts        # AABB overlap, forgiving player hitbox
    │   └── levelLinter.ts            # validates a level against §1.1 rules
    └── dev/
        ├── debugPlayerRig.ts         # TEMPORARY keyboard cube (Phase-C only)
        └── sceneDebugHud.ts          # FPS/pool/collision on-screen readout
```

### Deliberate scope boundary: `dev/debugPlayerRig.ts` is not `laneController.ts`

The plan's real player-movement controller (frame-rate-independent, jump
coyote-time forgiveness, wired to the actual character) is Phase D's job.
`debugPlayerRig.ts` exists only to prove Phase C's own systems — track
scrolling, obstacle spawning, collision — with *something* keyboard-driven
standing in for the player. It's a plain box, lane movement is a simple
eased lerp, and the jump is a fixed sine arc. Expect this file to be deleted
once Phase D lands.

### The Phase-C gate

> "track scrolls at 60 FPS with a keyboard-driven placeholder cube; no
> allocation spikes over a 3-minute run; obstacle patterns validated against
> the §1.1 spacing rules by an automated level linter."

- **60 FPS / renders correctly:** verified live in a real Chromium browser
  via Playwright (`tests/e2e/scene.spec.ts`) — jsdom has no WebGL context, so
  this genuinely could not be proven by the unit suite alone. A screenshot
  is captured for visual review on every run.
- **No allocation spikes over a long run:** `ObjectPool` never calls its
  `factory` beyond initial warmup once traffic reaches steady state;
  `ObstacleSpawner.reset()` restarts a segment's schedule without
  reconstructing any pool, material, or geometry. A unit test simulates a
  3-minute looped soak (`obstacleSpawner.test.ts`) asserting
  `totalConstructed` goes flat after the first lap, and the Playwright suite
  independently confirms the same thing in a **real browser over a real
  ~32-second run** (sampling `obstacles constructed` at two checkpoints 15s
  apart and asserting they're equal).
- **Level linter:** `levelLinter.ts` checks every consecutive obstacle pair
  against the two named rules from `game_implementation_plan.md` §1.1 (no
  3-lane blocker; ≥1.2s between a jump-type and lane-type obstacle) plus a
  general ≥1.6s minimum gap, structural checks (lane range, duplicates,
  negative time), and asserts the real authored `SEGMENT_1` has zero issues.
  `sceneMain.ts` calls `assertLevelSegmentValid(SEGMENT_1)` at startup so an
  invalid pattern fails loudly in dev rather than silently shipping.

### Two real bugs this phase's tests (and a screenshot) caught

The first Playwright screenshot showed the player and trees floating over
bare sky — no ground visible at all. Both root causes were genuine bugs, not
flaky rendering:

1. **A structural gap in the tile-recycling design.** Ground tiles reused
   `OBSTACLE_DESPAWN_Z` (6 units past the player) as their own recycle
   threshold with 40-unit tiles. Recycling happens one tile at a time, so
   right after any single recycle event the belt's nearest edge is
   temporarily a full tile-length short of that threshold — a "sawtooth" gap
   reaching back to -34, i.e. no ground under large stretches of the track,
   including whenever the gap swallowed the region right under the player
   and camera. Fixed with a ground-specific `GROUND_RECYCLE_THRESHOLD_Z`
   (20, comfortably past `CAMERA_POSITION.z`) and much finer tiles (10 units,
   12 of them) so the worst-case sawtooth dip is only 10 units and still
   lands past the camera. Caught by a new regression test
   (`jungleTrack.test.ts`: "never leaves the camera with no ground beneath
   it, at any point across a long run") that checks the belt's nearest edge
   on *every* simulated frame over 30 simulated seconds, not just sampled
   instants — the bug was invisible to spot-checks because it only appeared
   during part of each recycle cycle.
2. **The same fix's initial construction didn't start in the covered
   state.** Tiles were constructed with the frontmost tile's near edge at
   `z=0` (the player's position), but the camera sits further back at `z=8`
   — so even a perfectly gapless belt had a permanent gap between the last
   bit of ground and the camera until the belt organically scrolled forward
   enough to catch up. Fixed by starting tile positions at the recycle
   threshold (the belt's maximum extent) instead of at 0, so the invariant
   holds from the very first rendered frame.

Both are why the Playwright screenshot step matters: 132 unit tests were all
green throughout, because the pure recycling math (spacing, no
overlaps/gaps *between* tiles) was correct the whole time — the bug was in
where that correct-relative-spacing belt sat in *absolute* space relative to
the camera, which only a real render surfaces.

### Verified

- **132/132 Vitest unit tests** (up from 62 after Phase B), including:
  pure pooling/recycling/collision/linter/scheduling logic with no Three.js
  dependency, plus real `THREE.Scene`/`Mesh`/`Object3D` graph tests (Three's
  core scene-graph classes are plain JS with no WebGL requirement, so they
  run fine under jsdom — only `THREE.WebGLRenderer` needs a real GPU/canvas).
- **5/5 Playwright e2e tests** against a real Chromium browser: renders
  without console errors, keyboard lane-change and jump reach the debug rig
  and update the HUD, and the allocation-gate check holds over a real
  ~32-second run. Confirmed stable across repeated runs.
- `npm run typecheck` and `npm run build` (now producing two pages,
  `index.html` and `scene.html`) both pass clean.
- Playwright's Chromium build was installed (`npx playwright install
  chromium`) to make the above possible; `@playwright/test` was already a
  Phase-B devDependency.

### Running it

```powershell
cd cv_model\Nishit_Frontend\frontend
npm run test                          # 132 unit tests
npm run typecheck && npm run build    # both index.html and scene.html
npm run dev                           # http://127.0.0.1:5173/scene.html
npx playwright test tests/e2e/scene.spec.ts   # real-browser verification
```

A/D or ←/→ to change lanes, Space/W/↑ to jump. The on-screen HUD shows FPS,
current lane, airborne state, active/constructed obstacle counts, collision
count, and segment progress (Segment 1 loops automatically for soak testing).

### What's deliberately not here yet (Phase C)

- No real player character, animation, or `laneController.ts` — Phase D.
- No enemy, battle, HP, or damage model — Phase F. Collisions are detected
  and counted (`collisions` in the HUD) but nothing consumes them yet.
- No Segment 2/3 or battle markers in `levelDefinition.ts` — Phase H content,
  once there's a battle system to script against.
- No attack effects, particles, or audio — Phase G.
- The production `scene` JS chunk is ~533 KB (~135 KB gzipped), mostly
  Three.js itself; Vite's build warns about this. Not addressed now —
  code-splitting/tree-shaking is a reasonable Phase H polish item, not a
  Phase C blocker.
- `dev/debugPlayerRig.ts` and `dev/sceneDebugHud.ts` are explicitly
  throwaway; expect them to be deleted once Phase D's real character and the
  eventual merged game HUD exist.

  **Update (Phase D):** `debugPlayerRig.ts` has been deleted, exactly as
  predicted above — see below. `sceneDebugHud.ts` is still in use.

### Update (post-Phase D) — real jungle environment assets

After seeing the Phase-D screenshot, the user asked for the environment to
use a real asset they provided: `Character_3D_Models/Forest_Road/` (a
"Dirt Road Forest Scene" pack). Before wiring anything in, the actual road
mesh was extracted and rendered from a top-down angle to see its real shape
— **it's a winding, curved path**, not a straight lane, and cannot be sliced
into repeating tiles for our fixed 3-lane mechanic without rearchitecting
lanes/camera/collision to be curve-relative. This was surfaced to the user
with three options (rearchitect to follow the curve; use the whole pack as a
static backdrop only; or keep straight lanes and harvest just the pack's
textures and small props). **The user chose to harvest textures + props**,
keeping the straight recycling-lane mechanic intact.

What changed:

- `jungleTrack.ts`'s ground material is now the pack's real `Dirt_Road`
  diffuse/normal/roughness textures (tiled via `RepeatWrapping`), replacing
  the flat green color.
- `scene/forestProps.ts` (new) places four of the pack's real prop clusters
  — a fallen-log group, a mud pile, a broken-rock cluster, and a fence run —
  as one-time set-dressing near the start of the track.

Two things worth knowing about how these props behave:

1. **Each extracted "prop" is itself a merged multi-instance cluster.** The
   source scene was exported to OBJ with one mesh per *material*, not per
   instance — e.g. the "Wood_Log" object is many logs already baked into one
   ~100-unit-wide mesh, with no per-instance placement data recoverable from
   OBJ. Duplicating a whole cluster repeatedly across the recycled tile belt
   would be excessive, so these are placed once, scaled down to track scale,
   and are **not** part of the recycling pool — they will not reappear on
   later loops. An explicit scope decision, documented in
   `forestProps.ts`'s own doc comment, not an oversight.
2. **A real bug caught before it shipped:** the first version placed these
   props at a fixed world Z forever. Every other moving thing in the scene
   (track tiles, obstacles) simulates forward motion by shifting position
   every frame — a static object that never gets that treatment just sits
   there while the ground rushes past it, which would have looked broken the
   moment anyone watched for more than an instant (a single screenshot
   didn't reveal it). Fixed by adding `updateForestProps()`, called from the
   same per-frame loop that already moves the track and obstacles.

New assets, copied and renamed for clean paths (originals untouched):
`public/textures/forest-road/` (~11 MB: three texture sets — Dirt_Road,
Wall_Mud, Broken_Rocks — plus two single diffuse maps) and
`public/models/forest-props/` (~584 KB: four small extracted OBJ subsets).

Verified: **149/149 unit tests** (146 + 3 new for `updateForestProps`), all
5 Playwright e2e tests still pass (two jump-test timeouts were widened —
see below), and `npm run build` still succeeds. FPS in the harness dropped
from ~50s to ~15 under Playwright's 5-parallel-worker CPU contention with
the new ~11 MB of textures loading; this pushed a latent timing assumption
in `tests/e2e/scene.spec.ts`'s jump test past its old 2–3 s poll window
(sceneRoot.ts clamps each frame's delta to at most 1/15 s, so heavy
real-world contention makes the *in-game* clock run in slow motion relative
to wall-clock time) — fixed by widening those polls to 8–10 s with a comment
explaining why, not by changing any gameplay timing.

**Not addressed in this pass, worth knowing:**
- Texture files are used at their original resolution (~11 MB total); no
  image compression/resizing tool was available in this environment to
  shrink them. Fine for local dev; worth revisiting before any real
  deployment.
- The props' visual fit (scale/silhouette against the track) is a first
  pass, not polished — they read as "real assets," not yet "carefully
  art-directed."
- The pack's actual curved road, its tree/grass/cliff meshes, and its FBX
  variant remain unused, per the scope decision above.

---

## Phase D — Character & movement

Replaces the Phase-C placeholder cube with the real Naruto model you
provided (`cv_model/Character_3D_Models/naruto-sage/`, copied into
`frontend/public/models/naruto/`) and the real lane/jump controller.
`dev/debugPlayerRig.ts` is deleted, exactly as Phase C's own doc comment said
it would be.

```
frontend/
├── public/models/naruto/         # copied from Character_3D_Models/naruto-sage
│   ├── Naruto.dae                # rigged mesh + skeleton (no baked animations)
│   ├── Naruto.mtl, Naruto.png, nr2_tex01/02.png, eyes/*.png
└── src/
    ├── config/gameConfig.ts      # + JUMP_DURATION_S/JUMP_HEIGHT/JUMP_INPUT_BUFFER_S
    ├── game/
    │   ├── laneController.ts     # real lane/jump controller (replaces debugPlayerRig)
    │   └── characterPose.ts      # pure procedural leg-gait math
    └── scene/
        ├── characterRig.ts       # bone-name mapping for the Collada skeleton
        └── playerCharacter.ts    # loads the model, scales/orients it, animates it
```

### What's in the model file

`Naruto.dae` is a Wii game-asset rip: a 132-bone rigged humanoid skeleton
(full finger/hair bones included) shared identically across 3 skinned
sub-meshes (body, gear detail, eyes) — confirmed by loading it with
`ColladaLoader` and inspecting the parsed result directly, not by assuming
anything about the file. Two facts that shaped everything below:

- **No baked animation clips.** The file has a skeleton and a bind pose, but
  no `<library_animations>`. There is nothing to play — any run/jump/idle
  motion has to be computed, bone by bone, in code.
- **The bind pose is a raw T-pose**, not the crossed-arms stance from your
  reference photos. That crossed-arms look was evidently a separate
  animation applied in the original game, not something this file carries.

### The Phase-D gate

> "lane changes and jumps are frame-rate independent and identical at
> 30/60/144 Hz."

`laneController.ts` is a direct, tested port of Phase C's `debugPlayerRig`
timing model (time-fraction-based lane easing, sine jump arc) plus a real
addition: buffered jump input (`JUMP_INPUT_BUFFER_S` = 150 ms) so a jump
pressed slightly early while still airborne fires the instant the character
lands instead of being dropped — this is genuinely useful once Phase E adds
the CV pipeline's own ~150–200 ms latency on top of a player's reaction time.
`tests/unit/laneController.test.ts` proves frame-rate independence
numerically (a 1-step update vs. a 120-step update over the same duration
land on the same X/height, within floating-point tolerance) for both lane
changes and jumps, plus 6 tests specifically on the buffering behavior.

### A real bug the buffering tests caught before it shipped

The first `requestJump()`-while-airborne implementation only started aging
the buffered request *after landing*, not from the moment of the request —
so a jump buffered near the *start* of a long jump arc would, once it
finally landed, still be checked against a buffer age of effectively zero
and fire regardless of how stale it actually was. Caught while writing "a
buffered jump request older than `JUMP_INPUT_BUFFER_S` expires unconsumed"
(it didn't, on the first implementation) — fixed by aging the buffer from
the request itself every tick, independent of airborne/grounded state.

### Orienting and animating a rig with no baked clips

Two decisions came from actually loading the model and looking at it, not
from assumptions:

1. **Up-axis and facing.** The source rig is authored Z-up (confirmed by
   `Box3.setFromObject` on the parsed scene: a ~1.72-unit Z-extent matching a
   real human height, feet at `z≈0`). A single `-90°` rotation about world X
   converts that to Three.js's Y-up convention. Getting the *facing*
   direction right took two attempts: a clean T-pose screenshot (no
   animation active, so front/back is unambiguous) showed the face/headband
   pointing at the camera with only the up-axis fix applied, so an
   additional 180° world-space yaw was added on top. An earlier attempt at
   this same fix was tried, then reverted based on a side-view screenshot
   that seemed to contradict it — that read turned out to be unreliable
   because arm-posing deltas were still active in that shot, and arm
   direction is not a trustworthy facing cue. The clean T-pose render was
   the signal that was actually trusted in the end.
2. **Leg animation only, for now.** `characterPose.ts` computes a run-cycle
   (thigh swing + knee bend, hips and knees hinging on the world-X axis —
   verified visually, not just assumed) and a jump tuck, applied via
   `Object3D.rotateOnWorldAxis` rather than by setting local Euler angles.
   That choice matters here specifically: this rig's per-bone local axis
   conventions don't reduce to any simple rule once you trace the actual
   parent chain above `hip` (several intermediate helper nodes), so
   reasoning about a bone's *local* axis by hand is unreliable, while a
   *world*-space axis means the same thing regardless. Getting the arms from
   "straight out to the side" (T-pose) to a natural hang/swing turned out to
   need more than the equivalent single-axis rotation: a side-view check
   showed the bind-pose arm direction has a real forward (Z) component once
   the rig's full parent chain is composed, not just the sideways (X)
   component a clean T-pose would suggest, so the first attempt left the
   arms reaching in an unnatural direction instead of hanging down. Rather
   than keep guessing at arm axes, arms are left at the bind T-pose for now
   — correctly scaled, oriented, and grounded, with working leg animation,
   is a solid state to hand back for review; natural arm motion is flagged
   below as a follow-up.

### Verified

- **146/146 Vitest unit tests** (up from 132 after Phase C): pure gait math
  (`characterPose.test.ts`, including periodicity, symmetry, and "knee bend
  never negative"), the lane/jump controller including 6 buffering-specific
  tests, and everything from Phase C unchanged.
- **5/5 Playwright e2e tests** against a real Chromium browser, including the
  same 3-minute-soak allocation gate from Phase C (now with the real
  character loaded and animating alongside the obstacle pooling).
- `npm run typecheck` and `npm run build` both pass clean; the production
  `scene` bundle is ~623 KB (~160 KB gzipped, up from Phase C's 533 KB —
  `ColladaLoader` accounts for the difference). Model assets themselves
  (~658 KB: one `.dae`, one diffuse texture set) are served as static files,
  not bundled into JS.
- Visually confirmed with real Playwright screenshots at each step of the
  orientation fix, not just claimed: the final state clearly shows the
  correct back-view (spiky hair, headband ties, gear pack, no face) facing
  down the track, standing correctly on the now-visible ground.

### Running it

```powershell
cd cv_model\Nishit_Frontend\frontend
npm run test                          # 146 unit tests
npm run typecheck && npm run build    # both index.html and scene.html
npm run dev                           # http://127.0.0.1:5173/scene.html
npx playwright test tests/e2e/scene.spec.ts   # real-browser verification
```

Same controls as Phase C (A/D or ←/→ to change lanes, Space/W/↑ to jump); the
HUD is unchanged. A "Loading Naruto…" overlay covers the canvas for the brief
moment the model takes to load (local file, well under a second).

### What's deliberately not here yet (Phase D)

- **Arms stay at the T-pose bind pose.** Flagged above — this is the honest
  open item from this phase, not an oversight. Options for a follow-up pass:
  more iteration on the world-axis rotation needed to bring the arms down
  naturally (now that the same eyes-on-a-real-screenshot method has resolved
  two other axis questions, it should converge faster a third time), or if
  you can find/export a version of this rig with baked animation clips
  (idle/run/jump), swapping to `AnimationMixer`-driven playback would sidestep
  procedural posing for the arms entirely.
- No naruto_run speed boost, CV input, or real gesture control — Phase E.
- No enemy, battle, HP, or damage model — Phase F. Collisions are still just
  detected and counted, not consumed.
- No attack effects, particles, or audio — Phase G.
- The reference photos' crossed-arms idle stance was not reproduced exactly;
  see the arms note above.

---

## Visual & animation overhaul (`naruto_run_agent_brief.md`)

A review brief graded the Phase-D build "dev harness, visually unfinished"
and listed blockers P0.1-P2.6. This pass works through them. Stack confirmed
as the brief assumed: three.js in the browser, DOM HUD overlay.

### Asset inventory (brief §1)

| Folder | Contents | Used? |
|:---|:---|:---|
| `naruto-sage` | Character mesh + 132-bone rig, **no animation clips** | ✅ the player character |
| `Running_Style` | `run.fbx` — different character, 52-bone rig, **1 real run clip (0.56s)** | ✅ retargeted onto the character |
| `Forest_Road` | Photoreal forest scene (curved road) + texture library | ✅ textures/props only (see earlier section) |
| `Attack` | `Boule de feu` low-poly fireball + gradient texture | ⏳ unused — Phase G attack effects |
| `Enemy-Obito` | `tobi_v9.blend` + mask/suit textures | ⏳ unused — Phase F needs an enemy; **`.blend` is not loadable by three.js and must be exported to glTF/FBX first** |

### P0.1 — "the character is in T-pose; the animation does not exist"

Correct diagnosis, and the cause was the brief's own hypothesis #2: the
character model ships **zero** animation clips. The earlier build posed a few
leg bones procedurally each frame, which was too subtle to read and left the
arms at bind pose entirely.

The fix uses the real run cycle from `Running_Style/run.fbx`. That clip is
authored for a completely different skeleton, so it had to be retargeted, and
**two standard approaches were tried and measured before a third worked**:

1. **`SkeletonUtils.retargetClip`** (three's built-in) copies each source
   bone's world orientation onto the target. It assumes both rigs share
   bone-axis conventions. These don't — the result was a crumpled character,
   posed bounding box `0.87 x 0.99 x 1.28` (wider and deeper than tall) with
   feet pointing sideways.
2. **Rest-relative rotation transfer.** Got the character upright, but the
   legs swung *sideways* instead of fore/aft (measured Z-swing range only
   `-0.17 ... -0.01`) because the two rigs' bone rest frames differ by a
   twist about the limb axis.
3. **Direction-driven retargeting** (`scene/retargetRunClip.ts`) — what
   shipped. For each limb segment it reads the source's world-space
   bone-to-child direction and *aims* the target bone along it. Immune to
   bind-pose, bone-axis, and limb-length differences all at once. Result:
   upright (`up = [0, 0.98, 0.21]`), a real stride (Z-swing `-0.98 ... +0.24`),
   bounding box `1.65 x 1.75 x 0.54`.

Why that mattered: the source rig's **bind pose is itself malformed** — its
`hips->head` axis sits ~52° off vertical and its right leg points nearly
straight *backwards* (`hips->foot = [-0.12, -0.24, -0.96]`). Any method
defined relative to that bind pose inherits the mess, which is what sank
approaches 1 and 2.

Two further findings from measuring the source clip:

- Its **torso leans ~58° forward**, which read as doubled-over. Spine chains
  are damped to `0.1` weight, landing the lean at ~12° — the brief's target.
- It **barely animates the arms at all**: upper arms move only from
  `[0.28, 0.18, -0.94]` to `[0.38, 0.06, -0.92]` across the whole cycle, and
  both arms point the *same* way rather than counter-swinging. Transferring
  that would have reproduced the original "hands look stiff" complaint. Arms
  are therefore **synthesized** — hanging down, elbows bent, swinging fore/aft
  opposite their same-side leg, phase-locked to the leg cycle so they cannot
  drift out of sync.

**Cadence matching** (brief §A) is wired: `PlayerCharacter.setSpeed()` scales
the clip's `timeScale` in proportion to world scroll speed, so the feet don't
slide against the ground.

### A real bug this uncovered: the stale skinned bounding box

After the retarget landed, the character rendered **3.3x too large and lying
on its side**. Cause: `Box3.setFromObject` prefers a `SkinnedMesh`'s *cached*
`boundingBox`, and nothing invalidates that cache when the model is rotated.
Measuring right after the Z-up to Y-up fix returned the stale pre-rotation
box — height `0.534` (the model's original *depth*) instead of the true
`1.722`. Fixed by `measureBounds()`, which clears the cache before measuring.
Calling `updateMatrixWorld` first does **not** help — verified explicitly.

### Everything else in the brief

| Item | Status |
|:---|:---|
| P0.2 grounding | Feet pivot at y=0; soft contact-shadow blob that shrinks/fades with jump height; real shadow-mapped sun |
| P0.3 broken geometry | Eye sub-mesh given `alphaTest`, fixing the black bars across the face |
| P0.4 scale | Scale contract established: **1 unit = 1 metre**, character 1.75u, trees 6-14u, rocks 0.4-1.2u |
| P1.1 camera | New `scene/chaseCamera.ts`: above/behind, pitched ~12° down, exponential (frame-rate-independent) damping, **50% lateral follow** so lane changes stay visible |
| P1.2 environment | Real dirt-road textures, wide grass terrain, distance fog hiding the spawn seam, gradient sky dome, **4 distinct tree meshes** across 3 depth bands with randomised scale/rotation/tint, plus rocks and grass tufts |
| P1.3 lighting | Warm directional key with shadows + cool hemisphere fill, ACES tone mapping, shadow frustum tightened to the play area and kept centred on the player |
| P1.4 lane lean | Character banks up to 13° into a lane change and eases back to zero |
| P2.1 jump feel | Rebuilt as anticipation -> airborne -> landing, with crouch squash (feet stay planted), a fast-rise/slow-hang arc, and input buffering |
| P2.2 obstacles | Distinct silhouettes per hazard (lying log, boulder, sunken pit, tall barrier) with accent bands; grounded, not hovering |
| P2.3 collision feedback | Hit-stop (90ms), camera shake, red screen-edge flash, life loss, and 1.1s invulnerability — 4 of the brief's channels |
| P2.4 HUD | Player HUD (distance, speed, life pips, input status); dev panel kept but hidden behind **`H`** |

### Verified

- **168/168 unit tests**, including new suites for the chase camera
  (frame-rate-independent damping, partial lateral follow, and a **frustum
  check that at least 25 units of path really are visible**) and hit feedback
  (i-frames, hit-stop, life accounting).
- **8/8 Playwright e2e tests**, stable across repeat runs, including a new
  regression test that the **animation's playback head actually advances** —
  the P0.1 failure was silent, so it needs an explicit assertion.
- `npm run typecheck` and `npm run build` clean.
- Playwright parallelism dropped from 8 workers to 2: these are GPU-bound
  WebGL tests sharing one headless GPU, and over-parallelising made every
  test slower *and* pushed model loading past its timeouts. The suite now
  runs both faster (~49s vs ~56s) and reliably.

### Known gaps and judgement calls

- **Scale conflict, resolved toward the brief.** An earlier instruction was to
  shrink the character to 0.8u; the brief instead sets 1u = 1m with a 1.7-1.8u
  character and asks for a character-to-tree *ratio* of 1/3 to 1/4. Both
  address "the character is too big for the world"; the brief's version keeps
  a physically meaningful unit scale to author future assets against, so the
  trees grew rather than the character shrinking. Easy to flip if preferred.
- **Arm swing is synthesized, not authored.** Correct and readable, but
  parametric — it will look identical every cycle. A real arm-swinging run
  clip would be better if one turns up.
- **No audio, no speed lines/motion blur, no distinct stumble animation** —
  brief §I and §P2.6, left for a later pass.
- **Limb twist is not transferred** by the direction-based retarget (forearm
  roll is lost). Not perceptible from behind at run speed; a real limitation
  of the technique rather than a bug.
- The `scene` bundle is now ~718 KB (~189 KB gzipped), mostly three.js plus
  the FBX and Collada loaders.

---

## Obito boss + Game Over (`naruto_run_brief_02_boss_and_gameover.md`)

Strictly additive per brief §0: every new system lives in new files behind
`src/game/bossSystem.ts`, and both features sit behind flags in
`src/config/bossConfig.ts` (`ENABLE_BOSS`, `ENABLE_GAME_OVER`).

### Asset inventory (brief §1) — one blocking finding

The folder is `Character_3D_Models` (with the `s`).

| Asset | What it actually contains |
|:---|:---|
| `Enemy-Obito/tobi_v9.fbx` (24 MB) | **0 animation clips, 0 bones — completely un-rigged.** 15 static meshes, ~157k tris, 13 materials |
| `Attack/Boule de feu…fbx` | 2 static meshes (~2.9k tris), no animation, orange gradient texture |

This is the case brief §1 said to flag, and it is stronger than "no attack or
dodge clips": with no skeleton, nothing can be posed, and unlike Naruto
(132-bone rig) a clip cannot be retargeted onto him either.

**Owner decision: animate Obito by whole-object transform.** So his "idle"
is a breathing bob and sway, his dodge is a banked slide, his attack is a
lunge-and-recoil, his hit is a knockback plus red flash, and his defeat is a
topple plus dissolve. At duel distance, on a masked and cloaked character,
that reads acceptably. It would not survive a close-up.

### Spec conflicts (brief §2, §10) — resolved with the owner

1. **§2.1 damage contradiction.** 60/40/20 against 100 HP kills Obito on the
   *second* hit. Owner chose the brief's alternative: **Obito has 120 HP**
   and takes 60/40/20, so the bar empties exactly on the third blow. Naruto
   keeps 100 HP and takes 60/20/20. Both fall on hit three. Pinned by tests.
2. **§2.2 obstacle lives vs combat HP** — kept fully separate, as the brief
   recommends and as §0 requires (touching the obstacle system would not be
   additive). Obstacle hits consume lives only; Obito's attacks consume HP
   only; either reaching zero ends the run.
3. **§2.3 HP reset** — both fighters return to full at the start of each
   encounter.
4. **§10.6 visible lives** — already shipped: three pips in the running HUD.

### What was built

```
src/config/bossConfig.ts        feature flags + every tunable
src/game/gameState.ts           5-state machine + capability table
src/game/encounterScheduler.ts  fixed milestones + "clean moment" gating
src/game/combatController.ts    telegraph, damage, positional dodge
src/game/sealSequence.ts        randomised 3-seal attack input
src/game/encounterDirector.ts   intro -> combat -> victory/defeat
src/game/bossSystem.ts          facade: everything above + assets + UI
src/scene/obitoCharacter.ts     transform-animated boss
src/scene/healthBar.ts          billboarded bars, chip layer, colour shift
src/scene/attackVfx.ts          pooled, pre-warmed, tinted per fighter
src/scene/combatMarkers.ts      telegraph lane marker + damage numbers
src/ui/gameOverPanel.ts         Try Again panel
src/ui/combatHud.ts             seal prompts + banner
```

`sceneMain.ts` gained only a construction, one `boss.update()` call, one key
hook, and a few `caps.*` checks. `chaseCamera.ts` gained an *optional*
framing override that is a no-op at blend 0, so the running camera is
untouched — the brief asked to feed the existing camera different values
rather than write a second one.

**Suspend-and-restore (§0 rule 3)** is expressed as a `scrollFactor` the host
multiplies into its own unchanged speed, plus a per-state capability lookup.
Outside an encounter the factor is exactly 1 and every capability is true, so
returning to `RUNNING` restores everything by construction rather than by
remembering to undo things.

**No timers anywhere** (`setTimeout`/`setInterval`): everything advances on
the frame delta. That is what makes brief §4.5's warning — "an orphaned timer
from the previous run firing during the new one" — structurally impossible;
dropping the director drops all its state.

### Three real bugs found by looking at the actual output

1. **Obito's FBX ships a white PointLight at intensity 10.** It is inert
   while he is hidden (three skips invisible subtrees when gathering lights),
   then blew the entire environment out to near-white the instant an
   encounter started. Stripped on load; a character must never carry its own
   lighting. Caught only by screenshotting combat.
2. **None of Obito's textures were binding** — every material loaded with
   `map = none`, leaving the suit, cloak and mask flat white, so he read as a
   featureless pale figure. The FBX's texture paths do not resolve; the two
   PNGs are now attached explicitly, matched by the material names the file
   actually uses.
3. **An all-identical seal sequence ~11% of the time.** Measured 20/200, which
   matches uniform random on a 3-item pool — so not a randomness bug, but
   "SNAKE SNAKE SNAKE" is trivially easy and reads as broken. Consecutive
   repeats are now excluded.

### Verified

- **238 unit tests** (up from 168), including the state machine's capability
  table, the encounter scheduler's clean-moment gating, the combat damage
  curve and telegraph fairness (dodging the telegraphed lane provably takes
  zero damage), the seal sequence, and a non-regression suite that pins the
  additive contract itself.
- **17 Playwright e2e tests**, including: Game Over on both loss causes, the
  world genuinely freezing, Try Again by mouse *and* keyboard, **ten
  consecutive restarts with no drift**, intro→combat with HP at 100/120, the
  scroll suspended during combat, and a full keyboard seal sequence landing a
  hit on Obito.
- **Flags-off non-regression confirmed two ways**: the whole scene suite
  passes with both flags `false`, and a running screenshot with the flags
  `true` is visually identical to the approved build.
- `npm run typecheck` and `npm run build` clean.

### Known gaps

- **Hand-seal CV is not wired.** The brief describes it as an existing input
  layer, but this build has never had CV wired into the game: the recognition
  pipeline and the WebSocket transport are complete (Phases A/B), yet
  `scene.html` is keyboard-only — Phase E of the original plan was never
  done. Combat therefore uses the keyboard fallback the brief mandates
  (`1`/`2`/`3`), and the seal names deliberately match the pipeline's own
  vocabulary so wiring it later is a mapping change, not a redesign. **The
  §6 lean-vs-gesture contention risk is consequently untested.**
- No sound, no dedicated victory animation beyond the transform beat, and no
  "Main Menu" secondary button.
- Obito reads small at the duel distance; moved from z=-11 to z=-8, but a
  closer framing or a larger scale may still be wanted.
- The 24 MB Obito model is loaded in the background and an encounter is
  *postponed* (not skipped) if it is not ready. On a cold, slow connection
  the first encounter could arrive late. Decimating that model would help
  both load time and the shadow cost of 157k triangles.
- `K` skips the intro (brief §5.2 suggested it); `H` still toggles the dev
  panel.

---

## Obito's special attack + the Hare counter (`naruto_run_brief_04_special_attack.md`)

Strictly additive per brief §0: new logic lives in new files, and the whole
feature sits behind `ENABLE_SPECIAL_ATTACK` in `src/config/specialAttackConfig.ts`.

### ⚠️ Brief 03 is not in the repo

Brief 04 "amends Brief 03's work order" and expects Brief 03 to exist. It does
not — only briefs 02 and 04 are checked in. That matters because Brief 04
repeatedly refers to things Brief 03 was supposed to build:

| Brief 04 refers to | Actual state |
|:---|:---|
| `ikazuchi` / the fixed attack catalog (§2.1, §2.2) | Does not exist. Combat still uses brief 02's tiger/ram/snake seal sequence. |
| The §5.3 jutsu prompt component (§4.1) | Does not exist. The warning card was built from scratch. |
| The input abstraction layer (§4.2) | Does not exist. |
| The combat stance fix, step 1 of the revised order (§1) | Not done — Brief 03 §3 is not available to work from. |

So the revised work order in §1 could not be followed literally: **step 2 was
built without step 1**, because step 1's specification is missing. Everything
in Brief 04 itself is implemented; the parts that depend on Brief 03 are
called out under "Known gaps" below.

### The eight questions (§8), answered

1. **Context-gated priority — adopted.** The decision lives in exactly one
   place, `EncounterDirector.submitCounter()`, and is made by asking whether
   the window is open. Inside it, `hare` is the counter and the input is
   consumed entirely; outside it the call returns `false` and the input is
   free to mean whatever the catalog says.
2. **"Quickly change lanes" = a description of `ikazuchi`'s dash**, not a
   fourth behaviour. Nothing to build until the catalog exists.
3. **Rare-prompt gating reversed — `hare` is freely performable.** Pressing
   it any time during combat is accepted as practice. Rebalancing `ikazuchi`
   (flat 20 vs. a 4s cooldown) is deferred with Brief 03; **if the cooldown
   route is taken, note that the counter path never consults it**, so the
   §2.2 unfairness the brief warns about cannot occur here.
4. **Trigger on Obito at ≤60 HP — adopted.** He starts at 120 and the first
   jutsu deals 60, so this fires precisely after the player's first landed
   hit. Deterministic, always happens, never an ambush on arrival.
5. **A successful counter deals zero damage** to either fighter. Pinned by a
   test, so the clean 3-hit kill maths cannot drift.
6. **70% slowdown — yes**, and it was safe to take because the countdown is
   physically incapable of being scaled: the controller is fed `rawDt` and the
   world is fed `rawDt * worldTimeScale`. 2.5s is 2.5 real seconds by
   construction, not by discipline.
7. **First window 3.2s, second 2.5s**, so the mechanic teaches itself before
   it tightens.
8. (§2.3) **Once per encounter, max twice per run**, whatever the outcome.

### One finding worth the owner's attention

`hare` is the **weakest-separated class in the recognizer**. From
`cv_model/inference/hand_config.py`, it has the lowest confidence threshold of
all thirteen classes (0.32; the next lowest is `rat` at 0.45) and a margin
threshold of **0.02** — an order of magnitude below every other sign except
`rat`. Those numbers say the classifier can barely distinguish `hare` from its
runner-up and the calibration compensates by accepting almost anything.

Brief 04 makes that class the single input standing between the player and an
unavoidable death. Two consequences when CV is finally wired:

- **False negatives** — the player forms it correctly, it is not accepted, and
  they die to a mechanic with no second chance.
- **False positives** — with a 0.02 margin, some *other* sign gets read as
  `hare`. Outside the window that is a stray `ikazuchi`; inside it, it hands
  out a free save.

Nothing in this build is affected (it is keyboard-only), but a different
counter sign, or per-class recalibration for `hare` specifically, is worth
deciding before the CV wiring lands. Flagged rather than acted on, per §7.

### What was built

```
src/config/specialAttackConfig.ts   flag + every tunable, incl. the window
src/game/specialAttackController.ts pure state machine, no timers
src/scene/specialWarningMarker.ts   billboarded warning above Naruto's head
src/ui/specialAttackOverlay.ts      vignette + hare card + practice prompt
```

Additive edits only: `combatController.ts` gained one read-only getter and one
lethal-hit method; `obitoCharacter.ts` gained a `charging` motion; `attackVfx.ts`
gained a reserved special instance; `gameOverPanel.ts` gained one cause line;
`encounterDirector.ts` and `bossSystem.ts` gained the wiring; `sceneMain.ts`
gained one dependency, one debug field group and a key case; `sceneRoot.ts`
gained a third callback argument (see bug 4 below).

**The two dangerous leaks are structurally impossible.** §5 names them: a run
that starts with lean input still suppressed, or the world stuck at 70%.
Neither is a stored flag — `suppressLaneInput` and `worldTimeScale` are getters
computed from the controller's current state, so *every* exit restores them,
including exits nobody has thought of yet. `tearDownSpecial()` still exists as
the single imperative teardown (overlay, marker, wind-up, VFX) and runs on all
four exit paths: counter, strike, encounter end, and restart.

### ⚠️ The one that mattered most: the feature was unreachable in play

Shipped "green" — 284 unit tests, 28 e2e — and then the owner played the game
and **never saw the special attack at all**. Two independent causes, and the
same root reason neither was caught:

> **Every test reached the warning through `forceSpecialAttack()`**, a debug
> hook that chips Obito to the trigger HP *and* parks his ordinary attacks.
> Not one test played the actual fight. The feature worked perfectly along a
> path no player can take.

1. **The §2.2 practice guard was an off switch.** The brief says to
   "prompt them to try it first, *or* delay the special attack". It was
   implemented as *delay indefinitely*: a player who never pressed `H` sat in
   `awaiting_practice` for the rest of the run and the attack never came. Now
   bounded by `SPECIAL_PRACTICE_PATIENCE_S` (3s) — prompt, then fire anyway.
2. **The HP trigger does not guarantee the attack, despite §3 saying it
   does.** Firing at "Obito ≤60 HP" assumes the player lands a jutsu. That
   needs a completed three-seal sequence *and* Obito still standing in the
   lane it was aimed at — he deliberately moves out of it — while his own
   attacks kill Naruto in three hits at roughly 4.1s, 8.2s and 12.3s. A
   recorded end-to-end run of the real fight ended at 12s with Obito
   untouched on **120 HP**, the mechanic never triggered. Added
   `SPECIAL_COMBAT_FALLBACK_S` (5s of combat) as a floor, so it is now
   genuinely guaranteed once per encounter; the HP threshold usually still
   fires first and keeps the intended "retaliation" read.

Also made the practice banner much louder — it *was* on screen and was missed
completely, which is its own answer about how prominent it needed to be.

Two e2e tests now play the fight for real, with no debug trigger: one where
the player does nothing at all, and one where they land a genuine jutsu.

### Nine bugs found while building it

The last five were found only by looking at a rendered frame and by measuring
the running page — no test would have caught any of them.

1. **The teardown would never have run on the failure path.** Being struck
   transitions the encounter to `defeat` on the same frame, so a special
   attack advanced only from the `combat` branch would freeze mid-state and
   leave its overlay on screen through Game Over. The controller is now
   advanced on every phase and only *arms* in combat. Caught by writing the
   §5 reset test, then pinned by an integration test.
2. **Lane suppression did not actually suppress anything.** The frame loop
   read the resolved capabilities but `sceneMain`'s keydown handler read
   `boss.state.capabilities` — the raw table, which knows nothing about the
   warning window. Arrow keys worked fine mid-window. Only the end-to-end test
   caught it: the two sources diverge exclusively when a real key is pressed.
3. **The counter's own feedback was being swallowed.** The counter arrives
   from a key handler between frames, and `update()` zeroed the hit-stop and
   shake fields at the top of every frame — so the one piece of feedback
   confirming the player's save landed was discarded before anything read it.
   Those requests are now carried into the next frame.
4. **The countdown was not running in real seconds.** `SceneRoot` clamps the
   frame delta to 1/15s so a stall cannot teleport anything that integrates.
   Below 15fps that makes game time run slower than the clock — a 3.2s window
   measured **8.2 real seconds** at ~5fps. This broke §4.1's explicit
   acceptance criterion ("lasts exactly the configured duration in real
   seconds"). `SceneRoot` now also passes the unclamped delta, and *only* the
   special attack's countdown uses it; everything that integrates still uses
   the clamped one. Verified back down to 3.5s, and pinned by a unit test
   that drives the director at 5fps.
5. **The warning card covered the entire fight.** Centred and stacked, it sat
   exactly over Naruto, Obito and both health bars — including Obito's
   wind-up, which §4.1 specifically requires be readable from his body. Moved
   to a bottom strip and laid out as a row.
6. **The 卯 glyph rendered as a broken glyph** beside "HARE" on the
   canvas-drawn marker, where the CJK fallback is whatever the OS supplies.
   Dropped from both warning surfaces; flavour is not worth risking
   legibility on the one prompt the player cannot afford to misread.
7. **The hand schematic read as a candle.** Redrawn, and now paired with a
   written description ("clasp both hands — index and middle fingers straight
   up") so the words carry the meaning and the drawing only speeds up
   recognition.
8. **A stale "Perform the seals" prompt stayed on screen** through the
   window, competing with the counter — exactly what §4.2 forbids. The combat
   HUD is now hidden for the duration and restored afterwards, though not on
   the defeat path, where it would flash back on over the death beat.
9. **Then the world-space banner hid Obito instead.** Having moved the card
   out of the way, the banner above Naruto's head took over the same job:
   Obito stands 8 units down the track, so his head projects to almost
   exactly Naruto's screen height, and the banner draws with
   `depthTest: false`. Lifted to 2.8 units above head height — above the
   horizon line, where nothing in the fight competes with it. The fix for a
   framing bug can quietly recreate it somewhere else; only a second
   screenshot showed that.

### Verified

- **288 unit tests** (up from 238): the trigger guard rails, the practice
  gate, real-seconds timing under both a time scale and a clamped host delta,
  the stale-hold rejection, both outcomes, and a table-driven walk of every
  exit path asserting the two leaks are released.
- **13 Playwright e2e tests** for this brief, and **30/30 across the whole
  suite** (so briefs 02 and the visual overhaul are unregressed). They cover
  the rendered warning (screenshotted to `test-results/special-warning.png`),
  the draining ring, lane suppression and restore, jutsu-prompt cancel and
  restore, a counter damaging nobody, death at 100/100 with the correct cause
  line, and a full restart with no residual suppression.
- **The 👁 criteria were checked against real frames**, not just assertions.
  `special-warning.png` shows Naruto, Obito, both health bars, the banner
  above Naruto's head, the countdown ring, the hare reference and the danger
  vignette all legible at once; `special-countered.png` shows the counter
  resolving and the jutsu prompt restored.
- `npm run typecheck` and `npm run build` clean.

**The e2e file shares one page for all eleven tests.** Every test needs a live
encounter and Obito's model is ~24 MB, so Playwright's default per-test page
meant eleven cold loads of it — which starved the machine badly enough that
unrelated tests in *other* files started timing out. The file now runs
serially against a single page and resets between tests through the game's own
restart path, which incidentally exercises that path eleven more times. Suite
time went from 13.5 min to 5.5 min.

**§6's "60fps sustained" could not be verified in this environment.** Measured
frame times in headless software rendering: ~80 ms/frame while running (12fps)
and ~140 ms/frame in combat (7fps) — both *pre-existing*, driven by the
157k-triangle Obito and his shadow. The warning window adds ~19 ms on top of
that, so the feature is not the bottleneck, but the criterion needs a check on
real GPU hardware.

### Known gaps

- **Blocked on Brief 03:** the combat stance fix, the jutsu catalog and
  `ikazuchi`, the prompt component, and the input abstraction layer.
- **No `hare` reference photo.** §4.1 wants the sign shown large; the CV
  training set is not in the repo, so the card draws a schematic and will
  swap in a real image automatically if one is dropped at
  `public/textures/seals/hare.png`.
- **No audio.** §4.1's sting and rising tension cue are not implemented — the
  project has no audio layer at all yet.
- **The frame rate during an encounter is untested on real hardware** (see
  above). If it turns out to be a problem, decimating the 24 MB / 157k-triangle
  Obito is the single biggest win, and it would shorten the e2e suite too.
- `H` is the counter during an encounter and the dev-panel toggle elsewhere;
  `` ` `` toggles the panel unconditionally.

---

## Body-movement CV input (`naruto_run_brief_05_body_cv_integration.md`)

**Status: §11 steps 2–5 complete. No camera connected yet, by design.**

### Discovery (§2), in one paragraph

There is **no trained body model**. `body_tracking_prototype.py` is a
standalone heuristic demo — MediaPipe's pretrained `pose_landmarker_lite`
plus hard-coded thresholds, drawing to an OpenCV window, returning nothing.
The real implementation is `BodyMovementRecognizer` in
`cv_model/inference/combined_pipeline.py`, already wired into the Phase-A
transport (52/52 tests green). The prototype is untouched, per the owner.

### ⚠️ The mirroring inversion, and where it is fixed

The pipeline runs inference on the **raw, unmirrored** frame:
`framePublisher.ts` "never mirrors"; `run_combined_camera.py` flips only
*after* recognition; and `cv_model/tests/test_display_mirroring.py` asserts
that display mirroring leaves labels alone. In an unmirrored frame the
player's physical left sits at *higher* x, so leaning left yields
`lean_x > 0` → the pipeline calls it **`bending_right`**. Unfixed, leaning
left would move Naruto right.

`src/input/poseAdapter.ts` is **the one place any of this is flipped**, and it
handles both concerns together so neither can be done without the other:

- *gameplay* — negate `lean_x`, swap `bending_left` ⟷ `bending_right`;
- *display* — mirror landmark x, so the skeleton overlay matches the
  CSS-mirrored preview instead of rendering backwards over it.

One switch, `MIRROR_CAMERA_X`, governs both. The file says in as many words
that nothing else may flip.

### What was built

```
src/input/bodyTuning.ts        every constant + runtime JSON loader
src/input/poseAdapter.ts       THE mirroring point; pipeline -> player space
src/input/bodyCalibration.ts   neutral baseline; distance normalisation
src/input/leanDetector.ts      hysteresis + debounce + return-to-neutral
src/input/jumpDetector.ts      height + velocity + refractory
src/input/bodyInputSource.ts   composes the above; §6 state gating
src/input/inputRouter.ts       the intent layer (§5.4)
src/input/poseFixtures.ts      synthetic generator (§9.1 third source)
public/config/bodyTuning.json  edit + refresh, no rebuild
```

**Tuning without a rebuild**, as instructed: the JSON is fetched at startup
and overlaid on the defaults. Missing keys fall back; unknown keys, bad types
and a collapsed hysteresis gap are rejected loudly. `_`-prefixed keys are
comments, since JSON has none.

**What is deliberately *not* re-implemented.** The pipeline's `ConsensusFilter`
already debounces labels and already fires on transition, resetting when the
label returns to neutral — that is §5.2's `hasReturnedToNeutral`, already
built. The jump path therefore adds only normalisation, velocity and
refractory. The lean path bypasses the label entirely (below), so its
debounce is the *only* such layer, not a second one.

### Lean comes from the metric, not the label

`classify_metrics` is a priority chain: `jumping > naruto_run > bending_*`.
A jogging player satisfies `naruto_run`, so their lean never reaches the
label — and this is a running game, so the most engaged players would lose
lane control entirely. Lane intent is therefore derived from the `lean_x`
metric, which no label ranking can suppress. Fixture
`jog_in_place_with_lean` pins it: every frame labelled `naruto_run`, all four
lane changes still fire, zero phantom jumps.

### A real bug the tests caught: lean was not distance-normalised

T-11 failed on first run. Jump had been normalised by torso height but lean
had not — `lean_x` is a normalised-*image* offset, so the same physical lean
produces roughly half the value at 3m that it does at 1m. It worked at the
calibration distance and silently stopped working when the player stepped
back, which is exactly the failure §4.4 warns about. Lean is now divided by
the *current* frame's torso height (so drifting closer or further stays
calibrated), and the resting offset is stored already normalised.

### Fixture results — §9.3, all green

42 tests, no camera and no human: T-01 through T-16 plus the mandated
`jog_in_place_with_lean` and the tuning-loader cases. Highlights: boundary
hovering → **zero** lane changes; five-second held lean → **exactly one**;
jogging on the spot → **zero** jumps; two crossings in 600ms → **one** jump;
identical decisions at 1m and 3m; and both mirroring directions.

### Honest gaps before the camera goes on

- **The fixtures are synthetic, not recorded.** §9.2 wants clips of at least
  two people of different builds, run through the model once. That needs a
  camera and volunteers. These prove the *logic* is correct given a known
  input; they cannot prove real bodies produce the signals assumed. The same
  tests will run unchanged against recorded fixtures — both arrive as
  `PoseSample[]`.
- **§8's "degrade capture resolution" mitigation is dead.** Measured
  640×480 = 13.1ms/frame and 320×240 = 13.4ms — MediaPipe rescales
  internally, so dropping resolution buys nothing. Proposed replacement:
  **reduce the inference rate** (the scheduler already keeps only the latest
  frame), which trades responsiveness for load without touching accuracy.
- **End-to-end latency is still unmeasured** — the earlier 13ms was inference
  only, on empty synthetic frames. The per-stage instrumentation the budget
  needs comes with step 6.

### Camera integration (§11 steps 6–8)

**`ENABLE_CV_INPUT` ships OFF.** Play the camera build with
`scene.html?cv=1`; `?cv=0` forces it off. Turning it on makes the startup
gate block the run until a player is detected — the intended end state, but
it also means anyone without the recognition service running meets a gate
instead of a game. Defaulting it on broke every existing end-to-end test, and
those tests were right to break: it changes how the approved build starts.
Flip the default in `src/config/cvInputConfig.ts` once §9.7's manual protocol
has been run on the demo machine.

**One command to play** (§3.2): `.\cv_model\Nishit_Frontend\start_game.ps1`
starts the recognition service, waits for it to actually answer, starts the
dev server and opens the game. `-KeyboardOnly` skips the service. The manual
two-terminal path is documented at the top of the script.

```
src/config/cvInputConfig.ts   flag, transport URL, capture settings
src/input/cameraPoseSource.ts camera + frame upload + socket + latency stages
src/game/startupGate.ts       BOOT -> ... -> RUNNING, pure and testable
src/ui/startupGatePanel.ts    gate panels, in the Game Over visual language
src/ui/cameraHud.ts           bottom-right preview, skeleton, status ring
```

Reuses Phase B's `CameraController`, `FramePublisher` and `WebSocketClient`
rather than adding a second transport (§2.2).

**Capture is 640×360 at 20fps**, not 30. Inference costs ~13ms regardless of
resolution, so the lever that reduces load is *rate*, not size — the
replacement for §8's dead resolution-degradation mitigation. 20fps still
gives the 3-frame lean debounce a 150ms window.

#### A bug only the rendered frame caught

The gate panel reuses the Game Over panel's `.go-*` classes, as §4 asks. But
`.go-panel` is `opacity: 0` by default and the rule that reveals it is
**ID-scoped** to `#game-over.go-visible`. So the gate mounted, reported
`hidden: false`, `display: grid`, `opacity: 1` on its root — and painted
nothing at all. Playwright's `toBeVisible()` ignores opacity, so the test
passed too. Fixed by adding `#startup-gate` to both reveal rules; the test
now asserts computed opacity, not just visibility.

#### Verified

- **345 unit tests** (up from 330): the gate's state machine, sustained
  detection, the three distinguishable failures, guidance text, the keyboard
  escape and manual recalibrate.
- **8 new e2e tests**: R-01 flag-off (asserts `getUserMedia` is never
  called), I-01 full gate to RUNNING with the world frozen until it opens,
  I-02 keyboard escape playable, I-03 distinct service error, and §7's
  preview position/mirroring/toggle plus the debug overlay contents.
- 👁 `test-results/startup-gate.png` and `camera-hud.png`: the preview sits
  bottom-right and overlaps nothing; the reserved regions are untouched.

#### Still outstanding

- **No end-to-end latency measurement with a real person.** The chain is
  instrumented per stage (`LatencyBreakdown`) and surfaced in the debug
  overlay, but the numbers need a camera and the confirmed demo machine.
- **§9.7's manual protocol and the mirroring video** — the acceptance gate
  for the wrong-direction bug. Nothing automated can replace it.
- **§8 failure handling** (step 9): disconnect and frame-exit currently drive
  the status indicator but do not yet pause the run.
- Recorded fixtures with two people, still.
