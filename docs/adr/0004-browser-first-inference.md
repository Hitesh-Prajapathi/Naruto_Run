# 0004. Browser-first inference, Python as reference and eval engine

Status: Proposed (the current system is Python-served)

## Context

Today the browser captures the camera, encodes JPEG, and streams frames over a WebSocket (NCV1) to a Python process that runs the models. This requires a local server, adds JPEG encode and upload stages to the latency chain, and means a public demo cannot be hosted statically. Frames also leave the browser process, which complicates a privacy story.

## Decision

- Run the whole recogniser in a **Web Worker**: MediaPipe Tasks Vision (JS) for landmarks, onnxruntime-web for the image branch and fusion (WebGPU, falling back to WASM). Frames reach the worker through `requestVideoFrameCallback`, `OffscreenCanvas` and a transferred `ImageBitmap`.
- Keep the **Python service as the reference implementation and evaluation engine**. The temporal decoder and combo FSM are specified once and ported to TypeScript, with a **golden parity test** (recorded clip, Python events vs browser events within 1 event and 50 ms) in CI.
- The game selects `source = browser | websocket` through the existing `PipelineEventSource` abstraction.
- Model assets are served with sha256 checks and Cache Storage; the image branch is lazy-loaded.
- Privacy: in browser mode no frames leave the device (planned, not implemented).

## Consequences

- A static, public demo becomes possible, and the JPEG/upload stages disappear.
- Two implementations must stay in agreement; the parity test is the guard, and divergence is a standing risk.
- Performance varies widely across devices; a hardware table (FPS, P95 latency, bundle size, first-load time) is required before claims are made.
- Until Phase 5 lands, the WebSocket path remains the only working path.
