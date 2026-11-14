# Transport service

The transport service (`python -m narutocv.transport`, console script
`narutocv-serve`) is the bridge between the browser game and the Python
recognition pipeline. The browser owns the camera and streams unmirrored JPEG
frames over one binary WebSocket (`/ws`, wire format "NCV1"); the server runs
`narutocv.inference` and returns state snapshots and ordered events.

This document has two parts: the threat model as the code stood *before* the
Phase 6 hardening pass (what a malicious or buggy client could do), and the
hardening that answers it.

## 1. Threat model (pre-hardening baseline)

Assets: the CPU budget of the single recognition pipeline (~30-40 ms/frame),
the single "controller" slot, process memory, and the privacy of whatever the
camera sees (the server never sends video back, but events and landmarks derived
from it are sensitive).

Trust boundary: anything that can open a TCP connection to the port, plus every
web page the user visits (a browser will happily connect to
`ws://127.0.0.1:8765` on behalf of any site).

| # | Threat | How it worked in the baseline code |
|---|--------|------------------------------------|
| T1 | Cross-site WebSocket hijacking | `allowed_origins` defaulted to `None`, which means "accept any Origin". Binding to 127.0.0.1 does not help: the victim's own browser is the attacker's proxy, so any web page can take the controller slot, push frames, and read landmarks/events. |
| T2 | Decompression bomb / memory blowup | The header dimensions are client-declared. The only check that the JPEG matches them ran *after* `cv2.imdecode`, so a ~2 MiB JPEG whose SOF header says 65535x65535 forces a multi-GiB allocation before it is rejected. Non-JPEG payloads (PNG, TIFF, ...) were handed to OpenCV as well. |
| T3 | Event-loop starvation | JPEG decode ran inline on the asyncio loop, so a stream of large frames stalls every other connection and `/health`, which an orchestrator reads as "dead". |
| T4 | Unbounded frame rate | No per-connection rate limit. Frames beyond the pipeline's rate are only superseded by the latest-frame scheduler *after* a full JPEG decode, so a fast sender burns CPU even though most frames are discarded. |
| T5 | Amplification | Every bad frame and every bad control message produced one `error` reply, so a client can make the server send as fast as it can receive. |
| T6 | Controller-slot squatting | The slot is reserved during the handshake and there was no idle timeout afterwards: one silent client holds it forever, and a client that reconnects in a loop keeps winning it. TCP half-open connections were never detected (no WS heartbeat). |
| T7 | Expensive control commands | `reset` rebuilds the MediaPipe pose landmarker (~0.1 s) and `begin_calibration` clears the pipeline; both were unlimited, so a client could pin the controller thread by spamming them. |
| T8 | No version negotiation | The server parsed `client_hello` but never compared versions: a "2.0.0" client was accepted and then failed in confusing ways. The client could not learn which model bundle or class list the server runs. |
| T9 | Oversized WebSocket messages | `max_msg_size` was 4 MiB, twice what a legitimate frame can be, and applied to text messages as well. |
| T10 | Connection flooding | No cap on concurrent sockets; each one that is upgraded costs a task and buffers. |
| T11 | Information disclosure | `/health` returned `last_error` verbatim (may contain file paths). There was no structured log, so none of the above was observable either. |
| T12 | Hung shutdown | Only `on_cleanup` existed. `web.run_app` waits for open WebSocket handlers on SIGTERM, so the process stalled and was then SIGKILLed with the MediaPipe/ONNX handles still open. A failed `scheduler.stop` skipped `controller.close()`, and a failure partway through `SessionScopedPipeline.__init__` leaked the already-built hand landmarker. |

Out of scope: authentication and TLS. The service is a local sidecar bound to
loopback by default; if it is exposed beyond that, put it behind a reverse proxy
that terminates TLS and authenticates, and set an explicit origin allow-list.

## 2. Hardening

Every knob is a CLI flag and an environment variable (flag wins); defaults are
in `src/narutocv/transport/config.py`.

| Threat | Mitigation | Flag / env | Default |
|--------|-----------|------------|---------|
| T1 | Origin allow-list on the upgrade; a present, unlisted `Origin` (including `null`) gets HTTP 403. A *missing* `Origin` is allowed because only browsers send it and hijacking is a browser attack; `--require-origin` rejects those too. `*` allows any origin. | `--allow-origin` (repeat) / `NARUTOCV_ALLOWED_ORIGINS` (comma separated), `--require-origin` / `NARUTOCV_REQUIRE_ORIGIN` | `http://127.0.0.1:5173`, `http://localhost:5173`, and the same on `:4173` (Vite dev and preview) |
| T1 | Loopback bind (a warning is logged for anything else). | `--host` / `NARUTOCV_HOST`, `--port` / `NARUTOCV_PORT` | `127.0.0.1`, `8765` |
| T9 | `max_msg_size` = header + 2 MiB JPEG ceiling + 64 KiB margin (about 2.06 MiB). Larger messages are cut off by aiohttp (close 1009) before being buffered. Text messages over 16 KiB are rejected unparsed. | (constants in `config.py`) | |
| T2 | The JPEG's own SOF header is read (walking marker segments, so an EXIF thumbnail cannot mislead it) before `cv2.imdecode`. It must be a JPEG, be within 1920x1080, and match the dimensions declared in the frame header; otherwise the frame is rejected without decoding. | | |
| T3 | JPEG decode runs in the default executor, not on the event loop. | | |
| T4 | Per-connection token bucket on frames; excess frames are dropped *silently* (no error reply) and counted as `rate_limited`. A burst is never grounds for disconnecting. | `--max-fps` / `NARUTOCV_MAX_FPS`, `--frame-burst` / `NARUTOCV_FRAME_BURST` | 60 fps sustained (the web client sends 30), burst 30 |
| T5 | Error replies are themselves rate-limited per connection (suppressed ones are counted), and a connection producing too many invalid messages is closed with 4400. | | 5 replies/s, burst 10; close after 100 invalid messages |
| T7 | Per-connection token bucket on control messages (`reset` rebuilds the pose landmarker). Over the cap the message is not executed. | | 10/s, burst 20 |
| T10 | Connection cap, checked and taken before the first `await` and released in a `finally` on every exit path. Over the cap the upgrade is refused with HTTP 503. | `--max-sessions` / `NARUTOCV_MAX_SESSIONS` | 2 |
| T6 | Idle timeout (any frame, control or ping counts as activity; the web client pings every 15 s), plus protocol-level WebSocket ping/pong that detects half-open TCP. The handshake keeps its own 5 s timeout. | `--idle-timeout` / `NARUTOCV_IDLE_TIMEOUT`, `--heartbeat` / `NARUTOCV_HEARTBEAT` | 60 s, 15 s (0 disables the heartbeat) |
| T8 | Version and model-bundle handshake (below). | | |
| T11 | `/health` reports only the exception class name in `last_error`; the full traceback goes to the structured log. | | |
| T12 | Graceful shutdown (below). | `--shutdown-grace` / `NARUTOCV_SHUTDOWN_GRACE` | 5 s |

`max_sessions` is a cap on concurrently *upgraded sockets*, not on parallel
recognition. The service still runs exactly one pipeline and admits exactly one
controlling client ("keep one recognition producer"); with the default of 2, a
second socket is accepted just long enough to be told
`controller_already_connected` (close 4409), and a third is refused with 503.
Set it to 1 to refuse a second socket outright.

Residual risks, deliberately not addressed here: a non-browser process on the
same machine can still occupy the controller slot (no authentication; loopback
is the trust boundary), and `/health`, `/live` and `/metrics` are unauthenticated.

### Protocol version 1.1.0 and the handshake

The wire change is **additive**, so 1.0.0 clients and servers keep working.

- Client `client_hello`: `protocol_version` (`MAJOR.MINOR.PATCH`) as before, plus
  optional `expected_model_id`.
- Server `server_ready`: as before, plus `model` (`id`, `sha256`, per-file
  `files`), `classes` (`hand`, `body`, `attacks`) and `capabilities` (initial
  mode, supported modes, geometry, frame limits, timeouts).
- Compatibility rule: **majors must match**; minors/patches may differ in either
  direction, and unknown fields are ignored by both sides.
- A different major: error `protocol_version_mismatch` (message names both
  versions, `details` carries `server_protocol_version`), then close **4426**.
- An `expected_model_id` that differs from the server's: error `model_mismatch`,
  then close **4412**. A client that sends no `expected_model_id` accepts any bundle.
- The TS client treats 4412/4426 (and a mismatching `server_ready`) as fatal: it
  reports the error, enters the `error` connection state and does **not** retry.

Model bundle identity comes from `models/manifest.yaml` (override with
`NARUTOCV_MODEL_MANIFEST`): `sha256` is the digest of the sorted
`<name>:<sha256>` lines of every artifact, and `id` is `narutocv-` plus its first
12 hex characters. The digests are the manifest's pinned values; they are not
re-hashed from disk at runtime (the Docker build runs
`scripts/fetch_models.py --verify`). If the manifest cannot be read the bundle is
`unversioned`. Schemas: `schemas/transport_message_v1.schema.json` and
`transport_control_v1.schema.json` (same `v1` file names; additive change).

Close codes: 1001 server shutting down, 1009 message too big, 4400 protocol
error / handshake failed / too many invalid messages, 4408 idle timeout, 4409
controller already connected, 4412 model mismatch, 4426 protocol version
mismatch, 4429 client too slow (event backpressure).

### Observability

Logs are JSON lines on stderr (`--log-format text` for humans, `--log-level`):
`ts`, `level`, `logger`, `msg`, `conn` (a 12-hex connection id on every line
about a connection), and event-specific fields. Client-controlled strings are
length-capped and JSON-escaped. Frame payloads are never logged.

`GET /metrics` (Prometheus text format 0.0.4, hand-rolled, no dependency):

| Metric | Type | Labels |
|--------|------|--------|
| `narutocv_frames_received_total` | counter | |
| `narutocv_frames_processed_total`, `narutocv_frames_failed_total` | counter | (from scheduler stats) |
| `narutocv_frames_dropped_total` | counter | `reason` = `rate_limited`, `invalid`, `stale_sequence`, `rejected`, `superseded` |
| `narutocv_controls_dropped_total`, `narutocv_error_replies_suppressed_total` | counter | |
| `narutocv_connections_rejected_total` | counter | `reason` = `origin`, `shutting_down`, `not_ready`, `max_sessions`, `controller_busy` |
| `narutocv_connections_closed_total` | counter | `reason` = `client`, `idle_timeout`, `backpressure`, `message_too_big`, `invalid_messages`, `shutdown`, `handshake_failed`, `controller_busy` |
| `narutocv_handshake_failures_total` | counter | `reason` = `timeout`, `invalid`, `protocol_version`, `model_mismatch` |
| `narutocv_active_sessions`, `narutocv_open_connections`, `narutocv_ready` | gauge | |
| `narutocv_build_info` | gauge | `protocol_version`, `model_id` |
| `narutocv_stage_latency_ms` | histogram | `stage` = the pipeline's own `timings_ms` keys plus `queue_wait_ms`, `worker_elapsed_ms`, `end_to_end_ms` |

`superseded` counts frames the latest-frame scheduler replaced before they were
processed (normal under load, not an error). Only the newest result of each
burst is observed in the stage histograms.

Health endpoints: `GET /health` is **readiness** (200 only when the models are
loaded and the server is not draining; 503 while starting, degraded or shutting
down; existing fields are unchanged and `model_id`, `protocol_version`,
`open_connections` were added). `GET /live` is **liveness** (200 whenever the
event loop answers, including while degraded or draining, so an orchestrator
restarts on a hang rather than on a model that failed to load).

### Graceful shutdown

On SIGTERM/SIGINT aiohttp stops listening and runs `on_shutdown`, which marks the
server `shutting_down` (`/health` goes 503), closes every WebSocket with 1001, and
waits up to the grace period for handlers to finish. `on_cleanup` then stops the
scheduler, cancels the result pump, unsubscribes and closes the runtime (which
closes the MediaPipe hand/pose landmarkers). Each step is isolated, so a
scheduler that will not stop in time still cannot prevent the landmarkers from
being released. Leaks fixed while reading the close paths:
`SessionScopedPipeline.close()` closed the body landmarker only if the hand
landmarker's close did not raise, and a failure inside its constructor left the
two native landmarkers the base class had already opened unreleased.

### Docker

```sh
docker build -f docker/transport.Dockerfile -t narutocv-transport .
docker run --rm -p 127.0.0.1:8765:8765 \
  -e NARUTOCV_ALLOWED_ORIGINS=http://localhost:5173 narutocv-transport
```

Multi-stage and uv based (`uv sync --frozen`); runs as uid 10001; models are
downloaded and sha256-verified at build time by `scripts/fetch_models.py` (if the
build has no network, mount verified models at `/app/models`); `HEALTHCHECK`
polls `/live`; the entrypoint is exec-form so Python is PID 1 and receives
SIGTERM. The container binds `0.0.0.0` (via `NARUTOCV_HOST`) because it has to,
so publish the port on loopback only, as above, unless a TLS-terminating,
authenticating proxy is in front.
