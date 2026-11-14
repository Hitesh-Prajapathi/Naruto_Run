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
