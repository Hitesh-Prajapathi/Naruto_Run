"""Loopback HTTP + WebSocket transport around PipelineRuntimeController.

Implements the step-6 contract from game_implementation_plan.md and
docs/archive/context.md §9.1: the browser owns the camera and streams unmirrored JPEG
frames in over one binary WebSocket; this server returns capped state
snapshots and immediately-delivered ordered sparse events out. Every frame is
handed to the existing, locked PipelineRuntimeController/LatestFrameScheduler
unchanged -- this module carries bytes, it never decides what a gesture is.

Phase 6 hardening (see docs/transport.md): origin allow-list, message/frame
size limits, per-connection token buckets, a connection cap, idle timeout and
heartbeat, protocol/model handshake, Prometheus metrics, structured logs and
a graceful shutdown that drains sessions and releases native resources.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import logging
import time
import uuid
from typing import Any, Callable, FrozenSet, Optional

from aiohttp import WSMsgType, web

from narutocv.inference.combined_pipeline import ATTACKS, FALLBACK_HAND_CLASSES
from narutocv.inference.pipeline_config import DEFAULT_PIPELINE_CONFIG, PipelineConfig
from narutocv.inference.runtime import (
    PipelineFactory,
    PipelineRuntimeController,
    RuntimeConfig,
    RuntimeState,
)
from narutocv.inference.scheduler import LatestFrameScheduler

from . import logging_config
from .config import DEFAULT_ALLOWED_ORIGINS, TransportConfig
from .connection import ClientSession, EventQueueOverflow, StaleFrameError, TokenBucket
from .frame_decoder import decode_image
from .health import HealthStatus, ReadinessState
from .metrics import CONTENT_TYPE, TransportMetrics
from .model_info import ModelBundle, load_model_bundle
from .pipeline import session_scoped_pipeline_factory
from .protocol import (
    MAX_FRAME_HEIGHT,
    MAX_FRAME_PAYLOAD_BYTES,
    MAX_FRAME_WIDTH,
    PROTOCOL_VERSION,
    ControlCommand,
    ModelMismatchError,
    ProtocolError,
    VersionMismatchError,
    build_ack,
    build_error,
    build_pipeline_event,
    build_server_ready,
    build_state_snapshot,
    decode_frame,
    negotiate_hello,
    parse_client_hello_full,
    parse_control_message,
)

logger = logging.getLogger("narutocv.transport")

# Typed as Any rather than AppKey["TransportServer"]: TransportServer is
# defined later in this module and `from __future__ import annotations`
# only defers annotation evaluation, not this assignment's RHS.
TRANSPORT_SERVER_KEY: "web.AppKey[Any]" = web.AppKey("transport_server")

SNAPSHOT_INTERVAL_SECONDS = 1.0 / 15.0  # 15 Hz cap, per docs/archive/context.md step-6
SENDER_POLL_INTERVAL_SECONDS = 0.01
RESULT_WAIT_TIMEOUT_SECONDS = 0.5
MAX_MODE_CHARS = 32

# WebSocket close codes. 4xxx is the application range; the number echoes the
# closest HTTP status where there is one.
CLOSE_CODE_GOING_AWAY = 1001
CLOSE_CODE_PROTOCOL_ERROR = 4400
CLOSE_CODE_IDLE_TIMEOUT = 4408
CLOSE_CODE_CONTROLLER_TAKEN = 4409
CLOSE_CODE_MODEL_MISMATCH = 4412
CLOSE_CODE_VERSION_MISMATCH = 4426
CLOSE_CODE_BACKPRESSURE = 4429
# Close codes after which a client must NOT auto-reconnect: retrying cannot
# succeed until the client or server is upgraded.
FATAL_CLOSE_CODES = (CLOSE_CODE_MODEL_MISMATCH, CLOSE_CODE_VERSION_MISMATCH)

_UNSET: Any = object()


class TransportServer:
    """Owns exactly one PipelineRuntimeController and one controlling client.

    Multi-client ownership is explicitly out of scope for step 6: a second
    connection while one is controlling is rejected rather than silently
    mixed in (docs/archive/context.md: "Keep one recognition producer.").
    `config.max_sessions` bounds how many sockets may be open at once (the
    controller plus connections about to be told "controller busy"); it does
    not create parallel recognition pipelines.
    """

    def __init__(
        self,
        *,
        pipeline_factory: Optional[PipelineFactory] = None,
        runtime_config: Optional[RuntimeConfig] = None,
        allowed_origins: Optional[FrozenSet[str]] = _UNSET,
        event_queue_size: int = 256,
        body_only: bool = False,
        config: Optional[TransportConfig] = None,
        model_bundle: Optional[ModelBundle] = None,
        clock: Callable[[], float] = time.monotonic,
        pipeline_config: Optional[PipelineConfig] = None,
    ) -> None:
        # Defaults to the session-scoped pipeline so that a reset re-detects
        # the player instead of staying locked onto whoever was in front of
        # the camera first. See transport/pipeline.py.
        self._pipeline_factory = pipeline_factory or session_scoped_pipeline_factory(
            body_only=body_only, config=pipeline_config or DEFAULT_PIPELINE_CONFIG
        )
        # `include_geometry=True` by default here, unlike RuntimeConfig's own
        # default. The browser game needs the pose landmarks for three
        # things: deciding whether a player is present at all, deriving torso
        # height (which is what makes lean and jump work at any distance from
        # the camera), and drawing the skeleton overlay. With geometry off,
        # `body.geometry` is null, the landmark array is empty, and the
        # startup gate sits on "No one detected" forever while the camera
        # preview plainly shows a person -- which is exactly what happened.
        self._runtime_config = runtime_config or RuntimeConfig(include_geometry=True)
        self.config = config or TransportConfig()
        if allowed_origins is not _UNSET:
            self.config = dataclasses.replace(self.config, allowed_origins=allowed_origins)
        self.event_queue_size = event_queue_size
        self._clock = clock
        self._model_bundle = model_bundle
        self._body_only = body_only

        self.metrics = TransportMetrics()
        self.health = HealthStatus(protocol_version=PROTOCOL_VERSION)
        self.controller: Optional[PipelineRuntimeController] = None
        self.scheduler: Optional[LatestFrameScheduler] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._session: Optional[ClientSession] = None
        self._active_ws: Optional[web.WebSocketResponse] = None
        self._sockets: set[web.WebSocketResponse] = set()
        self._open_connections = 0
        self._result_task: Optional[asyncio.Task] = None
        self._subscription_id: Optional[int] = None
        self._server_info: dict[str, Any] = {}

    @property
    def allowed_origins(self) -> Optional[FrozenSet[str]]:
        return self.config.allowed_origins

    # --- lifecycle -----------------------------------------------------

    def build_app(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/health", self.handle_health)
        app.router.add_get("/live", self.handle_live)
        app.router.add_get("/metrics", self.handle_metrics)
        app.router.add_get("/ws", self.handle_ws)
        app.on_startup.append(self._on_startup)
        app.on_shutdown.append(self._on_shutdown)
        app.on_cleanup.append(self._on_cleanup)
        return app

    async def _on_startup(self, _app: web.Application) -> None:
        self._loop = asyncio.get_running_loop()
        if self._model_bundle is None:
            self._model_bundle = load_model_bundle()
        self.health.model_id = self._model_bundle.id
        self.metrics.build_info.set(
            1, protocol_version=PROTOCOL_VERSION, model_id=self._model_bundle.id
        )
        try:
            self.controller = PipelineRuntimeController(
                pipeline_factory=self._pipeline_factory,
                config=self._runtime_config,
            ).start()
        except Exception as error:  # keep the process alive; /health reports it
            self.health.state = ReadinessState.DEGRADED
            self.health.last_error = type(error).__name__
            logger.exception("failed to start the recognition runtime")
            return
        self.scheduler = LatestFrameScheduler(self.controller).start()
        self._subscription_id = self.controller.subscribe(self._on_pipeline_event)
        self._server_info = self._build_server_info()
        self.health.runtime_started = True
        self.health.state = ReadinessState.READY
        self.metrics.ready.set(1)
        self._result_task = asyncio.create_task(self._result_pump())
        logger.info(
            "transport ready",
            extra={
                "model_id": self._model_bundle.id,
                "protocol_version": PROTOCOL_VERSION,
                "max_sessions": self.config.max_sessions,
                "max_fps": self.config.max_fps,
                "origins": "*"
                if self.config.allowed_origins is None
                else sorted(self.config.allowed_origins),
            },
        )

    def _build_server_info(self) -> dict[str, Any]:
        assert self.controller is not None and self._model_bundle is not None
        pipeline = self.controller.pipeline
        hand_classes = getattr(getattr(pipeline, "hand", None), "classes", None)
        if not isinstance(hand_classes, (tuple, list)) or not hand_classes:
            hand_classes = FALLBACK_HAND_CLASSES
        body_labels = ("idle", "jumping", "naruto_run", "bending_left", "bending_right")
        supports_modes = hasattr(pipeline, "set_mode")
        cfg = self.config
        return {
            "model": self._model_bundle.to_dict(),
            "classes": {
                "hand": list(hand_classes),
                "body": list(body_labels),
                "attacks": [attack.name for attack in ATTACKS],
            },
            "capabilities": {
                "initial_mode": getattr(pipeline, "mode", "full"),
                "recognition_modes": ["body_only", "hand_only", "full"] if supports_modes else [],
                "geometry": self._runtime_config.include_geometry,
                "calibration": True,
                # 1.2.0: which decoder produces HAND_SEAL events ("legacy" | "v1"), read
                # from the live pipeline rather than the CLI so it cannot drift from it.
                "hand_decoder": getattr(getattr(pipeline, "hand", None), "decoder_mode", "legacy"),
                "max_fps": cfg.max_fps,
                "max_frame_bytes": MAX_FRAME_PAYLOAD_BYTES,
                "max_frame_width": MAX_FRAME_WIDTH,
                "max_frame_height": MAX_FRAME_HEIGHT,
                "idle_timeout_s": cfg.idle_timeout_s,
                "heartbeat_s": cfg.heartbeat_s,
            },
        }

    async def _on_shutdown(self, _app: web.Application) -> None:
        """Drain: stop admitting sessions, tell every client we are going away,
        and wait (bounded) for their handlers to finish before cleanup frees
        the pipeline underneath them."""
        self.health.state = ReadinessState.SHUTTING_DOWN
        self.metrics.ready.set(0)
        sockets = list(self._sockets)
        logger.info("shutdown: draining connections", extra={"open_connections": len(sockets)})
        if sockets:
            await asyncio.gather(
                *(
                    self._close_socket(ws, CLOSE_CODE_GOING_AWAY, b"server shutting down")
                    for ws in sockets
                ),
                return_exceptions=True,
            )
        deadline = self._clock() + self.config.shutdown_grace_s
        while self._open_connections > 0 and self._clock() < deadline:
            await asyncio.sleep(0.02)
        if self._open_connections > 0:
            logger.warning(
                "shutdown: connections still open after the grace period",
                extra={"open_connections": self._open_connections},
            )

    async def _close_socket(self, ws: web.WebSocketResponse, code: int, message: bytes) -> None:
        with contextlib.suppress(Exception):
            await asyncio.wait_for(ws.close(code=code, message=message), timeout=2.0)

    async def _on_cleanup(self, _app: web.Application) -> None:
        self.health.state = ReadinessState.SHUTTING_DOWN
        self.metrics.ready.set(0)
        loop = asyncio.get_running_loop()
        # Stop the scheduler first so the blocked wait_for_result() call in
        # _result_pump unblocks on its own (returns None once STOPPED)
        # instead of the task cancellation racing a still-running frame.
        # Every step is isolated: a scheduler that will not stop in time must
        # not stop the native landmarkers from being released.
        try:
            if self.scheduler is not None:
                await loop.run_in_executor(None, lambda: self.scheduler.stop(drain=False))
        except Exception:
            logger.exception("scheduler did not stop cleanly")
        if self._result_task is not None:
            self._result_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._result_task
        try:
            if self.controller is not None:
                if self._subscription_id is not None:
                    self.controller.unsubscribe(self._subscription_id)
                await loop.run_in_executor(None, self.controller.close)
                logger.info("recognition runtime closed")
        except Exception:
            logger.exception("recognition runtime did not close cleanly")

    # --- HTTP ------------------------------------------------------------

    async def handle_health(self, _request: web.Request) -> web.Response:
        """Readiness: 200 only once the models are loaded and we are not draining."""
        body = self.health.to_dict()
        body["open_connections"] = self._open_connections
        return web.json_response(body, status=self.health.http_status)

    async def handle_live(self, _request: web.Request) -> web.Response:
        """Liveness: 200 whenever the process can answer at all."""
        return web.json_response(self.health.live_dict())

    async def handle_metrics(self, _request: web.Request) -> web.Response:
        if self.scheduler is not None:
            stats = self.scheduler.stats()
            self.metrics.frames_processed.set_total(stats.processed_frames)
            self.metrics.frames_failed.set_total(stats.failed_frames)
            self.metrics.frames_dropped.set_total(stats.dropped_frames, reason="superseded")
        return web.Response(
            body=self.metrics.registry.render().encode(),
            headers={"Content-Type": CONTENT_TYPE},
        )

    # --- WebSocket ---------------------------------------------------------

    def _origin_allowed(self, request: web.Request) -> bool:
        origin = request.headers.get("Origin")
        if origin is None:
            return not self.config.require_origin
        if self.config.allowed_origins is None:
            return True
        return origin.rstrip("/") in self.config.allowed_origins

    def _reject(self, request: web.Request, status: int, reason: str, text: str) -> web.Response:
        self.metrics.connections_rejected.inc(reason=reason)
        logger.warning(
            "connection rejected",
            extra={
                "reason": reason,
                "status": status,
                "origin": request.headers.get("Origin"),
                "remote": request.remote,
            },
        )
        return web.Response(status=status, text=text)

    async def handle_ws(self, request: web.Request) -> web.StreamResponse:
        if not self._origin_allowed(request):
            return self._reject(request, 403, "origin", "origin not allowed")
        if self.health.state == ReadinessState.SHUTTING_DOWN:
            return self._reject(request, 503, "shutting_down", "server is shutting down")
        if self.controller is None or self.controller.state != RuntimeState.RUNNING:
            return self._reject(request, 503, "not_ready", "recognition runtime is not ready")
        # Checked and taken synchronously (no await between), so concurrent
        # upgrades cannot both slip under the cap. Released in the finally.
        if self._open_connections >= self.config.max_sessions:
            return self._reject(request, 503, "max_sessions", "max_sessions_reached")
        self._open_connections += 1
        self.metrics.open_connections.set(self._open_connections)
        conn_id = uuid.uuid4().hex[:12]
        token = logging_config.connection_id.set(conn_id)
        try:
            return await self._run_connection(request, conn_id)
        finally:
            self._open_connections -= 1
            self.metrics.open_connections.set(self._open_connections)
            logging_config.connection_id.reset(token)

    async def _run_connection(self, request: web.Request, conn_id: str) -> web.StreamResponse:
        cfg = self.config
        ws = web.WebSocketResponse(
            max_msg_size=cfg.max_ws_message_bytes,
            heartbeat=cfg.heartbeat_s or None,
        )
        await ws.prepare(request)
        self._sockets.add(ws)
        logger.info(
            "connection opened",
            extra={"remote": request.remote, "origin": request.headers.get("Origin")},
        )
        close_reason = "client"
        try:
            if self._active_ws is not None:
                await ws.send_json(
                    build_error(
                        code="controller_already_connected",
                        message="another client already controls this session",
                    )
                )
                self.metrics.connections_rejected.inc(reason="controller_busy")
                close_reason = "controller_busy"
                await ws.close(code=CLOSE_CODE_CONTROLLER_TAKEN)
                return ws

            # Reserve the controller slot synchronously, before the first await
            # below. Two clients can both observe `_active_ws is None` if the
            # reservation happens only after an await (e.g. inside _handshake);
            # setting it here, with no await in between, closes that race.
            self._active_ws = ws

            try:
                await self._handshake(ws)
            except (asyncio.TimeoutError, ProtocolError) as error:
                self._active_ws = None
                close_reason = "handshake_failed"
                await self._fail_handshake(ws, error)
                return ws

            session = ClientSession(
                event_queue_size=self.event_queue_size,
                connection_id=conn_id,
                frame_bucket=TokenBucket(cfg.max_fps, cfg.frame_burst, clock=self._clock),
                control_bucket=TokenBucket(cfg.control_rate, cfg.control_burst, clock=self._clock),
                error_bucket=TokenBucket(
                    cfg.error_reply_rate, cfg.error_reply_burst, clock=self._clock
                ),
            )
            session.is_controller = True
            self._session = session
            self.health.connected_clients = 1
            self.health.active_controller = True
            self.metrics.active_sessions.set(1)
            logger.info("session started", extra={"session_id": self.controller.session_id})

            try:
                close_reason = await self._serve_client(ws, session)
            finally:
                self._active_ws = None
                self._session = None
                self.health.connected_clients = 0
                self.health.active_controller = False
                self.metrics.active_sessions.set(0)
                # A reconnect starts fresh: old temporal/attack evidence must
                # never cross into whatever client controls the session next.
                # During shutdown the runtime may already be closing; that
                # must not turn a clean drain into a traceback.
                with contextlib.suppress(Exception):
                    await self._run_controller(
                        lambda: self.controller.reset(reason="client_disconnected")
                    )
            return ws
        finally:
            self._sockets.discard(ws)
            if self.health.state == ReadinessState.SHUTTING_DOWN:
                close_reason = "shutdown"
            self.metrics.connections_closed.inc(reason=close_reason)
            logger.info(
                "connection closed",
                extra={"reason": close_reason, "close_code": ws.close_code},
            )

    async def _fail_handshake(self, ws: web.WebSocketResponse, error: BaseException) -> None:
        if isinstance(error, VersionMismatchError):
            reason, code, close_code = (
                "protocol_version",
                "protocol_version_mismatch",
                CLOSE_CODE_VERSION_MISMATCH,
            )
            details: Optional[dict] = {
                "server_protocol_version": PROTOCOL_VERSION,
                "client_protocol_version": error.client_version[:32],
            }
        elif isinstance(error, ModelMismatchError):
            reason, code, close_code = "model_mismatch", "model_mismatch", CLOSE_CODE_MODEL_MISMATCH
            details = {
                "server_model_id": error.actual,
                "client_expected_model_id": error.expected[:80],
            }
        elif isinstance(error, asyncio.TimeoutError):
            reason, code, close_code, details = (
                "timeout",
                "handshake_failed",
                CLOSE_CODE_PROTOCOL_ERROR,
                None,
            )
        else:
            reason, code, close_code, details = (
                "invalid",
                "handshake_failed",
                CLOSE_CODE_PROTOCOL_ERROR,
                None,
            )
        self.metrics.handshake_failures.inc(reason=reason)
        message = str(error) or "handshake timed out waiting for client_hello"
        logger.warning("handshake failed", extra={"reason": reason, "detail": message})
        await self._safe_send_error(ws, code, message, details=details)
        await ws.close(code=close_code)

    async def _handshake(self, ws: web.WebSocketResponse) -> None:
        msg = await asyncio.wait_for(ws.receive(), timeout=self.config.handshake_timeout_s)
        if msg.type != WSMsgType.TEXT:
            raise ProtocolError("expected a text client_hello as the first message")
        if len(msg.data) > self.config.max_text_message_bytes:
            raise ProtocolError("client_hello is too large")
        try:
            raw = json.loads(msg.data)
        except json.JSONDecodeError as error:
            raise ProtocolError(f"client_hello was not valid JSON: {error}") from error
        if not isinstance(raw, dict):
            raise ProtocolError("client_hello must be a JSON object")
        hello = parse_client_hello_full(raw)
        assert self.controller is not None and self._model_bundle is not None
        negotiate_hello(hello, server_model_id=self._model_bundle.id)
        await ws.send_json(
            build_server_ready(
                session_id=self.controller.session_id,
                model=self._server_info.get("model"),
                classes=self._server_info.get("classes"),
                capabilities=self._server_info.get("capabilities"),
            )
        )

    async def _serve_client(self, ws: web.WebSocketResponse, session: ClientSession) -> str:
        """Run the message loop; returns the reason the connection ended."""
        cfg = self.config
        sender_task = asyncio.create_task(self._sender_loop(ws, session))
        reason = "client"
        try:
            while True:
                try:
                    msg = await ws.receive(timeout=cfg.idle_timeout_s)
                except asyncio.TimeoutError:
                    logger.info("idle timeout", extra={"idle_timeout_s": cfg.idle_timeout_s})
                    await ws.close(code=CLOSE_CODE_IDLE_TIMEOUT, message=b"idle timeout")
                    return "idle_timeout"
                if msg.type == WSMsgType.BINARY:
                    await self._handle_frame(ws, session, msg.data)
                elif msg.type == WSMsgType.TEXT:
                    if len(msg.data) > cfg.max_text_message_bytes:
                        await self._reply_error(
                            ws, session, "invalid_control", "control message too large"
                        )
                    else:
                        await self._handle_control(ws, session, msg.data)
                elif msg.type == WSMsgType.ERROR:
                    error = ws.exception()
                    if ws.close_code == 1009 or getattr(error, "code", None) == 1009:
                        reason = "message_too_big"
                    logger.warning(
                        "websocket error",
                        extra={"close_code": ws.close_code, "error": repr(ws.exception())},
                    )
                    break
                elif msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSING, WSMsgType.CLOSED):
                    break
                if session.invalid_messages >= cfg.max_invalid_messages:
                    logger.warning("too many invalid messages; closing")
                    await ws.close(
                        code=CLOSE_CODE_PROTOCOL_ERROR, message=b"too many invalid messages"
                    )
                    return "invalid_messages"
            if ws.close_code == CLOSE_CODE_BACKPRESSURE:
                return "backpressure"
            return reason
        finally:
            sender_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await sender_task

    async def _run_controller(self, call: Callable[[], Any]) -> Any:
        """Run a blocking PipelineRuntimeController call off the event loop.

        process_frame holds the same operation lock on the scheduler worker
        thread for the duration of one frame (~30-40 ms); running control
        commands inline on the event loop thread would stall every other
        connection's I/O for that long.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, call)

    async def _reply_error(
        self,
        ws: web.WebSocketResponse,
        session: ClientSession,
        code: str,
        message: str,
    ) -> None:
        """Count a bad message and answer it, within the per-connection error cap."""
        session.invalid_messages += 1
        bucket = session.error_bucket
        if bucket is not None and not bucket.try_consume():
            self.metrics.errors_suppressed.inc()
            return
        await self._safe_send_error(ws, code, message)

    async def _handle_frame(
        self, ws: web.WebSocketResponse, session: ClientSession, data: bytes
    ) -> None:
        assert self.scheduler is not None
        self.metrics.frames_received.inc()
        # Cheapest check first and silent: an error reply per dropped frame
        # would turn the cap into a traffic amplifier.
        if session.frame_bucket is not None and not session.frame_bucket.try_consume():
            self.metrics.frames_dropped.inc(reason="rate_limited")
            return
        try:
            envelope = decode_frame(data)
            session.frame_guard.accept(envelope.sequence)
        except StaleFrameError as error:
            self.metrics.frames_dropped.inc(reason="stale_sequence")
            await self._reply_error(ws, session, "invalid_frame", str(error))
            return
        except ProtocolError as error:
            self.metrics.frames_dropped.inc(reason="invalid")
            await self._reply_error(ws, session, "invalid_frame", str(error))
            return
        try:
            # JPEG decode is ~5-10 ms of CPU; keep it off the event loop so
            # /live, /metrics and other sockets stay responsive.
            decoded = await self._run_controller(lambda: decode_image(envelope))
        except ProtocolError as error:
            self.metrics.frames_dropped.inc(reason="invalid")
            await self._reply_error(ws, session, "invalid_frame", str(error))
            return
        try:
            self.scheduler.submit(decoded.image, captured_at_ms=decoded.captured_at_ms)
        except Exception as error:  # scheduler not accepting frames right now
            self.metrics.frames_dropped.inc(reason="rejected")
            await self._reply_error(ws, session, "frame_rejected", str(error))

    async def _handle_control(
        self, ws: web.WebSocketResponse, session: ClientSession, raw_text: str
    ) -> None:
        assert self.controller is not None
        if session.control_bucket is not None and not session.control_bucket.try_consume():
            self.metrics.controls_dropped.inc()
            await self._reply_error(ws, session, "rate_limited", "control messages too frequent")
            return
        try:
            payload = json.loads(raw_text)
            if not isinstance(payload, dict):
                raise ProtocolError("control message must be a JSON object")
            command, request_id = parse_control_message(payload)
            if command is ControlCommand.SET_MODE:
                mode = payload.get("mode", "body_only")
                if not isinstance(mode, str) or len(mode) > MAX_MODE_CHARS:
                    raise ProtocolError("mode must be a short string")
        except (json.JSONDecodeError, ProtocolError) as error:
            await self._reply_error(ws, session, "invalid_control", str(error))
            return

        controller = self.controller
        try:
            if command is ControlCommand.PING:
                pass
            elif command is ControlCommand.RESET:
                if self.scheduler is not None:
                    self.scheduler.clear_pending()
                await self._run_controller(lambda: controller.reset(reason="client_requested"))
            elif command is ControlCommand.BEGIN_CALIBRATION:
                if self.scheduler is not None:
                    self.scheduler.clear_pending()
                await self._run_controller(controller.begin_neutral_calibration)
            elif command is ControlCommand.FINISH_CALIBRATION:
                await self._run_controller(controller.finish_neutral_calibration)
            elif command is ControlCommand.CANCEL_CALIBRATION:
                await self._run_controller(controller.cancel_neutral_calibration)
            elif command is ControlCommand.SET_MODE:
                mode = payload.get("mode", "body_only")
                if hasattr(controller.pipeline, "set_mode"):
                    await self._run_controller(lambda: controller.pipeline.set_mode(mode))
        except Exception as error:
            await ws.send_json(
                build_ack(
                    command=command.value,
                    ok=False,
                    request_id=request_id,
                    error=str(error),
                )
            )
            return
        await ws.send_json(build_ack(command=command.value, ok=True, request_id=request_id))

    async def _sender_loop(self, ws: web.WebSocketResponse, session: ClientSession) -> None:
        """Deliver events immediately and snapshots at a capped rate.

        Runs as one task per connection so a slow client stalls only its own
        delivery; scheduler.submit() above never awaits this loop.
        """
        next_snapshot_at = time.monotonic()
        while True:
            now = time.monotonic()
            for event in session.events.pop_all():
                await ws.send_json(build_pipeline_event(event))
            if now >= next_snapshot_at and session.snapshot.has_pending:
                snapshot = session.snapshot.take()
                assert snapshot is not None
                await ws.send_json(snapshot)
                next_snapshot_at = now + SNAPSHOT_INTERVAL_SECONDS
            await asyncio.sleep(SENDER_POLL_INTERVAL_SECONDS)

    async def _result_pump(self) -> None:
        """Bridge the synchronous scheduler thread into the asyncio world.

        The blocking wait runs in the default executor so it never stalls the
        event loop; only the newest scheduled result is kept, matching the
        latest-frame contract the scheduler already enforces on submission.
        """
        assert self.scheduler is not None
        loop = asyncio.get_running_loop()
        while True:
            result = await loop.run_in_executor(
                None, self.scheduler.wait_for_result, RESULT_WAIT_TIMEOUT_SECONDS
            )
            if result is None or result.error is not None or result.runtime_frame is None:
                continue
            self._observe_timings(result)
            session = self._session
            if session is None:
                continue
            output = result.runtime_frame.output
            session.snapshot.replace(
                build_state_snapshot(
                    session_id=output["session_id"],
                    frame_id=output["frame_id"],
                    captured_at_ms=output["captured_at_ms"],
                    output=output,
                )
            )

    def _observe_timings(self, result: Any) -> None:
        """Feed the pipeline's per-stage `timings_ms` into the histograms.

        Stage names are the pipeline's own fixed keys (never client input), so
        the label set stays bounded. Observed on the event loop thread.
        """
        timings = dict(getattr(result.runtime_frame, "timings_ms", None) or {})
        timings["queue_wait_ms"] = result.queue_wait_ms
        timings["worker_elapsed_ms"] = result.worker_elapsed_ms
        timings["end_to_end_ms"] = result.end_to_end_ms
        for stage, value in timings.items():
            if isinstance(stage, str) and isinstance(value, (int, float)):
                self.metrics.stage_latency.observe(float(value), stage=stage)

    def _on_pipeline_event(self, event: dict) -> None:
        """Runs synchronously on the scheduler worker thread (see
        PipelineEventDispatcher._deliver). Hands off via call_soon_threadsafe
        so the bounded queue is only ever touched on the event loop thread.
        """
        session = self._session
        loop = self._loop
        if session is None or loop is None:
            return
        loop.call_soon_threadsafe(self._enqueue_event, session, event)

    def _enqueue_event(self, session: ClientSession, event: dict) -> None:
        try:
            session.events.push(event)
        except EventQueueOverflow:
            logger.warning(
                "client event queue overflowed; disconnecting client",
                extra={"conn": session.connection_id},
            )
            ws = self._active_ws
            if ws is not None and not ws.closed:
                asyncio.ensure_future(ws.close(code=CLOSE_CODE_BACKPRESSURE))

    async def _safe_send_error(
        self,
        ws: web.WebSocketResponse,
        code: str,
        message: str,
        *,
        details: Optional[dict] = None,
    ) -> None:
        if not ws.closed:
            with contextlib.suppress(ConnectionError):
                await ws.send_json(build_error(code=code, message=message, details=details))


def create_app(
    *,
    pipeline_factory: Optional[PipelineFactory] = None,
    runtime_config: Optional[RuntimeConfig] = None,
    allowed_origins: Optional[FrozenSet[str]] = _UNSET,
    body_only: bool = False,
    config: Optional[TransportConfig] = None,
    model_bundle: Optional[ModelBundle] = None,
    pipeline_config: Optional[PipelineConfig] = None,
) -> web.Application:
    server = TransportServer(
        pipeline_factory=pipeline_factory,
        runtime_config=runtime_config,
        allowed_origins=allowed_origins,
        body_only=body_only,
        config=config,
        model_bundle=model_bundle,
        pipeline_config=pipeline_config,
    )
    app = server.build_app()
    app[TRANSPORT_SERVER_KEY] = server
    return app


__all__ = [
    "CLOSE_CODE_BACKPRESSURE",
    "CLOSE_CODE_CONTROLLER_TAKEN",
    "CLOSE_CODE_GOING_AWAY",
    "CLOSE_CODE_IDLE_TIMEOUT",
    "CLOSE_CODE_MODEL_MISMATCH",
    "CLOSE_CODE_PROTOCOL_ERROR",
    "CLOSE_CODE_VERSION_MISMATCH",
    "DEFAULT_ALLOWED_ORIGINS",
    "FATAL_CLOSE_CODES",
    "TRANSPORT_SERVER_KEY",
    "TransportServer",
    "create_app",
]
