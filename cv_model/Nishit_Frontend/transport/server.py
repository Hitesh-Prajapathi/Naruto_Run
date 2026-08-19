"""Loopback HTTP + WebSocket transport around PipelineRuntimeController.

Implements the step-6 contract from game_implementation_plan.md and
context.md §9.1: the browser owns the camera and streams unmirrored JPEG
frames in over one binary WebSocket; this server returns capped state
snapshots and immediately-delivered ordered sparse events out. Every frame is
handed to the existing, locked PipelineRuntimeController/LatestFrameScheduler
unchanged -- this module carries bytes, it never decides what a gesture is.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from typing import Any, Callable, FrozenSet, Optional

from aiohttp import WSMsgType, web

from cv_model.inference.runtime import (
    PipelineFactory,
    PipelineRuntimeController,
    RuntimeConfig,
    RuntimeState,
)
from cv_model.inference.scheduler import LatestFrameScheduler

from .connection import ClientSession, EventQueueOverflow, StaleFrameError
from .frame_decoder import decode_image
from .pipeline import session_scoped_pipeline_factory
from .health import HealthStatus, ReadinessState
from .protocol import (
    ControlCommand,
    ProtocolError,
    build_ack,
    build_error,
    build_pipeline_event,
    build_server_ready,
    build_state_snapshot,
    decode_frame,
    parse_client_hello,
    parse_control_message,
)

logger = logging.getLogger("narutocv.transport")

# Typed as Any rather than AppKey["TransportServer"]: TransportServer is
# defined later in this module and `from __future__ import annotations`
# only defers annotation evaluation, not this assignment's RHS.
TRANSPORT_SERVER_KEY: "web.AppKey[Any]" = web.AppKey("transport_server")

HANDSHAKE_TIMEOUT_SECONDS = 5.0
SNAPSHOT_INTERVAL_SECONDS = 1.0 / 15.0  # 15 Hz cap, per context.md step-6
SENDER_POLL_INTERVAL_SECONDS = 0.01
RESULT_WAIT_TIMEOUT_SECONDS = 0.5
MAX_WS_MESSAGE_BYTES = 4 * 1024 * 1024  # headroom above MAX_FRAME_PAYLOAD_BYTES

CLOSE_CODE_PROTOCOL_ERROR = 4400
CLOSE_CODE_CONTROLLER_TAKEN = 4409
CLOSE_CODE_BACKPRESSURE = 4429


class TransportServer:
    """Owns exactly one PipelineRuntimeController and one controlling client.

    Multi-client ownership is explicitly out of scope for step 6: a second
    connection while one is controlling is rejected rather than silently
    mixed in (context.md: "Keep one recognition producer.").
    """

    def __init__(
        self,
        *,
        pipeline_factory: Optional[PipelineFactory] = None,
        runtime_config: Optional[RuntimeConfig] = None,
        allowed_origins: Optional[FrozenSet[str]] = None,
        event_queue_size: int = 256,
        body_only: bool = False,
    ) -> None:
        # Defaults to the session-scoped pipeline so that a reset re-detects
        # the player instead of staying locked onto whoever was in front of
        # the camera first. See transport/pipeline.py.
        self._pipeline_factory = pipeline_factory or session_scoped_pipeline_factory(
            body_only=body_only
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
        self.allowed_origins = allowed_origins
        self.event_queue_size = event_queue_size

        self.health = HealthStatus()
        self.controller: Optional[PipelineRuntimeController] = None
        self.scheduler: Optional[LatestFrameScheduler] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._session: Optional[ClientSession] = None
        self._active_ws: Optional[web.WebSocketResponse] = None
        self._result_task: Optional[asyncio.Task] = None
        self._subscription_id: Optional[int] = None

    # --- lifecycle -----------------------------------------------------

    def build_app(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/health", self.handle_health)
        app.router.add_get("/ws", self.handle_ws)
        app.on_startup.append(self._on_startup)
        app.on_cleanup.append(self._on_cleanup)
        return app

    async def _on_startup(self, _app: web.Application) -> None:
        self._loop = asyncio.get_running_loop()
        try:
            self.controller = PipelineRuntimeController(
                pipeline_factory=self._pipeline_factory,
                config=self._runtime_config,
            ).start()
        except Exception as error:  # keep the process alive; /health reports it
            self.health.state = ReadinessState.DEGRADED
            self.health.last_error = str(error)
            logger.exception("failed to start the recognition runtime")
            return
        self.scheduler = LatestFrameScheduler(self.controller).start()
        self._subscription_id = self.controller.subscribe(self._on_pipeline_event)
        self.health.runtime_started = True
        self.health.state = ReadinessState.READY
        self._result_task = asyncio.create_task(self._result_pump())

    async def _on_cleanup(self, _app: web.Application) -> None:
        self.health.state = ReadinessState.SHUTTING_DOWN
        # Stop the scheduler first so the blocked wait_for_result() call in
        # _result_pump unblocks on its own (returns None once STOPPED)
        # instead of the task cancellation racing a still-running frame.
        if self.scheduler is not None:
            self.scheduler.stop(drain=False)
        if self._result_task is not None:
            self._result_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._result_task
        if self.controller is not None:
            self.controller.close()

    # --- HTTP ------------------------------------------------------------

    async def handle_health(self, _request: web.Request) -> web.Response:
        return web.json_response(self.health.to_dict(), status=self.health.http_status)

    # --- WebSocket ---------------------------------------------------------

    def _origin_allowed(self, request: web.Request) -> bool:
        if self.allowed_origins is None:
            return True
        return request.headers.get("Origin") in self.allowed_origins

    async def handle_ws(self, request: web.Request) -> web.WebSocketResponse:
        if not self._origin_allowed(request):
            return web.Response(status=403, text="origin not allowed")
        if self.controller is None or self.controller.state != RuntimeState.RUNNING:
            return web.Response(status=503, text="recognition runtime is not ready")

        ws = web.WebSocketResponse(max_msg_size=MAX_WS_MESSAGE_BYTES)
        await ws.prepare(request)

        if self._active_ws is not None:
            await ws.send_json(
                build_error(
                    code="controller_already_connected",
                    message="another client already controls this session",
                )
            )
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
            await self._safe_send_error(ws, "handshake_failed", str(error))
            await ws.close(code=CLOSE_CODE_PROTOCOL_ERROR)
            return ws

        session = ClientSession(event_queue_size=self.event_queue_size)
        session.is_controller = True
        self._session = session
        self.health.connected_clients = 1
        self.health.active_controller = True

        try:
            await self._serve_client(ws, session)
        finally:
            self._active_ws = None
            self._session = None
            self.health.connected_clients = 0
            self.health.active_controller = False
            # A reconnect starts fresh: old temporal/attack evidence must
            # never cross into whatever client controls the session next.
            await self._run_controller(
                lambda: self.controller.reset(reason="client_disconnected")
            )
        return ws

    async def _handshake(self, ws: web.WebSocketResponse) -> None:
        msg = await asyncio.wait_for(ws.receive(), timeout=HANDSHAKE_TIMEOUT_SECONDS)
        if msg.type != WSMsgType.TEXT:
            raise ProtocolError("expected a text client_hello as the first message")
        try:
            raw = json.loads(msg.data)
        except json.JSONDecodeError as error:
            raise ProtocolError(f"client_hello was not valid JSON: {error}") from error
        parse_client_hello(raw)
        assert self.controller is not None
        await ws.send_json(build_server_ready(session_id=self.controller.session_id))

    async def _serve_client(
        self, ws: web.WebSocketResponse, session: ClientSession
    ) -> None:
        sender_task = asyncio.create_task(self._sender_loop(ws, session))
        try:
            async for msg in ws:
                if msg.type == WSMsgType.BINARY:
                    await self._handle_frame(ws, session, msg.data)
                elif msg.type == WSMsgType.TEXT:
                    await self._handle_control(ws, msg.data)
                elif msg.type in (WSMsgType.ERROR, WSMsgType.CLOSE, WSMsgType.CLOSING):
                    break
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

    async def _handle_frame(
        self, ws: web.WebSocketResponse, session: ClientSession, data: bytes
    ) -> None:
        assert self.scheduler is not None
        try:
            envelope = decode_frame(data)
            session.frame_guard.accept(envelope.sequence)
        except (ProtocolError, StaleFrameError) as error:
            await self._safe_send_error(ws, "invalid_frame", str(error))
            return
        try:
            decoded = decode_image(envelope)
        except ProtocolError as error:
            await self._safe_send_error(ws, "invalid_frame", str(error))
            return
        try:
            self.scheduler.submit(decoded.image, captured_at_ms=decoded.captured_at_ms)
        except Exception as error:  # scheduler not accepting frames right now
            await self._safe_send_error(ws, "frame_rejected", str(error))

    async def _handle_control(self, ws: web.WebSocketResponse, raw_text: str) -> None:
        assert self.controller is not None
        try:
            payload = json.loads(raw_text)
            command, request_id = parse_control_message(payload)
        except (json.JSONDecodeError, ProtocolError) as error:
            await self._safe_send_error(ws, "invalid_control", str(error))
            return

        controller = self.controller
        try:
            if command is ControlCommand.PING:
                pass
            elif command is ControlCommand.RESET:
                if self.scheduler is not None:
                    self.scheduler.clear_pending()
                await self._run_controller(
                    lambda: controller.reset(reason="client_requested")
                )
            elif command is ControlCommand.BEGIN_CALIBRATION:
                if self.scheduler is not None:
                    self.scheduler.clear_pending()
                await self._run_controller(controller.begin_neutral_calibration)
            elif command is ControlCommand.FINISH_CALIBRATION:
                await self._run_controller(controller.finish_neutral_calibration)
            elif command is ControlCommand.CANCEL_CALIBRATION:
                await self._run_controller(controller.cancel_neutral_calibration)
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

    async def _sender_loop(
        self, ws: web.WebSocketResponse, session: ClientSession
    ) -> None:
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
            logger.warning("client event queue overflowed; disconnecting client")
            ws = self._active_ws
            if ws is not None and not ws.closed:
                asyncio.ensure_future(ws.close(code=CLOSE_CODE_BACKPRESSURE))

    async def _safe_send_error(
        self, ws: web.WebSocketResponse, code: str, message: str
    ) -> None:
        if not ws.closed:
            await ws.send_json(build_error(code=code, message=message))


def create_app(
    *,
    pipeline_factory: Optional[PipelineFactory] = None,
    runtime_config: Optional[RuntimeConfig] = None,
    allowed_origins: Optional[FrozenSet[str]] = None,
    body_only: bool = False,
) -> web.Application:
    server = TransportServer(
        pipeline_factory=pipeline_factory,
        runtime_config=runtime_config,
        allowed_origins=allowed_origins,
        body_only=body_only,
    )
    app = server.build_app()
    app[TRANSPORT_SERVER_KEY] = server
    return app
