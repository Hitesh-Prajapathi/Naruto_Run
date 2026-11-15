"""Tunable limits and policy for the transport service.

Everything that bounds what a client can make the server do lives here, so it
is visible in one place, settable from the CLI or environment, and overridable
per test. Environment variables are read by `from_env`; CLI flags (see
`__main__.py`) take precedence over them.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Optional

from .protocol import FRAME_HEADER_SIZE, MAX_FRAME_PAYLOAD_BYTES

# The Vite dev/preview servers and the Playwright harness all use 127.0.0.1:5173
# (apps/web/vite.config.ts, playwright.config.ts); `localhost` is the same host
# typed differently in the address bar. Preview defaults to 4173.
DEFAULT_ALLOWED_ORIGINS: frozenset[str] = frozenset(
    {
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://127.0.0.1:4173",
        "http://localhost:4173",
    }
)

# Largest legitimate binary message (header + max JPEG) plus a small margin for
# future header growth. Anything larger is closed by aiohttp with code 1009
# before the payload is buffered further.
MAX_WS_MESSAGE_BYTES = FRAME_HEADER_SIZE + MAX_FRAME_PAYLOAD_BYTES + 64 * 1024
# Text messages are hello/control JSON, a few hundred bytes at most.
MAX_TEXT_MESSAGE_BYTES = 16 * 1024

ANY_ORIGIN = "*"


def parse_origins(raw: Optional[str]) -> Optional[frozenset[str]]:
    """Parse a comma-separated origin list. `*` anywhere means "allow any"
    (returned as None). Returns the default list for an empty/None value."""
    if raw is None or not raw.strip():
        return DEFAULT_ALLOWED_ORIGINS
    origins = {part.strip().rstrip("/") for part in raw.split(",") if part.strip()}
    if ANY_ORIGIN in origins:
        return None
    return frozenset(origins)


def _env_float(env: Mapping[str, str], name: str, default: float) -> float:
    raw = env.get(name)
    return float(raw) if raw not in (None, "") else default


def _env_int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = env.get(name)
    return int(raw) if raw not in (None, "") else default


def _env_bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw = env.get(name)
    if raw in (None, ""):
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class TransportConfig:
    # --- origin policy ---------------------------------------------------
    #: None means any origin (explicit "*"). Otherwise a request carrying an
    #: Origin header that is not listed gets HTTP 403.
    allowed_origins: Optional[frozenset[str]] = DEFAULT_ALLOWED_ORIGINS
    #: Browsers always send Origin on a WebSocket upgrade, so a *missing*
    #: header means a non-browser client (the diagnostics script, curl, tests),
    #: which cross-site hijacking cannot use. Set True to reject those too.
    require_origin: bool = False

    # --- size limits -------------------------------------------------------
    max_ws_message_bytes: int = MAX_WS_MESSAGE_BYTES
    max_text_message_bytes: int = MAX_TEXT_MESSAGE_BYTES

    # --- rate limits (token buckets, per connection) -----------------------
    #: Sustained frames/second accepted. The web client publishes at 30 fps,
    #: so the default is 2x that; frames over the cap are dropped and counted,
    #: never answered with an error and never grounds for disconnecting.
    max_fps: float = 60.0
    #: Frames allowed in an instantaneous burst before the cap bites.
    frame_burst: int = 30
    #: Control messages (reset/calibration/ping) per second and burst.
    control_rate: float = 10.0
    control_burst: int = 20
    #: Error replies per second sent to one client; the rest are suppressed so a
    #: bad client cannot make the server a traffic amplifier.
    error_reply_rate: float = 5.0
    error_reply_burst: int = 10
    #: A connection that sends this many invalid messages is closed (4400).
    max_invalid_messages: int = 100

    # --- connection limits -------------------------------------------------
    #: Concurrent upgraded WebSocket connections (see docs/transport.md: there is
    #: still exactly one *controlling* client; this bounds how many sockets can
    #: exist at once, including ones about to be told "controller busy").
    max_sessions: int = 2
    handshake_timeout_s: float = 5.0
    #: Close a connection that sends nothing (frames, controls, pings) for this
    #: long. Must comfortably exceed the web client's 15 s application ping.
    idle_timeout_s: float = 60.0
    #: Protocol-level WebSocket ping interval; a missing pong closes the socket.
    heartbeat_s: float = 15.0
    #: How long shutdown waits for open connections to close before giving up.
    shutdown_grace_s: float = 5.0

    def __post_init__(self) -> None:
        if self.max_sessions < 1:
            raise ValueError("max_sessions must be at least 1")
        if self.max_fps <= 0 or self.frame_burst < 1:
            raise ValueError("max_fps must be positive and frame_burst at least 1")
        if self.control_rate <= 0 or self.control_burst < 1:
            raise ValueError("control_rate must be positive and control_burst at least 1")
        if self.idle_timeout_s <= 0 or self.handshake_timeout_s <= 0:
            raise ValueError("timeouts must be positive")
        if self.heartbeat_s < 0:
            raise ValueError("heartbeat_s cannot be negative (0 disables it)")

    def with_overrides(self, **overrides) -> TransportConfig:
        return replace(self, **{k: v for k, v in overrides.items() if v is not None})

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> TransportConfig:
        env = os.environ if env is None else env
        defaults = cls()
        return cls(
            allowed_origins=parse_origins(env.get("NARUTOCV_ALLOWED_ORIGINS")),
            require_origin=_env_bool(env, "NARUTOCV_REQUIRE_ORIGIN", defaults.require_origin),
            max_fps=_env_float(env, "NARUTOCV_MAX_FPS", defaults.max_fps),
            frame_burst=_env_int(env, "NARUTOCV_FRAME_BURST", defaults.frame_burst),
            max_sessions=_env_int(env, "NARUTOCV_MAX_SESSIONS", defaults.max_sessions),
            idle_timeout_s=_env_float(env, "NARUTOCV_IDLE_TIMEOUT", defaults.idle_timeout_s),
            heartbeat_s=_env_float(env, "NARUTOCV_HEARTBEAT", defaults.heartbeat_s),
            shutdown_grace_s=_env_float(env, "NARUTOCV_SHUTDOWN_GRACE", defaults.shutdown_grace_s),
        )
