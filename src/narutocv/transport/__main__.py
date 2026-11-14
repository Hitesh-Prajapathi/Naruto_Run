"""Launcher for the local browser<->Python transport bridge (Phase A).

Binds to 127.0.0.1 by default -- external-network binding is not part of the
step-6 release (docs/archive/context.md section 9.1). Requires the package to
be installed (`uv sync --extra transport`), after which it runs from any cwd.

Every limit below can also be set by environment variable (flag wins):
NARUTOCV_ALLOWED_ORIGINS, NARUTOCV_REQUIRE_ORIGIN, NARUTOCV_MAX_FPS,
NARUTOCV_FRAME_BURST, NARUTOCV_MAX_SESSIONS, NARUTOCV_IDLE_TIMEOUT,
NARUTOCV_HEARTBEAT, NARUTOCV_SHUTDOWN_GRACE, NARUTOCV_HOST, NARUTOCV_PORT,
NARUTOCV_LOG_LEVEL, NARUTOCV_LOG_FORMAT, NARUTOCV_MODEL_MANIFEST. The hand-seal decoder
(`--hand-decoder legacy|v1`, default legacy) follows the same rule via NARUTOCV_HAND_DECODER
(and NARUTOCV_HAND_DECODER_PRIOR for the v1 grammar prior).

Usage:
    python -m narutocv.transport
    narutocv-serve --port 8765 --allow-origin http://localhost:5173
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
from collections.abc import Mapping

from aiohttp import web

from narutocv.inference.hand_decoder import (
    add_hand_decoder_arguments,
    pipeline_config_from_args,
)
from narutocv.transport.config import TransportConfig, parse_origins
from narutocv.transport.logging_config import configure_logging
from narutocv.transport.server import create_app


def parse_args(
    argv: list[str] | None = None, env: Mapping[str, str] | None = None
) -> argparse.Namespace:
    env = os.environ if env is None else env
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host",
        default=env.get("NARUTOCV_HOST", "127.0.0.1"),
        help="bind address (default: 127.0.0.1, loopback only)",
    )
    parser.add_argument(
        "--port", type=int, default=int(env.get("NARUTOCV_PORT", "8765")), help="bind port"
    )
    parser.add_argument(
        "--allow-origin",
        action="append",
        dest="allowed_origins",
        default=None,
        help="allowed browser Origin header; repeat to allow multiple, or '*' for any. "
        "Default: the local Vite dev/preview origins (127.0.0.1/localhost on 5173 and "
        "4173). Overrides NARUTOCV_ALLOWED_ORIGINS (comma separated).",
    )
    parser.add_argument(
        "--require-origin",
        action="store_true",
        default=None,
        help="also reject upgrades that carry no Origin header (non-browser clients)",
    )
    parser.add_argument("--max-fps", type=float, default=None, help="per-connection frame cap")
    parser.add_argument("--frame-burst", type=int, default=None, help="frame cap burst size")
    parser.add_argument(
        "--max-sessions", type=int, default=None, help="max concurrent WebSocket connections"
    )
    parser.add_argument(
        "--idle-timeout", type=float, default=None, help="close silent connections after N seconds"
    )
    parser.add_argument(
        "--heartbeat", type=float, default=None, help="WebSocket ping interval seconds (0 = off)"
    )
    parser.add_argument(
        "--shutdown-grace",
        type=float,
        default=None,
        help="seconds to wait for connections to drain on SIGTERM",
    )
    parser.add_argument(
        "--body-only",
        action="store_true",
        help="skip hand recognition. The browser game reads only body movement, "
        "and hand inference is over half the per-frame cost -- skipping it roughly "
        "doubles the frame rate and halves input latency. Leave off if you are "
        "using the diagnostics page, which does show hand signs.",
    )
    parser.add_argument(
        "--log-level",
        default=env.get("NARUTOCV_LOG_LEVEL", "INFO"),
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    parser.add_argument(
        "--log-format",
        default=env.get("NARUTOCV_LOG_FORMAT", "json"),
        choices=("json", "text"),
        help="structured JSON lines (default) or human-readable text",
    )
    add_hand_decoder_arguments(parser)
    args = parser.parse_args(argv)
    args.pipeline_config = pipeline_config_from_args(args, parser, env)
    return args


def build_config(args: argparse.Namespace, env: Mapping[str, str] | None = None) -> TransportConfig:
    base = TransportConfig.from_env(env)
    if args.allowed_origins:
        base = dataclasses.replace(
            base, allowed_origins=parse_origins(",".join(args.allowed_origins))
        )
    return base.with_overrides(
        require_origin=args.require_origin,
        max_fps=args.max_fps,
        frame_burst=args.frame_burst,
        max_sessions=args.max_sessions,
        idle_timeout_s=args.idle_timeout,
        heartbeat_s=args.heartbeat,
        shutdown_grace_s=args.shutdown_grace,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    configure_logging(args.log_level, args.log_format)
    log = logging.getLogger("narutocv.transport")
    config = build_config(args)
    if args.body_only:
        log.info("body-only mode: hand recognition skipped (roughly double the frame rate)")
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        log.warning(
            "binding to a non-loopback address; put a TLS-terminating, authenticating proxy "
            "in front and keep an explicit origin allow-list",
            extra={"host": args.host},
        )
    pipeline_config = args.pipeline_config
    if pipeline_config.hand_decoder == "v1":
        log.warning(
            "hand decoder v1 is selected: untested on a real camera and its thresholds are "
            "an identity placeholder (models/calibration_v0.json)",
            extra={"use_prior": pipeline_config.hand_decoder_use_prior},
        )
    app = create_app(config=config, body_only=args.body_only, pipeline_config=pipeline_config)
    web.run_app(
        app,
        host=args.host,
        port=args.port,
        # The server's own drain (app.on_shutdown) bounds shutdown; this is the
        # backstop for handlers that ignore it.
        shutdown_timeout=config.shutdown_grace_s + 2.0,
        access_log=None,
        print=None,
    )


if __name__ == "__main__":
    main()
