"""Launcher for the local browser<->Python transport bridge (Phase A).

Binds to 127.0.0.1 by default -- external-network binding is not part of the
step-6 release (docs/archive/context.md section 9.1). Requires the package to
be installed (`uv sync --extra transport`), after which it runs from any cwd.

Usage:
    python -m narutocv.transport
    narutocv-serve --port 8765 --allow-origin http://localhost:5173
"""

from __future__ import annotations

import argparse
import logging

from aiohttp import web

from narutocv.transport.server import create_app


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="bind address (default: 127.0.0.1, loopback only)",
    )
    parser.add_argument("--port", type=int, default=8765, help="bind port (default: 8765)")
    parser.add_argument(
        "--allow-origin",
        action="append",
        dest="allowed_origins",
        default=None,
        help="allowed browser Origin header; repeat to allow multiple. "
        "Omit to allow any origin (fine for loopback-only local dev).",
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
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    allowed_origins = frozenset(args.allowed_origins) if args.allowed_origins else None
    if args.body_only:
        logging.getLogger("narutocv.transport").info(
            "body-only mode: hand recognition skipped (roughly double the frame rate)"
        )
    app = create_app(allowed_origins=allowed_origins, body_only=args.body_only)
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
