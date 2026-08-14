#!/usr/bin/env python3
"""Launcher for the local browser<->Python transport bridge (Phase A).

Binds to 127.0.0.1 by default -- external-network binding is not part of the
step-6 release (context.md §9.1). Run from anywhere; this script puts the
repository root on sys.path itself so `cv_model.inference` and
`cv_model.Nishit_Frontend.transport` both resolve regardless of cwd.

Usage:
    Nishit_Frontend/venv/Scripts/python serve_transport.py
    Nishit_Frontend/venv/Scripts/python serve_transport.py --port 8765 --allow-origin http://localhost:5173
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aiohttp import web  # noqa: E402  (import after sys.path bootstrap)

from cv_model.Nishit_Frontend.transport.server import create_app  # noqa: E402


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
    allowed_origins = (
        frozenset(args.allowed_origins) if args.allowed_origins else None
    )
    if args.body_only:
        logging.getLogger("narutocv.transport").info(
            "body-only mode: hand recognition skipped (roughly double the frame rate)"
        )
    app = create_app(allowed_origins=allowed_origins, body_only=args.body_only)
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
