"""Structured (JSON-lines) logging for the transport service, stdlib only.

Each record is one JSON object per line: `ts`, `level`, `logger`, `msg`, the
per-connection id as `conn` when there is one, any `extra={...}` fields, and
`exc` for exceptions. Client-controlled strings are length-capped and JSON
escaped, so a hostile client cannot forge log lines or flood the log with one
giant message.

The connection id lives in a ContextVar set by the WebSocket handler. It does
not propagate into executor/scheduler threads, so code on those threads passes
`extra={"conn": ...}` explicitly.
"""

from __future__ import annotations

import contextvars
import json
import logging
import sys
import time
from typing import Any, Optional

connection_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "narutocv_connection_id", default=None
)

MAX_FIELD_CHARS = 500
MAX_EXC_CHARS = 4000

# Attributes every LogRecord has; anything else came from `extra=`.
_STANDARD_ATTRS = frozenset(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys()
    | {"message", "asctime", "taskName"}
)


def _clip(value: Any, limit: int = MAX_FIELD_CHARS) -> Any:
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + "...(truncated)"
    return value


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": _clip(record.getMessage()),
        }
        conn = getattr(record, "conn", None) or connection_id.get()
        if conn:
            payload["conn"] = conn
        for key, value in record.__dict__.items():
            if key in _STANDARD_ATTRS or key == "conn" or key.startswith("_"):
                continue
            payload[key] = _clip(value)
        if record.exc_info:
            payload["exc"] = _clip(self.formatException(record.exc_info), MAX_EXC_CHARS)
        return json.dumps(payload, default=str, separators=(",", ":"))


class TextFormatter(logging.Formatter):
    """Human-readable fallback for local development (`--log-format text`)."""

    def format(self, record: logging.LogRecord) -> str:
        conn = getattr(record, "conn", None) or connection_id.get()
        prefix = f"[{conn}] " if conn else ""
        record.message = prefix + record.getMessage()
        base = f"{self.formatTime(record)} {record.levelname} {record.name}: {record.message}"
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


def configure_logging(level: str = "INFO", fmt: str = "json") -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter() if fmt == "json" else TextFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(getattr(logging, level))
