"""Process/model/runtime readiness reporting for the GET /health endpoint."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ReadinessState(str, Enum):
    STARTING = "starting"
    READY = "ready"
    DEGRADED = "degraded"
    SHUTTING_DOWN = "shutting_down"


@dataclass
class HealthStatus:
    """Mutable, single-writer health snapshot owned by the transport server."""

    state: ReadinessState = ReadinessState.STARTING
    runtime_started: bool = False
    active_controller: bool = False
    connected_clients: int = 0
    last_error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "state": self.state.value,
            "runtime_started": self.runtime_started,
            "active_controller": self.active_controller,
            "connected_clients": self.connected_clients,
            "last_error": self.last_error,
        }

    @property
    def http_status(self) -> int:
        """503 while starting/degraded so orchestration/dev scripts can poll
        this endpoint instead of guessing a startup delay."""
        return 200 if self.state == ReadinessState.READY else 503
