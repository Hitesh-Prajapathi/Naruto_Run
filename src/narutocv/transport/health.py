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
    #: Exception class name only: the full message (which can contain file
    #: paths) goes to the structured log, not to an unauthenticated endpoint.
    last_error: Optional[str] = None
    model_id: Optional[str] = None
    protocol_version: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "protocol_version": self.protocol_version,
            "state": self.state.value,
            "runtime_started": self.runtime_started,
            "active_controller": self.active_controller,
            "connected_clients": self.connected_clients,
            "last_error": self.last_error,
        }

    def live_dict(self) -> dict:
        """Liveness payload for GET /live: the process and event loop are
        responding. Deliberately 200 even while degraded or draining, so an
        orchestrator restarts on a hang, not on a model that failed to load
        (that is /health's job)."""
        return {"status": "alive", "state": self.state.value}

    @property
    def http_status(self) -> int:
        """503 while starting/degraded so orchestration/dev scripts can poll
        this endpoint instead of guessing a startup delay."""
        return 200 if self.state == ReadinessState.READY else 503
