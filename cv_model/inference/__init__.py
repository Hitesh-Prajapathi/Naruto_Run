"""Reusable inference components for the NarutoCV camera pipeline.

Import concrete components from their modules so lightweight tools such as the
accuracy reporter do not initialize MediaPipe as a side effect.
"""

from .output_schema import (
    SCHEMA_VERSION,
    PipelineOutputSerializer,
    PipelineOutputV1,
    SerializationError,
    validate_pipeline_output_v1,
)
from .events import (
    DispatchFailure,
    DispatchReport,
    PipelineEventDispatcher,
    PipelineEventType,
    PipelineEventV1,
)
from .runtime import (
    CalibrationOutcome,
    PipelineRuntimeController,
    RuntimeConfig,
    RuntimeFrame,
    RuntimeState,
    RuntimeStateError,
)
from .scheduler import (
    LatestFrameScheduler,
    ScheduledResult,
    SchedulerState,
    SchedulerStateError,
    SchedulerStats,
)

__all__ = (
    "SCHEMA_VERSION",
    "PipelineOutputSerializer",
    "PipelineOutputV1",
    "SerializationError",
    "validate_pipeline_output_v1",
    "DispatchFailure",
    "DispatchReport",
    "PipelineEventDispatcher",
    "PipelineEventType",
    "PipelineEventV1",
    "CalibrationOutcome",
    "PipelineRuntimeController",
    "RuntimeConfig",
    "RuntimeFrame",
    "RuntimeState",
    "RuntimeStateError",
    "LatestFrameScheduler",
    "ScheduledResult",
    "SchedulerState",
    "SchedulerStateError",
    "SchedulerStats",
)
