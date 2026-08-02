"""Validated runtime configuration for the backend recognition pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class HandTemporalConfig:
    window_size: int = 5
    required_votes: int = 3
    neutral_frames: int = 2
    calibrated_required_votes: int = 4

    def __post_init__(self) -> None:
        if self.window_size < 1:
            raise ValueError("hand window_size must be positive")
        if not 1 <= self.required_votes <= self.window_size:
            raise ValueError("hand required_votes must be within window_size")
        if not 1 <= self.neutral_frames <= self.window_size:
            raise ValueError("hand neutral_frames must be within window_size")
        if not self.required_votes <= self.calibrated_required_votes <= self.window_size:
            raise ValueError("calibrated hand votes must be within the evidence window")


@dataclass(frozen=True)
class BodyMovementConfig:
    consensus_window: int = 5
    consensus_votes: int = 3
    jump_history_frames: int = 30
    jump_baseline_frames: int = 15
    jump_height_threshold: float = 0.045
    torso_angle_threshold: float = 25.0
    arms_back_depth_threshold: float = 0.05
    lean_threshold: float = 0.06

    def __post_init__(self) -> None:
        if self.consensus_window < 1:
            raise ValueError("body consensus_window must be positive")
        if not 1 <= self.consensus_votes <= self.consensus_window:
            raise ValueError("body consensus_votes must be within consensus_window")
        if self.jump_history_frames < self.jump_baseline_frames or self.jump_baseline_frames < 1:
            raise ValueError("jump history must contain the baseline window")
        for name in (
            "jump_height_threshold",
            "torso_angle_threshold",
            "arms_back_depth_threshold",
            "lean_threshold",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class AttackQueueConfig:
    seal_timeout: float = 3.0
    trigger_cooldown: float = 2.0
    max_seals: int = 3
    max_noise_events: int = 1

    def __post_init__(self) -> None:
        if self.seal_timeout <= 0:
            raise ValueError("seal_timeout must be positive")
        if self.trigger_cooldown < 0:
            raise ValueError("trigger_cooldown cannot be negative")
        if self.max_seals < 1:
            raise ValueError("max_seals must be positive")
        if not 0 <= self.max_noise_events < self.max_seals:
            raise ValueError("max_noise_events must be between zero and max_seals - 1")


@dataclass(frozen=True)
class PipelineConfig:
    hand: HandTemporalConfig = field(default_factory=HandTemporalConfig)
    body: BodyMovementConfig = field(default_factory=BodyMovementConfig)
    attacks: AttackQueueConfig = field(default_factory=AttackQueueConfig)


DEFAULT_PIPELINE_CONFIG = PipelineConfig()
