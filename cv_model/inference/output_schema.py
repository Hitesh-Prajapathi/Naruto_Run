"""Versioned, JSON-safe output contract for the NarutoCV recognition pipeline."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Optional, TypedDict

if TYPE_CHECKING:
    from .combined_pipeline import Classification, FrameResult


SCHEMA_VERSION = "1.0.0"


class SerializationError(ValueError):
    """Raised when a pipeline result cannot satisfy the public V1 contract."""


class PredictionOutputV1(TypedDict):
    label: str
    confidence: float
    second_label: str
    second_confidence: float
    margin: float


class HandGeometryV1(TypedDict):
    classification_bbox: list[int]
    hand_bbox: Optional[list[int]]
    landmarks: list[list[list[float]]]


class HandOutputV1(TypedDict):
    raw: PredictionOutputV1
    center: PredictionOutputV1
    roi: Optional[PredictionOutputV1]
    accepted_label: str
    rejection_reason: Optional[str]
    stable_label: str
    emitted_seal: Optional[str]
    fusion_status: str
    detected_hand_count: int
    geometry: Optional[HandGeometryV1]


class BodyGeometryV1(TypedDict):
    landmarks: list[list[float]]


class BodyOutputV1(TypedDict):
    raw_label: str
    stable_label: str
    emitted_movement: Optional[str]
    metrics: dict[str, float]
    geometry: Optional[BodyGeometryV1]


class QueueOutputV1(TypedDict):
    seals: list[str]
    accepted_seal: Optional[str]
    duplicate_ignored: bool
    timeout_cleared: bool
    max_length_cleared: bool
    cooldown_suppressed: bool


class AttackOutputV1(TypedDict):
    name: str
    display_name: str
    recognized_at_ms: int


class PipelineOutputV1(TypedDict):
    schema_version: str
    session_id: str
    frame_id: int
    captured_at_ms: int
    processing_ms: float
    hand: HandOutputV1
    body: BodyOutputV1
    queue: QueueOutputV1
    attack: Optional[AttackOutputV1]


def _prediction(prediction: Classification) -> PredictionOutputV1:
    return {
        "label": str(prediction.label),
        "confidence": float(prediction.confidence),
        "second_label": str(prediction.second_label),
        "second_confidence": float(prediction.second_confidence),
        "margin": float(prediction.margin),
    }


def _bbox(values: tuple[int, int, int, int]) -> list[int]:
    return [int(value) for value in values]


def _hand_geometry(result: FrameResult) -> HandGeometryV1:
    hand = result.hand
    return {
        "classification_bbox": _bbox(hand.classification_bbox),
        "hand_bbox": _bbox(hand.hand_bbox) if hand.hand_bbox is not None else None,
        "landmarks": [
            [[float(x), float(y)] for x, y in landmarks]
            for landmarks in hand.landmarks
        ],
    }


def _body_geometry(result: FrameResult) -> BodyGeometryV1:
    return {
        "landmarks": [[float(x), float(y)] for x, y in result.body.landmarks],
    }


def _ensure_json_compatible(value: Any, path: str = "output") -> None:
    """Reject values that standard JSON cannot represent safely."""
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SerializationError(f"{path} must be finite")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _ensure_json_compatible(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise SerializationError(f"{path} contains a non-string key")
            _ensure_json_compatible(item, f"{path}.{key}")
        return
    raise SerializationError(f"{path} contains unsupported type {type(value).__name__}")


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], path: str
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise SerializationError(
            f"{path} keys do not match V1 contract; missing={missing}, extra={extra}"
        )


def _require_string(value: Any, path: str, *, nullable: bool = False) -> None:
    if value is None and nullable:
        return
    if not isinstance(value, str):
        raise SerializationError(f"{path} must be a string")


def _require_number(
    value: Any,
    path: str,
    *,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SerializationError(f"{path} must be a number")
    if not math.isfinite(float(value)):
        raise SerializationError(f"{path} must be finite")
    if minimum is not None and value < minimum:
        raise SerializationError(f"{path} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise SerializationError(f"{path} must be at most {maximum}")


def _validate_prediction(value: Any, path: str) -> None:
    if not isinstance(value, Mapping):
        raise SerializationError(f"{path} must be an object")
    _require_exact_keys(
        value,
        {"label", "confidence", "second_label", "second_confidence", "margin"},
        path,
    )
    _require_string(value["label"], f"{path}.label")
    _require_string(value["second_label"], f"{path}.second_label")
    for key in ("confidence", "second_confidence", "margin"):
        _require_number(value[key], f"{path}.{key}", minimum=0.0, maximum=1.0)


def _validate_bbox(value: Any, path: str, *, nullable: bool = False) -> None:
    if value is None and nullable:
        return
    if not isinstance(value, list) or len(value) != 4:
        raise SerializationError(f"{path} must contain four integers")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise SerializationError(f"{path} must contain four integers")


def _validate_points(value: Any, path: str, *, nested: bool) -> None:
    if not isinstance(value, list):
        raise SerializationError(f"{path} must be an array")
    point_groups = value if nested else [value]
    for group_index, group in enumerate(point_groups):
        if not isinstance(group, list):
            raise SerializationError(f"{path}[{group_index}] must be an array")
        for point_index, point in enumerate(group):
            point_path = f"{path}[{group_index}][{point_index}]"
            if not isinstance(point, list) or len(point) != 2:
                raise SerializationError(f"{point_path} must contain x and y")
            _require_number(point[0], f"{point_path}[0]")
            _require_number(point[1], f"{point_path}[1]")


def _validate_hand_geometry(value: Any) -> None:
    if value is None:
        return
    if not isinstance(value, Mapping):
        raise SerializationError("output.hand.geometry must be an object or null")
    _require_exact_keys(
        value,
        {"classification_bbox", "hand_bbox", "landmarks"},
        "output.hand.geometry",
    )
    _validate_bbox(value["classification_bbox"], "output.hand.geometry.classification_bbox")
    _validate_bbox(value["hand_bbox"], "output.hand.geometry.hand_bbox", nullable=True)
    _validate_points(value["landmarks"], "output.hand.geometry.landmarks", nested=True)


def _validate_body_geometry(value: Any) -> None:
    if value is None:
        return
    if not isinstance(value, Mapping):
        raise SerializationError("output.body.geometry must be an object or null")
    _require_exact_keys(value, {"landmarks"}, "output.body.geometry")
    _validate_points(value["landmarks"], "output.body.geometry.landmarks", nested=False)


def validate_pipeline_output_v1(output: Mapping[str, Any]) -> None:
    """Validate the stable shape and fundamental invariants of a V1 output."""
    if not isinstance(output, Mapping):
        raise SerializationError("output must be an object")
    _require_exact_keys(
        output,
        {
            "schema_version",
            "session_id",
            "frame_id",
            "captured_at_ms",
            "processing_ms",
            "hand",
            "body",
            "queue",
            "attack",
        },
        "output",
    )
    if output["schema_version"] != SCHEMA_VERSION:
        raise SerializationError(
            f"unsupported schema version: {output['schema_version']!r}"
        )
    if not isinstance(output["session_id"], str) or not output["session_id"]:
        raise SerializationError("session_id must be a non-empty string")
    for key in ("frame_id", "captured_at_ms"):
        if (
            isinstance(output[key], bool)
            or not isinstance(output[key], int)
            or output[key] < 0
        ):
            raise SerializationError(f"{key} must be a non-negative integer")
    processing_ms = output["processing_ms"]
    if (
        isinstance(processing_ms, bool)
        or not isinstance(processing_ms, (int, float))
        or processing_ms < 0
    ):
        raise SerializationError("processing_ms must be a non-negative number")

    hand = output["hand"]
    body = output["body"]
    queue = output["queue"]
    if not all(isinstance(value, Mapping) for value in (hand, body, queue)):
        raise SerializationError("hand, body, and queue must be objects")
    _require_exact_keys(
        hand,
        {
            "raw",
            "center",
            "roi",
            "accepted_label",
            "rejection_reason",
            "stable_label",
            "emitted_seal",
            "fusion_status",
            "detected_hand_count",
            "geometry",
        },
        "output.hand",
    )
    _require_exact_keys(
        body,
        {"raw_label", "stable_label", "emitted_movement", "metrics", "geometry"},
        "output.body",
    )
    _require_exact_keys(
        queue,
        {
            "seals",
            "accepted_seal",
            "duplicate_ignored",
            "timeout_cleared",
            "max_length_cleared",
            "cooldown_suppressed",
        },
        "output.queue",
    )
    _validate_prediction(hand["raw"], "output.hand.raw")
    _validate_prediction(hand["center"], "output.hand.center")
    if hand["roi"] is not None:
        _validate_prediction(hand["roi"], "output.hand.roi")
    for key in ("accepted_label", "stable_label", "fusion_status"):
        _require_string(hand[key], f"output.hand.{key}")
    for key in ("rejection_reason", "emitted_seal"):
        _require_string(hand[key], f"output.hand.{key}", nullable=True)
    detected_hand_count = hand["detected_hand_count"]
    if (
        isinstance(detected_hand_count, bool)
        or not isinstance(detected_hand_count, int)
        or not 0 <= detected_hand_count <= 2
    ):
        raise SerializationError(
            "output.hand.detected_hand_count must be from zero to two"
        )
    _validate_hand_geometry(hand["geometry"])

    for key in ("raw_label", "stable_label"):
        _require_string(body[key], f"output.body.{key}")
    _require_string(
        body["emitted_movement"],
        "output.body.emitted_movement",
        nullable=True,
    )
    if not isinstance(body["metrics"], Mapping):
        raise SerializationError("output.body.metrics must be an object")
    for name, value in body["metrics"].items():
        _require_string(name, "output.body.metrics key")
        _require_number(value, f"output.body.metrics.{name}")
    _validate_body_geometry(body["geometry"])

    seals = queue["seals"]
    if not isinstance(seals, list) or len(seals) > 3:
        raise SerializationError("output.queue.seals must contain at most three labels")
    for index, seal in enumerate(seals):
        _require_string(seal, f"output.queue.seals[{index}]")
    _require_string(
        queue["accepted_seal"],
        "output.queue.accepted_seal",
        nullable=True,
    )
    for key in (
        "duplicate_ignored",
        "timeout_cleared",
        "max_length_cleared",
        "cooldown_suppressed",
    ):
        if not isinstance(queue[key], bool):
            raise SerializationError(f"output.queue.{key} must be a boolean")

    attack = output["attack"]
    if attack is not None:
        if not isinstance(attack, Mapping):
            raise SerializationError("output.attack must be an object or null")
        _require_exact_keys(
            attack,
            {"name", "display_name", "recognized_at_ms"},
            "output.attack",
        )
        _require_string(attack["name"], "output.attack.name")
        _require_string(attack["display_name"], "output.attack.display_name")
        recognized_at_ms = attack["recognized_at_ms"]
        if (
            isinstance(recognized_at_ms, bool)
            or not isinstance(recognized_at_ms, int)
            or recognized_at_ms < 0
        ):
            raise SerializationError(
                "output.attack.recognized_at_ms must be a non-negative integer"
            )
    _ensure_json_compatible(dict(output))


@dataclass(frozen=True)
class PipelineOutputSerializer:
    """Convert internal frame results into the stable public V1 contract."""

    include_geometry: bool = False

    def to_dict(
        self,
        result: FrameResult,
        *,
        session_id: str,
        frame_id: int,
        captured_at_ms: int,
    ) -> PipelineOutputV1:
        hand = result.hand
        body = result.body
        queue = result.queue_update
        center = hand.center_prediction or hand.raw
        output: PipelineOutputV1 = {
            "schema_version": SCHEMA_VERSION,
            "session_id": session_id,
            "frame_id": frame_id,
            "captured_at_ms": captured_at_ms,
            "processing_ms": float(result.processing_ms),
            "hand": {
                "raw": _prediction(hand.raw),
                "center": _prediction(center),
                "roi": _prediction(hand.roi_prediction) if hand.roi_prediction else None,
                "accepted_label": str(hand.accepted_label),
                "rejection_reason": hand.rejection_reason,
                "stable_label": str(hand.stable_label),
                "emitted_seal": hand.emitted_seal,
                "fusion_status": str(hand.fusion_status),
                "detected_hand_count": len(hand.landmarks),
                "geometry": _hand_geometry(result) if self.include_geometry else None,
            },
            "body": {
                "raw_label": str(body.raw_label),
                "stable_label": str(body.stable_label),
                "emitted_movement": body.emitted_movement,
                "metrics": {
                    str(name): float(value) for name, value in body.metrics.items()
                },
                "geometry": _body_geometry(result) if self.include_geometry else None,
            },
            "queue": {
                "seals": [str(seal) for seal in result.seal_history],
                "accepted_seal": queue.accepted_seal,
                "duplicate_ignored": bool(queue.duplicate_ignored),
                "timeout_cleared": bool(queue.timeout_cleared),
                "max_length_cleared": bool(queue.max_length_cleared),
                "cooldown_suppressed": bool(queue.cooldown_suppressed),
            },
            "attack": (
                {
                    "name": str(result.attack.name),
                    "display_name": str(result.attack.display_name),
                    "recognized_at_ms": int(round(result.attack.timestamp * 1000.0)),
                }
                if result.attack is not None
                else None
            ),
        }
        validate_pipeline_output_v1(output)
        return output

    def to_json(
        self,
        result: FrameResult,
        *,
        session_id: str,
        frame_id: int,
        captured_at_ms: int,
    ) -> str:
        output = self.to_dict(
            result,
            session_id=session_id,
            frame_id=frame_id,
            captured_at_ms=captured_at_ms,
        )
        return json.dumps(
            output,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
