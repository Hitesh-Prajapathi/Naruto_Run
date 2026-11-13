from __future__ import annotations

import copy
import json
import math
import unittest
from dataclasses import replace
from pathlib import Path

from cv_model.inference.combined_pipeline import (
    AttackEvent,
    BodyResult,
    Classification,
    FrameResult,
    HandResult,
    QueueUpdate,
)
from cv_model.inference.output_schema import (
    SCHEMA_VERSION,
    PipelineOutputSerializer,
    SerializationError,
    validate_pipeline_output_v1,
)


SCHEMA_PATH = (
    Path(__file__).resolve().parents[1]
    / "schemas"
    / "pipeline_output_v1.schema.json"
)


def _frame_result() -> FrameResult:
    attack = AttackEvent("shippu", "SHIPPU / WIND", 123.456)
    raw = Classification("rat", 0.875, "ram", 0.125)
    center = Classification("ram", 0.75, "rat", 0.20)
    roi = Classification("rat", 0.9375, "ram", 0.03125)
    hand = HandResult(
        raw=raw,
        accepted_label="rat",
        rejection_reason=None,
        stable_label="rat",
        emitted_seal="rat",
        classification_bbox=(0, 0, 224, 224),
        hand_bbox=(30, 40, 190, 200),
        landmarks=(((0.25, 0.50), (0.75, 0.50)),),
        center_prediction=center,
        roi_prediction=roi,
        fusion_status="center_roi_fused",
    )
    body = BodyResult(
        raw_label="jumping",
        stable_label="jumping",
        emitted_movement="jumping",
        metrics={"jump_height": 0.0625, "lean_x": 0.0},
        landmarks=((0.40, 0.30), (0.60, 0.30)),
    )
    queue = QueueUpdate(
        attack=attack,
        accepted_seal="rat",
        duplicate_ignored=False,
        timeout_cleared=False,
        max_length_cleared=False,
        cooldown_suppressed=False,
        queue=(),
    )
    return FrameResult(
        hand=hand,
        body=body,
        attack=attack,
        seal_history=(),
        queue_update=queue,
        processing_ms=41.25,
    )


class PipelineOutputSchemaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.result = _frame_result()
        self.serializer = PipelineOutputSerializer()

    def serialize(self, **overrides: object) -> dict[str, object]:
        values: dict[str, object] = {
            "session_id": "camera-test-001",
            "frame_id": 42,
            "captured_at_ms": 123456,
        }
        values.update(overrides)
        return self.serializer.to_dict(self.result, **values)

    def test_golden_v1_output_without_geometry(self) -> None:
        self.assertEqual(
            self.serialize(),
            {
                "schema_version": "1.0.0",
                "session_id": "camera-test-001",
                "frame_id": 42,
                "captured_at_ms": 123456,
                "processing_ms": 41.25,
                "hand": {
                    "raw": {
                        "label": "rat",
                        "confidence": 0.875,
                        "second_label": "ram",
                        "second_confidence": 0.125,
                        "margin": 0.75,
                    },
                    "center": {
                        "label": "ram",
                        "confidence": 0.75,
                        "second_label": "rat",
                        "second_confidence": 0.20,
                        "margin": 0.55,
                    },
                    "roi": {
                        "label": "rat",
                        "confidence": 0.9375,
                        "second_label": "ram",
                        "second_confidence": 0.03125,
                        "margin": 0.90625,
                    },
                    "accepted_label": "rat",
                    "rejection_reason": None,
                    "stable_label": "rat",
                    "emitted_seal": "rat",
                    "fusion_status": "center_roi_fused",
                    "detected_hand_count": 1,
                    "geometry": None,
                },
                "body": {
                    "raw_label": "jumping",
                    "stable_label": "jumping",
                    "emitted_movement": "jumping",
                    "metrics": {"jump_height": 0.0625, "lean_x": 0.0},
                    "geometry": None,
                },
                "queue": {
                    "seals": [],
                    "accepted_seal": "rat",
                    "duplicate_ignored": False,
                    "timeout_cleared": False,
                    "max_length_cleared": False,
                    "cooldown_suppressed": False,
                },
                "attack": {
                    "name": "shippu",
                    "display_name": "SHIPPU / WIND",
                    "recognized_at_ms": 123456,
                },
            },
        )

    def test_geometry_can_be_included_without_changing_recognition(self) -> None:
        output = PipelineOutputSerializer(include_geometry=True).to_dict(
            self.result,
            session_id="geometry",
            frame_id=1,
            captured_at_ms=10,
        )

        self.assertEqual(output["hand"]["geometry"]["hand_bbox"], [30, 40, 190, 200])
        self.assertEqual(
            output["hand"]["geometry"]["landmarks"],
            [[[0.25, 0.5], [0.75, 0.5]]],
        )
        self.assertEqual(
            output["body"]["geometry"]["landmarks"],
            [[0.4, 0.3], [0.6, 0.3]],
        )

    def test_absent_roi_body_event_and_attack_are_explicit_nulls(self) -> None:
        hand = replace(
            self.result.hand,
            roi_prediction=None,
            emitted_seal=None,
            hand_bbox=None,
            landmarks=(),
        )
        body = replace(self.result.body, emitted_movement=None, landmarks=())
        queue = replace(self.result.queue_update, attack=None, accepted_seal=None)
        result = replace(
            self.result,
            hand=hand,
            body=body,
            attack=None,
            queue_update=queue,
        )

        output = self.serializer.to_dict(
            result,
            session_id="empty-events",
            frame_id=0,
            captured_at_ms=0,
        )

        self.assertIsNone(output["hand"]["roi"])
        self.assertIsNone(output["hand"]["emitted_seal"])
        self.assertEqual(output["hand"]["detected_hand_count"], 0)
        self.assertIsNone(output["body"]["emitted_movement"])
        self.assertIsNone(output["attack"])

    def test_every_queue_transition_flag_is_serialized(self) -> None:
        queue = replace(
            self.result.queue_update,
            duplicate_ignored=True,
            timeout_cleared=True,
            max_length_cleared=True,
            cooldown_suppressed=True,
            queue=("bird", "ram"),
        )
        output = self.serializer.to_dict(
            replace(self.result, queue_update=queue, seal_history=queue.queue),
            session_id="queue",
            frame_id=7,
            captured_at_ms=70,
        )

        self.assertEqual(output["queue"]["seals"], ["bird", "ram"])
        for name in (
            "duplicate_ignored",
            "timeout_cleared",
            "max_length_cleared",
            "cooldown_suppressed",
        ):
            self.assertTrue(output["queue"][name])

    def test_json_output_is_compact_and_deterministic(self) -> None:
        first = self.serializer.to_json(
            self.result,
            session_id="stable",
            frame_id=9,
            captured_at_ms=90,
        )
        second = self.serializer.to_json(
            self.result,
            session_id="stable",
            frame_id=9,
            captured_at_ms=90,
        )

        self.assertEqual(first, second)
        self.assertEqual(json.loads(first)["schema_version"], SCHEMA_VERSION)
        self.assertNotIn(": ", first)
        self.assertNotIn(", ", first)

    def test_non_finite_values_are_rejected(self) -> None:
        with self.assertRaises(SerializationError):
            self.serializer.to_dict(
                replace(self.result, processing_ms=math.nan),
                session_id="invalid",
                frame_id=1,
                captured_at_ms=1,
            )

    def test_version_and_unexpected_fields_are_rejected(self) -> None:
        output = self.serialize()
        wrong_version = copy.deepcopy(output)
        wrong_version["schema_version"] = "2.0.0"
        with self.assertRaises(SerializationError):
            validate_pipeline_output_v1(wrong_version)

        extra_field = copy.deepcopy(output)
        extra_field["hand"]["experimental"] = True
        with self.assertRaises(SerializationError):
            validate_pipeline_output_v1(extra_field)

    def test_out_of_range_confidence_and_oversized_queue_are_rejected(self) -> None:
        invalid_confidence = copy.deepcopy(self.serialize())
        invalid_confidence["hand"]["raw"]["confidence"] = 1.01
        with self.assertRaises(SerializationError):
            validate_pipeline_output_v1(invalid_confidence)

        oversized_queue = copy.deepcopy(self.serialize())
        oversized_queue["queue"]["seals"] = ["bird", "ram", "rat", "dog"]
        with self.assertRaises(SerializationError):
            validate_pipeline_output_v1(oversized_queue)

    def test_machine_readable_schema_matches_contract_version(self) -> None:
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

        self.assertEqual(schema["$schema"], "https://json-schema.org/draft/2020-12/schema")
        self.assertEqual(schema["properties"]["schema_version"]["const"], SCHEMA_VERSION)
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(set(schema["required"]), set(self.serialize()))


if __name__ == "__main__":
    unittest.main()
