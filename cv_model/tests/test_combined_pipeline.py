from __future__ import annotations

import unittest

import numpy as np

from cv_model.inference.combined_pipeline import (
    ATTACKS,
    AttackRecognizer,
    BodyMovementRecognizer,
    BodyResult,
    Classification,
    CombinedNarutoPipeline,
    EvidenceHandFilter,
    HandResult,
    HandSignRecognizer,
)
from cv_model.inference.pipeline_config import (
    AttackQueueConfig,
    BodyMovementConfig,
    HandFusionConfig,
    HandTemporalConfig,
)


class HandTemporalTests(unittest.TestCase):
    def test_sign_requires_fresh_evidence_after_neutral(self) -> None:
        hand_filter = EvidenceHandFilter(
            HandTemporalConfig(window_size=5, required_votes=3, neutral_frames=2)
        )

        outputs = [hand_filter.update("ram") for _ in range(3)]
        self.assertEqual(outputs[-1], ("ram", "ram"))
        self.assertEqual(hand_filter.update("zero")[0], "ram")
        self.assertEqual(hand_filter.update("zero"), ("zero", None))

        self.assertEqual(hand_filter.update("ram"), ("zero", None))
        self.assertEqual(hand_filter.update("ram"), ("zero", None))
        self.assertEqual(hand_filter.update("ram"), ("ram", "ram"))

    def test_held_sign_emits_once(self) -> None:
        hand_filter = EvidenceHandFilter()
        emitted = [hand_filter.update("bird")[1] for _ in range(8)]
        self.assertEqual([item for item in emitted if item], ["bird"])


class HandFusionTests(unittest.TestCase):
    def make_recognizer(self, **config: object) -> HandSignRecognizer:
        recognizer = object.__new__(HandSignRecognizer)
        recognizer.classes = ("dog", "ox", "zero")
        recognizer.fusion_config = HandFusionConfig(**config)
        recognizer._smoothed_hand_bbox = None
        recognizer._missing_hand_frames = 0
        return recognizer

    def test_probability_fusion_can_correct_center_view(self) -> None:
        recognizer = self.make_recognizer(center_weight=0.60)
        center = Classification("ox", 0.60, "dog", 0.30, (0.30, 0.60, 0.10))
        roi = Classification("dog", 0.80, "ox", 0.15, (0.80, 0.15, 0.05))

        fused = recognizer._fuse_predictions(center, roi)

        self.assertEqual(fused.label, "dog")
        self.assertAlmostEqual(fused.confidence, 0.50)
        self.assertEqual(len(fused.probabilities), 3)

    def test_hand_bbox_uses_exponential_smoothing(self) -> None:
        recognizer = self.make_recognizer(bbox_smoothing_alpha=0.50)
        self.assertEqual(recognizer._smooth_hand_bbox((0, 0, 100, 100)), (0, 0, 100, 100))
        self.assertEqual(recognizer._smooth_hand_bbox((10, 20, 110, 120)), (5, 10, 105, 110))

    def test_delayed_no_hand_gate_preserves_known_detector_fallbacks(self) -> None:
        recognizer = self.make_recognizer(absence_grace_frames=3)
        dog = Classification("dog", 0.90, "zero", 0.05)
        hare = Classification("hare", 0.90, "zero", 0.05)

        recognizer._missing_hand_frames = 2
        self.assertFalse(recognizer._should_gate_without_hands(dog))
        recognizer._missing_hand_frames = 3
        self.assertTrue(recognizer._should_gate_without_hands(dog))
        self.assertFalse(recognizer._should_gate_without_hands(hare))

    def test_invalid_fusion_configuration_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            HandFusionConfig(center_weight=1.1)
        with self.assertRaises(ValueError):
            HandFusionConfig(absence_grace_frames=0)


class HandAcceptanceCalibrationTests(unittest.TestCase):
    def make_recognizer(self) -> HandSignRecognizer:
        recognizer = object.__new__(HandSignRecognizer)
        recognizer.confidence_override = None
        recognizer.confidence_thresholds = {
            "dog": 0.68,
            "dragon": 0.50,
            "ram": 0.92,
            "rat": 0.45,
            "zero": 0.45,
        }
        recognizer.margin_thresholds = {
            "dog": 0.40,
            "dragon": 0.20,
            "ram": 0.55,
            "rat": 0.05,
            "zero": 0.05,
        }
        return recognizer

    def test_lower_confidence_dog_requires_camera_validated_runner_up(self) -> None:
        recognizer = self.make_recognizer()

        accepted = Classification("dog", 0.70, "hare", 0.12)
        tiger_confusion = Classification("dog", 0.70, "snake", 0.12)

        self.assertEqual(recognizer._accept(accepted, 2), ("dog", None))
        self.assertEqual(
            recognizer._accept(tiger_confusion, 1), ("zero", "dog_pair_guard")
        )

    def test_dragon_uses_live_camera_confidence_range(self) -> None:
        recognizer = self.make_recognizer()
        dragon = Classification("dragon", 0.52, "monkey", 0.25)

        self.assertEqual(recognizer._accept(dragon, 2), ("dragon", None))

    def test_two_detected_hands_do_not_rewrite_clear_ram(self) -> None:
        recognizer = self.make_recognizer()
        ram = Classification("ram", 0.97, "dog", 0.01)

        self.assertEqual(recognizer._accept(ram, 2), ("ram", None))

    def test_ambiguous_ram_rat_pair_still_recovers_rat(self) -> None:
        recognizer = self.make_recognizer()
        ambiguous = Classification("ram", 0.80, "rat", 0.14)

        self.assertEqual(
            recognizer._accept(ambiguous, 2),
            ("rat", "resolved_ram_to_rat_pair_probability"),
        )


class AttackQueueTests(unittest.TestCase):
    def test_authoritative_attack_catalog(self) -> None:
        self.assertEqual(
            [(attack.display_name, attack.seals) for attack in ATTACKS],
            [
                ("FIRE ATTACK", ("tiger", "horse")),
                ("LIGHTNING DODGE", ("hare",)),
                ("WATER ATTACK", ("snake", "dragon")),
                ("SAND ATTACK", ("monkey", "ox")),
                ("WIND ATTACK", ("dog", "rat")),
            ],
        )

    def test_every_attack_sequence(self) -> None:
        for definition in ATTACKS:
            with self.subTest(attack=definition.name):
                recognizer = AttackRecognizer()
                event = None
                for index, seal in enumerate(definition.seals):
                    event = recognizer.update(seal, 1.0 + index * 0.1)
                self.assertIsNotNone(event)
                self.assertEqual(event.name, definition.name)
                self.assertEqual(recognizer.seal_labels, ())

    def test_leading_noise_cannot_be_skipped_to_trigger_an_attack(self) -> None:
        recognizer = AttackRecognizer()

        self.assertIsNone(recognizer.update("ox", 1.0))
        self.assertIsNone(recognizer.update("hare", 1.1))
        self.assertEqual(recognizer.seal_labels, ("ox", "hare"))

    def test_out_of_order_two_seal_combo_does_not_trigger(self) -> None:
        recognizer = AttackRecognizer()

        self.assertIsNone(recognizer.update("horse", 1.0))
        self.assertIsNone(recognizer.update("tiger", 1.1))
        self.assertEqual(recognizer.seal_labels, ("horse", "tiger"))

    def test_adjacent_duplicate_is_ignored(self) -> None:
        recognizer = AttackRecognizer()
        recognizer.update_detailed("ram", 1.0)
        update = recognizer.update_detailed("ram", 1.1)
        self.assertTrue(update.duplicate_ignored)
        self.assertIsNone(update.accepted_seal)
        self.assertEqual(update.queue, ("ram",))

    def test_timeout_starts_fresh_queue(self) -> None:
        recognizer = AttackRecognizer()
        recognizer.update_detailed("bird", 1.0)
        update = recognizer.update_detailed("ram", 4.1)
        self.assertTrue(update.timeout_cleared)
        self.assertEqual(update.queue, ("ram",))

    def test_third_nonmatching_seal_clears_queue(self) -> None:
        recognizer = AttackRecognizer()
        recognizer.update_detailed("bird", 1.0)
        recognizer.update_detailed("horse", 1.1)
        update = recognizer.update_detailed("ox", 1.2)
        self.assertTrue(update.max_length_cleared)
        self.assertEqual(update.queue, ())

    def test_attack_cooldown_is_reported(self) -> None:
        recognizer = AttackRecognizer()
        self.assertIsNotNone(recognizer.update_detailed("hare", 1.0).attack)
        update = recognizer.update_detailed("hare", 1.5)
        self.assertTrue(update.cooldown_suppressed)
        self.assertIsNone(update.attack)
        self.assertEqual(update.queue, ())

    def test_invalid_queue_configuration_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            AttackRecognizer(config=AttackQueueConfig(max_seals=1))


class BodyMovementTests(unittest.TestCase):
    def setUp(self) -> None:
        self.recognizer = object.__new__(BodyMovementRecognizer)
        self.recognizer.config = BodyMovementConfig()

    def classify(self, **overrides: object) -> str:
        values = {
            "jump_height": 0.0,
            "torso_angle": 0.0,
            "arms_back": False,
            "lean_x": 0.0,
        }
        values.update(overrides)
        return self.recognizer.classify_metrics(**values)

    def test_supported_body_movements(self) -> None:
        self.assertEqual(self.classify(jump_height=0.06), "jumping")
        self.assertEqual(
            self.classify(torso_angle=30.0, arms_back=True), "naruto_run"
        )
        self.assertEqual(self.classify(lean_x=0.08), "bending_right")
        self.assertEqual(self.classify(lean_x=-0.08), "bending_left")
        self.assertEqual(self.classify(), "idle")

    def test_attack_names_are_not_body_movements(self) -> None:
        labels = {
            self.classify(jump_height=0.06),
            self.classify(torso_angle=30.0, arms_back=True),
            self.classify(lean_x=0.08),
            self.classify(lean_x=-0.08),
            self.classify(),
        }
        self.assertNotIn("lightning_dodge", labels)
        self.assertNotIn("stone_defense", labels)


class _FakeHandRecognizer:
    def __init__(self, seals: list[str | None]) -> None:
        self.seals = iter(seals)

    def process(self, _frame: np.ndarray, _timestamp: float) -> HandResult:
        seal = next(self.seals)
        label = seal or "zero"
        return HandResult(
            raw=Classification(label, 1.0, "zero", 0.0),
            accepted_label=label,
            rejection_reason=None,
            stable_label=label,
            emitted_seal=seal,
            classification_bbox=(0, 0, 1, 1),
            hand_bbox=None,
        )


class _FakeBodyRecognizer:
    def __init__(self, labels: list[str]) -> None:
        self.labels = iter(labels)

    def process(self, _frame: np.ndarray, _timestamp: float) -> BodyResult:
        label = next(self.labels)
        return BodyResult(label, label, label, {})


class CombinedPipelineIsolationTests(unittest.TestCase):
    def test_body_events_never_gate_or_trigger_attacks(self) -> None:
        pipeline = object.__new__(CombinedNarutoPipeline)
        pipeline.hand = _FakeHandRecognizer(["dog", "rat"])
        pipeline.body = _FakeBodyRecognizer(
            ["jumping", "naruto_run"]
        )
        pipeline.attacks = AttackRecognizer()
        frame = np.zeros((2, 2, 3), dtype=np.uint8)

        results = [pipeline.process(frame, 1.0 + index * 0.1) for index in range(2)]
        self.assertIsNone(results[0].attack)
        self.assertEqual(results[1].attack.name, "shippu")
        self.assertIn("attack_queue_ms", results[1].timings_ms)
        self.assertIn("recognition_total_ms", results[1].timings_ms)

        movement_only = object.__new__(CombinedNarutoPipeline)
        movement_only.hand = _FakeHandRecognizer([None])
        movement_only.body = _FakeBodyRecognizer(["naruto_run"])
        movement_only.attacks = AttackRecognizer()
        result = movement_only.process(frame, 2.0)
        self.assertIsNone(result.attack)
        self.assertEqual(result.seal_history, ())


if __name__ == "__main__":
    unittest.main()
