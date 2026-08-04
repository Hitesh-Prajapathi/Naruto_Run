"""Trial-level live-camera accuracy and attack diagnostics for NarutoCV."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

from .combined_pipeline import ATTACKS, FALLBACK_HAND_CLASSES


EVALUATION_VERSION = "1.1.0"
HAND_LABELS = tuple(FALLBACK_HAND_CLASSES)
BODY_LABELS = ("idle", "jumping", "naruto_run", "bending_left", "bending_right")
ATTACK_LABELS = tuple(attack.name for attack in ATTACKS)
ATTACK_SEALS = {attack.name: attack.seals for attack in ATTACKS}


def _dominant(counter: Counter[str]) -> Optional[str]:
    if not counter:
        return None
    return sorted(counter.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _sample_summary(values: list[float]) -> dict[str, Optional[float] | int]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p95": None, "max": None}
    ordered = sorted(values)
    position = (len(ordered) - 1) * 0.95
    lower = math.floor(position)
    upper = math.ceil(position)
    p95 = ordered[lower]
    if lower != upper:
        p95 += (ordered[upper] - ordered[lower]) * (position - lower)
    return {
        "count": len(ordered),
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "p95": p95,
        "max": ordered[-1],
    }


@dataclass
class LabelAttemptEvaluator:
    """Accumulate one guided hand-sign or body-movement attempt."""

    mode: str
    expected_label: str
    attempt_index: int
    action_started_at_ms: int
    prepare_raw_counts: Counter[str] = field(default_factory=Counter)
    prepare_events: list[str] = field(default_factory=list)
    action_raw_counts: Counter[str] = field(default_factory=Counter)
    action_accepted_counts: Counter[str] = field(default_factory=Counter)
    action_stable_counts: Counter[str] = field(default_factory=Counter)
    action_events: list[str] = field(default_factory=list)
    rejection_reasons: Counter[str] = field(default_factory=Counter)
    expected_confidences: list[float] = field(default_factory=list)
    expected_margins: list[float] = field(default_factory=list)
    first_match_at_ms: Optional[int] = None

    def __post_init__(self) -> None:
        if self.mode not in {"hand", "body"}:
            raise ValueError("label attempts support only hand or body mode")
        allowed = HAND_LABELS if self.mode == "hand" else BODY_LABELS
        if self.expected_label not in allowed:
            raise ValueError(f"unsupported {self.mode} label: {self.expected_label}")
        if self.attempt_index < 1:
            raise ValueError("attempt_index must be positive")

    @property
    def neutral_label(self) -> str:
        return "zero" if self.mode == "hand" else "idle"

    def start_action(self, captured_at_ms: int) -> None:
        """Anchor latency to the actual post-reset action boundary."""
        self.action_started_at_ms = int(captured_at_ms)
        self.first_match_at_ms = None

    def record(self, output: Mapping[str, Any], *, phase: str) -> None:
        if phase not in {"prepare", "action"}:
            raise ValueError("phase must be prepare or action")
        captured_at_ms = int(output["captured_at_ms"])
        if self.mode == "hand":
            section = output["hand"]
            raw_label = str(section["raw"]["label"])
            accepted_label = str(section["accepted_label"])
            stable_label = str(section["stable_label"])
            emitted = section["emitted_seal"]
            rejection = section["rejection_reason"]
        else:
            section = output["body"]
            raw_label = str(section["raw_label"])
            accepted_label = raw_label
            stable_label = str(section["stable_label"])
            emitted = section["emitted_movement"]

        if phase == "prepare":
            self.prepare_raw_counts[raw_label] += 1
            if emitted and emitted != self.neutral_label:
                self.prepare_events.append(str(emitted))
            return

        if self.mode == "hand":
            if rejection:
                self.rejection_reasons[str(rejection)] += 1
            if raw_label == self.expected_label:
                self.expected_confidences.append(float(section["raw"]["confidence"]))
                self.expected_margins.append(float(section["raw"]["margin"]))

        self.action_raw_counts[raw_label] += 1
        self.action_accepted_counts[accepted_label] += 1
        self.action_stable_counts[stable_label] += 1
        if emitted and emitted != self.neutral_label:
            emitted_label = str(emitted)
            self.action_events.append(emitted_label)
            if emitted_label == self.expected_label and self.first_match_at_ms is None:
                self.first_match_at_ms = captured_at_ms
        if self.expected_label == self.neutral_label:
            if stable_label == self.expected_label and self.first_match_at_ms is None:
                self.first_match_at_ms = captured_at_ms

    def finish(self) -> dict[str, Any]:
        if self.expected_label == self.neutral_label:
            system_prediction = _dominant(self.action_stable_counts)
        else:
            system_prediction = self.action_events[0] if self.action_events else None
        first_event_success = system_prediction == self.expected_label
        target_event_detected = (
            first_event_success
            if self.expected_label == self.neutral_label
            else self.expected_label in self.action_events
        )
        unexpected_action_events = [
            event for event in self.action_events if event != self.expected_label
        ]
        duplicate_emissions = sum(
            current == previous
            for previous, current in zip(self.action_events, self.action_events[1:])
        )
        latency_ms = (
            self.first_match_at_ms - self.action_started_at_ms
            if self.first_match_at_ms is not None
            else None
        )
        return {
            "attempt_type": "label",
            "mode": self.mode,
            "expected_label": self.expected_label,
            "attempt_index": self.attempt_index,
            "success": first_event_success,
            "first_event_success": first_event_success,
            "target_event_detected": target_event_detected,
            "system_prediction": system_prediction,
            "first_event_prediction": system_prediction,
            "dominant_raw_prediction": _dominant(self.action_raw_counts),
            "dominant_accepted_prediction": _dominant(self.action_accepted_counts),
            "dominant_stable_prediction": _dominant(self.action_stable_counts),
            "detection_latency_ms": latency_ms,
            "prepare_false_positive_events": list(self.prepare_events),
            "action_events": list(self.action_events),
            "unexpected_action_events": unexpected_action_events,
            "duplicate_emissions": duplicate_emissions,
            "target_emission_count": self.action_events.count(self.expected_label),
            "prepare_raw_counts": dict(self.prepare_raw_counts),
            "action_raw_counts": dict(self.action_raw_counts),
            "action_accepted_counts": dict(self.action_accepted_counts),
            "action_stable_counts": dict(self.action_stable_counts),
            "rejection_reasons": dict(self.rejection_reasons),
            "expected_confidence": _sample_summary(self.expected_confidences),
            "expected_margin": _sample_summary(self.expected_margins),
        }


class LabelEvaluationSession:
    """Aggregate guided label attempts into per-label metrics and confusion."""

    def __init__(self, mode: str) -> None:
        if mode not in {"hand", "body"}:
            raise ValueError("mode must be hand or body")
        self.mode = mode
        self.attempts: list[dict[str, Any]] = []

    @property
    def neutral_label(self) -> str:
        return "zero" if self.mode == "hand" else "idle"

    def add(self, attempt: Mapping[str, Any]) -> None:
        if attempt.get("mode") != self.mode:
            raise ValueError("attempt mode does not match session mode")
        self.attempts.append(dict(attempt))

    def summary(self) -> dict[str, Any]:
        allowed = HAND_LABELS if self.mode == "hand" else BODY_LABELS
        confusion: dict[str, Counter[str]] = {}
        for attempt in self.attempts:
            expected = str(attempt["expected_label"])
            prediction = attempt["system_prediction"] or "no_detection"
            confusion.setdefault(expected, Counter())[str(prediction)] += 1

        metrics: dict[str, Any] = {}
        for label in allowed:
            expected_attempts = [
                attempt
                for attempt in self.attempts
                if attempt["expected_label"] == label
            ]
            tp = sum(bool(attempt["first_event_success"]) for attempt in expected_attempts)
            fn = len(expected_attempts) - tp
            first_cross_label_fp = sum(
                attempt["expected_label"] != label
                and attempt["system_prediction"] == label
                for attempt in self.attempts
            )
            emitted_cross_label_fp = sum(
                event == label
                for attempt in self.attempts
                if attempt["expected_label"] != label
                for event in attempt["action_events"]
            )
            neutral_fp = sum(
                event == label
                for attempt in self.attempts
                for event in attempt["prepare_false_positive_events"]
            )
            fp = first_cross_label_fp + neutral_fp
            eventual_tp = sum(
                bool(attempt["target_event_detected"])
                for attempt in expected_attempts
            )
            action_event_count = sum(
                len(attempt["action_events"]) for attempt in expected_attempts
            )
            unexpected_event_count = sum(
                len(attempt["unexpected_action_events"])
                for attempt in expected_attempts
            )
            duplicate_emissions = sum(
                int(attempt["duplicate_emissions"])
                for attempt in expected_attempts
            )
            latencies = [
                float(attempt["detection_latency_ms"])
                for attempt in expected_attempts
                if attempt["detection_latency_ms"] is not None
            ]
            metrics[label] = {
                "attempts": len(expected_attempts),
                "true_positives": tp,
                "first_event_true_positives": tp,
                "eventual_true_positives": eventual_tp,
                "false_positives": fp,
                "cross_label_false_positives": first_cross_label_fp,
                "all_emitted_cross_label_false_positives": emitted_cross_label_fp,
                "neutral_phase_false_positives": neutral_fp,
                "false_negatives": fn,
                "precision": tp / (tp + fp) if tp + fp else None,
                "recall": tp / (tp + fn) if tp + fn else None,
                "first_event_recall": tp / len(expected_attempts) if expected_attempts else None,
                "eventual_recall": (
                    eventual_tp / len(expected_attempts)
                    if expected_attempts
                    else None
                ),
                "action_event_count": action_event_count,
                "unexpected_action_event_count": unexpected_event_count,
                "false_event_rate": (
                    unexpected_event_count / action_event_count
                    if action_event_count
                    else 0.0
                ),
                "duplicate_emissions": duplicate_emissions,
                "detection_latency_ms": _sample_summary(latencies),
            }
        prepare_frame_count = sum(
            sum(attempt["prepare_raw_counts"].values()) for attempt in self.attempts
        )
        prepare_false_events = sum(
            len(attempt["prepare_false_positive_events"])
            for attempt in self.attempts
        )
        prepare_false_attempts = sum(
            bool(attempt["prepare_false_positive_events"])
            for attempt in self.attempts
        )
        return {
            "evaluation_version": EVALUATION_VERSION,
            "mode": self.mode,
            "attempt_count": len(self.attempts),
            "overall": {
                "first_event_successes": sum(
                    bool(attempt["first_event_success"])
                    for attempt in self.attempts
                ),
                "eventual_target_detections": sum(
                    bool(attempt["target_event_detected"])
                    for attempt in self.attempts
                ),
                "duplicate_emissions": sum(
                    int(attempt["duplicate_emissions"])
                    for attempt in self.attempts
                ),
                "unexpected_action_events": sum(
                    len(attempt["unexpected_action_events"])
                    for attempt in self.attempts
                ),
            },
            "preparation": {
                "frame_count": prepare_frame_count,
                "false_event_count": prepare_false_events,
                "attempts_with_false_events": prepare_false_attempts,
                "false_event_attempt_rate": (
                    prepare_false_attempts / len(self.attempts)
                    if self.attempts
                    else 0.0
                ),
                "false_events_per_100_frames": (
                    100.0 * prepare_false_events / prepare_frame_count
                    if prepare_frame_count
                    else 0.0
                ),
            },
            "metrics": metrics,
            "confusion": {
                expected: dict(predictions)
                for expected, predictions in sorted(confusion.items())
            },
            "attempts": list(self.attempts),
        }


@dataclass
class AttackAttemptEvaluator:
    """Explain one guided attack attempt using existing queue transitions."""

    expected_attack: str
    attempt_index: int
    emitted_seals: list[str] = field(default_factory=list)
    accepted_seals: list[str] = field(default_factory=list)
    attacks_observed: list[str] = field(default_factory=list)
    rejection_reasons: Counter[str] = field(default_factory=Counter)
    raw_labels: Counter[str] = field(default_factory=Counter)
    duplicate_ignored: int = 0
    timeout_cleared: int = 0
    max_length_cleared: int = 0
    cooldown_suppressed: int = 0

    def __post_init__(self) -> None:
        if self.expected_attack not in ATTACK_LABELS:
            raise ValueError(f"unsupported attack: {self.expected_attack}")
        if self.attempt_index < 1:
            raise ValueError("attempt_index must be positive")

    @property
    def expected_seals(self) -> tuple[str, ...]:
        return ATTACK_SEALS[self.expected_attack]

    def record(self, output: Mapping[str, Any], *, phase: str = "action") -> None:
        if phase != "action":
            return
        hand = output["hand"]
        queue = output["queue"]
        raw_label = str(hand["raw"]["label"])
        self.raw_labels[raw_label] += 1
        if hand["rejection_reason"]:
            self.rejection_reasons[str(hand["rejection_reason"])] += 1
        if hand["emitted_seal"]:
            self.emitted_seals.append(str(hand["emitted_seal"]))
        if queue["accepted_seal"]:
            self.accepted_seals.append(str(queue["accepted_seal"]))
        self.duplicate_ignored += int(bool(queue["duplicate_ignored"]))
        self.timeout_cleared += int(bool(queue["timeout_cleared"]))
        self.max_length_cleared += int(bool(queue["max_length_cleared"]))
        self.cooldown_suppressed += int(bool(queue["cooldown_suppressed"]))
        if output["attack"] is not None:
            self.attacks_observed.append(str(output["attack"]["name"]))

    def _matched_prefix_length(self) -> int:
        matched = 0
        for observed in self.accepted_seals:
            if matched < len(self.expected_seals) and observed == self.expected_seals[matched]:
                matched += 1
        return matched

    def finish(self) -> dict[str, Any]:
        success = self.expected_attack in self.attacks_observed
        if success:
            failure_reason = None
        elif self.attacks_observed:
            failure_reason = "wrong_attack"
        elif self.cooldown_suppressed:
            failure_reason = "cooldown_suppressed"
        elif self.max_length_cleared:
            failure_reason = "max_length_cleared"
        elif self.timeout_cleared:
            failure_reason = "seal_timeout"
        elif not self.accepted_seals:
            failure_reason = "no_seals_recognized"
        else:
            failure_reason = "incomplete_sequence"
        matched = self._matched_prefix_length()
        return {
            "attempt_type": "attack",
            "expected_attack": self.expected_attack,
            "expected_seals": list(self.expected_seals),
            "attempt_index": self.attempt_index,
            "success": success,
            "failure_reason": failure_reason,
            "matched_prefix_length": matched,
            "missing_expected_seals": list(self.expected_seals[matched:]),
            "emitted_seals": list(self.emitted_seals),
            "accepted_seals": list(self.accepted_seals),
            "attacks_observed": list(self.attacks_observed),
            "dominant_raw_prediction": _dominant(self.raw_labels),
            "raw_label_counts": dict(self.raw_labels),
            "rejection_reasons": dict(self.rejection_reasons),
            "queue_transitions": {
                "duplicate_ignored": self.duplicate_ignored,
                "timeout_cleared": self.timeout_cleared,
                "max_length_cleared": self.max_length_cleared,
                "cooldown_suppressed": self.cooldown_suppressed,
            },
        }


class AttackEvaluationSession:
    def __init__(self) -> None:
        self.attempts: list[dict[str, Any]] = []

    def add(self, attempt: Mapping[str, Any]) -> None:
        self.attempts.append(dict(attempt))

    def summary(self) -> dict[str, Any]:
        metrics: dict[str, Any] = {}
        for attack in ATTACK_LABELS:
            attempts = [
                attempt
                for attempt in self.attempts
                if attempt["expected_attack"] == attack
            ]
            successes = sum(bool(attempt["success"]) for attempt in attempts)
            failure_reasons = Counter(
                str(attempt["failure_reason"])
                for attempt in attempts
                if attempt["failure_reason"] is not None
            )
            metrics[attack] = {
                "expected_seals": list(ATTACK_SEALS[attack]),
                "attempts": len(attempts),
                "successes": successes,
                "success_rate": successes / len(attempts) if attempts else None,
                "failure_reasons": dict(failure_reasons),
            }
        return {
            "evaluation_version": EVALUATION_VERSION,
            "mode": "attack",
            "attempt_count": len(self.attempts),
            "metrics": metrics,
            "attempts": list(self.attempts),
        }


EVALUATION_CSV_FIELDS = (
    "mode",
    "target",
    "attempt_index",
    "success",
    "first_event_success",
    "target_event_detected",
    "prediction",
    "failure_reason",
    "detection_latency_ms",
    "expected_sequence",
    "observed_sequence",
    "prepare_false_positive_events",
    "unexpected_action_events",
    "duplicate_emissions",
    "details_json",
)


def write_evaluation_reports(
    summary: Mapping[str, Any],
    *,
    json_path: Optional[Path] = None,
    csv_path: Optional[Path] = None,
) -> None:
    """Write summary JSON and compact per-attempt CSV without camera images."""
    if json_path is None and csv_path is None:
        return
    if json_path is not None and csv_path is not None:
        if json_path.resolve() == csv_path.resolve():
            raise ValueError("evaluation JSON and CSV must use different files")
    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if csv_path is None:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=EVALUATION_CSV_FIELDS)
        writer.writeheader()
        for attempt in summary["attempts"]:
            is_attack = attempt["attempt_type"] == "attack"
            row = {
                "mode": "attack" if is_attack else attempt["mode"],
                "target": (
                    attempt["expected_attack"]
                    if is_attack
                    else attempt["expected_label"]
                ),
                "attempt_index": attempt["attempt_index"],
                "success": attempt["success"],
                "first_event_success": attempt.get(
                    "first_event_success",
                    attempt["success"],
                ),
                "target_event_detected": attempt.get(
                    "target_event_detected",
                    attempt["success"],
                ),
                "prediction": (
                    "|".join(attempt["attacks_observed"])
                    if is_attack
                    else attempt["system_prediction"]
                ),
                "failure_reason": attempt.get("failure_reason"),
                "detection_latency_ms": attempt.get("detection_latency_ms"),
                "expected_sequence": (
                    ">".join(attempt["expected_seals"]) if is_attack else ""
                ),
                "observed_sequence": (
                    ">".join(attempt["accepted_seals"])
                    if is_attack
                    else "|".join(attempt["action_events"])
                ),
                "prepare_false_positive_events": "|".join(
                    attempt.get("prepare_false_positive_events", [])
                ),
                "unexpected_action_events": "|".join(
                    attempt.get("unexpected_action_events", [])
                ),
                "duplicate_emissions": attempt.get("duplicate_emissions", 0),
                "details_json": json.dumps(attempt, separators=(",", ":")),
            }
            writer.writerow(row)
