"""Combined hand-sign, body-movement, and attack recognition pipeline.

This module contains no UI or simulation code.  It accepts OpenCV BGR frames
and returns structured predictions that a camera test runner (or a future
frontend integration) can consume.
"""

from __future__ import annotations

import ast
import math
import time
import urllib.request
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Optional, Sequence

import cv2
import mediapipe as mp
import numpy as np
import onnxruntime as ort
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from .hand_config import (
    HAND_CONFIDENCE_THRESHOLDS,
    HAND_MARGIN_THRESHOLDS,
)
from .pipeline_config import (
    DEFAULT_PIPELINE_CONFIG,
    AttackQueueConfig,
    BodyMovementConfig,
    HandFusionConfig,
    HandTemporalConfig,
    PipelineConfig,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HAND_MODEL = REPO_ROOT / "cv_model/models/best_model_A.onnx"
DEFAULT_HAND_LANDMARKER = REPO_ROOT / "cv_model/models/hand_landmarker.task"
DEFAULT_POSE_MODEL = REPO_ROOT / "pose_landmarker.task"
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)

FALLBACK_HAND_CLASSES = (
    "bird",
    "boar",
    "dog",
    "dragon",
    "hare",
    "horse",
    "monkey",
    "ox",
    "ram",
    "rat",
    "snake",
    "tiger",
    "zero",
)


@dataclass(frozen=True)
class Classification:
    label: str
    confidence: float
    second_label: str
    second_confidence: float
    probabilities: tuple[float, ...] = ()

    @property
    def margin(self) -> float:
        return self.confidence - self.second_confidence


@dataclass(frozen=True)
class HandResult:
    raw: Classification
    accepted_label: str
    rejection_reason: Optional[str]
    stable_label: str
    emitted_seal: Optional[str]
    classification_bbox: tuple[int, int, int, int]
    hand_bbox: Optional[tuple[int, int, int, int]]
    landmarks: tuple[tuple[tuple[float, float], ...], ...] = ()
    center_prediction: Optional[Classification] = None
    roi_prediction: Optional[Classification] = None
    fusion_status: str = "center_only"
    timings_ms: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class BodyResult:
    raw_label: str
    stable_label: str
    emitted_movement: Optional[str]
    metrics: dict[str, float]
    landmarks: tuple[tuple[float, float], ...] = ()
    timings_ms: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class AttackEvent:
    name: str
    display_name: str
    timestamp: float


@dataclass(frozen=True)
class QueueUpdate:
    """Result of one attack-queue update, including non-attack state changes."""

    attack: Optional[AttackEvent]
    accepted_seal: Optional[str]
    duplicate_ignored: bool
    timeout_cleared: bool
    max_length_cleared: bool
    cooldown_suppressed: bool
    queue: tuple[str, ...]


@dataclass(frozen=True)
class FrameResult:
    hand: HandResult
    body: BodyResult
    attack: Optional[AttackEvent]
    seal_history: tuple[str, ...]
    queue_update: QueueUpdate
    processing_ms: float
    timings_ms: dict[str, float] = field(default_factory=dict)


class ConsensusFilter:
    """Sliding-window consensus with edge-triggered event emission."""

    def __init__(self, window_size: int, minimum_votes: int, neutral: str) -> None:
        if minimum_votes > window_size:
            raise ValueError("minimum_votes cannot exceed window_size")
        self.window: Deque[str] = deque(maxlen=window_size)
        self.minimum_votes = minimum_votes
        self.neutral = neutral
        self.stable = neutral
        self._last_emitted = neutral

    def update(self, label: str) -> tuple[str, Optional[str]]:
        self.window.append(label)
        counts = Counter(self.window)
        candidate, votes = counts.most_common(1)[0]
        if len(self.window) == self.window.maxlen and votes >= self.minimum_votes:
            self.stable = candidate

        emitted = None
        if self.stable == self.neutral:
            self._last_emitted = self.neutral
        elif self.stable != self._last_emitted:
            emitted = self.stable
            self._last_emitted = self.stable
        return self.stable, emitted

    def reset(self) -> None:
        self.window.clear()
        self.stable = self.neutral
        self._last_emitted = self.neutral


class EvidenceHandFilter:
    """Low-latency hysteresis using accepted evidence in recent frames."""

    def __init__(self, config: HandTemporalConfig = DEFAULT_PIPELINE_CONFIG.hand) -> None:
        self.config = config
        self.history: Deque[str] = deque(maxlen=config.window_size)
        self.stable = "zero"
        self._last_emitted = "zero"
        self.required_votes: dict[str, int] = {}

    def update(self, label: str) -> tuple[str, Optional[str]]:
        self.history.append(label)
        recent = list(self.history)
        neutral_frames = self.config.neutral_frames
        if (
            label == "zero"
            and len(recent) >= neutral_frames
            and recent[-neutral_frames:] == ["zero"] * neutral_frames
        ):
            self.stable = "zero"
            # Do not let evidence from the previous held sign make the next
            # recognition fire after only one new observation.
            self.history.clear()
            self.history.extend(["zero"] * neutral_frames)
        else:
            non_neutral = Counter(item for item in recent if item != "zero")
            if non_neutral:
                candidate, votes = non_neutral.most_common(1)[0]
                required = self.required_votes.get(candidate, self.config.required_votes)
                if votes >= required:
                    self.stable = candidate

        emitted = None
        if self.stable == "zero":
            self._last_emitted = "zero"
        elif self.stable != self._last_emitted:
            emitted = self.stable
            self._last_emitted = self.stable
        return self.stable, emitted

    def reset(self) -> None:
        self.history.clear()
        self.stable = "zero"
        self._last_emitted = "zero"

    def make_stricter(self, labels: Sequence[str]) -> None:
        for label in labels:
            self.required_votes[label] = self.config.calibrated_required_votes


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values)


def _square_bbox(
    xs: Sequence[float],
    ys: Sequence[float],
    width: int,
    height: int,
    padding: float = 0.28,
) -> tuple[int, int, int, int]:
    x_min, x_max = min(xs) * width, max(xs) * width
    y_min, y_max = min(ys) * height, max(ys) * height
    side = max(x_max - x_min, y_max - y_min, 32.0) * (1.0 + 2.0 * padding)
    cx, cy = (x_min + x_max) / 2.0, (y_min + y_max) / 2.0
    x1, y1 = int(round(cx - side / 2)), int(round(cy - side / 2))
    x2, y2 = int(round(cx + side / 2)), int(round(cy + side / 2))

    # Shift the square back into the frame before clipping it.
    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > width:
        x1 -= x2 - width
        x2 = width
    if y2 > height:
        y1 -= y2 - height
        y2 = height
    return max(0, x1), max(0, y1), min(width, x2), min(height, y2)


def _center_square_bbox(width: int, height: int) -> tuple[int, int, int, int]:
    """Match training preprocessing: largest centered square from the frame."""
    side = min(width, height)
    x1 = (width - side) // 2
    y1 = (height - side) // 2
    return x1, y1, x1 + side, y1 + side


class HandSignRecognizer:
    """Training-matched ONNX classification with stabilized hand-ROI fusion."""

    def __init__(
        self,
        model_path: Path = DEFAULT_HAND_MODEL,
        landmarker_path: Path = DEFAULT_HAND_LANDMARKER,
        confidence_threshold: Optional[float] = None,
        temporal_config: HandTemporalConfig = DEFAULT_PIPELINE_CONFIG.hand,
        fusion_config: HandFusionConfig = DEFAULT_PIPELINE_CONFIG.hand_fusion,
    ) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Hand-sign ONNX model not found: {model_path}")
        if not landmarker_path.exists():
            raise FileNotFoundError(f"MediaPipe hand model not found: {landmarker_path}")

        self.confidence_override = confidence_threshold
        self.confidence_thresholds = dict(HAND_CONFIDENCE_THRESHOLDS)
        self.margin_thresholds = dict(HAND_MARGIN_THRESHOLDS)
        self.session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        metadata = self.session.get_modelmeta().custom_metadata_map
        self.classes = self._parse_class_names(metadata.get("names"))

        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(landmarker_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.45,
            min_hand_presence_confidence=0.45,
            min_tracking_confidence=0.45,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.filter = EvidenceHandFilter(temporal_config)
        self.fusion_config = fusion_config
        self._smoothed_hand_bbox: Optional[np.ndarray] = None
        self._missing_hand_frames = 0
        self._last_timestamp_ms = -1

    @staticmethod
    def _parse_class_names(raw_names: Optional[str]) -> tuple[str, ...]:
        if raw_names:
            parsed = ast.literal_eval(raw_names)
            if isinstance(parsed, dict):
                return tuple(str(parsed[index]) for index in sorted(parsed))
        return FALLBACK_HAND_CLASSES

    def _classify_crop_timed(
        self, crop_bgr: np.ndarray
    ) -> tuple[Classification, dict[str, float]]:
        started = time.perf_counter()
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
        tensor = resized.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
        inference_started = time.perf_counter()
        output = np.asarray(
            self.session.run([self.output_name], {self.input_name: tensor})[0]
        ).reshape(-1)
        inference_finished = time.perf_counter()
        probabilities = output if np.isclose(output.sum(), 1.0, atol=1e-3) else _softmax(output)
        prediction = self._classification_from_probabilities(probabilities)
        finished = time.perf_counter()
        return prediction, {
            "preprocess_ms": (inference_started - started) * 1000.0,
            "onnx_ms": (inference_finished - inference_started) * 1000.0,
            "postprocess_ms": (finished - inference_finished) * 1000.0,
        }

    def _classify_crop(self, crop_bgr: np.ndarray) -> Classification:
        prediction, _timings = self._classify_crop_timed(crop_bgr)
        return prediction

    def _classification_from_probabilities(
        self, probabilities: Sequence[float]
    ) -> Classification:
        values = np.asarray(probabilities, dtype=np.float64).reshape(-1)
        total = float(values.sum())
        if total <= 0.0:
            raise ValueError("classification probabilities must have a positive sum")
        values = values / total
        ranking = np.argsort(values)[::-1]
        first, second = int(ranking[0]), int(ranking[1])
        return Classification(
            label=self.classes[first],
            confidence=float(values[first]),
            second_label=self.classes[second],
            second_confidence=float(values[second]),
            probabilities=tuple(float(value) for value in values),
        )

    def _smooth_hand_bbox(
        self, bbox: tuple[int, int, int, int]
    ) -> tuple[int, int, int, int]:
        current = np.asarray(bbox, dtype=np.float64)
        if self._smoothed_hand_bbox is None:
            smoothed = current
        else:
            alpha = self.fusion_config.bbox_smoothing_alpha
            smoothed = alpha * current + (1.0 - alpha) * self._smoothed_hand_bbox
        self._smoothed_hand_bbox = smoothed
        return tuple(int(round(value)) for value in smoothed)

    def _fuse_predictions(
        self, center: Classification, roi: Classification
    ) -> Classification:
        if not center.probabilities or not roi.probabilities:
            return center
        center_values = np.asarray(center.probabilities, dtype=np.float64)
        roi_values = np.asarray(roi.probabilities, dtype=np.float64)
        if center_values.shape != roi_values.shape:
            raise ValueError("center and ROI predictions must have matching classes")
        weight = self.fusion_config.center_weight
        return self._classification_from_probabilities(
            weight * center_values + (1.0 - weight) * roi_values
        )

    def _should_gate_without_hands(self, prediction: Classification) -> bool:
        allowed = {"zero", *self.fusion_config.no_hand_fallback_labels}
        return (
            self._missing_hand_frames >= self.fusion_config.absence_grace_frames
            and prediction.label not in allowed
        )

    def _accept(
        self, prediction: Classification, detected_hands: int
    ) -> tuple[str, Optional[str]]:
        # Rat and Ram are the model's dominant live-camera confusion pair.
        # Do not use the detected-hand count alone: the camera evaluator showed
        # genuine Ram alternating between one and two detected hands.  Resolve
        # only when the model itself exposes meaningful Rat probability.
        if (
            prediction.label == "ram"
            and prediction.second_label == "rat"
            and prediction.second_confidence >= 0.12
        ):
            return "rat", "resolved_ram_to_rat_pair_probability"

        # In the recorded camera trials, genuine lower-confidence Dog frames
        # consistently had Hare, Monkey, or Ox as runner-up.  Tiger frames
        # misclassified as Dog instead had Snake (or Boar) as runner-up.  Keep
        # the original strict floor for those unsafe pairings while allowing
        # the camera-tested Dog cluster to use its calibrated lower threshold.
        if (
            prediction.label == "dog"
            and prediction.confidence < 0.80
            and prediction.second_label not in {"hare", "monkey", "ox"}
        ):
            return "zero", "dog_pair_guard"

        confidence_threshold = (
            self.confidence_override
            if self.confidence_override is not None
            else self.confidence_thresholds[prediction.label]
        )
        if prediction.confidence < confidence_threshold:
            return "zero", f"confidence<{confidence_threshold:.2f}"
        margin_threshold = self.margin_thresholds[prediction.label]
        if prediction.margin < margin_threshold:
            return "zero", f"margin<{margin_threshold:.2f}"
        if prediction.label == "zero":
            return "zero", "model_zero"
        return prediction.label, None

    def calibrate_neutral(
        self, samples: Sequence[Classification]
    ) -> dict[str, tuple[float, float]]:
        """Tighten session thresholds for false labels seen in neutral poses."""
        adjustments: dict[str, tuple[float, float]] = {}
        for label in self.classes:
            if label == "zero":
                continue
            matching = [sample for sample in samples if sample.label == label]
            if not matching:
                continue
            confidence = min(
                0.95,
                self.confidence_thresholds[label] + 0.12,
                max(sample.confidence for sample in matching) + 0.015,
            )
            margin = min(
                0.80,
                self.margin_thresholds[label] + 0.15,
                max(sample.margin for sample in matching) + 0.015,
            )
            new_confidence = max(self.confidence_thresholds[label], confidence)
            new_margin = max(self.margin_thresholds[label], margin)
            if (
                new_confidence > self.confidence_thresholds[label]
                or new_margin > self.margin_thresholds[label]
            ):
                self.confidence_thresholds[label] = new_confidence
                self.margin_thresholds[label] = new_margin
                adjustments[label] = (new_confidence, new_margin)
        self.filter.make_stricter(adjustments)
        return adjustments

    def _video_timestamp_ms(self, timestamp: Optional[float]) -> int:
        candidate = int((time.monotonic() if timestamp is None else timestamp) * 1000)
        self._last_timestamp_ms = max(candidate, self._last_timestamp_ms + 1)
        return self._last_timestamp_ms

    def process(
        self, frame_bgr: np.ndarray, timestamp: Optional[float] = None
    ) -> HandResult:
        started = time.perf_counter()
        height, width = frame_bgr.shape[:2]
        classification_bbox = _center_square_bbox(width, height)
        cx1, cy1, cx2, cy2 = classification_bbox
        center_prediction, center_timings = self._classify_crop_timed(
            frame_bgr[cy1:cy2, cx1:cx2]
        )

        hand_preprocess_started = time.perf_counter()
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        hand_detection_started = time.perf_counter()
        detected = self.landmarker.detect_for_video(
            mp_image, self._video_timestamp_ms(timestamp)
        )
        hand_detection_finished = time.perf_counter()
        detected_hands = len(detected.hand_landmarks)
        prediction = center_prediction
        roi_prediction = None
        roi_timings = {"preprocess_ms": 0.0, "onnx_ms": 0.0, "postprocess_ms": 0.0}
        hand_bbox = None
        if detected.hand_landmarks:
            self._missing_hand_frames = 0
            points = [point for hand in detected.hand_landmarks for point in hand]
            raw_hand_bbox = _square_bbox(
                [point.x for point in points],
                [point.y for point in points],
                width,
                height,
            )
            hand_bbox = self._smooth_hand_bbox(raw_hand_bbox)
            hx1, hy1, hx2, hy2 = hand_bbox
            if hx2 > hx1 and hy2 > hy1:
                roi_prediction, roi_timings = self._classify_crop_timed(
                    frame_bgr[hy1:hy2, hx1:hx2]
                )
                prediction = self._fuse_predictions(center_prediction, roi_prediction)
            if roi_prediction is None:
                fusion_status = "center_only_invalid_roi"
            elif center_prediction.label == roi_prediction.label:
                fusion_status = "center_roi_agree"
            else:
                fusion_status = "center_roi_fused"
        else:
            self._missing_hand_frames += 1
            if self._missing_hand_frames >= self.fusion_config.absence_grace_frames:
                self._smoothed_hand_bbox = None
                fusion_status = "center_no_hands_gated"
            else:
                fusion_status = "center_no_hands_grace"

        if self._should_gate_without_hands(prediction):
            accepted_label, rejection_reason = "zero", "no_hand_landmarks"
        else:
            accepted_label, rejection_reason = self._accept(prediction, detected_hands)
        stable, emitted = self.filter.update(accepted_label)
        landmark_points = tuple(
            tuple((point.x, point.y) for point in hand)
            for hand in detected.hand_landmarks
        )
        finished = time.perf_counter()
        measured_ms = (
            sum(center_timings.values())
            + (hand_detection_started - hand_preprocess_started) * 1000.0
            + (hand_detection_finished - hand_detection_started) * 1000.0
            + sum(roi_timings.values())
        )
        hand_total_ms = (finished - started) * 1000.0
        timings_ms = {
            "hand_center_preprocess_ms": center_timings["preprocess_ms"],
            "hand_center_onnx_ms": center_timings["onnx_ms"],
            "hand_center_postprocess_ms": center_timings["postprocess_ms"],
            "hand_landmarker_preprocess_ms": (
                hand_detection_started - hand_preprocess_started
            )
            * 1000.0,
            "hand_landmarker_ms": (
                hand_detection_finished - hand_detection_started
            )
            * 1000.0,
            "hand_roi_preprocess_ms": roi_timings["preprocess_ms"],
            "hand_roi_onnx_ms": roi_timings["onnx_ms"],
            "hand_roi_postprocess_ms": roi_timings["postprocess_ms"],
            "hand_postprocess_ms": max(0.0, hand_total_ms - measured_ms),
            "hand_total_ms": hand_total_ms,
        }
        return HandResult(
            raw=prediction,
            accepted_label=accepted_label,
            rejection_reason=rejection_reason,
            stable_label=stable,
            emitted_seal=emitted,
            classification_bbox=classification_bbox,
            hand_bbox=hand_bbox,
            landmarks=landmark_points,
            center_prediction=center_prediction,
            roi_prediction=roi_prediction,
            fusion_status=fusion_status,
            timings_ms=timings_ms,
        )

    def reset(self) -> None:
        self.filter.reset()
        self._smoothed_hand_bbox = None
        self._missing_hand_frames = 0

    def close(self) -> None:
        self.landmarker.close()


def _point_distance(a: object, b: object) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def _midpoint(a: object, b: object) -> np.ndarray:
    return np.asarray([(a.x + b.x) / 2, (a.y + b.y) / 2, (a.z + b.z) / 2])


class BodyMovementRecognizer:
    """Stateful geometric body-movement recognizer based on MediaPipe Pose."""

    def __init__(
        self,
        pose_model_path: Path = DEFAULT_POSE_MODEL,
        config: BodyMovementConfig = DEFAULT_PIPELINE_CONFIG.body,
    ) -> None:
        if not pose_model_path.exists():
            pose_model_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Downloading MediaPipe pose model to {pose_model_path} ...")
            urllib.request.urlretrieve(POSE_MODEL_URL, pose_model_path)

        options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(pose_model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.50,
            min_pose_presence_confidence=0.50,
            min_tracking_confidence=0.50,
            output_segmentation_masks=False,
        )
        self.config = config
        self.landmarker = vision.PoseLandmarker.create_from_options(options)
        self.filter = ConsensusFilter(
            window_size=config.consensus_window,
            minimum_votes=config.consensus_votes,
            neutral="idle",
        )
        self.hip_history: Deque[float] = deque(maxlen=config.jump_history_frames)
        self.previous_shoulder_x: Optional[float] = None
        self.previous_time: Optional[float] = None
        self._last_timestamp_ms = -1

    def process(self, frame_bgr: np.ndarray, timestamp: float) -> BodyResult:
        started = time.perf_counter()
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        detection_started = time.perf_counter()
        timestamp_ms = max(int(timestamp * 1000), self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms
        detected = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        detection_finished = time.perf_counter()
        if not detected.pose_landmarks:
            stable, emitted = self.filter.update("idle")
            self.previous_time = timestamp
            finished = time.perf_counter()
            return BodyResult(
                "idle",
                stable,
                emitted,
                {},
                timings_ms={
                    "pose_preprocess_ms": (detection_started - started) * 1000.0,
                    "pose_landmarker_ms": (
                        detection_finished - detection_started
                    )
                    * 1000.0,
                    "body_postprocess_ms": (finished - detection_finished) * 1000.0,
                    "body_total_ms": (finished - started) * 1000.0,
                },
            )

        landmarks = detected.pose_landmarks[0]
        world = detected.pose_world_landmarks[0] if detected.pose_world_landmarks else landmarks
        ls, rs = landmarks[11], landmarks[12]
        le, re = landmarks[13], landmarks[14]
        lw, rw = landmarks[15], landmarks[16]
        lh, rh = landmarks[23], landmarks[24]

        shoulder = _midpoint(ls, rs)
        hip = _midpoint(lh, rh)
        shoulder_width = max(_point_distance(ls, rs), 1e-4)
        hip_y = float(hip[1])
        lean_x = float(shoulder[0] - hip[0])

        dt = timestamp - self.previous_time if self.previous_time is not None else 0.0
        lateral_velocity = 0.0
        if self.previous_shoulder_x is not None and dt > 1e-3:
            lateral_velocity = abs(float(shoulder[0]) - self.previous_shoulder_x) / dt

        jump_height = 0.0
        if len(self.hip_history) >= self.config.jump_baseline_frames:
            baseline = float(np.median(self.hip_history))
            jump_height = baseline - hip_y
        self.hip_history.append(hip_y)

        wls, wrs = world[11], world[12]
        wlh, wrh = world[23], world[24]
        wlw, wrw = world[15], world[16]
        torso = _midpoint(wls, wrs) - _midpoint(wlh, wrh)
        torso_norm = max(float(np.linalg.norm(torso)), 1e-6)
        cosine = float(np.clip(np.dot(torso, np.asarray([0.0, -1.0, 0.0])) / torso_norm, -1.0, 1.0))
        torso_angle = math.degrees(math.acos(cosine))
        depth = self.config.arms_back_depth_threshold
        arms_back = wlw.z > wlh.z + depth and wrw.z > wrh.z + depth

        crossed_distance = (
            _point_distance(lw, re) + _point_distance(rw, le)
        ) / (2.0 * shoulder_width)

        # Body movement events remain independent from hand-seal attacks. Dodge
        # and stone are attacks in the hand catalog, not body movement labels.
        raw_label = self.classify_metrics(
            jump_height=jump_height,
            torso_angle=torso_angle,
            arms_back=arms_back,
            lean_x=lean_x,
        )

        stable, emitted = self.filter.update(raw_label)
        self.previous_shoulder_x = float(shoulder[0])
        self.previous_time = timestamp
        metrics = {
            "lean_x": lean_x,
            "jump_height": jump_height,
            "lateral_velocity": lateral_velocity,
            "torso_angle": torso_angle,
            "crossed_distance": crossed_distance,
        }
        landmark_points = tuple((point.x, point.y) for point in landmarks)
        finished = time.perf_counter()
        return BodyResult(
            raw_label,
            stable,
            emitted,
            metrics,
            landmark_points,
            timings_ms={
                "pose_preprocess_ms": (detection_started - started) * 1000.0,
                "pose_landmarker_ms": (detection_finished - detection_started)
                * 1000.0,
                "body_postprocess_ms": (finished - detection_finished) * 1000.0,
                "body_total_ms": (finished - started) * 1000.0,
            },
        )

    def classify_metrics(
        self,
        *,
        jump_height: float,
        torso_angle: float,
        arms_back: bool,
        lean_x: float,
    ) -> str:
        """Classify already-computed geometry without invoking MediaPipe."""
        if jump_height > self.config.jump_height_threshold:
            return "jumping"
        if torso_angle > self.config.torso_angle_threshold and arms_back:
            return "naruto_run"
        if lean_x > self.config.lean_threshold:
            return "bending_right"
        if lean_x < -self.config.lean_threshold:
            return "bending_left"
        return "idle"

    def reset(self) -> None:
        self.filter.reset()
        self.hip_history.clear()
        self.previous_shoulder_x = None
        self.previous_time = None

    def close(self) -> None:
        self.landmarker.close()


@dataclass(frozen=True)
class AttackDefinition:
    name: str
    display_name: str
    seals: tuple[str, ...]


ATTACKS = (
    AttackDefinition("homura", "FIRE ATTACK", ("tiger", "horse")),
    AttackDefinition("ikazuchi", "LIGHTNING DODGE", ("hare",)),
    AttackDefinition("ryusui", "WATER ATTACK", ("snake", "dragon")),
    AttackDefinition("daichi", "SAND ATTACK", ("monkey", "ox")),
    AttackDefinition("shippu", "WIND ATTACK", ("dog", "rat")),
)


class AttackRecognizer:
    """Matches debounced hand-seal sequences into attacks."""

    def __init__(
        self,
        seal_timeout: Optional[float] = None,
        *,
        config: AttackQueueConfig = DEFAULT_PIPELINE_CONFIG.attacks,
    ) -> None:
        if seal_timeout is not None:
            config = AttackQueueConfig(
                seal_timeout=seal_timeout,
                trigger_cooldown=config.trigger_cooldown,
                max_seals=config.max_seals,
            )
        longest_attack = max(len(attack.seals) for attack in ATTACKS)
        if config.max_seals < longest_attack:
            raise ValueError("max_seals cannot be shorter than the longest attack")
        self.config = config
        self.seals: Deque[tuple[str, float]] = deque(maxlen=config.max_seals)
        self.last_triggered: dict[str, float] = {}

    @property
    def seal_labels(self) -> tuple[str, ...]:
        return tuple(label for label, _ in self.seals)

    def _sequence_matches(self, attack: AttackDefinition) -> bool:
        events = list(self.seals)
        if tuple(label for label, _ in events) != attack.seals:
            return False
        return all(
            events[index][1] - events[index - 1][1] <= self.config.seal_timeout
            for index in range(1, len(events))
        )

    def _trigger(
        self, attack: AttackDefinition, now: float
    ) -> tuple[Optional[AttackEvent], bool]:
        self.seals.clear()
        if (
            now - self.last_triggered.get(attack.name, -math.inf)
            < self.config.trigger_cooldown
        ):
            return None, True
        self.last_triggered[attack.name] = now
        return AttackEvent(attack.name, attack.display_name, now), False

    def update_detailed(
        self,
        emitted_seal: Optional[str],
        now: float,
    ) -> QueueUpdate:
        timeout_cleared = False
        if self.seals and now - self.seals[-1][1] > self.config.seal_timeout:
            self.seals.clear()
            timeout_cleared = True

        accepted_seal = None
        duplicate_ignored = False
        max_length_cleared = False
        cooldown_suppressed = False
        attack = None

        if emitted_seal and emitted_seal != "zero":
            if self.seals and self.seals[-1][0] == emitted_seal:
                duplicate_ignored = True
            else:
                accepted_seal = emitted_seal
                self.seals.append((emitted_seal, now))
                for definition in ATTACKS:
                    if not self._sequence_matches(definition):
                        continue
                    attack, cooldown_suppressed = self._trigger(definition, now)
                    break
                if attack is None and not cooldown_suppressed:
                    if len(self.seals) >= self.config.max_seals:
                        self.seals.clear()
                        max_length_cleared = True

        return QueueUpdate(
            attack=attack,
            accepted_seal=accepted_seal,
            duplicate_ignored=duplicate_ignored,
            timeout_cleared=timeout_cleared,
            max_length_cleared=max_length_cleared,
            cooldown_suppressed=cooldown_suppressed,
            queue=self.seal_labels,
        )

    def update(
        self,
        emitted_seal: Optional[str],
        now: float,
    ) -> Optional[AttackEvent]:
        return self.update_detailed(emitted_seal, now).attack

    def clear_queue(self) -> None:
        self.seals.clear()

    def reset(self) -> None:
        self.seals.clear()
        self.last_triggered.clear()


class CombinedNarutoPipeline:
    """Top-level frame processor shared by camera and future integrations."""

    def __init__(
        self,
        hand_model: Path = DEFAULT_HAND_MODEL,
        hand_landmarker: Path = DEFAULT_HAND_LANDMARKER,
        pose_model: Path = DEFAULT_POSE_MODEL,
        hand_confidence: Optional[float] = None,
        config: PipelineConfig = DEFAULT_PIPELINE_CONFIG,
    ) -> None:
        self.config = config
        self.hand = HandSignRecognizer(
            model_path=Path(hand_model),
            landmarker_path=Path(hand_landmarker),
            confidence_threshold=hand_confidence,
            temporal_config=config.hand,
            fusion_config=config.hand_fusion,
        )
        self.body = BodyMovementRecognizer(Path(pose_model), config.body)
        self.attacks = AttackRecognizer(config=config.attacks)

    def process(self, frame_bgr: np.ndarray, timestamp: Optional[float] = None) -> FrameResult:
        started = time.perf_counter()
        now = time.monotonic() if timestamp is None else timestamp
        hand_result = self.hand.process(frame_bgr, now)
        body_result = self.body.process(frame_bgr, now)
        attack_started = time.perf_counter()
        queue_update = self.attacks.update_detailed(hand_result.emitted_seal, now)
        attack_finished = time.perf_counter()
        attack = queue_update.attack
        processing_ms = (time.perf_counter() - started) * 1000.0
        attack_queue_ms = (attack_finished - attack_started) * 1000.0
        timings_ms = {
            **hand_result.timings_ms,
            **body_result.timings_ms,
            "attack_queue_ms": attack_queue_ms,
            "pipeline_overhead_ms": max(
                0.0,
                processing_ms
                - hand_result.timings_ms.get("hand_total_ms", 0.0)
                - body_result.timings_ms.get("body_total_ms", 0.0)
                - attack_queue_ms,
            ),
            "recognition_total_ms": processing_ms,
        }
        return FrameResult(
            hand=hand_result,
            body=body_result,
            attack=attack,
            seal_history=queue_update.queue,
            queue_update=queue_update,
            processing_ms=processing_ms,
            timings_ms=timings_ms,
        )

    def reset(self) -> None:
        self.hand.reset()
        self.body.reset()
        self.attacks.reset()

    def close(self) -> None:
        self.hand.close()
        self.body.close()

    def __enter__(self) -> "CombinedNarutoPipeline":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
