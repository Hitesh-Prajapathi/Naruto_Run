"""The v1 hand decoder wiring: flag/env/CLI selection, calibration file, ``expected_next``,
the recognizer's v1 stage on synthetic probability streams, and the output/event contract.

Nothing here needs a camera, MediaPipe models or ONNX weights.  The recognizer is built with
``object.__new__`` and stubbed classifier/landmarker, so the real ``HandSignRecognizer.process``
glue runs.  What none of this can show is v1 behaviour on a real camera.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import yaml
from narutocv.inference import combined_pipeline
from narutocv.inference.combined_pipeline import (
    ATTACKS,
    AttackDefinition,
    AttackRecognizer,
    BodyResult,
    Classification,
    CombinedNarutoPipeline,
    EvidenceHandFilter,
    FrameResult,
    HandResult,
    HandSignRecognizer,
    QueueUpdate,
)
from narutocv.inference.events import PipelineEventDispatcher
from narutocv.inference.hand_decoder import (
    DEFAULT_CALIBRATION_PATH,
    HAND_DECODER_ENV,
    HAND_DECODER_PRIOR_ENV,
    V1HandDecoder,
    build_pipeline_config,
    calibrate_probabilities,
    load_calibration,
    resolve_hand_decoder,
    resolve_hand_decoder_prior,
)
from narutocv.inference.output_schema import (
    PipelineOutputSerializer,
    SerializationError,
    validate_pipeline_output_v1,
)
from narutocv.inference.pipeline_config import (
    DEFAULT_PIPELINE_CONFIG,
    HandFusionConfig,
    PipelineConfig,
)
from narutocv.models.calibration import TemperatureScaler
from narutocv.temporal.decoder import DEFAULT_LABELS, DecoderConfig, SealEvent

ROOT = Path(__file__).resolve().parents[2]
LABELS = tuple(DEFAULT_LABELS)
ZERO = LABELS.index("zero")


def _load_script(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(
        f"_script_{name}", ROOT / "scripts" / f"{name}.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def onehot(label: str, level: float = 0.92) -> list[float]:
    """``level`` on ``label``; the remainder on the idle class."""
    p = np.zeros(len(LABELS))
    p[ZERO] = 1.0 - level
    p[LABELS.index(label)] += level
    return p.tolist()


def run_stream(
    decoder: V1HandDecoder, frames: list[tuple[float, list[float]]]
) -> list[tuple[float, SealEvent]]:
    out = []
    for t_ms, probs in frames:
        step = decoder.update(probs, t_ms)
        if step.event is not None:
            out.append((t_ms, step.event))
    return out


def stream(label: str, *, idle_before: int, hold: int, idle_after: int, fps: float = 30.0):
    step = 1000.0 / fps
    seq = (
        [onehot("zero", 0.95)] * idle_before
        + [onehot(label)] * hold
        + [onehot("zero", 0.95)] * idle_after
    )
    return [(1000.0 + i * step, p) for i, p in enumerate(seq)]


# ---------------------------------------------------------------------------------------
# Flag selection: config, env, CLI
# ---------------------------------------------------------------------------------------
class TestFlagSelection:
    def test_default_is_legacy_everywhere(self) -> None:
        assert DEFAULT_PIPELINE_CONFIG.hand_decoder == "legacy"
        assert PipelineConfig().hand_decoder == "legacy"
        assert PipelineConfig().hand_decoder_use_prior is None
        assert resolve_hand_decoder(None, {}) == "legacy"
        assert build_pipeline_config(env={}).hand_decoder == "legacy"

    def test_config_validates_the_mode(self) -> None:
        assert PipelineConfig(hand_decoder="v1").hand_decoder == "v1"
        with pytest.raises(ValueError, match="hand_decoder"):
            PipelineConfig(hand_decoder="v2")

    def test_flag_beats_env_beats_default(self) -> None:
        env = {HAND_DECODER_ENV: "v1"}
        assert resolve_hand_decoder(None, env) == "v1"
        assert resolve_hand_decoder("legacy", env) == "legacy"
        assert resolve_hand_decoder("v1", {}) == "v1"
        assert resolve_hand_decoder(None, {HAND_DECODER_ENV: "  V1 "}) == "v1"
        assert resolve_hand_decoder(None, {HAND_DECODER_ENV: ""}) == "legacy"

    def test_invalid_values_are_errors(self) -> None:
        with pytest.raises(ValueError, match="hand_decoder"):
            resolve_hand_decoder("v3", {})
        with pytest.raises(ValueError, match=HAND_DECODER_ENV):
            resolve_hand_decoder(None, {HAND_DECODER_ENV: "fast"})
        with pytest.raises(ValueError, match=HAND_DECODER_PRIOR_ENV):
            resolve_hand_decoder_prior(None, {HAND_DECODER_PRIOR_ENV: "maybe"})

    def test_prior_flag_env(self) -> None:
        assert resolve_hand_decoder_prior(None, {}) is None
        assert resolve_hand_decoder_prior(None, {HAND_DECODER_PRIOR_ENV: "1"}) is True
        assert resolve_hand_decoder_prior(None, {HAND_DECODER_PRIOR_ENV: "off"}) is False
        assert resolve_hand_decoder_prior(False, {HAND_DECODER_PRIOR_ENV: "1"}) is False
        cfg = build_pipeline_config("v1", use_prior=True, env={})
        assert (cfg.hand_decoder, cfg.hand_decoder_use_prior) == ("v1", True)

    def test_transport_cli(self) -> None:
        from narutocv.transport.__main__ import parse_args

        assert parse_args([], {}).pipeline_config.hand_decoder == "legacy"
        assert parse_args(["--hand-decoder", "v1"], {}).pipeline_config.hand_decoder == "v1"
        env = {HAND_DECODER_ENV: "v1"}
        assert parse_args([], env).pipeline_config.hand_decoder == "v1"
        assert (
            parse_args(["--hand-decoder", "legacy"], env).pipeline_config.hand_decoder == "legacy"
        )
        args = parse_args(["--hand-decoder", "v1", "--hand-decoder-prior"], {})
        assert args.pipeline_config.hand_decoder_use_prior is True
        with pytest.raises(SystemExit):
            parse_args([], {HAND_DECODER_ENV: "nope"})
        with pytest.raises(SystemExit):
            parse_args(["--hand-decoder", "nope"], {})

    def test_transport_factory_receives_the_config(self) -> None:
        from narutocv.transport import pipeline as transport_pipeline

        captured: dict[str, Any] = {}

        class Spy:
            def __init__(self, **kwargs: Any) -> None:
                captured.update(kwargs)

        original = transport_pipeline.SessionScopedPipeline
        transport_pipeline.SessionScopedPipeline = Spy  # type: ignore[misc]
        try:
            cfg = PipelineConfig(hand_decoder="v1")
            transport_pipeline.session_scoped_pipeline_factory(body_only=True, config=cfg)()
        finally:
            transport_pipeline.SessionScopedPipeline = original  # type: ignore[misc]
        assert captured["config"].hand_decoder == "v1"
        assert captured["body_only"] is True

    @pytest.mark.parametrize("script", ["run_combined_camera", "evaluate_live_camera"])
    def test_script_cli(self, script: str, monkeypatch: pytest.MonkeyPatch) -> None:
        module = _load_script(script)
        extra = ["--mode", "hand"] if script == "evaluate_live_camera" else []
        monkeypatch.delenv(HAND_DECODER_ENV, raising=False)
        monkeypatch.delenv(HAND_DECODER_PRIOR_ENV, raising=False)

        monkeypatch.setattr(sys, "argv", [script, *extra])
        assert module.parse_args().pipeline_config.hand_decoder == "legacy"

        monkeypatch.setattr(sys, "argv", [script, *extra, "--hand-decoder", "v1"])
        assert module.parse_args().pipeline_config.hand_decoder == "v1"

        monkeypatch.setenv(HAND_DECODER_ENV, "v1")
        monkeypatch.setattr(sys, "argv", [script, *extra])
        assert module.parse_args().pipeline_config.hand_decoder == "v1"
        monkeypatch.setattr(sys, "argv", [script, *extra, "--hand-decoder", "legacy"])
        assert module.parse_args().pipeline_config.hand_decoder == "legacy"

        monkeypatch.setenv(HAND_DECODER_ENV, "bogus")
        monkeypatch.setattr(sys, "argv", [script, *extra])
        with pytest.raises(SystemExit):
            module.parse_args()

    def test_recognizer_rejects_an_unknown_mode_before_loading_anything(self) -> None:
        with pytest.raises(ValueError, match="hand_decoder"):
            HandSignRecognizer(hand_decoder="v2")


# ---------------------------------------------------------------------------------------
# Calibration file and manifest
# ---------------------------------------------------------------------------------------
class TestCalibrationFile:
    def test_placeholder_contents(self) -> None:
        bundle = load_calibration()
        assert bundle.path == DEFAULT_CALIBRATION_PATH
        assert bundle.scaler.temperature == 1.0
        # DecoderConfig defaults, except the grammar prior, which stays off so that
        # measurements are honest unless it is enabled explicitly.
        assert bundle.decoder == DecoderConfig(use_prior=False)
        assert bundle.provenance.startswith("identity placeholder")
        assert "contaminated" in bundle.provenance

    def test_use_prior_override(self) -> None:
        assert load_calibration(use_prior=True).decoder.use_prior is True
        assert load_calibration(use_prior=None).decoder.use_prior is False

    def test_unknown_or_missing_keys_are_errors(self, tmp_path: Path) -> None:
        path = tmp_path / "c.json"
        path.write_text(json.dumps({"temperature": 1.0, "decoder": {}, "bogus": 1}))
        with pytest.raises(ValueError, match="unknown"):
            load_calibration(path)
        path.write_text(json.dumps({"temperature": 1.0}))
        with pytest.raises(ValueError, match="missing"):
            load_calibration(path)

    def test_identity_at_temperature_one(self) -> None:
        rng = np.random.default_rng(0)
        scaler = TemperatureScaler(1.0)
        for _ in range(20):
            p = rng.dirichlet(np.ones(len(LABELS)) * 0.4)
            np.testing.assert_allclose(calibrate_probabilities(p, scaler), p, atol=1e-9)
        edge = np.zeros(len(LABELS))
        edge[3] = 1.0  # exact zeros are floored, not -inf
        out = calibrate_probabilities(edge, scaler)
        assert np.isfinite(out).all() and out[3] == pytest.approx(1.0)

    def test_temperature_above_one_flattens(self) -> None:
        p = np.array(onehot("tiger", 0.8))
        flat = calibrate_probabilities(p, TemperatureScaler(2.0))
        assert flat.max() < p.max() and flat.argmax() == p.argmax()

    def test_class_order_mismatch_fails_loudly(self) -> None:
        with pytest.raises(ValueError, match="class order"):
            V1HandDecoder.from_file(tuple(reversed(LABELS)))
        with pytest.raises(ValueError, match="class order"):
            V1HandDecoder.from_file(("a", "b"))

    def test_manifest_tracks_the_calibration_file(self) -> None:
        entries = yaml.safe_load((ROOT / "models" / "manifest.yaml").read_text())["models"]
        entry = next(e for e in entries if e["name"] == "calibration_v0.json")
        assert entry["url"] is None
        data = DEFAULT_CALIBRATION_PATH.read_bytes()
        assert entry["sha256"] == hashlib.sha256(data).hexdigest()
        assert entry["size"] == len(data)

    def test_fetch_models_verifies_it_and_detects_tampering(self, tmp_path: Path) -> None:
        fetch = _load_script_from(ROOT / "scripts" / "fetch_models.py", "_fetch_models")
        entries = yaml.safe_load(fetch.MANIFEST.read_text())["models"]
        entry = next(e for e in entries if e["name"] == "calibration_v0.json")
        assert fetch.check(fetch.MODELS_DIR / entry["name"], entry["sha256"], entry["size"]) is None
        copy = tmp_path / "calibration_v0.json"
        copy.write_bytes(DEFAULT_CALIBRATION_PATH.read_bytes() + b" ")
        assert fetch.check(copy, entry["sha256"], entry["size"]) is not None

    def test_downloaded_files_are_world_readable(self, tmp_path: Path) -> None:
        fetch = _load_script_from(ROOT / "scripts" / "fetch_models.py", "_fetch_models2")
        source = tmp_path / "src.bin"
        source.write_bytes(b"model bytes")
        dest = tmp_path / "out" / "model.bin"
        fetch.download(source.as_uri(), dest)
        assert dest.read_bytes() == b"model bytes"
        assert dest.stat().st_mode & 0o777 == 0o644
        assert not list(dest.parent.glob("*.part"))


def _load_script_from(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------------------
# AttackRecognizer.expected_next
# ---------------------------------------------------------------------------------------
class TestExpectedNext:
    def test_empty_queue_has_no_prior(self) -> None:
        assert AttackRecognizer().expected_next() == ()
        assert AttackRecognizer().expected_next(now=5.0) == ()

    @pytest.mark.parametrize(
        ("first", "following"),
        [("tiger", ("horse",)), ("snake", ("dragon",)), ("monkey", ("ox",)), ("dog", ("rat",))],
    )
    def test_second_seal_of_each_real_attack(self, first: str, following: tuple[str, ...]) -> None:
        recognizer = AttackRecognizer()
        recognizer.update(first, 1.0)
        assert recognizer.expected_next() == following
        assert recognizer.expected_next(now=1.5) == following

    def test_catalog_derived_not_hard_coded(self) -> None:
        for attack in ATTACKS:
            recognizer = AttackRecognizer()
            for index, seal in enumerate(attack.seals[:-1]):
                recognizer.update(seal, 1.0 + 0.1 * index)
            if len(attack.seals) > 1:
                assert recognizer.expected_next() == (attack.seals[-1],), attack.name

    def test_single_seal_attack_never_leaves_a_queue(self) -> None:
        recognizer = AttackRecognizer()
        assert recognizer.update("hare", 1.0) is not None
        assert recognizer.expected_next() == ()

    def test_unrelated_or_non_prefix_queue_has_no_prior(self) -> None:
        recognizer = AttackRecognizer()
        recognizer.update("bird", 1.0)
        assert recognizer.expected_next() == ()
        recognizer = AttackRecognizer()
        recognizer.update("horse", 1.0)  # second half of FIRE only, not a prefix
        assert recognizer.expected_next() == ()

    def test_strict_prefix_only_completed_sequences_are_excluded(self) -> None:
        recognizer = AttackRecognizer()
        recognizer.update("tiger", 1.0)
        recognizer.update("horse", 1.1)  # completes FIRE, queue clears
        assert recognizer.seal_labels == ()
        assert recognizer.expected_next() == ()

    def test_union_over_every_attack_with_the_prefix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        extended = (
            *ATTACKS,
            AttackDefinition("extra", "EXTRA", ("tiger", "rat")),
            AttackDefinition("long", "LONG", ("tiger", "snake", "ox")),
            AttackDefinition("dup", "DUP", ("tiger", "horse", "ram")),
        )
        monkeypatch.setattr(combined_pipeline, "ATTACKS", extended)
        recognizer = AttackRecognizer(config=DEFAULT_PIPELINE_CONFIG.attacks.__class__(max_seals=4))
        recognizer.update("tiger", 1.0)
        assert recognizer.expected_next() == ("horse", "rat", "snake")  # sorted, de-duplicated

    def test_expired_queue_counts_as_empty(self) -> None:
        recognizer = AttackRecognizer()  # seal_timeout 3.0
        recognizer.update("tiger", 1.0)
        assert recognizer.expected_next(now=3.9) == ("horse",)
        assert recognizer.expected_next(now=4.1) == ()
        # without `now` the queue is taken as is
        assert recognizer.expected_next() == ("horse",)


# ---------------------------------------------------------------------------------------
# The v1 stage on synthetic probability streams
# ---------------------------------------------------------------------------------------
def make_v1_recognizer(*, use_prior: bool | None = None) -> HandSignRecognizer:
    recognizer = object.__new__(HandSignRecognizer)
    recognizer.classes = LABELS
    recognizer.decoder_mode = "v1"
    recognizer.v1 = V1HandDecoder.from_file(LABELS, use_prior=use_prior)
    return recognizer


class TestV1Stage:
    def test_clean_seal_emits_exactly_one_event_with_times(self) -> None:
        recognizer = make_v1_recognizer()
        events = []
        stables = []
        for t_ms, probs in stream("tiger", idle_before=10, hold=25, idle_after=15):
            accepted, reason, stable, emitted, event = recognizer._decode_v1(probs, t_ms)
            stables.append(stable)
            if emitted is not None:
                assert event is not None and event.label == emitted
                events.append((t_ms, event))
        assert len(events) == 1
        t_frame, event = events[0]
        assert event.label == "tiger"
        assert event.t_emit_ms == t_frame
        assert event.t_start_ms < event.t_emit_ms
        assert event.t_emit_ms - event.t_start_ms >= 120.0  # min_dwell_ms
        assert 0.6 <= event.confidence <= 1.0
        assert stables[0] == "zero"
        assert "tiger" in stables
        assert stables[-1] == "zero"  # released

    def test_sixty_fps_gives_the_same_single_event(self) -> None:
        recognizer = make_v1_recognizer()
        frames = stream("ram", idle_before=20, hold=50, idle_after=30, fps=60.0)
        events = [e for _, e in run_stream(recognizer.v1, frames)]
        assert [e.label for e in events] == ["ram"]

    def test_flicker_while_held_does_not_chatter(self) -> None:
        recognizer = make_v1_recognizer()
        seq = [onehot("zero", 0.95)] * 8
        for i in range(60):
            seq.append(onehot("monkey", 0.92) if i % 4 else onehot("monkey", 0.35))
        seq += [onehot("zero", 0.95)] * 10
        frames = [(1000.0 + i * 33.3, p) for i, p in enumerate(seq)]
        events = [e for _, e in run_stream(recognizer.v1, frames)]
        assert [e.label for e in events] == ["monkey"]

    def test_flicker_around_the_threshold_never_fires(self) -> None:
        recognizer = make_v1_recognizer()
        seq = []
        for i in range(90):
            seq.append(onehot("tiger", 0.9) if i % 2 else onehot("zero", 0.9))
        frames = [(1000.0 + i * 33.3, p) for i, p in enumerate(seq)]
        assert run_stream(recognizer.v1, frames) == []

    def test_second_seal_after_release_fires_again(self) -> None:
        recognizer = make_v1_recognizer()
        seq = (
            [onehot("zero", 0.95)] * 6
            + [onehot("tiger")] * 20
            + [onehot("zero", 0.95)] * 20
            + [onehot("horse")] * 20
            + [onehot("zero", 0.95)] * 6
        )
        frames = [(1000.0 + i * 33.3, p) for i, p in enumerate(seq)]
        events = [e for _, e in run_stream(recognizer.v1, frames)]
        assert [e.label for e in events] == ["tiger", "horse"]

    def test_no_hands_gate_feeds_idle(self) -> None:
        recognizer = make_v1_recognizer()
        for i in range(60):
            *_, emitted, event = recognizer._decode_v1(
                onehot("tiger", 0.95), 1000.0 + i * 33.3, no_hands=True
            )
            assert emitted is None and event is None
        accepted, reason, stable, *_ = recognizer._decode_v1(
            onehot("tiger", 0.95), 1000.0 + 61 * 33.3, no_hands=True
        )
        assert (accepted, reason, stable) == ("zero", "no_hand_landmarks", "zero")

    def test_grammar_prior_is_off_by_default_and_enabled_on_request(self) -> None:
        ambiguous = [0.0] * len(LABELS)
        ambiguous[LABELS.index("horse")] = 0.46
        ambiguous[LABELS.index("ram")] = 0.50
        ambiguous[ZERO] = 0.04

        def labels_for(use_prior: bool | None) -> list[str]:
            recognizer = make_v1_recognizer(use_prior=use_prior)
            out = []
            for i in range(30):
                *_, emitted, _event = recognizer._decode_v1(
                    ambiguous, 1000.0 + i * 33.3, expected_next=("horse",)
                )
                if emitted:
                    out.append(emitted)
            return out

        assert make_v1_recognizer().decoder_uses_prior is False
        assert make_v1_recognizer(use_prior=True).decoder_uses_prior is True
        assert labels_for(None) == []  # ram 0.50 never reaches tau_on 0.6
        assert labels_for(False) == []
        assert labels_for(True) == ["horse"]  # boosted x3 and renormalised

    def test_calibrate_neutral_is_a_no_op_in_v1(self) -> None:
        recognizer = make_v1_recognizer()
        recognizer.confidence_thresholds = {"dog": 0.68}
        recognizer.margin_thresholds = {"dog": 0.4}
        recognizer.filter = EvidenceHandFilter()
        samples = (Classification("dog", 0.9, "zero", 0.05),)
        assert recognizer.calibrate_neutral(samples) == {}
        assert recognizer.confidence_thresholds == {"dog": 0.68}

    def test_reset_clears_the_decoder(self) -> None:
        recognizer = make_v1_recognizer()
        recognizer.filter = EvidenceHandFilter()
        recognizer._smoothed_hand_bbox = None
        recognizer._missing_hand_frames = 0
        for t_ms, probs in stream("tiger", idle_before=3, hold=15, idle_after=0):
            recognizer._decode_v1(probs, t_ms)
        assert recognizer.v1 is not None and recognizer.v1.active_label == "tiger"
        recognizer.reset()
        assert recognizer.v1.active_label is None
        # timestamps may restart after a reset
        assert recognizer._decode_v1(onehot("zero", 0.95), 5.0)[2] == "zero"


class _StubLandmarker:
    def __init__(self, hands: bool) -> None:
        self.hands = hands
        self.timestamps: list[int] = []

    def detect_for_video(self, _image: Any, timestamp_ms: int) -> Any:
        self.timestamps.append(timestamp_ms)
        if not self.hands:
            return SimpleNamespace(hand_landmarks=[], handedness=[])
        points = [SimpleNamespace(x=0.4 + 0.01 * i, y=0.4 + 0.01 * i, z=0.0) for i in range(21)]
        return SimpleNamespace(hand_landmarks=[points], handedness=[])


def make_process_recognizer(label_stream: list[str], *, hands: bool, v1: bool):
    recognizer = make_v1_recognizer() if v1 else object.__new__(HandSignRecognizer)
    recognizer.classes = LABELS
    recognizer.fusion_config = HandFusionConfig()
    recognizer.filter = EvidenceHandFilter()
    recognizer.confidence_override = None
    recognizer.confidence_thresholds = {name: 0.5 for name in LABELS}
    recognizer.margin_thresholds = {name: 0.1 for name in LABELS}
    recognizer._smoothed_hand_bbox = None
    recognizer._missing_hand_frames = 0
    recognizer._last_timestamp_ms = -1
    recognizer.landmarker = _StubLandmarker(hands)
    labels = iter(label_stream)
    current: dict[str, str] = {}

    def classify(_crop: Any) -> Any:
        # The centre crop is classified first on each frame; reuse its label for the ROI crop.
        if "label" not in current or current.get("used") == "roi":
            current["label"] = next(labels)
            current["used"] = "center"
        else:
            current["used"] = "roi"
        prediction = recognizer._classification_from_probabilities(onehot(current["label"], 0.93))
        return prediction, {"preprocess_ms": 0.0, "onnx_ms": 0.0, "postprocess_ms": 0.0}

    recognizer._classify_crop_timed = classify
    return recognizer


class TestProcessGlue:
    FRAME = np.zeros((480, 640, 3), dtype=np.uint8)

    def run(self, recognizer: HandSignRecognizer, labels: list[str], fps: float = 30.0):
        return [recognizer.process(self.FRAME, 100.0 + i / fps) for i, _ in enumerate(labels)]

    def test_v1_process_emits_one_seal_with_pipeline_clock_times(self) -> None:
        labels = ["zero"] * 8 + ["tiger"] * 25 + ["zero"] * 15
        recognizer = make_process_recognizer(labels, hands=True, v1=True)
        results = self.run(recognizer, labels)
        emitted = [(i, r) for i, r in enumerate(results) if r.emitted_seal]
        assert len(emitted) == 1
        index, result = emitted[0]
        assert result.emitted_seal == "tiger" and result.decoder == "v1"
        event = result.seal_event
        assert event is not None
        # raw capture time in ms, not MediaPipe's bumped integer timestamp
        assert event.t_emit_ms == pytest.approx((100.0 + index / 30.0) * 1000.0)
        assert event.t_start_ms < event.t_emit_ms
        assert all(r.decoder == "v1" for r in results)
        assert all(r.seal_event is None for i, r in enumerate(results) if i != index)
        assert results[-1].stable_label == "zero"
        # the legacy filter was never touched
        assert not recognizer.filter.history

    def test_v1_ignores_the_per_class_thresholds(self) -> None:
        labels = ["zero"] * 4 + ["tiger"] * 20 + ["zero"] * 10
        recognizer = make_process_recognizer(labels, hands=True, v1=True)
        recognizer.confidence_thresholds = {name: 0.999 for name in LABELS}  # would reject all
        results = self.run(recognizer, labels)
        assert [r.emitted_seal for r in results if r.emitted_seal] == ["tiger"]

    def test_legacy_process_is_unchanged(self) -> None:
        labels = ["zero"] * 4 + ["tiger"] * 12 + ["zero"] * 6
        recognizer = make_process_recognizer(labels, hands=True, v1=False)
        results = self.run(recognizer, labels)
        assert [r.emitted_seal for r in results if r.emitted_seal] == ["tiger"]
        assert all(r.decoder == "legacy" and r.seal_event is None for r in results)
        assert recognizer.filter.history  # the legacy filter did the work

    def test_absent_hands_never_produce_a_seal_in_v1(self) -> None:
        labels = ["tiger"] * 60
        recognizer = make_process_recognizer(labels, hands=False, v1=True)
        results = self.run(recognizer, labels)
        assert not any(r.emitted_seal for r in results)
        assert results[-1].rejection_reason == "no_hand_landmarks"

    def test_prior_receives_expected_next(self) -> None:
        recognizer = make_process_recognizer(["zero"] * 3, hands=True, v1=True)
        seen: list[tuple[str, ...]] = []
        original = recognizer._decode_v1

        def spy(probabilities, t_ms, expected_next=(), **kwargs):
            seen.append(tuple(expected_next))
            return original(probabilities, t_ms, expected_next, **kwargs)

        recognizer._decode_v1 = spy
        recognizer.process(self.FRAME, 1.0, expected_next=("horse",))
        recognizer.process(self.FRAME, 1.1)
        assert seen == [("horse",), ()]


class _FakeHand:
    def __init__(self, uses_prior: bool) -> None:
        if uses_prior:
            self.decoder_uses_prior = True
        self.calls: list[tuple[float, dict[str, Any]]] = []

    def process(self, _frame: Any, timestamp: float, **kwargs: Any) -> HandResult:
        self.calls.append((timestamp, kwargs))
        return HandResult(
            raw=Classification("zero", 1.0, "bird", 0.0),
            accepted_label="zero",
            rejection_reason=None,
            stable_label="zero",
            emitted_seal=None,
            classification_bbox=(0, 0, 1, 1),
            hand_bbox=None,
        )


class _IdleBody:
    def process(self, _frame: Any, _timestamp: float) -> BodyResult:
        return BodyResult("idle", "idle", None, {})


class TestPipelinePassesTheGrammar:
    def make(self, uses_prior: bool) -> CombinedNarutoPipeline:
        pipeline = object.__new__(CombinedNarutoPipeline)
        pipeline.hand = _FakeHand(uses_prior)
        pipeline.body = _IdleBody()
        pipeline.attacks = AttackRecognizer()
        return pipeline

    def test_prior_enabled_gets_expected_next_from_the_queue(self) -> None:
        pipeline = self.make(True)
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        pipeline.process(frame, 1.0)
        pipeline.attacks.update("tiger", 1.05)
        pipeline.process(frame, 1.1)
        pipeline.process(frame, 5.0)  # queue expired by then
        assert [kw for _, kw in pipeline.hand.calls] == [
            {"expected_next": ()},
            {"expected_next": ("horse",)},
            {"expected_next": ()},
        ]

    def test_legacy_call_signature_is_untouched(self) -> None:
        pipeline = self.make(False)
        pipeline.process(np.zeros((2, 2, 3), dtype=np.uint8), 1.0)
        assert pipeline.hand.calls == [(1.0, {})]

    def test_session_scoped_hand_only_path_passes_it_too(self) -> None:
        from narutocv.transport.pipeline import SessionScopedPipeline

        pipeline = object.__new__(SessionScopedPipeline)
        pipeline.hand = _FakeHand(True)
        pipeline.body = _IdleBody()
        pipeline.attacks = AttackRecognizer()
        pipeline.attacks.update("snake", 1.0)
        pipeline._mode = "hand_only"
        pipeline._idle_hand = None
        pipeline._idle_body = BodyResult("idle", "idle", None, {})
        pipeline.process(np.zeros((2, 2, 3), dtype=np.uint8), 1.2)
        assert pipeline.hand.calls == [(1.2, {"expected_next": ("dragon",)})]


# ---------------------------------------------------------------------------------------
# Output and event contract
# ---------------------------------------------------------------------------------------
def frame_result(
    *, decoder: str, event: SealEvent | None, accepted_seal: str | None = "tiger"
) -> FrameResult:
    hand = HandResult(
        raw=Classification("tiger", 0.9, "zero", 0.05),
        accepted_label="tiger",
        rejection_reason=None,
        stable_label="tiger",
        emitted_seal=accepted_seal,
        classification_bbox=(0, 0, 2, 2),
        hand_bbox=None,
        decoder=decoder,
        seal_event=event,
    )
    queue = QueueUpdate(
        attack=None,
        accepted_seal=accepted_seal,
        duplicate_ignored=False,
        timeout_cleared=False,
        max_length_cleared=False,
        cooldown_suppressed=False,
        queue=(accepted_seal,) if accepted_seal else (),
    )
    return FrameResult(
        hand=hand,
        body=BodyResult("idle", "idle", None, {}),
        attack=None,
        seal_history=queue.queue,
        queue_update=queue,
        processing_ms=1.0,
    )


def hand_seal_payload(result: FrameResult) -> dict[str, Any]:
    output = PipelineOutputSerializer().to_dict(
        result, session_id="s", frame_id=1, captured_at_ms=12345
    )
    report = PipelineEventDispatcher().dispatch_frame(output)
    return next(e for e in report.events if e["event_type"] == "HAND_SEAL")["payload"]


class TestOutputAndEvents:
    EVENT = SealEvent("tiger", 12100.0, 12345.0, 0.87)

    def test_legacy_output_and_payload_have_no_decoder_object(self) -> None:
        result = frame_result(decoder="legacy", event=None)
        output = PipelineOutputSerializer().to_dict(
            result, session_id="s", frame_id=1, captured_at_ms=1
        )
        assert "decoder" not in output["hand"]
        payload = hand_seal_payload(result)
        assert payload == {
            "seal": "tiger",
            "stable_label": "tiger",
            "confidence": 0.9,
            "decoder": "legacy",
        }

    def test_v1_output_carries_the_decoder_event(self) -> None:
        output = PipelineOutputSerializer().to_dict(
            frame_result(decoder="v1", event=self.EVENT),
            session_id="s",
            frame_id=1,
            captured_at_ms=1,
        )
        assert output["hand"]["decoder"] == {
            "mode": "v1",
            "event": {
                "label": "tiger",
                "t_start_ms": 12100.0,
                "t_emit_ms": 12345.0,
                "confidence": 0.87,
            },
        }
        json.dumps(output, allow_nan=False)

    def test_v1_payload_has_the_new_fields(self) -> None:
        payload = hand_seal_payload(frame_result(decoder="v1", event=self.EVENT))
        assert payload == {
            "seal": "tiger",
            "stable_label": "tiger",
            "confidence": 0.9,  # still the raw frame confidence
            "decoder": "v1",
            "t_start_ms": 12100.0,
            "t_emit_ms": 12345.0,
            "decoder_confidence": 0.87,
        }

    def test_event_for_another_label_is_not_attached(self) -> None:
        other = SealEvent("ram", 1.0, 2.0, 0.7)
        payload = hand_seal_payload(frame_result(decoder="v1", event=other))
        assert payload["decoder"] == "v1" and "t_emit_ms" not in payload

    def test_v1_frame_without_an_event_has_a_null_event(self) -> None:
        result = frame_result(decoder="v1", event=None, accepted_seal=None)
        output = PipelineOutputSerializer().to_dict(
            result, session_id="s", frame_id=1, captured_at_ms=1
        )
        assert output["hand"]["decoder"] == {"mode": "v1", "event": None}

    def test_validation_rejects_malformed_decoder_objects(self) -> None:
        output = PipelineOutputSerializer().to_dict(
            frame_result(decoder="v1", event=self.EVENT),
            session_id="s",
            frame_id=1,
            captured_at_ms=1,
        )
        for mutate in (
            lambda h: h["decoder"].update(mode="legacy"),
            lambda h: h["decoder"]["event"].update(t_emit_ms=1.0),  # before t_start
            lambda h: h["decoder"]["event"].update(confidence=1.5),
            lambda h: h["decoder"].update(extra=1),
            lambda h: h["decoder"]["event"].pop("label"),
        ):
            broken = json.loads(json.dumps(output))
            mutate(broken["hand"])
            with pytest.raises(SerializationError):
                validate_pipeline_output_v1(broken)

    def test_json_schemas_accept_both_shapes(self) -> None:
        jsonschema = pytest.importorskip("jsonschema")
        output_schema = json.loads(
            (ROOT / "schemas" / "pipeline_output_v1.schema.json").read_text()
        )
        event_schema = json.loads((ROOT / "schemas" / "pipeline_event_v1.schema.json").read_text())
        for decoder, event in (("legacy", None), ("v1", self.EVENT), ("v1", None)):
            result = frame_result(decoder=decoder, event=event)
            output = PipelineOutputSerializer().to_dict(
                result, session_id="s", frame_id=1, captured_at_ms=12345
            )
            jsonschema.Draft202012Validator(output_schema).validate(output)
            report = PipelineEventDispatcher().dispatch_frame(output)
            for emitted in report.events:
                jsonschema.Draft202012Validator(event_schema).validate(emitted)
        bad = {"seal": "tiger", "stable_label": "t", "confidence": 0.5, "decoder": "v2"}
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.Draft202012Validator(event_schema).validate(
                {
                    "schema_version": "1.0.0",
                    "event_id": "s:1",
                    "event_sequence": 1,
                    "event_type": "HAND_SEAL",
                    "session_id": "s",
                    "frame_id": 1,
                    "captured_at_ms": 1,
                    "payload": bad,
                }
            )
