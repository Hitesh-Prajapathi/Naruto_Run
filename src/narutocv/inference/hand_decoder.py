"""The ``v1`` hand-seal decoding stage: temperature calibration + :class:`SealDecoder`.

Selected with ``hand_decoder: "v1"`` (config), ``--hand-decoder v1`` (CLIs and the transport)
or ``NARUTOCV_HAND_DECODER=v1`` (environment); the default is ``legacy``, which never touches
this module or ``models/calibration_v0.json``.

Per frame the recognizer feeds its fused per-class probabilities (centre crop + hand ROI) in
here::

    p (fused) -> TemperatureScaler -> SealDecoder.update(p, t_ms, expected_next) -> SealEvent | None

Design notes and known limits:

* ``TemperatureScaler`` expects logits; the live path only has the fused *probabilities*
  (a weighted mix of two softmaxes), so ``log(clip(p))`` stands in for the logits.  With the
  shipped ``temperature = 1.0`` this is an exact identity (``softmax(log p) = p``).  A fitted
  ``T != 1`` would act as ``p ** (1 / T)``, renormalised -- a monotone sharpening/flattening,
  but not the calibration that was fitted on single-branch logits, so refit the temperature on
  *fused probabilities* once real held-out data exists.
* ``StackedFusion`` is not used yet (there is one branch); see :data:`STACKED_FUSION_HOOK`.
* ``SealEvent.t_start_ms`` is the first update of the uninterrupted run with the *smoothed*
  probability at or above ``tau_on``.  It is not corrected for the smoother's rise time, so it
  lags the true onset by roughly the EMA rise time (about 100 ms at the default half-life).
  ``t_emit_ms`` is the timestamp of the emitting frame.  Both are on the pipeline clock
  (``timestamp * 1000``, which in the transport equals the frame's ``captured_at_ms``).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

from narutocv.models.calibration import TemperatureScaler
from narutocv.temporal.decoder import DecoderConfig, SealDecoder, SealEvent

from .pipeline_config import (
    DEFAULT_HAND_DECODER,
    DEFAULT_PIPELINE_CONFIG,
    HAND_DECODER_MODES,
    PipelineConfig,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CALIBRATION_PATH = REPO_ROOT / "models" / "calibration_v0.json"

HAND_DECODER_ENV = "NARUTOCV_HAND_DECODER"
HAND_DECODER_PRIOR_ENV = "NARUTOCV_HAND_DECODER_PRIOR"

#: Probability floor for ``log`` when turning fused probabilities into pseudo-logits.
LOG_EPS = 1e-12

#: Reserved extension point.  Once a second branch (a landmark MLP) exists, replace the
#: ``fused probabilities -> TemperatureScaler`` step in :meth:`V1HandDecoder.update` with
#: ``build_fusion_features([p_image, p_landmarks], meta) -> StackedFusion.predict_logits ->
#: TemperatureScaler`` (``narutocv.models.calibration``; TS port ``apps/web/src/recognition``).
#: Nothing consumes it yet; ``fusion`` in the calibration file is intentionally absent.
STACKED_FUSION_HOOK = "not used: single image branch"

Array = NDArray[np.float64]

_TRUE = ("1", "true", "yes", "on")
_FALSE = ("0", "false", "no", "off")


def resolve_hand_decoder(
    value: Optional[str] = None, env: Optional[Mapping[str, str]] = None
) -> str:
    """Flag value if given, else ``NARUTOCV_HAND_DECODER``, else ``legacy``.

    Raises ``ValueError`` for an unknown mode (from either source).
    """
    env = os.environ if env is None else env
    source = "hand_decoder"
    if value is None:
        raw = env.get(HAND_DECODER_ENV)
        if raw is None or not raw.strip():
            return DEFAULT_HAND_DECODER
        value, source = raw.strip().lower(), HAND_DECODER_ENV
    if value not in HAND_DECODER_MODES:
        raise ValueError(f"{source} must be one of {', '.join(HAND_DECODER_MODES)}, got {value!r}")
    return value


def resolve_hand_decoder_prior(
    value: Optional[bool] = None, env: Optional[Mapping[str, str]] = None
) -> Optional[bool]:
    """Flag value if given, else ``NARUTOCV_HAND_DECODER_PRIOR``, else ``None`` (use the file)."""
    if value is not None:
        return value
    env = os.environ if env is None else env
    raw = env.get(HAND_DECODER_PRIOR_ENV)
    if raw is None or not raw.strip():
        return None
    lowered = raw.strip().lower()
    if lowered in _TRUE:
        return True
    if lowered in _FALSE:
        return False
    raise ValueError(f"{HAND_DECODER_PRIOR_ENV} must be a boolean, got {raw!r}")


def build_pipeline_config(
    hand_decoder: Optional[str] = None,
    *,
    use_prior: Optional[bool] = None,
    env: Optional[Mapping[str, str]] = None,
    base: PipelineConfig = DEFAULT_PIPELINE_CONFIG,
) -> PipelineConfig:
    """``base`` with the hand decoder resolved from the flag, then the environment."""
    return dataclasses.replace(
        base,
        hand_decoder=resolve_hand_decoder(hand_decoder, env),
        hand_decoder_use_prior=resolve_hand_decoder_prior(use_prior, env),
    )


def add_hand_decoder_arguments(parser: argparse.ArgumentParser) -> None:
    """``--hand-decoder`` and ``--hand-decoder-prior`` for every entry point that builds a pipeline."""
    parser.add_argument(
        "--hand-decoder",
        choices=HAND_DECODER_MODES,
        default=None,
        help=(
            "hand-seal decoder: 'legacy' (per-class thresholds + 5-frame vote; default) or "
            "'v1' (temperature calibration + SealDecoder, thresholds bypassed). "
            f"Falls back to ${HAND_DECODER_ENV}. v1 has not been tuned on multi-subject data."
        ),
    )
    parser.add_argument(
        "--hand-decoder-prior",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "v1 only: boost the next seal(s) of combos in progress. Default: the calibration "
            f"file's value (off), or ${HAND_DECODER_PRIOR_ENV}."
        ),
    )


def pipeline_config_from_args(
    args: argparse.Namespace,
    parser: Optional[argparse.ArgumentParser] = None,
    env: Optional[Mapping[str, str]] = None,
) -> PipelineConfig:
    """Resolve ``--hand-decoder`` / ``--hand-decoder-prior`` / env into a ``PipelineConfig``."""
    try:
        return build_pipeline_config(args.hand_decoder, use_prior=args.hand_decoder_prior, env=env)
    except ValueError as error:
        if parser is not None:
            parser.error(str(error))
        raise


@dataclass(frozen=True)
class CalibrationBundle:
    """Contents of ``models/calibration_v0.json``."""

    scaler: TemperatureScaler
    decoder: DecoderConfig
    provenance: str
    path: Optional[Path] = None


def load_calibration(
    path: Optional[Path] = None, *, use_prior: Optional[bool] = None
) -> CalibrationBundle:
    """Read the calibration file; ``use_prior`` (if not ``None``) overrides the file's value."""
    source = Path(path) if path is not None else DEFAULT_CALIBRATION_PATH
    data = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{source}: calibration file must be a JSON object")
    unknown = set(data) - {"temperature", "decoder", "provenance"}
    if unknown:
        raise ValueError(f"{source}: unknown calibration keys {sorted(unknown)}")
    missing = {"temperature", "decoder"} - set(data)
    if missing:
        raise ValueError(f"{source}: missing calibration keys {sorted(missing)}")
    decoder_dict = dict(data["decoder"])
    if use_prior is not None:
        decoder_dict["use_prior"] = bool(use_prior)
    return CalibrationBundle(
        scaler=TemperatureScaler(float(data["temperature"])),
        decoder=DecoderConfig.from_dict(decoder_dict),
        provenance=str(data.get("provenance", "")),
        path=source,
    )


def calibrate_probabilities(probabilities: ArrayLike, scaler: TemperatureScaler) -> Array:
    """Temperature-scale a probability vector via ``log`` pseudo-logits (identity at T = 1)."""
    p = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    return scaler.transform(np.log(np.clip(p, LOG_EPS, None)))


@dataclass(frozen=True)
class DecoderStep:
    """Everything the recognizer needs back from one decoder update."""

    event: Optional[SealEvent]
    stable_label: str
    accepted_label: str
    rejection_reason: Optional[str]


class V1HandDecoder:
    """Calibrated probabilities -> :class:`SealDecoder`, one frame at a time."""

    def __init__(self, bundle: CalibrationBundle, classes: Sequence[str]) -> None:
        if tuple(classes) != tuple(bundle.decoder.labels):
            raise ValueError(
                "the hand model's class order must equal the decoder labels: "
                f"model={tuple(classes)!r} decoder={tuple(bundle.decoder.labels)!r}"
            )
        self.bundle = bundle
        self.config = bundle.decoder
        self.scaler = bundle.scaler
        self.decoder = SealDecoder(bundle.decoder)
        self._labels = tuple(bundle.decoder.labels)
        self._idle_label = bundle.decoder.idle_label
        self._idle_index = self._labels.index(self._idle_label) if self._idle_label else None

    @classmethod
    def from_file(
        cls,
        classes: Sequence[str],
        path: Optional[Path] = None,
        *,
        use_prior: Optional[bool] = None,
    ) -> V1HandDecoder:
        return cls(load_calibration(path, use_prior=use_prior), classes)

    @property
    def idle_label(self) -> str:
        return self._idle_label or "zero"

    @property
    def active_label(self) -> Optional[str]:
        return self.decoder.active_label

    def reset(self) -> None:
        self.decoder.reset()

    def idle_vector(self) -> Array:
        """One-hot idle probabilities (used when the structural no-hands gate fires)."""
        vector = np.full(len(self._labels), 0.0)
        vector[self._idle_index if self._idle_index is not None else 0] = 1.0
        return vector

    def update(
        self,
        probabilities: ArrayLike,
        t_ms: float,
        expected_next: Sequence[str] = (),
        *,
        no_hands: bool = False,
    ) -> DecoderStep:
        """Fold in one frame; ``t_ms`` is the raw pipeline clock in milliseconds."""
        if no_hands and self._idle_index is not None:
            calibrated = self.idle_vector()
        else:
            calibrated = calibrate_probabilities(probabilities, self.scaler)
        events = self.decoder.update(calibrated, t_ms, expected_next=expected_next or None)
        event = events[0] if events else None
        accepted, reason = self._frame_candidate(no_hands=no_hands)
        return DecoderStep(
            event=event,
            stable_label=self.decoder.active_label or self.idle_label,
            accepted_label=accepted,
            rejection_reason=reason,
        )

    def _frame_candidate(self, *, no_hands: bool) -> tuple[str, Optional[str]]:
        """The decoder's per-frame view, for diagnostics (not an emission)."""
        if no_hands:
            return self.idle_label, "no_hand_landmarks"
        smoothed = self.decoder.smoothed
        if smoothed.size == 0:
            return self.idle_label, "decoder_warmup"
        top = int(np.argmax(smoothed))
        if top == self._idle_index:
            return self.idle_label, "model_zero"
        if float(smoothed[top]) < self.config.tau_on:
            return self.idle_label, "decoder_below_tau_on"
        return self._labels[top], None
