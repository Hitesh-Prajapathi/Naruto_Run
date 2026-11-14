# 0003. Calibrated probabilities and one hysteresis decoder

Status: Proposed

## Context

`src/narutocv/inference/hand_config.py` holds 13 per-class confidence floors (0.32 to 0.92) and 13 margin floors, plus special-case guards (rat/ram, tiger/dog). On top of this, `EvidenceHandFilter` uses a 5-frame window with 3 required votes. The result is slow (median time-to-detect about 1.0 to 2.7 s in the v0 live tests) and brittle: each threshold fixed one session and broke another, which is what a model that does not generalise looks like when patched.

## Decision

- Make the model output **calibrated probabilities**: temperature scaling fit on validation subjects, reported with ECE and reliability diagrams (target ECE below 0.05).
- Replace per-class constants with **one global threshold pair** (enter tau_on, exit tau_off below tau_on) plus an **open-set reject** into `zero`/`unknown`.
- Replace the frame-count vote with an **exponentially smoothed, hysteresis decoder with minimum dwell in milliseconds**, so behaviour is identical at 20 and 60 fps. An HMM/Viterbi layer and a grammar-aware prior for mid-combo are optional; metrics are always reported with and without the prior.
- Tune tau_on, tau_off and dwell on validation subjects against a target of at most 0.5 false activations per minute while maximising combo recall; publish the trade-off curve.
- Thresholds ship in the model bundle (`calibration.json`); `hand_config.py` is deleted.

## Consequences

- Fewer knobs to explain and tune; behaviour is portable across frame rates and between the Python reference and the browser port.
- Calibration is only as good as the validation subjects; with few subjects it needs confidence intervals and must be re-fit per model version.
- Requires continuous labeled video with natural non-seal motion to measure false activations per minute.
- None of this is implemented yet; the numeric targets (FA/min, ECE, time-to-detect) are goals, not results.
