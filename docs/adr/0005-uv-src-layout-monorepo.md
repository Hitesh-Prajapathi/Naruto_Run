# 0005. uv + src-layout monorepo

Status: Accepted (implemented)

## Context

The original repo had two Python virtual environments (about 1.4 GB and 446 MB), a duplicated model file, tracked weights, tests that ran only via `unittest discover`, one failing import in the transport tests, per-platform launcher scripts, a person's name in module paths, and large unreachable git objects. A stranger could not clone and run it.

## Decision

- One Python toolchain: a single `pyproject.toml` managed by **uv** with optional extras and a committed `uv.lock`.
- **src layout:** an importable, typed package `src/narutocv/` (`inference/`, `transport/` today; later `data/`, `features/`, `models/`, `temporal/`, `eval/`).
- Monorepo directories by concern: `apps/web` (game), `tools/collect` (data collection), `training/`, `schemas/`, `models/manifest.yaml`, `docs/`, `tests/`.
- Models are not stored in git (except the legacy ONNX); `make fetch-models` downloads and sha256-verifies them from `models/manifest.yaml`.
- A `Makefile` replaces launcher scripts; pytest replaces unittest; pre-commit hooks for formatting, linting and large-file guarding.
- Renames done with `git mv` to preserve history; historical docs moved to `docs/archive/` with a superseded notice.

## Consequences

- Fresh clone to running demo is five commands (see the README).
- Larger one-time churn in imports and paths; mitigated by landing it with all tests green.
- mypy strictness is still incomplete (`make mypy` documents known failures).
- Raw data and personal prep notes stay outside the public repository.
