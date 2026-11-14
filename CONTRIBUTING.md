# Contributing

## Setup

Requires [uv](https://docs.astral.sh/uv/) (Python version pinned in `.python-version`) and Node.js.

```bash
make setup          # uv sync --all-extras + npm ci (apps/web)
make fetch-models   # sha256-verified downloads from models/manifest.yaml
make test           # pytest + vitest
```

| Target | What it does |
|---|---|
| `make test-py` / `make test-web` | Python (pytest) / web (vitest) tests only |
| `make lint` | ruff check + ruff format --check + eslint + tsc |
| `make fmt` | ruff format + prettier |
| `make mypy` | type check of `src/narutocv` (known failures are listed in `pyproject.toml`) |
| `make serve` | transport server on `127.0.0.1:8765` |
| `make web` | Vite dev server on `127.0.0.1:5173` |
| `make camera` | live camera tracker (no browser) |
| `make build-web` | production web build |

The collection tool is a separate npm project: see `tools/collect/README.md`.

## Pre-commit

A `.pre-commit-config.yaml` is at the repo root. Install the hooks once with `uv run pre-commit install` and run them on everything with `uv run pre-commit run --all-files`. Do not commit large binaries (the hooks guard file size); models are fetched via the manifest, datasets stay out of git.

## Branches and commits

- Branch from `main`: `feat/<topic>`, `fix/<topic>`, `docs/<topic>`, `exp/<topic>` for experiments.
- Small, logical commits with imperative subjects, ideally Conventional Commits style (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`).
- Use `git mv` for renames so history is kept. Keep tests green in each commit.
- Never put numbers in docs that are not backed by a committed file under `docs/results/`. Unmeasured values are written `TBD`.
- `docs/archive/` is historical and superseded; do not copy claims from it.

## Adding a seal

1. Edit `tools/collect/src/content/seals.json` (description, `confusable_with`, `status`) and `tools/collect/src/protocol.ts` (`SEALS`; bump `PROTOCOL_VERSION`).
2. Add the owner's reference photo as `tools/collect/public/seals/<seal>.jpg`.
3. Update `docs/seals.md` (the canonical spec) and the class list in the recogniser (`src/narutocv/inference/`) and `apps/web/src/config/bossConfig.ts`.
4. A new class needs new multi-subject data and a new model version (below); the v0 model cannot be extended.
5. If the seal is used in a combo, update `ATTACKS` in `src/narutocv/inference/combined_pipeline.py` and the matching game config, with tests.

## Adding a model version

1. Train from a config in `training/` (config-driven runs are planned; the v0 notebook is exploration only). Record the git SHA and the dataset version.
2. Evaluate with subject-disjoint (leave-one-subject-out) splits, never random frame splits. Put the report in `docs/results/` and link it from `docs/results/README.md`.
3. Fill in `docs/model_card_template.md` as `docs/models/<name>-<version>.md`.
4. Add the artifact to `models/manifest.yaml` with `name`, `version`, `url`, `sha256`, `size`. `make fetch-models` verifies it.
5. Calibration parameters (temperature, thresholds, dwell) ship with the model; do not hard-code them in `hand_config.py`.

## Licensing note for assets

The MIT license covers the code. The 3D models and textures in `apps/web/public/models` and `apps/web/public/textures` have unverified provenance. Before any public release, confirm each asset's license or replace it. Do not add new third-party assets without recording their source and license here.
