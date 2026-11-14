.PHONY: ci test-collect help setup fetch-models test test-py test-web lint fmt mypy camera serve web build-web

WEB := apps/web
COLLECT := tools/collect

help:
	@grep -E "^[a-z-]+:.*##" $(MAKEFILE_LIST) | sed -E "s/:.*## /\t/"

setup: ## uv sync --all-extras + npm ci
	uv sync --all-extras  # local dev: includes the heavy `train` extra
	cd $(WEB) && npm ci

fetch-models: ## download + sha256-verify models listed in models/manifest.yaml
	uv run python scripts/fetch_models.py

test: test-py test-web ## all tests

test-py: ## python tests (pytest)
	uv run pytest

test-web: ## web unit tests (vitest)
	cd $(WEB) && npx vitest run

test-collect: ## data-collection tool: vitest + build + receiver self-test
	cd $(COLLECT) && npx vitest run && npm run build
	uv run python $(COLLECT)/server/receive.py --self-test

ci: lint test-py test-web test-collect ## everything CI runs (minus Playwright e2e/smoke and mypy)

lint: ## ruff + eslint + tsc
	uv run ruff check .
	uv run ruff format --check .
	cd $(WEB) && npm run lint && npm run typecheck

fmt: ## ruff format + prettier
	uv run ruff format .
	cd $(WEB) && npm run format

mypy: ## strict type check of narutocv.inference (known failures, see pyproject)
	uv run mypy src/narutocv

camera: ## live camera tracker (replaces run_combined_tracker.command)
	uv run python scripts/run_combined_camera.py

serve: ## transport server for the browser game (127.0.0.1:8765)
	uv run python -m narutocv.transport

web: ## vite dev server (http://127.0.0.1:5173)
	cd $(WEB) && npm run dev

build-web: ## production build of the web app
	cd $(WEB) && npm run build
