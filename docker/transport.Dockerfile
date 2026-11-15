# syntax=docker/dockerfile:1.7
#
# NarutoCV transport service (browser game <-> recognition pipeline).
#
#   docker build -f docker/transport.Dockerfile -t narutocv-transport .
#   docker run --rm -p 127.0.0.1:8765:8765 \
#     -e NARUTOCV_ALLOWED_ORIGINS=http://localhost:5173 narutocv-transport
#
# Publish the port on loopback only (as above) unless a TLS-terminating,
# authenticating reverse proxy sits in front: the service has no auth of its own.
# See docs/transport.md for every NARUTOCV_* setting.

ARG PYTHON_VERSION=3.12

# ---- build stage: resolve deps with uv, fetch + verify models ----------------
FROM python:${PYTHON_VERSION}-slim-bookworm AS build

COPY --from=ghcr.io/astral-sh/uv:0.9.25 /uv /usr/local/bin/uv

# The package must live at the same path in both stages: the editable install
# points back at /app/src and narutocv locates models/ relative to its source.
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv

# Dependencies first (cached until pyproject.toml / uv.lock change).
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project --extra transport

COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra transport

# Models: download from models/manifest.yaml and verify sha256 + size. This
# fails the build on any mismatch. (Mount a volume at /app/models instead if the
# build has no network access.)
COPY models ./models
COPY scripts/fetch_models.py ./scripts/fetch_models.py
RUN /app/.venv/bin/python scripts/fetch_models.py \
    && /app/.venv/bin/python scripts/fetch_models.py --verify \
    && chmod -R a+rX /app/models

# ---- runtime stage -------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

# opencv-python (the `transport` extra) is the non-headless wheel and needs
# these shared libraries even though no window is ever opened; MediaPipe's
# native library additionally links libEGL/libGLESv2.
# Debian point releases drop old pinned package versions; the base image tag pins the suite.
# hadolint ignore=DL3008
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libegl1 libgles2 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --system --gid 10001 narutocv \
    && useradd --system --uid 10001 --gid narutocv --no-create-home --shell /usr/sbin/nologin narutocv

WORKDIR /app
COPY --from=build /app/.venv /app/.venv
COPY --from=build /app/src /app/src
COPY --from=build /app/models /app/models
COPY --from=build /app/pyproject.toml /app/pyproject.toml

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    PYTHONDONTWRITEBYTECODE=1 \
    NARUTOCV_HOST=0.0.0.0 \
    NARUTOCV_PORT=8765 \
    NARUTOCV_LOG_FORMAT=json

USER 10001:10001
EXPOSE 8765

# /live answers as long as the event loop does; /health (readiness) is for
# orchestrators that route traffic. A long start period covers model loading.
HEALTHCHECK --interval=15s --timeout=3s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import os,urllib.request;urllib.request.urlopen('http://127.0.0.1:%s/live' % os.environ.get('NARUTOCV_PORT','8765'),timeout=2)"]

# Exec form so python is PID 1 and receives SIGTERM directly (graceful drain).
ENTRYPOINT ["python", "-m", "narutocv.transport"]
