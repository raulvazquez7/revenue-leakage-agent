# syntax=docker/dockerfile:1

# ---- build: resolve the locked environment with uv ----
FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.8.17 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON=/usr/local/bin/python3.12

WORKDIR /app

# Dependencies first so source edits don't invalidate this layer.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-dev --no-install-project

COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

# ---- runtime: no uv, no build cache, non-root ----
FROM python:3.12-slim

RUN groupadd --system app \
    && useradd --system --gid app --home-dir /app --no-create-home app

WORKDIR /app
COPY --from=builder --chown=app:app /app /app
COPY --chown=app:app data ./data
# Writable state (sandbox ledgers, SQLite checkpoints); compose mounts a volume.
RUN mkdir -p /app/state && chown app:app /app/state

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    SANDBOX_DIR=/app/state/sandbox \
    CHECKPOINT_DB=/app/state/checkpoints.sqlite

USER app
EXPOSE 8501 8000

CMD ["streamlit", "run", "src/revenue_leakage_agent/interfaces/streamlit_app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]
