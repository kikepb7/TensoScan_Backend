# syntax=docker/dockerfile:1

# ---- build: resolve and install the locked dependencies into a virtualenv ----
FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.11.32 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app

# Set to 1 only while the legacy on-server OCR (/ocr routes) is still needed; it pulls in TensorFlow.
ARG WITH_LEGACY_OCR=0

COPY pyproject.toml uv.lock README.md ./
COPY app ./app
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$WITH_LEGACY_OCR" = "1" ]; then \
        uv sync --frozen --no-dev --extra legacy-ocr; \
    else \
        uv sync --frozen --no-dev; \
    fi

# ---- runtime: no build tools, no secrets, no root ----
FROM python:3.12-slim
ARG WITH_LEGACY_OCR=0
RUN if [ "$WITH_LEGACY_OCR" = "1" ]; then \
        apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 && rm -rf /var/lib/apt/lists/*; \
    fi \
    && useradd --create-home --uid 10001 api
WORKDIR /app
COPY --from=build --chown=api:api /app /app
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1
USER api
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"
CMD ["uvicorn", "app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
