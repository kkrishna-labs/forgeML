# ---------------------------------------------------------------------------
# ForgeML inference image.
#
# Multi-stage: the builder compiles wheels, the runtime keeps none of the build
# toolchain. That is worth several hundred MB, and a smaller image is not just
# tidiness — it is cold-start time on every serverless platform.
# ---------------------------------------------------------------------------

# --- stage 1: build --------------------------------------------------------
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/

# CPU-only torch. The CUDA build is ~2.5 GB and buys nothing on a CPU host;
# override the index URL at build time when targeting a GPU runtime.
RUN pip install --prefix=/install --extra-index-url https://download.pytorch.org/whl/cpu \
    ".[api]" torch transformers peft accelerate

# --- stage 2: runtime ------------------------------------------------------
FROM python:3.11-slim AS runtime

# Never run as root. On Hugging Face Spaces uid 1000 is also what the platform
# expects to own the writable cache directories.
RUN useradd --create-home --uid 1000 forge

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/home/forge/.cache/huggingface \
    FORGEML_LAZY_LOAD=0 \
    PORT=8000

COPY --from=builder /install /usr/local

WORKDIR /app
COPY --chown=forge:forge src/ ./src/
COPY --chown=forge:forge api/ ./api/
COPY --chown=forge:forge configs/ ./configs/

RUN mkdir -p /home/forge/.cache/huggingface && chown -R forge:forge /home/forge /app

USER forge
ENV PYTHONPATH=/app/src:/app

EXPOSE 8000

# start-period is generous on purpose: the first boot may download weights, and
# a health check that fires too early puts the container in a restart loop.
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=4).status==200 else 1)"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
