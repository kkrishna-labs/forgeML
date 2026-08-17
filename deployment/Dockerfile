# ---------------------------------------------------------------------------
# ForgeML inference image.
#
# Multi-stage: the builder compiles wheels into a self-contained venv, the
# runtime copies that venv and keeps none of the build toolchain. That is worth
# several hundred MB, and a smaller image is not just tidiness — it is
# cold-start time on every serverless platform, paid on every scale-up.
# ---------------------------------------------------------------------------

# --- stage 1: build --------------------------------------------------------
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Build into a venv rather than `pip install --prefix`. The difference matters:
# a --prefix target is not on sys.path, so a later pip install cannot see what
# is already there and re-resolves everything. A venv is visible to subsequent
# installs and is self-contained, so the runtime stage can copy it wholesale.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/

# Torch first, from the CPU index ONLY.
#
# --index-url, not --extra-index-url: with an extra index pip considers both and
# resolves by version, so it happily picks PyPI's CUDA-bundled wheel — ~2.5 GB of
# nvidia-* dependencies that buy nothing on a CPU host.
#
# Doing it as its own step also means peft and accelerate, which declare torch as
# a hard dependency, find it already satisfied in the venv on the next line and
# cannot drag the CUDA build back in.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

RUN pip install ".[api]" transformers peft accelerate

# Fail the build rather than shipping a 3 GB image if the CPU pin ever slips.
RUN python -c "import torch; assert '+cpu' in torch.__version__ or not torch.version.cuda, \
    f'expected a CPU-only torch build, got {torch.__version__}'; print(torch.__version__)"

# --- stage 2: runtime ------------------------------------------------------
FROM python:3.11-slim AS runtime

# Never run as root. On Hugging Face Spaces uid 1000 is also what the platform
# expects to own the writable cache directories.
RUN useradd --create-home --uid 1000 forge

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/home/forge/.cache/huggingface \
    FORGEML_LAZY_LOAD=0 \
    PORT=8000 \
    PATH="/opt/venv/bin:$PATH"

# The venv path must match the builder's, because a venv's scripts and
# pyvenv.cfg hardcode absolute paths.
COPY --from=builder /opt/venv /opt/venv

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
