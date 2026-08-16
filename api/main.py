"""ForgeML inference API.

Four endpoints, and each one exists for a reason:

``GET  /health``   liveness — is the process up? Cheap enough for a 1s probe.
``GET  /model``    which model is actually being served, with its scorecard.
``POST /predict``  generate.
``GET  /metrics``  in-process request counters.

``/model`` is the endpoint that makes this an ML service rather than a web
service. An API that will not tell you which model version answered you is not
operable — you cannot correlate a bad answer with a bad deployment.
"""

from __future__ import annotations

import os
import statistics
import time
from collections import deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.model_loader import get_predictor, is_loaded, uptime_seconds
from api.schemas import (
    ErrorResponse,
    HealthResponse,
    MetricsResponse,
    ModelResponse,
    PredictRequest,
    PredictResponse,
)
from forgeml import __version__
from forgeml.logging_utils import configure_logging, get_logger

configure_logging()
log = get_logger(__name__)

# Bounded so a long-lived process cannot grow without limit; 1000 samples is
# plenty for a stable p95.
_latencies: deque[float] = deque(maxlen=1000)
_counters = {"requests_total": 0, "errors_total": 0, "tokens_generated_total": 0}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the model at startup, not at import.

    ``FORGEML_LAZY_LOAD=1`` defers loading to the first request instead — useful on
    Hugging Face Spaces, where a slow startup can trip the platform's own health
    check before the weights have finished downloading.
    """
    if os.getenv("FORGEML_LAZY_LOAD", "").lower() not in ("1", "true", "yes"):
        get_predictor()
    else:
        log.info("lazy loading enabled — the first request will pay for the load")
    yield
    log.info("shutting down")


app = FastAPI(
    title="ForgeML Inference API",
    description=(
        "Serves the champion model selected by the ForgeML optimization pipeline. "
        "The model behind this endpoint was chosen automatically by a weighted "
        "trade-off across quality, latency, memory and cost — not by picking the "
        "highest score."
    ),
    version=__version__,
    lifespan=lifespan,
    responses={500: {"model": ErrorResponse}},
)

# Wide open because this is a public read-only demo with no user data and no
# authenticated state. A service with either would pin the origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def track_requests(request: Request, call_next: Any) -> Any:
    """Count requests and record server-side latency for ``/metrics``."""
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        _counters["errors_total"] += 1
        raise

    if request.url.path == "/predict":
        _counters["requests_total"] += 1
        _latencies.append((time.perf_counter() - started) * 1000)
        if response.status_code >= 400:
            _counters["errors_total"] += 1

    return response


@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health() -> HealthResponse:
    """Liveness probe. Reports healthy even before weights finish loading."""
    return HealthResponse(
        status="healthy",
        version=__version__,
        model_loaded=is_loaded(),
        uptime_seconds=round(uptime_seconds(), 2),
    )


@app.get("/model", response_model=ModelResponse, tags=["model"])
async def model_info() -> ModelResponse:
    """Which model is serving, and how it scored when it was selected."""
    return ModelResponse(**get_predictor().info().to_dict())


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
async def predict(request: PredictRequest) -> PredictResponse:
    """Generate an answer for one instruction."""
    predictor = get_predictor()
    try:
        result = predictor.predict(
            prompt=request.prompt,
            context=request.context,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )
    except Exception as exc:
        log.exception("prediction failed")
        raise HTTPException(
            status_code=500, detail=f"inference failed: {type(exc).__name__}"
        ) from exc

    _counters["tokens_generated_total"] += result.completion_tokens
    return PredictResponse(**result.to_dict())


@app.get("/metrics", response_model=MetricsResponse, tags=["ops"])
async def metrics() -> MetricsResponse:
    """In-process counters. Resets on restart; not a substitute for Prometheus."""
    samples = list(_latencies)
    mean = statistics.fmean(samples) if samples else 0.0
    p95 = 0.0
    if samples:
        ordered = sorted(samples)
        p95 = ordered[min(round(0.95 * (len(ordered) - 1)), len(ordered) - 1)]

    return MetricsResponse(
        requests_total=_counters["requests_total"],
        errors_total=_counters["errors_total"],
        latency_ms_mean=round(mean, 2),
        latency_ms_p95=round(p95, 2),
        uptime_seconds=round(uptime_seconds(), 2),
        tokens_generated_total=_counters["tokens_generated_total"],
    )


@app.get("/", include_in_schema=False)
async def root() -> JSONResponse:
    return JSONResponse(
        {
            "name": "ForgeML Inference API",
            "version": __version__,
            "docs": "/docs",
            "endpoints": ["/health", "/model", "/predict", "/metrics"],
        }
    )


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=os.getenv("FORGEML_HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("FORGEML_RELOAD", "").lower() in ("1", "true"),
    )
