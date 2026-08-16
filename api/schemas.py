"""Request and response schemas.

Pydantic models rather than bare dicts, for three reasons: FastAPI turns them into
OpenAPI docs automatically, invalid input gets a 422 with a precise error instead
of a 500 deep in the model code, and the schema is a contract the tests can assert
against.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """``POST /predict`` input."""

    prompt: str = Field(
        ...,
        min_length=1,
        max_length=8000,
        description="The instruction to answer.",
        examples=["Explain overfitting in machine learning."],
    )
    context: str | None = Field(
        default=None,
        max_length=16000,
        description="Optional grounding context the answer should be based on.",
    )
    max_new_tokens: int = Field(
        default=256, ge=1, le=1024, description="Upper bound on generated tokens."
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="0 means greedy decoding, which is deterministic and reproducible.",
    )
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "prompt": "Explain overfitting in machine learning.",
                    "max_new_tokens": 256,
                    "temperature": 0.0,
                }
            ]
        }
    }


class PredictResponse(BaseModel):
    """``POST /predict`` output."""

    model: str
    prediction: str
    latency_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = "stop"


class HealthResponse(BaseModel):
    """``GET /health`` — a liveness probe, deliberately trivial and cheap.

    Kept separate from readiness: a container that is up but still loading weights
    should report healthy so the orchestrator does not kill it mid-load, while
    ``/model`` reveals whether a real model is actually behind the endpoint.
    """

    status: str = "healthy"
    version: str
    model_loaded: bool = False
    uptime_seconds: float = 0.0


class ModelResponse(BaseModel):
    """``GET /model`` — the model card, served."""

    name: str
    method: str
    base_model: str
    version: str
    source: str
    quality: float | None = None
    latency_p95_ms: float | None = None
    parameters: int | None = None
    quantization: str | None = None
    loaded: bool = False

    model_config = {"extra": "allow"}


class MetricsResponse(BaseModel):
    """``GET /metrics`` — counters for this process only.

    In-process and reset on restart. Enough to demonstrate the endpoint and to
    sanity-check a demo; a real deployment would export Prometheus instead.
    """

    requests_total: int = 0
    errors_total: int = 0
    latency_ms_mean: float = 0.0
    latency_ms_p95: float = 0.0
    uptime_seconds: float = 0.0
    tokens_generated_total: int = 0


class ErrorResponse(BaseModel):
    detail: str
    error_type: str | None = None
    extra: dict[str, Any] | None = None
