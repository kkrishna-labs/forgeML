"""Cost modelling.

This is an **estimate built from measured throughput and a stated hourly rate**,
not a vendor quote. Being explicit about that is the point: the number is only as
good as its assumptions, so the assumptions travel with it in every artifact.

::

    cost per 1k requests = (1000 / requests_per_second) / 3600 * gpu_hourly_usd

What the model deliberately ignores: cold starts, autoscaling headroom, network
egress, and the fact that a real endpoint is idle most of the time. It is a
*relative* measure — good for ranking Model A against Model B on identical
hardware, useless as a budget forecast.
"""

from __future__ import annotations

from dataclasses import dataclass

SECONDS_PER_HOUR = 3600.0


@dataclass
class CostEstimate:
    """Serving cost derived from measured latency."""

    gpu_hourly_usd: float
    assumed_gpu: str
    latency_ms: float
    batch_size: int = 1
    tokens_per_second: float = 0.0

    @property
    def requests_per_hour(self) -> float:
        """Sequential capacity — one request at a time, no queueing model."""
        if self.latency_ms <= 0:
            return 0.0
        return SECONDS_PER_HOUR / (self.latency_ms / 1000.0) * self.batch_size

    @property
    def cost_per_request_usd(self) -> float:
        if self.requests_per_hour <= 0:
            return float("inf")
        return self.gpu_hourly_usd / self.requests_per_hour

    @property
    def cost_per_1k_requests_usd(self) -> float:
        return self.cost_per_request_usd * 1000

    @property
    def cost_per_1m_tokens_usd(self) -> float:
        """The unit hosted LLM providers price in, so comparisons are possible."""
        if self.tokens_per_second <= 0:
            return float("inf")
        tokens_per_hour = self.tokens_per_second * SECONDS_PER_HOUR
        return self.gpu_hourly_usd / tokens_per_hour * 1_000_000

    def as_metrics(self, prefix: str = "cost") -> dict[str, float]:
        return {
            f"{prefix}/per_1k_requests_usd": round(self.cost_per_1k_requests_usd, 6),
            f"{prefix}/per_1m_tokens_usd": round(self.cost_per_1m_tokens_usd, 4),
            f"{prefix}/requests_per_hour": round(self.requests_per_hour, 2),
        }

    def summary(self) -> str:
        return (
            f"${self.cost_per_1k_requests_usd:.4f}/1k requests "
            f"(${self.cost_per_1m_tokens_usd:.2f}/1M tokens) "
            f"assuming {self.assumed_gpu} at ${self.gpu_hourly_usd:.2f}/h"
        )


def estimate_training_cost(runtime_seconds: float, gpu_hourly_usd: float) -> float:
    """One-off cost of producing the model. Amortised over nothing — it is sunk.

    Reported separately from serving cost because the two answer different
    questions: training cost decides whether an experiment was worth running,
    serving cost decides whether a model is worth deploying.
    """
    return runtime_seconds / SECONDS_PER_HOUR * gpu_hourly_usd
