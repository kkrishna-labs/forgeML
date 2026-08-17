"""Memory and model-size measurement.

"Memory" is three different numbers and conflating them is the usual mistake:

* **weights** — parameters x bytes-per-parameter. Static, predictable.
* **peak allocated during inference** — weights + activations + the KV cache.
  This is what decides whether the model fits on the GPU you can afford.
* **peak reserved** — what the CUDA caching allocator took from the driver.
  Always >= allocated, and it is the number ``nvidia-smi`` shows you, which is why
  ``nvidia-smi`` and your logs never seem to agree.

All three are reported. The selector uses peak allocated, because that is the one
that predicts whether the next size up of hardware is required.
"""

from __future__ import annotations

import gc
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from forgeml.logging_utils import get_logger

log = get_logger(__name__)


@dataclass
class MemoryResult:
    """A memory snapshot for one measured workload."""

    weights_mb: float = 0.0
    peak_allocated_mb: float = 0.0
    peak_reserved_mb: float = 0.0
    host_rss_mb: float = 0.0
    gpu_total_mb: float = 0.0
    device: str = "cpu"

    @property
    def activation_overhead_mb(self) -> float:
        """Everything that is not weights: activations, KV cache, workspaces."""
        return max(self.peak_allocated_mb - self.weights_mb, 0.0)

    @property
    def gpu_utilization_pct(self) -> float:
        if not self.gpu_total_mb:
            return 0.0
        return 100.0 * self.peak_allocated_mb / self.gpu_total_mb

    def as_metrics(self, prefix: str = "memory") -> dict[str, float]:
        return {
            f"{prefix}/weights_mb": round(self.weights_mb, 2),
            f"{prefix}/peak_allocated_mb": round(self.peak_allocated_mb, 2),
            f"{prefix}/peak_reserved_mb": round(self.peak_reserved_mb, 2),
            f"{prefix}/activation_overhead_mb": round(self.activation_overhead_mb, 2),
            f"{prefix}/host_rss_mb": round(self.host_rss_mb, 2),
            f"{prefix}/gpu_utilization_pct": round(self.gpu_utilization_pct, 2),
        }

    def summary(self) -> str:
        return (
            f"weights {self.weights_mb:.0f}MB | peak {self.peak_allocated_mb:.0f}MB "
            f"(+{self.activation_overhead_mb:.0f}MB activations) on {self.device}"
        )


def weights_size_mb(model: Any) -> float:
    """Bytes actually occupied by parameters and buffers, in MB.

    Reads ``element_size()`` per tensor rather than assuming a dtype, so a 4-bit
    quantized model reports its real footprint instead of an fp16 estimate.
    """
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    total += sum(b.numel() * b.element_size() for b in model.buffers())
    return total / 1024**2


def checkpoint_size_mb(path: str | Path) -> float:
    """Total size of a saved checkpoint directory, in MB — the download size."""
    directory = Path(path)
    if not directory.exists():
        return 0.0
    if directory.is_file():
        return directory.stat().st_size / 1024**2
    return sum(f.stat().st_size for f in directory.rglob("*") if f.is_file()) / 1024**2


def host_rss_mb() -> float:
    """Resident set size of this process, in MB.

    ``psutil`` is not a hard dependency, so fall back to the Linux proc interface
    and return 0.0 where neither is available (Windows without psutil).
    """
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / 1024**2
    except ImportError:
        pass

    # os.sysconf does not exist on Windows, hence getattr rather than a direct
    # call — a bare os.sysconf(...) is an AttributeError there and a type error
    # in any checker running on Windows.
    sysconf = getattr(os, "sysconf", None)
    if sysconf is None:
        return 0.0

    try:
        with open(f"/proc/{os.getpid()}/statm", encoding="utf-8") as fh:
            pages = int(fh.read().split()[1])
        return pages * sysconf("SC_PAGE_SIZE") / 1024**2
    except Exception:  # noqa: BLE001
        return 0.0


@contextmanager
def track_peak_memory() -> Iterator[dict[str, float]]:
    """Context manager recording peak CUDA memory over the enclosed block.

    ::

        with track_peak_memory() as peak:
            model.generate(...)
        print(peak["peak_allocated_mb"])

    Resets the peak counters on entry, so nested or sequential measurements do not
    inherit an earlier spike.
    """
    stats: dict[str, float] = {"peak_allocated_mb": 0.0, "peak_reserved_mb": 0.0}

    try:
        import torch
    except ImportError:  # pragma: no cover
        # No torch means no CUDA counters. Yield zeros rather than raising, so
        # callers never have to special-case a CPU-only environment.
        yield stats
        return

    if not torch.cuda.is_available():
        yield stats
        return

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    try:
        yield stats
    finally:
        torch.cuda.synchronize()
        stats["peak_allocated_mb"] = torch.cuda.max_memory_allocated() / 1024**2
        stats["peak_reserved_mb"] = torch.cuda.max_memory_reserved() / 1024**2


def measure_inference_memory(
    model: Any,
    tokenizer: Any,
    prompt: str = "Explain gradient descent in simple terms.",
    max_new_tokens: int = 64,
    batch_size: int = 1,
) -> MemoryResult:
    """Measure peak memory for one realistic generation call."""
    import torch

    device = next(model.parameters()).device
    result = MemoryResult(weights_mb=weights_size_mb(model), device=str(device))

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(device.index or 0)
        result.gpu_total_mb = props.total_memory / 1024**2

    encoded = tokenizer([prompt] * batch_size, return_tensors="pt", padding=True)
    encoded = {k: v.to(device) for k, v in encoded.items()}

    model.eval()
    with track_peak_memory() as peak, torch.no_grad():
        model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    result.peak_allocated_mb = peak["peak_allocated_mb"]
    result.peak_reserved_mb = peak["peak_reserved_mb"]
    result.host_rss_mb = host_rss_mb()

    # On CPU there is no allocator to query, so weights are the best estimate
    # available and reporting 0 would be actively misleading.
    if result.peak_allocated_mb == 0.0:
        result.peak_allocated_mb = result.weights_mb

    log.info("memory: %s", result.summary())
    return result
