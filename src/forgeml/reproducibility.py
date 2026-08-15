"""Seeding and environment capture.

"Reproducible" is a claim, and a claim needs evidence. Two things make it real:
seed everything that has an RNG, and record the environment that produced the run.
Both are cheap; skipping either makes every number in the project unfalsifiable.
"""

from __future__ import annotations

import os
import platform
import random
import subprocess
from typing import Any

from forgeml.logging_utils import get_logger

log = get_logger(__name__)


def seed_everything(seed: int, deterministic: bool = False) -> int:
    """Seed Python, NumPy and torch (CPU + CUDA).

    ``deterministic=True`` additionally forces cuDNN into deterministic mode. It
    costs real throughput, so it is off by default: for this project run-to-run
    variance is something to *measure* (train each arm twice, look at the spread),
    not something to hide behind a flag.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    except ImportError:  # pragma: no cover
        pass

    log.info("seeded everything with %d (deterministic=%s)", seed, deterministic)
    return seed


def _safe(fn: Any, default: str = "unknown") -> str:
    try:
        value = fn()
        return str(value) if value is not None else default
    except Exception:  # noqa: BLE001 - environment capture must never break a run
        return default


def git_commit() -> str:
    """Short SHA of the working tree, with a ``-dirty`` suffix when uncommitted.

    Logged as an MLflow tag. This is what lets you check out the exact code that
    produced run #17 instead of guessing.
    """

    def _read() -> str:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        return f"{sha}-dirty" if dirty else sha

    return _safe(_read)


def environment_snapshot() -> dict[str, str]:
    """Everything about this machine that could change a number."""
    snapshot: dict[str, str] = {
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.release()}",
        "processor": platform.processor() or "unknown",
        "git_commit": git_commit(),
    }

    try:
        import torch

        snapshot["torch"] = torch.__version__
        snapshot["cuda_available"] = str(torch.cuda.is_available())
        if torch.cuda.is_available():
            snapshot["cuda"] = torch.version.cuda or "unknown"
            snapshot["gpu_name"] = torch.cuda.get_device_name(0)
            snapshot["gpu_count"] = str(torch.cuda.device_count())
            total_bytes = torch.cuda.get_device_properties(0).total_memory
            snapshot["gpu_total_mb"] = str(round(total_bytes / 1024**2))
    except ImportError:
        snapshot["torch"] = "not installed"

    for module in ("transformers", "peft", "datasets", "mlflow", "bitsandbytes"):
        try:
            snapshot[module] = __import__(module).__version__
        except Exception:  # noqa: BLE001
            snapshot[module] = "not installed"

    # Databricks injects these; harmless and empty elsewhere.
    for env_key in ("DATABRICKS_RUNTIME_VERSION", "DB_CLUSTER_ID"):
        if os.getenv(env_key):
            snapshot[env_key.lower()] = os.environ[env_key]

    return snapshot
