"""Process-wide model lifecycle.

The model is loaded once at startup and shared by every request. A per-request
load would be catastrophic — several seconds and several gigabytes each time — so
the loader is a module-level singleton guarded by a lock.

Loading happens in the FastAPI lifespan handler rather than at import time so the
process can bind its port immediately: a container that answers ``/health`` while
weights are still downloading survives its orchestrator's startup probe, whereas
one that blocks on import gets killed and restarted forever.
"""

from __future__ import annotations

import os
import threading
import time
from typing import TYPE_CHECKING

from forgeml.logging_utils import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from forgeml.inference.predictor import BasePredictor

log = get_logger(__name__)

_predictor: BasePredictor | None = None
_lock = threading.Lock()
_started_at = time.time()


def get_predictor() -> BasePredictor:
    """Return the shared predictor, loading it on first call.

    Double-checked locking: the fast path is a plain read with no lock, and only
    the very first concurrent callers pay for synchronization.
    """
    global _predictor
    if _predictor is not None:
        return _predictor

    with _lock:
        if _predictor is None:
            _predictor = _build()
    return _predictor


def _build() -> BasePredictor:
    from forgeml.inference.predictor import load_predictor

    uri = os.getenv("FORGEML_MODEL_URI")
    log.info("loading predictor (FORGEML_MODEL_URI=%s)", uri or "<unset -> stub>")

    started = time.perf_counter()
    predictor = load_predictor(allow_stub=True)
    log.info("predictor ready in %.1fs", time.perf_counter() - started)

    if os.getenv("FORGEML_WARMUP", "1").lower() in ("1", "true", "yes"):
        try:
            predictor.warmup()
        except Exception as exc:  # noqa: BLE001 - warmup is best-effort
            log.warning("warmup failed (serving anyway): %s", exc)

    return predictor


def reset_predictor() -> None:
    """Drop the cached predictor. Used by tests to swap in a stub."""
    global _predictor
    with _lock:
        _predictor = None


def set_predictor(predictor: BasePredictor) -> None:
    """Inject a predictor directly — the seam the API tests use."""
    global _predictor
    with _lock:
        _predictor = predictor


def uptime_seconds() -> float:
    return time.time() - _started_at


def is_loaded() -> bool:
    return _predictor is not None and _predictor.info().loaded
