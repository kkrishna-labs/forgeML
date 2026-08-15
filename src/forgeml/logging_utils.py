"""Logging that behaves the same in a terminal, in CI and in a Databricks notebook.

Databricks captures stdout per-cell, so we log to stdout (not stderr) and keep the
format single-line and greppable.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

_CONFIGURED = False
_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-28s | %(message)s"
_DATEFMT = "%H:%M:%S"


def configure_logging(level: str | int | None = None) -> None:
    """Install a single stdout handler on the root logger. Idempotent."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    resolved = level or os.getenv("FORGEML_LOG_LEVEL", "INFO")
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(resolved)

    # These three are extremely chatty during training and drown out our own logs.
    for noisy in ("urllib3", "filelock", "datasets", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Module-level logger. Call this at import time; it configures on first use."""
    configure_logging()
    return logging.getLogger(name)


def log_kv(logger: logging.Logger, title: str, payload: dict[str, Any]) -> None:
    """Print a small aligned key/value block — used for run headers."""
    if not payload:
        return
    width = max(len(str(k)) for k in payload)
    logger.info("%s", title)
    for key, value in payload.items():
        logger.info("  %s : %s", str(key).ljust(width), value)
