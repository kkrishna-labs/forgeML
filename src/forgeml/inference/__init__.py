"""Inference: the predictor shared by the REST API and the public demo."""

from __future__ import annotations

from forgeml.inference.predictor import (
    ModelInfo,
    Prediction,
    Predictor,
    StubPredictor,
    load_predictor,
)

__all__ = [
    "ModelInfo",
    "Prediction",
    "Predictor",
    "StubPredictor",
    "load_predictor",
]
