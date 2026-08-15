"""ForgeML — a Databricks-powered LLM fine-tuning, evaluation and optimization platform.

The package is deliberately layered so that the *decision logic* (data validation,
metric computation, model selection) never imports torch. That keeps CI cheap and
makes the interesting parts unit-testable on a laptop.

Layers
------
``forgeml.data``          load / validate / split / version an instruction dataset
``forgeml.training``      full, LoRA and QLoRA fine-tuning entry points  (needs torch)
``forgeml.evaluation``    quality, latency, memory and size measurement
``forgeml.optimization``  multi-objective champion selection
``forgeml.tracking``      MLflow helpers (experiments, runs, artifacts)
``forgeml.registry``      MLflow / Unity Catalog model registry helpers
``forgeml.inference``     the predictor used by the API and the demo
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["__version__"]
