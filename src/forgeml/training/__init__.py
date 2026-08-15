"""Fine-tuning: full, LoRA and QLoRA.

All three arms share one code path. The *only* thing that differs is how the model
is constructed in :mod:`forgeml.training.model_factory` — same data, same collator,
same trainer, same evaluation. That is what makes the comparison in the results
table a comparison of methods rather than a comparison of accidents.

Importing this subpackage pulls in torch, so keep it out of anything that has to
run in a CPU-only CI job.
"""

from __future__ import annotations

from forgeml.training.dataset import SupervisedCollator, SupervisedDataset, build_dataset
from forgeml.training.model_factory import (
    ParameterCount,
    count_parameters,
    load_model,
    load_tokenizer,
)
from forgeml.training.trainer import TrainingResult, run_training

__all__ = [
    "ParameterCount",
    "SupervisedCollator",
    "SupervisedDataset",
    "TrainingResult",
    "build_dataset",
    "count_parameters",
    "load_model",
    "load_tokenizer",
    "run_training",
]
