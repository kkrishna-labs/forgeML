# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — Training
# MAGIC
# MAGIC One notebook, every arm. Pass a config and it fine-tunes, evaluates and
# MAGIC logs to MLflow.
# MAGIC
# MAGIC ```
# MAGIC configs/full_ft.yaml    full fine-tuning
# MAGIC configs/lora.yaml       LoRA r=8
# MAGIC configs/lora_r16.yaml   LoRA r=16 + MLP projections
# MAGIC configs/qlora.yaml      4-bit NF4 base + LoRA r=8
# MAGIC configs/qlora_r16.yaml  4-bit NF4 base + LoRA r=16
# MAGIC ```
# MAGIC
# MAGIC **Why not one notebook per method?** Because `03_lora_training` and
# MAGIC `04_qlora_training` would be 95% identical, and the 5% would drift. Someone
# MAGIC fixes the prompt template in one and not the other, and six weeks later the
# MAGIC comparison table is measuring notebook divergence rather than method
# MAGIC difference. The methods differ only in configuration, so they differ only
# MAGIC in configuration here.

# COMMAND ----------

# MAGIC %pip install --quiet "transformers>=4.44" "peft>=0.12" "accelerate>=0.33" "trl>=0.9" "bitsandbytes>=0.43" "mlflow>=2.16"
# MAGIC %restart_python

# COMMAND ----------

import sys

REPO_ROOT = "/Workspace/Repos/forgeML"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, f"{REPO_ROOT}/src")

dbutils.widgets.text("config", "configs/lora.yaml", "Config path")
dbutils.widgets.text("catalog", "workspace", "Unity Catalog catalog")
dbutils.widgets.text("experiment", "/Shared/forgeml", "MLflow experiment")
dbutils.widgets.text("overrides", "", "Overrides, e.g. training.epochs=3,lora.r=32")

# COMMAND ----------

import yaml

from forgeml.config import load_config

overrides = {
    "data.delta.enabled": True,
    "data.delta.catalog": dbutils.widgets.get("catalog"),
    "tracking.experiment_name": dbutils.widgets.get("experiment"),
}

# Sweeps run the same notebook with a different `overrides` widget value, so a
# rank sweep is a job parameter rather than five copies of a file.
raw_overrides = dbutils.widgets.get("overrides").strip()
for pair in filter(None, (p.strip() for p in raw_overrides.split(","))):
    key, _, value = pair.partition("=")
    overrides[key.strip()] = yaml.safe_load(value)

config = load_config(f"{REPO_ROOT}/{dbutils.widgets.get('config')}", overrides)

print(f"run      : {config.run_slug()}")
print(f"method   : {config.training.method}")
print(f"lr       : {config.training.learning_rate}")
print(f"batch    : {config.training.effective_batch_size} (effective)")
if config.lora:
    print(f"lora     : r={config.lora.r} alpha={config.lora.alpha}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Check the hardware before anything expensive
# MAGIC
# MAGIC QLoRA needs CUDA — bitsandbytes has no CPU kernels. Failing here costs
# MAGIC seconds; failing forty minutes into a job costs a cluster.

# COMMAND ----------

import torch

if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"GPU  : {props.name}")
    print(f"VRAM : {props.total_memory / 1024**3:.1f} GB")
else:
    print("no GPU detected — CPU training only")

assert not (config.quantization.enabled and not torch.cuda.is_available()), (
    "QLoRA requires a CUDA GPU (bitsandbytes has no CPU kernels). Attach GPU "
    "compute, or run configs/lora.yaml instead."
)

# COMMAND ----------

from forgeml.data.schema import InstructionRecord


def read_split(name: str) -> list[InstructionRecord]:
    delta = config.data.delta
    table = delta.fqn("gold", getattr(delta, f"table_{name}"))
    return [
        InstructionRecord(
            id=row["id"],
            instruction=row["instruction"],
            context=row["context"] or "",
            response=row["response"],
            category=row["category"] or "",
        )
        for row in spark.read.table(table).collect()
    ]


train_records = read_split("train")
val_records = read_split("validation")
test_records = read_split("test")
print(f"train={len(train_records)} val={len(val_records)} test={len(test_records)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train, evaluate, log
# MAGIC
# MAGIC All three happen inside one MLflow run, because a training run whose
# MAGIC evaluation lives somewhere else is a run you cannot rank later.

# COMMAND ----------

import json

import mlflow

from forgeml.data.versioning import fingerprint_records
from forgeml.evaluation.evaluator import evaluate_model
from forgeml.tracking.mlflow_utils import (
    log_dataset_fingerprint,
    log_evaluation_report,
    log_metrics,
    start_run,
)
from forgeml.training.trainer import run_training

all_records = train_records + val_records + test_records
fingerprint = fingerprint_records(all_records, config.data)

with start_run(config, extra_tags={"arm": config.training.method}) as run:
    log_dataset_fingerprint(fingerprint)

    model, tokenizer, result = run_training(config, train_records, val_records)
    log_metrics(result.as_metrics())

    report = evaluate_model(model, tokenizer, test_records, config)
    log_evaluation_report(report)

    try:
        mlflow.transformers.log_model(
            transformers_model={"model": model, "tokenizer": tokenizer},
            artifact_path="model",
            task="text-generation",
        )
    except Exception as exc:
        print(f"transformers flavor failed ({exc}); logging the raw checkpoint")
        mlflow.log_artifacts(result.output_dir, artifact_path="model")

    run_id = run.info.run_id

print(report.summary())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Result
# MAGIC
# MAGIC The interesting number on a PEFT run is not the loss — it is the trainable
# MAGIC parameter percentage next to the quality it bought.

# COMMAND ----------

primary = config.evaluation.primary_metric
summary = {
    "run_id": run_id,
    "run_name": config.run_slug(),
    "method": config.training.method,
    "quality": report.quality.scores.get(primary, 0.0),
    "trainable_params": result.params_trainable,
    "trainable_pct": round(result.params_trainable_pct, 4),
    "train_runtime_s": round(result.train_runtime_s, 1),
    "peak_gpu_memory_mb": round(result.peak_gpu_memory_mb, 1),
    "latency_p95_ms": report.latency.p95_ms if report.latency else None,
    "model_size_mb": report.model_size_mb,
}
print(json.dumps(summary, indent=2))

dbutils.notebook.exit(json.dumps(summary))
