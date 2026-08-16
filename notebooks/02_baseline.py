# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — Baseline
# MAGIC
# MAGIC Evaluate the **untuned** base model.
# MAGIC
# MAGIC This notebook trains nothing, and it is not optional. Without it the
# MAGIC project cannot answer the only question that matters — *did fine-tuning
# MAGIC actually improve anything?* A LoRA run scoring 0.84 is meaningless until
# MAGIC you know whether the base model already scored 0.83.
# MAGIC
# MAGIC It also establishes the latency and memory floor that every fine-tuned arm
# MAGIC will be compared against.

# COMMAND ----------

# MAGIC %pip install --quiet "transformers>=4.44" "peft>=0.12" "accelerate>=0.33" "mlflow>=2.16"
# MAGIC %restart_python

# COMMAND ----------

import sys

REPO_ROOT = "/Workspace/Repos/forgeml"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, f"{REPO_ROOT}/src")

dbutils.widgets.text("config", "configs/baseline.yaml", "Config path")
dbutils.widgets.text("catalog", "workspace", "Unity Catalog catalog")
dbutils.widgets.text("experiment", "/Shared/forgeml", "MLflow experiment")

# COMMAND ----------

from forgeml.config import load_config
from forgeml.data.schema import InstructionRecord

config = load_config(
    f"{REPO_ROOT}/{dbutils.widgets.get('config')}",
    {
        "data.delta.enabled": True,
        "data.delta.catalog": dbutils.widgets.get("catalog"),
        "tracking.experiment_name": dbutils.widgets.get("experiment"),
    },
)


def read_split(name: str) -> list[InstructionRecord]:
    """Read a gold table back into canonical records."""
    delta = config.data.delta
    table = delta.fqn("gold", getattr(delta, f"table_{name}"))
    rows = spark.read.table(table).collect()
    return [
        InstructionRecord(
            id=row["id"],
            instruction=row["instruction"],
            context=row["context"] or "",
            response=row["response"],
            category=row["category"] or "",
        )
        for row in rows
    ]


test_records = read_split("test")
print(f"{len(test_records)} test examples")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate the untouched checkpoint
# MAGIC
# MAGIC Same evaluation function, same prompt formatting and same decoding
# MAGIC settings that every fine-tuned arm will use. That identity is the entire
# MAGIC value of the baseline — measure it differently and the comparison is void.

# COMMAND ----------

from forgeml.evaluation.evaluator import evaluate_model
from forgeml.training.model_factory import load_model, load_tokenizer
from forgeml.tracking.mlflow_utils import log_evaluation_report, start_run

with start_run(config, run_name="baseline-zero-shot", extra_tags={"arm": "baseline"}) as run:
    tokenizer = load_tokenizer(config)
    model = load_model(config, for_training=False)

    report = evaluate_model(model, tokenizer, test_records, config, run_name="baseline")
    log_evaluation_report(report)

    baseline_run_id = run.info.run_id
    print(report.summary())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Record the floor
# MAGIC
# MAGIC Write these numbers down. Every later notebook is trying to beat them, and
# MAGIC the selector's `min_quality_ratio_vs_baseline` gate is expressed against
# MAGIC this exact run.

# COMMAND ----------

import json

primary = config.evaluation.primary_metric
summary = {
    "baseline_run_id": baseline_run_id,
    "quality": report.quality.scores.get(primary, 0.0),
    "primary_metric": primary,
    "latency_p95_ms": report.latency.p95_ms if report.latency else None,
    "peak_memory_mb": report.memory.peak_allocated_mb if report.memory else None,
    "model_size_mb": report.model_size_mb,
    "perplexity": report.perplexity.perplexity if report.perplexity else None,
}
print(json.dumps(summary, indent=2))

dbutils.notebook.exit(json.dumps(summary))
