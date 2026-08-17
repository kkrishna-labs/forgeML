# Databricks notebook source
# MAGIC %md
# MAGIC # 04 — Cross-run evaluation
# MAGIC
# MAGIC Reads every finished run out of MLflow and builds the comparison table.
# MAGIC
# MAGIC No model is loaded here and no GPU is needed. All the measuring already
# MAGIC happened during training; this notebook only assembles it. That separation
# MAGIC is what lets you re-cut the comparison a dozen times without re-running a
# MAGIC single experiment.
# MAGIC
# MAGIC Output is a Delta table, which is what the dashboard reads.

# COMMAND ----------

# MAGIC %pip install --quiet "mlflow>=2.16" "pydantic>=2.6" "pyyaml>=6.0"
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ./_bootstrap

# COMMAND ----------

dbutils.widgets.text("config", "configs/base.yaml", "Config path")
dbutils.widgets.text("catalog", "workspace", "Unity Catalog catalog")
dbutils.widgets.text("experiment", "/Shared/forgeml", "MLflow experiment")

# COMMAND ----------

from forgeml.config import load_config
from forgeml.tracking.mlflow_utils import candidates_from_experiment

config = load_config(
    f"{REPO_ROOT}/{dbutils.widgets.get('config')}",
    {
        "data.delta.catalog": dbutils.widgets.get("catalog"),
        "tracking.experiment_name": dbutils.widgets.get("experiment"),
    },
)

candidates = candidates_from_experiment(config.tracking.experiment_name)
assert candidates, (
    f"no finished runs with a quality metric in {config.tracking.experiment_name} — "
    "run notebook 02 and 03 first"
)
print(f"{len(candidates)} candidate run(s)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## The comparison table
# MAGIC
# MAGIC The shape to look for: quality rising steeply from baseline to the first
# MAGIC fine-tune, then flattening, while memory and latency vary by a factor of
# MAGIC two or more. That gap is where the engineering decision lives.

# COMMAND ----------

import pandas as pd

frame = pd.DataFrame(
    [
        {
            "run_name": c.run_name,
            "method": c.method,
            "quality": round(c.quality, 4),
            "latency_p95_ms": round(c.latency_ms, 1),
            "peak_memory_mb": round(c.memory_mb, 1),
            "model_size_mb": round(c.model_size_mb, 1),
            "cost_per_1k_usd": round(c.cost_per_1k_usd, 5),
            "is_baseline": c.is_baseline,
            "run_id": c.run_id,
            "dataset_version": c.metadata.get("dataset_version", ""),
            "git_commit": c.metadata.get("git_commit", ""),
        }
        for c in candidates
    ]
).sort_values("quality", ascending=False)

display(frame)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Relative to baseline
# MAGIC
# MAGIC Absolute scores are hard to reason about; ratios are not. "94% of the
# MAGIC quality at 45% of the memory" is a sentence that makes a decision obvious.

# COMMAND ----------

baseline = next((c for c in candidates if c.is_baseline), None)

if baseline is None:
    print("no baseline run found — run notebook 02 to make these numbers meaningful")
else:
    relative = frame[~frame["is_baseline"]].copy()
    relative["quality_vs_baseline"] = (
        relative["quality"] / baseline.quality if baseline.quality else float("nan")
    )
    relative["latency_vs_baseline"] = relative["latency_p95_ms"] / baseline.latency_ms
    relative["memory_vs_baseline"] = relative["peak_memory_mb"] / baseline.memory_mb
    relative["size_vs_baseline"] = relative["model_size_mb"] / baseline.model_size_mb

    display(
        relative[
            [
                "run_name",
                "method",
                "quality_vs_baseline",
                "latency_vs_baseline",
                "memory_vs_baseline",
                "size_vs_baseline",
            ]
        ].round(3)
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## The Pareto frontier
# MAGIC
# MAGIC Anything **off** the frontier is beaten by another candidate on every
# MAGIC single axis, so no weighting of priorities could justify deploying it.
# MAGIC This is the part of the analysis that survives people disagreeing about
# MAGIC what matters.

# COMMAND ----------

from forgeml.optimization.selector import pareto_frontier

frontier = pareto_frontier([c for c in candidates if not c.is_baseline])
frontier_names = {c.run_name for c in frontier}

print(f"{len(frontier)} of {len(candidates) - 1} candidates are non-dominated:\n")
for candidate in sorted(frontier, key=lambda c: -c.quality):
    print(
        f"  {candidate.run_name:<40} quality {candidate.quality:.4f}  "
        f"latency {candidate.latency_ms:>6.0f}ms  memory {candidate.memory_mb:>6.0f}MB"
    )

dominated = [c.run_name for c in candidates if not c.is_baseline and c.run_name not in frontier_names]
if dominated:
    print(f"\ndominated (never deploy these): {', '.join(dominated)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Persist for the dashboard

# COMMAND ----------

frame["on_pareto_frontier"] = frame["run_name"].isin(frontier_names)

results_table = f"{config.data.delta.catalog}.{config.data.delta.schema_gold}.experiment_results"
spark.createDataFrame(frame).write.mode("overwrite").option(
    "overwriteSchema", "true"
).saveAsTable(results_table)

print(f"wrote {results_table}")
print("\nBuild the dashboard on top of this table with:")
print(f"  SELECT * FROM {results_table} ORDER BY quality DESC")

# COMMAND ----------

import json

dbutils.notebook.exit(
    json.dumps(
        {
            "num_candidates": len(candidates),
            "pareto_frontier": sorted(frontier_names),
            "results_table": results_table,
        }
    )
)
