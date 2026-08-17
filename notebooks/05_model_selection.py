# Databricks notebook source
# MAGIC %md
# MAGIC # 05 — Champion selection and registration
# MAGIC
# MAGIC The end of the pipeline: apply the selection policy, register the winner,
# MAGIC point the `@champion` alias at it.
# MAGIC
# MAGIC **No human picks the model.** A person picking "the one that looks best"
# MAGIC is not a pipeline — it is a habit that stops working the moment they go on
# MAGIC holiday. The policy is code, it is versioned, and it explains itself.
# MAGIC
# MAGIC ```
# MAGIC runs -> hard constraints -> normalise -> weighted utility -> champion
# MAGIC                                                                 |
# MAGIC                                                    registry @champion
# MAGIC ```

# COMMAND ----------

# MAGIC %pip install --quiet "mlflow>=2.16" "pydantic>=2.6" "pyyaml>=6.0"
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ./_bootstrap

# COMMAND ----------

dbutils.widgets.text("config", "configs/base.yaml", "Config path")
dbutils.widgets.text("catalog", "workspace", "Unity Catalog catalog")
dbutils.widgets.text("experiment", "/Shared/forgeml", "MLflow experiment")
dbutils.widgets.text("model_name", "forgeml_champion", "Registered model name")
dbutils.widgets.dropdown("dry_run", "false", ["true", "false"], "Dry run")

# COMMAND ----------

from forgeml.config import load_config

model_name = dbutils.widgets.get("model_name")
catalog = dbutils.widgets.get("catalog")
dry_run = dbutils.widgets.get("dry_run") == "true"

# Unity Catalog model names are three-level (catalog.schema.model). Registering
# into UC also requires the registry URI to be set before any registry call.
use_unity_catalog = "." not in model_name
if use_unity_catalog:
    model_name = f"{catalog}.forgeml_gold.{model_name}"

config = load_config(
    f"{REPO_ROOT}/{dbutils.widgets.get('config')}",
    {
        "data.delta.catalog": catalog,
        "tracking.experiment_name": dbutils.widgets.get("experiment"),
        "tracking.registered_model_name": model_name,
        "tracking.registry_uri": "databricks-uc",
    },
)
print(f"registering into: {model_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## The policy
# MAGIC
# MAGIC Constraints first, preferences second. A gate expresses a requirement and
# MAGIC no amount of weighted score should be able to buy its way past one.

# COMMAND ----------

selection_config = config.selection

print("hard constraints")
print(f"  min quality vs baseline : {selection_config.min_quality_ratio_vs_baseline:.0%}")
print(f"  must beat baseline      : {selection_config.require_beats_baseline}")
print(f"  max latency             : {selection_config.max_latency_ms or 'unconstrained'}")
print(f"  max memory              : {selection_config.max_memory_mb or 'unconstrained'}")
print("\nweights (applied to normalised metrics)")
for name, weight in selection_config.weights.items():
    print(f"  {name:<12} {weight}")

# COMMAND ----------

from forgeml.optimization.selector import select_champion
from forgeml.tracking.mlflow_utils import candidates_from_experiment, setup_mlflow

setup_mlflow(config)
candidates = candidates_from_experiment(config.tracking.experiment_name)
assert candidates, "no candidate runs found — run notebooks 02 and 03 first"

selection = select_champion(candidates, selection_config)

print(selection.to_markdown())

# COMMAND ----------

# MAGIC %md
# MAGIC ## The decision
# MAGIC
# MAGIC Note when the champion is *not* the highest-quality candidate. That is the
# MAGIC pipeline working, not the pipeline malfunctioning.

# COMMAND ----------

if selection.champion is None:
    print("NO CHAMPION — every candidate failed the constraints:\n")
    for candidate in selection.rejected:
        if candidate.is_baseline:
            continue
        print(f"  {candidate.run_name}: {'; '.join(candidate.rejected_reasons)}")
    print(
        "\nThis is a valid outcome. A pipeline that always ships something will "
        "eventually ship a regression."
    )
else:
    champion = selection.champion
    best_quality = max(c.quality for c in selection.ranked)

    print(f"champion : {champion.run_name}")
    print(f"utility  : {champion.utility:.4f}")
    print(f"quality  : {champion.quality:.4f}")
    print(f"latency  : {champion.latency_ms:.0f} ms (p95)")
    print(f"memory   : {champion.memory_mb:.0f} MB")
    print(f"size     : {champion.model_size_mb:.0f} MB")

    if champion.quality < best_quality:
        print(
            f"\nThe champion is NOT the highest-quality candidate "
            f"({champion.quality:.4f} vs {best_quality:.4f}). It won on the "
            "trade-off — which is the entire point of the selector."
        )

    gain = selection.quality_gain_vs_baseline()
    speedup = selection.speedup_vs_baseline()
    if gain is not None:
        print(f"\nvs baseline: {gain:+.4f} quality")
    if speedup is not None:
        print(f"vs baseline: {speedup:.2f}x latency")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register and promote
# MAGIC
# MAGIC The alias is what serving points at. Moving `@champion` to a new version is
# MAGIC atomic, so promotion and rollback are the same one-line operation and no
# MAGIC consumer ever hardcodes a version number.

# COMMAND ----------

from forgeml.registry.model_registry import register_champion

registered = register_champion(selection, model_name, dry_run=dry_run)

if registered:
    print(f"serving URI : {registered.uri}")
    print(f"pinned URI  : {registered.pinned_uri}")
    print(f"version     : {registered.version}")
    print("\nRoll back with:")
    print(f"  from forgeml.registry.model_registry import rollback")
    print(f"  rollback('{model_name}', to_version='<previous>')")
elif dry_run:
    print("dry run — nothing was registered")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model card
# MAGIC
# MAGIC Generated from the evaluation artifacts of the winning run, so it cannot
# MAGIC drift from the model it describes.

# COMMAND ----------

import json

import mlflow

if selection.champion is not None:
    champion_run = mlflow.get_run(selection.champion.run_id)

    card_lines = [
        f"# {model_name}",
        "",
        f"Selected automatically on {selection.champion.utility:.4f} utility.",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Quality ({config.evaluation.primary_metric}) | {selection.champion.quality:.4f} |",
        f"| Latency p95 | {selection.champion.latency_ms:.0f} ms |",
        f"| Peak memory | {selection.champion.memory_mb:.0f} MB |",
        f"| Model size | {selection.champion.model_size_mb:.0f} MB |",
        f"| Cost / 1k requests | ${selection.champion.cost_per_1k_usd:.4f} |",
        "",
        f"- **Method**: {selection.champion.method}",
        f"- **Base model**: {champion_run.data.tags.get('base_model', 'unknown')}",
        f"- **Dataset version**: {champion_run.data.tags.get('dataset_version', 'unknown')}",
        f"- **Code commit**: {champion_run.data.tags.get('env.git_commit', 'unknown')}",
        f"- **MLflow run**: {selection.champion.run_id}",
        "",
        "## Selection",
        "",
        selection.to_markdown(),
    ]
    displayHTML("<pre>" + "\n".join(card_lines) + "</pre>")

dbutils.notebook.exit(
    json.dumps(
        {
            "champion": selection.champion.run_name if selection.champion else None,
            "champion_run_id": selection.champion.run_id if selection.champion else None,
            "registered_version": registered.version if registered else None,
            "model_uri": registered.uri if registered else None,
            "rejected": [c.run_name for c in selection.rejected if not c.is_baseline],
        }
    )
)
