# Databricks notebook source
# MAGIC %md
# MAGIC # 01 — Data preparation
# MAGIC
# MAGIC Load, validate, split and fingerprint the instruction dataset, then write
# MAGIC the medallion tables.
# MAGIC
# MAGIC **This notebook trains nothing.** Keeping data preparation in its own job
# MAGIC means a training run never silently re-derives its own dataset, which is
# MAGIC how two runs end up incomparable without anyone noticing.
# MAGIC
# MAGIC ```
# MAGIC raw -> validate -> dedupe -> split -> fingerprint -> delta
# MAGIC ```

# COMMAND ----------

# MAGIC %pip install --quiet --upgrade "pydantic>=2.6" "pyyaml>=6.0" "datasets>=3.0" "huggingface_hub>=0.34" "fsspec>=2024.10"
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC Note that `datasets`, `huggingface_hub` and `fsspec` are upgraded *together*.
# MAGIC Upgrading only `datasets` leaves the runtime's older pinned `huggingface_hub`
# MAGIC in place, and the mismatch surfaces later as:
# MAGIC
# MAGIC ```
# MAGIC TypeError: HfFileSystem.find() got multiple values for keyword 'maxdepth'
# MAGIC ```
# MAGIC
# MAGIC which points at neither the real cause nor the fix. A partial upgrade of a
# MAGIC tightly-coupled trio is worse than no upgrade at all.
# MAGIC
# MAGIC If it still fails, the loader falls back to downloading the data files
# MAGIC directly and parsing them itself — no fsspec involved.

# COMMAND ----------

# MAGIC %run ./_bootstrap

# COMMAND ----------

dbutils.widgets.text("config", "configs/base.yaml", "Config path")
dbutils.widgets.text("catalog", "workspace", "Unity Catalog catalog")
dbutils.widgets.text("max_examples", "5000", "Max examples (blank = all)")

# COMMAND ----------

import json

from forgeml.config import load_config
from forgeml.data import (
    fingerprint_records,
    load_raw_records,
    records_to_frame,
    split_records,
    validate_records,
)

config_path = f"{REPO_ROOT}/{dbutils.widgets.get('config')}"
overrides = {"data.delta.enabled": True, "data.delta.catalog": dbutils.widgets.get("catalog")}

raw_max = dbutils.widgets.get("max_examples").strip()
if raw_max:
    overrides["data.max_examples"] = int(raw_max)

config = load_config(config_path, overrides)
print(f"dataset : {config.data.source}")
print(f"version : {config.data.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Bronze — raw, exactly as it arrived
# MAGIC
# MAGIC The bronze layer is deliberately unfiltered. If a cleaning rule turns out
# MAGIC to be wrong three weeks from now, you re-run silver from bronze instead of
# MAGIC re-downloading and hoping the upstream dataset has not changed.

# COMMAND ----------

records = load_raw_records(config.data)
print(f"loaded {len(records)} raw records")

delta = config.data.delta
for schema in (delta.schema_bronze, delta.schema_silver, delta.schema_gold):
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {delta.catalog}.{schema}")

bronze_table = delta.fqn("bronze", delta.table_raw)
spark.createDataFrame(records_to_frame(records)).write.mode("overwrite").option(
    "overwriteSchema", "true"
).saveAsTable(bronze_table)
print(f"wrote {bronze_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Silver — validated and deduplicated
# MAGIC
# MAGIC The report below is the artifact worth keeping. "cleaned 5,000 -> 4,510"
# MAGIC tells you nothing; "412 dropped for `response_missing`" tells you whether
# MAGIC the dataset changed or your rule did.

# COMMAND ----------

clean, report = validate_records(records, config.data)
print(report.summary())

assert report.is_healthy, (
    f"keep rate {report.keep_rate:.1%} is too low — a validation rule is probably "
    "wrong. Fix it before spending GPU time on this data."
)

silver_table = delta.fqn("silver", delta.table_clean)
spark.createDataFrame(records_to_frame(clean)).write.mode("overwrite").option(
    "overwriteSchema", "true"
).saveAsTable(silver_table)
print(f"wrote {silver_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gold — the three splits
# MAGIC
# MAGIC Split assignment is a pure function of each example's content hash, not a
# MAGIC shuffle. Growing the dataset later leaves every existing example in the
# MAGIC split it was already in, which is the only way "v1.0 vs v2.0" means
# MAGIC anything.

# COMMAND ----------

splits = split_records(clean, config.data)
splits.assert_disjoint()

for name, table in (
    ("train", delta.table_train),
    ("validation", delta.table_validation),
    ("test", delta.table_test),
):
    target = delta.fqn("gold", table)
    spark.createDataFrame(records_to_frame(splits.as_dict()[name])).write.mode(
        "overwrite"
    ).option("overwriteSchema", "true").saveAsTable(target)
    print(f"{name:<11} {len(splits.as_dict()[name]):>6}  ->  {target}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fingerprint
# MAGIC
# MAGIC A version string is a label a human typed. A content hash over every
# MAGIC example is a claim you can verify: re-run this notebook, recompute it,
# MAGIC compare. Every training run logs this hash, so "which data produced run
# MAGIC #17" stops being guesswork.

# COMMAND ----------

fingerprint = fingerprint_records(clean, config.data, splits)
print(json.dumps(fingerprint.to_dict(), indent=2))

# Written where the training notebooks can read it without re-deriving anything.
output_dir = f"/Volumes/{delta.catalog}/{delta.schema_gold}/artifacts"
try:
    dbutils.fs.mkdirs(output_dir)
    dbutils.fs.put(
        f"{output_dir}/fingerprint_{config.data.version}.json",
        json.dumps(fingerprint.to_dict(), indent=2),
        overwrite=True,
    )
    print(f"fingerprint -> {output_dir}")
except Exception as exc:
    # Free Edition workspaces may not expose Volumes; the Delta tables are the
    # source of truth regardless, so this is not fatal.
    print(f"could not write to a Volume ({exc}); fingerprint lives in the task output")

dbutils.notebook.exit(
    json.dumps(
        {
            "dataset_version": config.data.version,
            "content_hash": fingerprint.content_hash,
            "num_examples": fingerprint.num_examples,
            "splits": splits.sizes,
        }
    )
)
