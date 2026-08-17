# Databricks setup

## What runs where

```
GitHub                          Databricks                    Hugging Face
------                          ----------                    ------------
source of truth      -- Git -->  notebooks + jobs
tests / CI                       training (GPU)
                                 MLflow tracking
                                 model registry
                                        |
                                        +--- champion checkpoint --> Space (demo)
```

Databricks owns training, tracking and the registry. Hugging Face owns the
public URL. Neither depends on the other staying up.

---

## 1. Workspace

Sign up for **Databricks Free Edition** (the old Community Edition was retired
in 2025). Free Edition gives you serverless compute, MLflow and Unity Catalog.
It does **not** give you GPU model serving, and GPU compute availability is
limited — which is exactly why this project is designed around a ~0.5B model and
why the public demo lives elsewhere.

## 2. Connect the repo

Workspace → **Repos** → Add repo → paste the GitHub URL.

Nothing is installed; the package is imported straight from source, so a
`git pull` in the Git folder is the entire deploy step for code changes.

**You do not need to know where the checkout landed.** Databricks puts Git
folders under `/Workspace/Repos/<repo>`, `/Workspace/Repos/<email>/<repo>` or
`/Workspace/Users/<email>/<repo>` depending on workspace vintage, so every
notebook runs `%run ./_bootstrap`, which derives the root from its own path and
verifies the package is really there.

The **job** definition is the one place that still needs a literal path, because
a job task has no calling notebook to derive from. Find yours:

```python
print(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())
```

then pass it at deploy time:

```bash
databricks bundle deploy --target dev --var notebook_path=/Workspace/Users/<you>/forgeML/notebooks
```

## 3. Catalog and schemas

Notebook `01` creates these, but you can create them by hand:

```sql
CREATE SCHEMA IF NOT EXISTS workspace.forgeml_bronze;
CREATE SCHEMA IF NOT EXISTS workspace.forgeml_silver;
CREATE SCHEMA IF NOT EXISTS workspace.forgeml_gold;
```

Free Edition exposes a catalog named `workspace`. If yours differs, pass
`--var catalog=<yours>` or change the `catalog` widget — it is the only place
the name appears.

## 4. Run it by hand first

Before deploying a job, run the notebooks in order and watch each one. A job
that fails on task 4 of 7 is far more annoying to debug than a notebook that
fails in a cell you are already looking at.

```
01_data_preparation   ->  02_baseline  ->  03_train (x4)  ->  04_evaluation  ->  05_model_selection
```

## 5. Deploy the job

Install the CLI, authenticate, deploy:

```bash
databricks auth login --host https://<your-workspace>.cloud.databricks.com
```

```bash
databricks bundle deploy --target dev
```

```bash
databricks bundle run forgeml_training_pipeline
```

The bundle is in [`databricks.yml`](databricks.yml). The `dev` target prefixes
everything with `[dev <you>]` and pauses schedules, so deploying cannot
accidentally start a recurring GPU job.

---

## Cost control

This matters more than it sounds on a free tier.

- **Auto-terminate.** Job clusters terminate when the job ends; interactive
  clusters do not. Set an idle timeout on any cluster you attach by hand.
- **Start small.** `max_examples: 5000` and one epoch. Prove the pipeline, then
  scale the data — not the other way round.
- **Run the smoke config first.** `configs/smoke.yaml` completes in minutes on a
  tiny random-init model and catches every wiring bug for free.
- **Keep schedules paused** until you actually want unattended retraining.
- **Watch `max_concurrent_runs`.** It is 1 here on purpose.

## GPU availability

If GPU node types are unavailable in your workspace:

- run `configs/lora.yaml` on CPU — slow but it completes on a small model;
- skip the QLoRA arms, since bitsandbytes has no CPU kernels (notebook `03`
  asserts this up front rather than failing forty minutes in);
- or train elsewhere (Colab, a rented GPU) and point `MLFLOW_TRACKING_URI` at
  the Databricks workspace so tracking and the registry still live here.

That last option is worth knowing: the tracking store does not care where the
GPU is.

## Model serving

Free Edition has serving limitations and no GPU endpoints. The registry half is
already automated — `05_model_selection` registers the champion and moves the
`@champion` alias. Creating an endpoint on top of a registered model is a
workspace operation:

Serving → Create endpoint → select `workspace.forgeml_gold.forgeml_champion`
→ choose the version aliased `@champion`.

Treat this as the architecture you can describe and demonstrate, with the
Hugging Face Space as the URL you actually hand people.
