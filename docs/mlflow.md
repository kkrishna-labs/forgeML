# MLflow in ForgeML

## The mental model

Five concepts, and the distinctions between them are the whole thing:

| Concept | Is | Mutable? | Use for |
|---|---|---|---|
| **Experiment** | a folder of runs | — | one per project question |
| **Run** | one attempt | frozen when it ends | one per training arm |
| **Param** | an input you *chose* | no | learning rate, rank, dataset version |
| **Metric** | an output you *measured* | append-only time series | loss, ROUGE, latency |
| **Tag** | a label | yes | `arm=qlora`, git commit, GPU model |

The rule that resolves almost every "where does this go?" question:

> **You chose it → param. You measured it → metric. You might want to
> recategorise it later → tag.**

Params are immutable, so a run cannot be quietly rewritten. Metrics accept a
`step`, which is how a loss curve is stored. Tags are mutable, which is why
"which arm is this" belongs in a tag rather than a param.

## What ForgeML logs

### Params — the full resolved config

`ForgeConfig.flat_params()` flattens the entire config to `section.key` pairs.
That is ~100 params per run, and that is correct: any of them could explain a
result, and MLflow's UI lets you diff two runs' params directly.

```
model.name                    Qwen/Qwen2.5-0.5B-Instruct
training.method               qlora
training.learning_rate        0.0002
training.effective_batch_size 16
lora.r                        8
lora.alpha                    16
lora.scaling                  2.0
quantization.quant_type       nf4
dataset.version               v1.0
dataset.content_hash          a3f8b21c9e4d7f06
```

`lora.scaling` is derived, not configured. Logging derived values as first-class
params is worth doing whenever the derivation is the thing you actually reason
about.

### Metrics — namespaced by axis

```
train/loss                    per step
train/runtime_s
train/peak_gpu_memory_mb
eval/loss
quality/rouge_l
quality/token_f1
quality/perplexity
latency/p95_ms
latency/tokens_per_second
memory/peak_allocated_mb
memory/activation_overhead_mb
model/size_mb
model/params_trainable_pct
cost/per_1k_requests_usd
```

The `axis/name` convention groups them in the UI and, more importantly, gives
the selector stable names to query. Transformers' built-in MLflow integration is
disabled (`report_to: []`) precisely so these names are ours and cannot change
under a library upgrade.

### Tags — provenance

```
method              qlora
arm                 qlora
base_model          Qwen/Qwen2.5-0.5B-Instruct
dataset_version     v1.0
env.git_commit      a1b2c3d-dirty
env.torch           2.4.0
env.gpu_name        Tesla T4
env.cuda            12.1
```

`env.git_commit` carries a `-dirty` suffix when the working tree had uncommitted
changes. A dirty run is not reproducible and the tag says so rather than
pretending otherwise.

### Artifacts — the things a number cannot carry

```
config/resolved_config.yaml     the fully merged config, not the override
data/dataset_fingerprint.json   content hash + per-split hashes + length stats
evaluation/evaluation_report.json
evaluation/samples.md           the 10 worst and 5 best generations
model/                          the model itself
```

`samples.md` is the most useful artifact in the run. An aggregate score tells you
*that* a model is weak; ten concrete failures tell you *how*, and that is what
decides the next experiment.

## Setup

### Locally

Nothing to configure. MLflow writes to `./mlruns`:

```bash
mlflow ui
```

Then <http://localhost:5000>.

### On Databricks

The tracking URI is already correct inside a notebook — **do not override it**.
Setting it manually is the fastest route to runs written to the driver's local
disk and lost when the cluster terminates.

```python
config = load_config("configs/lora.yaml", {"tracking.experiment_name": "/Shared/forgeml"})
```

### From a laptop into a Databricks workspace

Useful when your GPU is elsewhere (Colab, a rented box) but you want tracking and
the registry to stay in the workspace:

```bash
export DATABRICKS_HOST=https://your-workspace.cloud.databricks.com
```

```bash
export MLFLOW_TRACKING_URI=databricks
```

The tracking store does not care where the GPU is.

## Reading runs back out

This is the part most tutorials skip, and it is what makes the pipeline
re-runnable:

```python
from forgeml.tracking.mlflow_utils import candidates_from_experiment
from forgeml.optimization.selector import select_champion

candidates = candidates_from_experiment("/Shared/forgeml")
result = select_champion(candidates, config.selection)
```

Selection reads from the tracking store, not from objects held in memory by
whatever notebook trained the models. Change the weights, re-run, get a new
decision — no GPU, no retraining.

## The registry

```
models:/forgeml_champion@champion     # what serving points at
models:/forgeml_champion/7            # what an audit log records
```

**Aliases, not stages.** `Staging`/`Production` are deprecated and Unity Catalog
does not support them. An alias is an atomic pointer:

```python
promote_alias("forgeml_champion", version="8", alias="champion")   # deploy
promote_alias("forgeml_champion", version="7", alias="champion")   # roll back
```

Same call. That symmetry is why rollback is not a special procedure.

Unity Catalog model names are three-level — `catalog.schema.model` — and require
`mlflow.set_registry_uri("databricks-uc")` before any registry call.

## Practical notes

**Log the model inside the run that trained it.** A model artifact in a
different run than its metrics cannot be ranked later, because the selector has
nothing to join on.

**Never let logging kill a run.** Every logging call in
`forgeml.tracking.mlflow_utils` is wrapped and degrades to a warning. Losing an
hour of GPU work because an artifact upload timed out is an unforced error.

**Non-finite metrics are dropped, loudly.** `NaN` and `inf` are legal floats that
poison the UI's charts and any downstream `max()`. Better a warning than a
silently corrupted comparison.

**Params are capped at 500 characters.** `flat_params()` truncates rather than
raising, so a long `target_modules` list cannot fail an otherwise good run.

## Common problems

| Symptom | Cause |
|---|---|
| Runs vanish after the cluster stops | tracking URI was overridden to a local path |
| `INVALID_PARAMETER_VALUE: Param value too long` | a param over 500 chars — use `flat_params()` |
| Cannot set an alias | registry URI is not `databricks-uc`, or the name is not three-level |
| Metrics missing from the comparison | the run did not finish; `candidates_from_experiment` filters on `status = 'FINISHED'` |
| Two runs look identical but scored differently | check `env.git_commit` — one of them was dirty |
