# Databricks notebooks

These are **source-format** notebooks (`.py` with `# COMMAND ----------`
separators), not `.ipynb`. That is deliberate:

- they diff properly in git — an `.ipynb` diff is unreadable JSON with embedded
  base64 outputs;
- they cannot leak outputs into the repo, so no accidental dataset samples or
  tokens in a committed cell;
- Databricks imports and exports this format natively, and Git folders sync it
  bidirectionally.

## Order

| Notebook | Does | Needs GPU |
|---|---|---|
| `01_data_preparation` | load, validate, split, fingerprint, write Delta | no |
| `02_baseline` | evaluate the untuned base model | helps |
| `03_train` | fine-tune one arm, evaluate, log to MLflow | yes for QLoRA |
| `04_evaluation` | read all runs, build the comparison + Pareto frontier | no |
| `05_model_selection` | apply the policy, register the champion | no |

`03_train` is one notebook for every method. Two notebooks that are 95%
identical will drift in the 5%, and then the comparison table measures notebook
divergence instead of method difference. The arms differ only in configuration,
so they differ only in configuration here too.

## Running them

Attach the repo as a **Git folder** (Workspace → Repos → Add repo) so
`/Workspace/Repos/forgeML/src` is importable. Every notebook puts that on
`sys.path` in its second cell; adjust `REPO_ROOT` if you clone it elsewhere.

Then either run them in order by hand, or deploy the job:

```bash
databricks bundle deploy --target dev
```

```bash
databricks bundle run forgeml_training_pipeline
```

See [`../databricks/README.md`](../databricks/README.md).

## Widgets

Every notebook is parameterised, so the job passes different values to the same
code rather than duplicating notebooks:

| Widget | Meaning |
|---|---|
| `config` | path to a YAML config, relative to the repo root |
| `catalog` | Unity Catalog catalog holding the medallion schemas |
| `experiment` | MLflow experiment path |
| `overrides` | `key=value` pairs, comma separated — how sweeps are run |

A rank sweep is therefore five job tasks pointing at one notebook with different
`overrides`, not five notebooks.
