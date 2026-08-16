# Architecture

## The shape of the problem

Fine-tuning an LLM is not an optimization problem with one objective. A model
that scores two points higher and runs three times slower is worse for almost
every real deployment, and the only way to know that is to measure all of it.

ForgeML is built around that: **train several arms identically, measure four
axes identically, then apply an explicit policy.**

## Flow

```
                        dataset (Hugging Face / JSONL / Delta)
                                     |
                          normalise to one record shape
                                     |
                     validate  ->  dedupe  ->  hash-bucket split
                                     |
                          content fingerprint (sha256)
                                     |
                    +----------------+----------------+
                    |                |                |
                  bronze          silver            gold
                (raw delta)   (clean delta)   (train/val/test)
                                     |
      +---------------+--------------+--------------+---------------+
      |               |              |              |               |
  baseline        full FT          LoRA           LoRA r16      QLoRA r8/r16
  (no training)                                                (4-bit NF4)
      |               |              |              |               |
      +---------------+--------------+--------------+---------------+
                                     |
                               one evaluator
                    quality | latency | memory | size | cost
                                     |
                                  MLflow
                     params · metrics · artifacts · model
                                     |
                              selection policy
                 constraints -> normalise -> weighted utility
                                     |
                        +------------+------------+
                        |                         |
                    champion                 challenger
                        |
                model registry  @champion
                        |
              +---------+---------+
              |                   |
     Databricks Serving    Hugging Face Space
      (architecture)          (public URL)
```

## Layering

The package is split along one line: **does it need a GPU?**

| Layer | Torch? | Why it matters |
|---|---|---|
| `forgeml.data` | no | data bugs are cheapest to catch and most expensive to discover on a GPU |
| `forgeml.evaluation.quality` | no | metrics you can read and unit-test |
| `forgeml.optimization` | no | the deployment decision is re-runnable without a GPU |
| `forgeml.tracking` / `registry` | no | reads runs back out of MLflow |
| `forgeml.training` | yes | the only genuinely GPU-bound part |
| `forgeml.inference` | optional | stub predictor keeps the API testable |

CI asserts this: it imports the decision layer and fails if `torch` appears in
`sys.modules`. The guarantee is enforced, not documented.

## Design decisions

### One training path for all three methods

`run_training()` does not know which arm it is running. Full FT, LoRA and QLoRA
differ only in how `model_factory` constructs the model — same data, same
collator, same trainer, same evaluator.

If each method had its own script, a fix to the prompt template would land in
one and not the others, and the comparison table would silently start measuring
script divergence. The single path is what makes the results table a statement
about *methods*.

### Hash-bucket splitting instead of shuffling

An example's split is `sha256(version + content) % 10000`. Adding 10k rows to a
5k dataset leaves all 5k original examples in their original splits.

With `random.shuffle`, growing the dataset reshuffles everything, so the new test
set contains examples the previous model trained on. Two "comparable" runs are
then not comparable at all, and nothing warns you.

### Completion-only loss masking

Prompt tokens are set to `-100`, which cross-entropy ignores. Without this the
model spends most of its gradient budget learning to predict text it is always
given at inference. On small models and small datasets, this is frequently the
difference between a fine-tune that helps and one that does nothing.

### Constraints before preferences

The selector applies hard gates first and only then ranks by weighted utility.
Gates express requirements; weights express preferences, and no amount of
preference should buy past a requirement. A candidate at 60% of baseline quality
is not eligible no matter how fast it is.

### Aliases, not stages

`@champion` is an atomic pointer to a model version. Promotion and rollback are
the same call with different arguments, and serving code never hardcodes a
version. MLflow's `Staging`/`Production` stages are deprecated and Unity Catalog
does not support them.

### The Pareto frontier is reported but does not decide

The frontier says which candidates are *defensible* — anything off it loses on
every axis to something else. The weights say which one ships. Reporting both is
honest: it separates "this is a fact about the models" from "this is our
priority ordering", and only the second is arguable.

## What is deliberately not here

- **Distributed training.** One GPU is the target. Adding DeepSpeed would add
  complexity without changing any conclusion at this model size.
- **Near-duplicate detection.** Exact dedupe removes most leakage in public
  instruction sets; MinHash and embedding-based dedupe need tuning that would
  become its own project.
- **An LLM judge.** It would improve the quality signal and destroy
  reproducibility and cost predictability. The limitations of n-gram metrics are
  stated in the model card instead.
- **Prometheus / OpenTelemetry.** `/metrics` is in-process counters. Real
  observability is a different project.
