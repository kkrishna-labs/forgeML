# Experiments

## Protocol

Every arm is trained and measured identically. Only the variable under test
changes. That sounds obvious and is the single easiest thing to get wrong.

| Held constant | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-0.5B-Instruct` |
| Dataset | `databricks/databricks-dolly-15k`, version `v1.0` |
| Splits | 80 / 10 / 10, assigned by content hash |
| Max sequence length | 1024 tokens |
| Effective batch size | 16 |
| Prompt format | tokenizer chat template, identical in train and eval |
| Loss | completion-only (prompt tokens masked to `-100`) |
| Decoding | greedy, `max_new_tokens=256` |
| Eval examples | 200 from the test split |
| Seed | 42 |

| Varied | Across |
|---|---|
| Method | none (baseline), full FT, LoRA, QLoRA |
| LoRA rank | 8, 16 |
| Target modules | attention only, attention + MLP |
| Quantization | none, 4-bit NF4 |

Learning rate is **not** held constant, and that is deliberate: full FT uses
`2e-5` and the PEFT arms use `2e-4`. Training freshly-initialised adapters wants
a rate an order of magnitude higher than nudging pretrained weights. Holding it
constant would be "fair" in a way that makes the comparison meaningless.

## Metrics

### Quality

| Metric | What it captures | Where it lies to you |
|---|---|---|
| ROUGE-L | longest common subsequence — rewards saying the right things in the right order | a correct paraphrase scores badly |
| Token F1 | bag-of-tokens overlap | order-blind; word salad can score well |
| Exact match | strict string equality after normalisation | near-zero on open-ended generation |
| Perplexity | how surprised the model was by the true continuation | comparable only within one tokenizer |
| Length ratio | diagnostic, not a score | explains a bad ROUGE faster than ROUGE does |

Three metrics rather than one because each fails differently. When ROUGE and
perplexity disagree, that disagreement is information.

### Latency

Measured over `generate()` with a fixed prompt length and a fixed number of new
tokens, `min_new_tokens == max_new_tokens` so an early EOS cannot make a model
look fast by refusing to answer. Three warmup runs discarded, twenty measured,
`torch.cuda.synchronize()` before stopping the clock.

**p95, not mean.** A mean hides the tail, and capacity is planned on the tail.

### Memory

Three numbers, because conflating them is the usual mistake:

- **weights** — parameters x bytes each; static and predictable;
- **peak allocated** — weights + activations + KV cache; decides which GPU you need;
- **peak reserved** — what the caching allocator took from the driver; this is
  what `nvidia-smi` shows, which is why `nvidia-smi` never agrees with your logs.

Selection uses peak allocated.

### Cost

```
cost per 1k requests = (1000 / requests_per_second) / 3600 * gpu_hourly_usd
```

An estimate from measured throughput and a stated rate, not a vendor quote. It
ignores cold starts, idle time and autoscaling headroom, so it ranks models
against each other and should never be used as a budget.

## Results

<!-- Paste the output of SelectionResult.to_markdown() here. -->

| Run | Method | Quality | Latency p95 | Memory | Size | Utility | Status |
|---|---|---:|---:|---:|---:|---:|---|
| | | | | | | | |

### Reading the table

Three things to look for:

1. **Does fine-tuning beat the baseline at all?** If not, stop and check the
   prompt format before touching hyperparameters. A format mismatch between
   training and evaluation looks exactly like a failed fine-tune.
2. **Where does quality flatten?** Usually between rank 8 and 16. Past that you
   are buying VRAM, not accuracy.
3. **What does 4-bit actually cost you?** The interesting comparison is
   `lora-r8` against `qlora-r8` — same rank, same data, same everything, one
   quantized base. Typically ~4x less base-weight memory for a small quality
   drop, and often *slower* per training step because every forward pass
   dequantizes on the fly.

That last one is worth internalising: **QLoRA saves memory, not time.**

## Interpreting a null result

If the champion is only marginally better than the baseline, that is a finding,
not a failure. Small instruction datasets on already-instruction-tuned
checkpoints often produce small gains — the base model has already seen this
kind of data.

The honest write-up says so, and the pipeline supports saying so: the selector
returns no champion when nothing clears the bar, and registration refuses to
promote. A pipeline that always ships something will eventually ship a
regression.

## Reproducing

```bash
forgeml data prepare --config configs/base.yaml
```

```bash
for cfg in baseline lora lora_r16 qlora qlora_r16; do forgeml train --config configs/$cfg.yaml; done
```

```bash
forgeml select --output reports/selection.json
```

Check that your dataset hash matches the one recorded in the runs. If it does
not, you are not reproducing the experiment — you are running a new one.
