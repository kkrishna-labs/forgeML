# Deployment

Two targets, on purpose.

| | Databricks Model Serving | Hugging Face Space |
|---|---|---|
| Role | the production architecture | the public demo |
| Runs | inside the workspace | free, public URL |
| Model source | Unity Catalog registry | checkpoint pulled at boot |
| Cost | consumes workspace compute | free tier |
| GPU | not available on Free Edition | not on the free CPU tier |

The pipeline is designed so neither is load-bearing for the other. Training,
tracking and the registry live on Databricks; the public URL that a recruiter can
actually click lives on Hugging Face. If the Databricks workspace is paused, the
demo keeps working.

---

## Local

```bash
docker build -f deployment/Dockerfile -t forgeml:latest .
```

```bash
docker run --rm -p 8000:8000 -e FORGEML_STUB=1 forgeml:latest
```

Then open <http://localhost:8000/docs>.

To serve real weights, mount a checkpoint and point the loader at it:

```bash
docker run --rm -p 8000:8000 -v "$PWD/outputs:/models:ro" -e FORGEML_MODEL_URI=/models/champion forgeml:latest
```

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `FORGEML_MODEL_URI` | unset | `models:/name@champion`, a run URI, or a local path. Unset means stub mode. |
| `FORGEML_BASE_MODEL` | unset | Base checkpoint for a bare adapter directory. |
| `FORGEML_STUB` | unset | `1` forces canned responses with no model. |
| `FORGEML_LOAD_4BIT` | unset | `1` loads the base in 4-bit (needs CUDA). |
| `FORGEML_LAZY_LOAD` | `0` | `1` defers loading to the first request. |
| `FORGEML_WARMUP` | `1` | Run one throwaway generation at startup. |
| `PORT` | `8000` | Listen port. |

## Hugging Face Space

A Space is a git repository, so deployment is a push. Use a **Docker** Space,
add `deployment/Dockerfile` as the Space's `Dockerfile`, and set the port to
`7860` (Spaces' convention) either in the `README.md` front-matter or by
overriding `PORT`.

```yaml
---
title: ForgeML
emoji: 🔨
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---
```

Two things that will bite you on the free CPU tier:

* **A Space sleeps when idle.** The first request after a sleep pays the full
  cold start. `FORGEML_LAZY_LOAD=1` makes the container answer `/health`
  immediately so the platform does not kill it mid-load.
* **Disk and RAM are limited.** Keep the served model small. A merged 0.5B model
  in fp32 is roughly 2 GB; bf16 halves that, and it is the reason the base model
  in `configs/base.yaml` is deliberately small.

## Databricks Model Serving

```
MLflow run -> registered model -> @champion alias -> serving endpoint -> REST
```

The registry half of this is already automated by `forgeml register`. Creating
the endpoint itself is a workspace operation — see `docs/databricks.md`. Free
Edition has serving limitations and no GPU endpoints, so treat this path as the
architecture you can *describe and demonstrate*, with the Space as the URL you
hand people.
