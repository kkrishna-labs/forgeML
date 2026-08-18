"""MLflow helpers.

The mental model that makes MLflow click:

* **Experiment** — a folder. One per project question. ForgeML uses one.
* **Run** — one attempt. Immutable once finished.
* **Params** — the *inputs* you chose. Strings. Written once. Searchable.
* **Metrics** — the *outputs* you measured. Floats. Can have a value per step,
  which is how a loss curve is stored.
* **Tags** — mutable labels for organising and filtering after the fact.
* **Artifacts** — files. Plots, configs, JSON reports, the model itself.

The distinction that trips everyone up: **params are immutable, metrics are
time-series, tags are mutable.** Log the learning rate as a param (you chose it),
the loss as a metric (you measured it), and "arm=qlora" as a tag (you may want to
recategorise later).

Everything here degrades gracefully when MLflow is not installed or no run is
active, because a logging failure must never take down a training job that has
already burned an hour of GPU.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from forgeml.logging_utils import get_logger
from forgeml.reproducibility import environment_snapshot

if TYPE_CHECKING:  # pragma: no cover
    from forgeml.config import ForgeConfig
    from forgeml.data.versioning import DatasetFingerprint
    from forgeml.evaluation.evaluator import EvaluationReport
    from forgeml.optimization.selector import ModelCandidate

log = get_logger(__name__)


def _mlflow() -> Any:
    import mlflow

    return mlflow


def is_databricks() -> bool:
    """True when running *on* Databricks compute (a notebook or job cluster)."""
    return bool(os.getenv("DATABRICKS_RUNTIME_VERSION"))


def tracking_is_databricks() -> bool:
    """True when the tracking *store* is a Databricks workspace.

    Deliberately different from :func:`is_databricks`, and the difference is the
    whole point: you can train on Colab or a rented GPU while logging into a
    Databricks workspace. Compute location and tracking location are
    independent, and conflating them mangles the experiment path — Databricks
    requires an absolute workspace path like ``/Shared/forgeml``, so rewriting
    it to ``Shared-forgeml`` because the GPU happens to be elsewhere gets the
    run rejected outright.
    """
    return str(_mlflow().get_tracking_uri()).startswith("databricks")


def setup_mlflow(config: ForgeConfig) -> str:
    """Point MLflow at the right tracking store, registry and experiment.

    On Databricks the tracking URI is already correct, so we leave it alone —
    overriding it is the fastest way to end up with runs written to the driver's
    local disk and lost when the cluster terminates.
    """
    mlflow = _mlflow()

    if config.tracking.tracking_uri:
        mlflow.set_tracking_uri(config.tracking.tracking_uri)
    elif is_databricks():
        mlflow.set_tracking_uri("databricks")

    if config.tracking.registry_uri:
        mlflow.set_registry_uri(config.tracking.registry_uri)

    experiment_name = config.tracking.experiment_name
    # An absolute workspace path only means something to a Databricks tracking
    # server; against a local ./mlruns store it would create a directory
    # literally named "/Shared/forgeml". Key the decision off the *tracking
    # store*, not off where the code happens to be running.
    if experiment_name.startswith("/") and not tracking_is_databricks():
        experiment_name = experiment_name.strip("/").replace("/", "-")

    mlflow.set_experiment(experiment_name)
    log.info(
        "mlflow tracking=%s experiment=%s",
        mlflow.get_tracking_uri(),
        experiment_name,
    )
    return experiment_name


@contextmanager
def start_run(
    config: ForgeConfig,
    run_name: str | None = None,
    nested: bool = False,
    extra_tags: dict[str, str] | None = None,
) -> Iterator[Any]:
    """Start a run with config, environment and tags already recorded.

    Using this instead of bare ``mlflow.start_run()`` guarantees that no run in the
    experiment is ever missing the metadata the selector and the model card need.
    """
    mlflow = _mlflow()
    setup_mlflow(config)

    name = run_name or config.tracking.run_name or config.run_slug()
    tags = {
        "method": config.training.method,
        "base_model": config.model.name,
        "dataset_version": config.data.version,
        "forgeml_version": __import__("forgeml").__version__,
        **config.tracking.tags,
        **(extra_tags or {}),
    }

    with mlflow.start_run(run_name=name, nested=nested, tags=tags) as run:
        log.info("mlflow run %s (%s)", name, run.info.run_id)
        log_forge_config(config)
        _log_environment()
        yield run


def log_forge_config(config: ForgeConfig) -> None:
    """Log the flattened config as params, and the full YAML as an artifact.

    Both, not either: params are searchable and comparable in the UI, while the
    YAML artifact is the thing you can actually re-run six months from now.
    """
    mlflow = _mlflow()
    try:
        mlflow.log_params(config.flat_params())
    except Exception as exc:  # noqa: BLE001
        log.warning("failed to log params: %s", exc)

    from forgeml.config import dump_config

    with tempfile.TemporaryDirectory() as tmp:
        path = dump_config(config, Path(tmp) / "resolved_config.yaml")
        _log_artifact(path, "config")


def _log_environment() -> None:
    snapshot = environment_snapshot()
    mlflow = _mlflow()
    try:
        mlflow.set_tags({f"env.{k}": v for k, v in snapshot.items()})
    except Exception as exc:  # noqa: BLE001
        log.warning("failed to log environment tags: %s", exc)


def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
    """Log a metric dict, dropping non-finite values with a warning.

    NaN and inf are legal floats but they poison the MLflow UI's charts and any
    downstream ``max()``. Better to drop them loudly than to store them quietly.
    """
    mlflow = _mlflow()
    import math

    clean: dict[str, float] = {}
    for key, value in metrics.items():
        numeric = float(value)
        if math.isnan(numeric) or math.isinf(numeric):
            log.warning("dropping non-finite metric %s=%s", key, value)
            continue
        clean[key] = numeric

    if clean:
        try:
            mlflow.log_metrics(clean, step=step)
        except Exception as exc:  # noqa: BLE001
            log.warning("failed to log metrics: %s", exc)


def log_json_artifact(payload: Any, filename: str, artifact_path: str | None = None) -> None:
    """Write a dict/list as a JSON artifact."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / filename
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        _log_artifact(path, artifact_path)


def log_text_artifact(text: str, filename: str, artifact_path: str | None = None) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / filename
        path.write_text(text, encoding="utf-8")
        _log_artifact(path, artifact_path)


def _log_artifact(path: Path, artifact_path: str | None = None) -> None:
    mlflow = _mlflow()
    try:
        mlflow.log_artifact(str(path), artifact_path=artifact_path)
    except Exception as exc:  # noqa: BLE001
        log.warning("failed to log artifact %s: %s", path.name, exc)


def log_dataset_fingerprint(fingerprint: DatasetFingerprint) -> None:
    """Record dataset identity as params (searchable) plus a JSON artifact."""
    mlflow = _mlflow()
    try:
        mlflow.log_params(fingerprint.mlflow_params())
    except Exception as exc:  # noqa: BLE001
        log.warning("failed to log dataset params: %s", exc)
    log_json_artifact(fingerprint.to_dict(), "dataset_fingerprint.json", "data")


def log_evaluation_report(report: EvaluationReport) -> None:
    """Log every metric plus the best/worst generation samples."""
    log_metrics(report.as_metrics())
    log_json_artifact(report.to_dict(), "evaluation_report.json", "evaluation")
    if report.samples:
        log_text_artifact(_samples_to_markdown(report.samples), "samples.md", "evaluation")


def _samples_to_markdown(samples: Sequence[dict[str, str]]) -> str:
    lines = ["# Generation samples", ""]
    for sample in samples:
        lines += [
            f"## [{sample['bucket']}] score = {sample['score']}",
            "",
            "**Instruction**",
            "",
            f"> {sample['instruction']}",
            "",
            "**Reference**",
            "",
            f"> {sample['reference']}",
            "",
            "**Prediction**",
            "",
            f"> {sample['prediction']}",
            "",
            "---",
            "",
        ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Reading runs back out
# ---------------------------------------------------------------------------


def candidates_from_experiment(
    experiment_name: str,
    primary_metric: str = "quality/rouge_l",
    latency_metric: str = "latency/p95_ms",
    memory_metric: str = "memory/peak_allocated_mb",
    size_metric: str = "model/size_mb",
    cost_metric: str = "cost/per_1k_requests_usd",
    filter_string: str = "attributes.status = 'FINISHED'",
    max_results: int = 200,
) -> list[ModelCandidate]:
    """Read finished runs out of MLflow and adapt them into selector candidates.

    This is the seam that makes selection re-runnable: the champion decision reads
    from the tracking store, not from objects held in memory by whatever notebook
    happened to train the models. Change the weights, re-run this, get a new
    decision — no GPU required.
    """
    mlflow = _mlflow()
    from forgeml.optimization.selector import ModelCandidate

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        log.warning("experiment %s does not exist", experiment_name)
        return []

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=filter_string,
        max_results=max_results,
        output_format="list",
    )

    candidates: list[ModelCandidate] = []
    for run in runs:
        metrics = run.data.metrics
        if primary_metric not in metrics:
            log.debug("skipping run %s: no %s metric", run.info.run_name, primary_metric)
            continue

        tags = run.data.tags
        candidates.append(
            ModelCandidate(
                run_id=run.info.run_id,
                run_name=run.info.run_name or run.info.run_id[:8],
                method=tags.get("method", "unknown"),
                quality=metrics.get(primary_metric, 0.0),
                latency_ms=metrics.get(latency_metric, 0.0),
                memory_mb=metrics.get(memory_metric, 0.0),
                model_size_mb=metrics.get(size_metric, 0.0),
                cost_per_1k_usd=metrics.get(cost_metric, 0.0),
                is_baseline=tags.get("arm") == "baseline",
                metadata={
                    "start_time": str(run.info.start_time),
                    "git_commit": tags.get("env.git_commit", ""),
                    "dataset_version": tags.get("dataset_version", ""),
                },
            )
        )

    log.info("loaded %d candidate run(s) from %s", len(candidates), experiment_name)
    return candidates
