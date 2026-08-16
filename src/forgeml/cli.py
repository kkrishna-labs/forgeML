"""``forgeml`` command line interface.

Every stage of the pipeline is reachable from here, which means the Databricks
notebooks stay thin — they call the same commands a developer runs locally. When
a notebook and a script disagree about how training works, the notebook always
wins and nobody notices; keeping one entry point removes that failure mode.

::

    forgeml data prepare --config configs/base.yaml
    forgeml train        --config configs/qlora.yaml
    forgeml evaluate     --config configs/qlora.yaml --model outputs/...
    forgeml select       --experiment forgeml
    forgeml register     --experiment forgeml
    forgeml serve        --stub
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

from forgeml import __version__
from forgeml.logging_utils import configure_logging, get_logger

app = typer.Typer(
    name="forgeml",
    help="LLM fine-tuning, evaluation and optimization platform.",
    no_args_is_help=True,
    add_completion=False,
)
data_app = typer.Typer(help="Dataset preparation.", no_args_is_help=True)
app.add_typer(data_app, name="data")

log = get_logger(__name__)

ConfigOption = typer.Option("configs/base.yaml", "--config", "-c", help="Path to a YAML config.")


def _parse_overrides(pairs: list[str] | None) -> dict[str, Any]:
    """``--set training.epochs=3`` -> ``{"training.epochs": 3}``.

    Values are parsed as YAML so ``3``, ``3.0``, ``true`` and ``null`` arrive with
    the right type instead of as strings.
    """
    import yaml

    overrides: dict[str, Any] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise typer.BadParameter(f"--set expects key=value, got {pair!r}")
        key, raw = pair.split("=", 1)
        overrides[key.strip()] = yaml.safe_load(raw)
    return overrides


@app.callback()
def main(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    configure_logging("DEBUG" if verbose else "INFO")


@app.command()
def version() -> None:
    """Print the installed version."""
    typer.echo(f"forgeml {__version__}")


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------


@data_app.command("prepare")
def data_prepare(
    config_path: str = ConfigOption,
    output_dir: str | None = typer.Option(None, "--output", "-o"),
    set_: list[str] | None = typer.Option(None, "--set", help="key=value override."),
) -> None:
    """Load, validate, split and fingerprint the dataset."""
    from forgeml.config import load_config
    from forgeml.data import (
        fingerprint_records,
        load_raw_records,
        split_records,
        validate_records,
        write_jsonl,
    )

    config = load_config(config_path, _parse_overrides(set_))
    target = Path(output_dir or config.data.local_dir)

    records = load_raw_records(config.data)
    clean, report = validate_records(records, config.data)
    typer.echo(report.summary())

    splits = split_records(clean, config.data)
    fingerprint = fingerprint_records(clean, config.data, splits)

    for name, split in splits.as_dict().items():
        write_jsonl(split, target / f"{name}.jsonl")
    fingerprint.save(target / "fingerprint.json")
    (target / "validation_report.json").write_text(
        json.dumps(report.to_dict(), indent=2), encoding="utf-8"
    )

    typer.echo(f"\nwrote {splits.total} records to {target}")
    typer.echo(f"dataset hash: {fingerprint.content_hash[:16]}")


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------


@app.command()
def train(
    config_path: str = ConfigOption,
    data_dir: str | None = typer.Option(None, "--data", "-d"),
    no_mlflow: bool = typer.Option(False, "--no-mlflow", help="Skip MLflow logging."),
    set_: list[str] | None = typer.Option(None, "--set"),
) -> None:
    """Fine-tune, evaluate and log one run end to end."""
    from forgeml.config import load_config
    from forgeml.data.loader import read_jsonl
    from forgeml.evaluation.evaluator import evaluate_model
    from forgeml.training.trainer import run_training

    config = load_config(config_path, _parse_overrides(set_))
    source = Path(data_dir or config.data.local_dir)
    if not (source / "train.jsonl").exists():
        raise typer.BadParameter(f"no prepared data in {source} — run `forgeml data prepare` first")

    train_records = read_jsonl(source / "train.jsonl")
    val_records = read_jsonl(source / "validation.jsonl")
    test_records = read_jsonl(source / "test.jsonl")

    if no_mlflow:
        model, tokenizer, result = run_training(
            config, train_records, val_records, log_to_mlflow=False
        )
        report = evaluate_model(model, tokenizer, test_records, config)
        typer.echo(report.summary())
        return

    from forgeml.tracking.mlflow_utils import (
        log_evaluation_report,
        log_metrics,
        start_run,
    )

    with start_run(config):
        model, tokenizer, result = run_training(config, train_records, val_records)
        log_metrics(result.as_metrics())
        report = evaluate_model(model, tokenizer, test_records, config)
        log_evaluation_report(report)
        _log_model(config, model, tokenizer, result)
        typer.echo(report.summary())


def _log_model(config: Any, model: Any, tokenizer: Any, result: Any) -> None:
    """Attach the trained model to the active MLflow run.

    Logged as a transformers flavor so the registry can serve it without our code
    being importable at load time — a model that only loads inside its own repo is
    not really registered.
    """
    if not config.tracking.log_model:
        return

    import mlflow

    try:
        mlflow.transformers.log_model(
            transformers_model={"model": model, "tokenizer": tokenizer},
            artifact_path="model",
            task="text-generation",
        )
        log.info("logged model artifact to the active run")
    except Exception as exc:  # noqa: BLE001
        # Falling back to the raw checkpoint keeps the run usable even when the
        # transformers flavor cannot serialise a PEFT-wrapped model.
        log.warning("mlflow.transformers.log_model failed (%s); logging raw files", exc)
        try:
            mlflow.log_artifacts(result.output_dir, artifact_path="model")
        except Exception as inner:  # noqa: BLE001
            log.error("could not log model artifacts at all: %s", inner)


# ---------------------------------------------------------------------------
# evaluate / select / register
# ---------------------------------------------------------------------------


@app.command()
def evaluate(
    config_path: str = ConfigOption,
    model_path: str = typer.Option(..., "--model", "-m", help="Checkpoint dir or model URI."),
    data_dir: str | None = typer.Option(None, "--data", "-d"),
    output: str | None = typer.Option(None, "--output", "-o"),
) -> None:
    """Evaluate an existing checkpoint without retraining it."""
    from forgeml.config import load_config
    from forgeml.data.loader import read_jsonl
    from forgeml.evaluation.evaluator import evaluate_model
    from forgeml.inference.predictor import Predictor

    config = load_config(config_path)
    source = Path(data_dir or config.data.local_dir)
    test_records = read_jsonl(source / "test.jsonl")

    predictor = Predictor(model_uri=model_path)
    report = evaluate_model(
        predictor.model, predictor.tokenizer, test_records, config, run_name=model_path
    )
    typer.echo(report.summary())

    if output:
        typer.echo(f"wrote {report.save(output)}")


@app.command()
def select(
    config_path: str = ConfigOption,
    experiment: str | None = typer.Option(None, "--experiment", "-e"),
    output: str | None = typer.Option(None, "--output", "-o"),
    set_: list[str] | None = typer.Option(None, "--set"),
) -> None:
    """Rank the runs in an experiment and print the champion decision."""
    from forgeml.config import load_config
    from forgeml.optimization.selector import select_champion
    from forgeml.tracking.mlflow_utils import candidates_from_experiment, setup_mlflow

    config = load_config(config_path, _parse_overrides(set_))
    experiment_name = experiment or setup_mlflow(config)

    candidates = candidates_from_experiment(experiment_name)
    if not candidates:
        typer.secho(f"no candidate runs in {experiment_name}", fg=typer.colors.RED)
        raise typer.Exit(1)

    result = select_champion(candidates, config.selection)
    typer.echo("\n" + result.to_markdown())

    if output:
        typer.echo(f"\nwrote {result.save(output)}")
    if result.champion is None:
        raise typer.Exit(2)


@app.command()
def register(
    config_path: str = ConfigOption,
    experiment: str | None = typer.Option(None, "--experiment", "-e"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Select a champion and promote it in the model registry."""
    from forgeml.config import load_config
    from forgeml.optimization.selector import select_champion
    from forgeml.registry.model_registry import register_champion
    from forgeml.tracking.mlflow_utils import candidates_from_experiment, setup_mlflow

    config = load_config(config_path)
    experiment_name = experiment or setup_mlflow(config)

    candidates = candidates_from_experiment(experiment_name)
    result = select_champion(candidates, config.selection)
    typer.echo(result.to_markdown())

    if result.champion is None:
        typer.secho(
            "\nno champion cleared the constraints — nothing registered", fg=typer.colors.YELLOW
        )
        raise typer.Exit(2)

    registered = register_champion(result, config.tracking.registered_model_name, dry_run=dry_run)
    if registered:
        typer.secho(f"\nregistered {registered.uri}", fg=typer.colors.GREEN)
        typer.echo(f"pinned:     {registered.pinned_uri}")


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port", "-p"),
    model_uri: str | None = typer.Option(None, "--model", "-m"),
    stub: bool = typer.Option(False, "--stub", help="Serve canned responses, no model."),
    reload: bool = typer.Option(False, "--reload"),
) -> None:
    """Run the FastAPI inference server."""
    import os

    import uvicorn

    if stub:
        os.environ["FORGEML_STUB"] = "1"
    if model_uri:
        os.environ["FORGEML_MODEL_URI"] = model_uri

    typer.echo(f"http://{host}:{port}/docs")
    uvicorn.run("api.main:app", host=host, port=port, reload=reload)


if __name__ == "__main__":  # pragma: no cover
    app()
