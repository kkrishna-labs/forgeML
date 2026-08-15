"""Model registry operations.

The registry is where an experiment stops being an experiment. A run says "this is
what happened"; a registered model version says "this is what we will serve".

**Aliases, not stages.** MLflow's old ``Staging``/``Production`` stages are
deprecated and Unity Catalog does not support them at all. Aliases are strictly
better anyway: an alias is a named pointer (``@champion``, ``@challenger``) that
you move between versions atomically, so a rollback is one call and consumers
never hardcode a version number.

::

    models:/forgeml_champion@champion     # always the current production model
    models:/forgeml_champion/7            # a specific, frozen version

Serving code should reference the alias. Audit trails reference the version.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from forgeml.logging_utils import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from forgeml.optimization.selector import SelectionResult

log = get_logger(__name__)

CHAMPION_ALIAS = "champion"
CHALLENGER_ALIAS = "challenger"


@dataclass
class RegisteredChampion:
    """What a successful promotion produced."""

    name: str
    version: str
    run_id: str
    alias: str = CHAMPION_ALIAS
    uri: str = ""

    def __post_init__(self) -> None:
        if not self.uri:
            self.uri = f"models:/{self.name}@{self.alias}"

    @property
    def pinned_uri(self) -> str:
        """Version-pinned URI — what belongs in an audit log, not in serving code."""
        return f"models:/{self.name}/{self.version}"

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "version": self.version,
            "run_id": self.run_id,
            "alias": self.alias,
            "uri": self.uri,
            "pinned_uri": self.pinned_uri,
        }


def _client() -> Any:
    from mlflow.tracking import MlflowClient

    return MlflowClient()


def is_unity_catalog(model_name: str) -> bool:
    """UC model names are three-level: ``catalog.schema.model``."""
    return model_name.count(".") == 2


def register_model_from_run(
    run_id: str,
    model_name: str,
    artifact_path: str = "model",
    description: str | None = None,
    tags: dict[str, str] | None = None,
) -> str:
    """Register a logged model from a run and return the new version number."""
    import mlflow

    source_uri = f"runs:/{run_id}/{artifact_path}"
    log.info("registering %s as %s", source_uri, model_name)

    version = mlflow.register_model(model_uri=source_uri, name=model_name, tags=tags)

    if description:
        _client().update_model_version(
            name=model_name, version=version.version, description=description
        )

    log.info("registered %s version %s", model_name, version.version)
    return str(version.version)


def promote_alias(model_name: str, version: str, alias: str = CHAMPION_ALIAS) -> None:
    """Point ``alias`` at ``version``.

    Atomic and idempotent: setting an alias that already exists just moves it, so
    promotion and rollback are the same operation with different arguments.
    """
    client = _client()
    client.set_registered_model_alias(name=model_name, alias=alias, version=version)
    log.info("alias @%s -> %s version %s", alias, model_name, version)


def get_champion_uri(model_name: str, alias: str = CHAMPION_ALIAS) -> str | None:
    """Resolve ``models:/name@alias`` if the alias exists, else None."""
    try:
        version = _client().get_model_version_by_alias(name=model_name, alias=alias)
    except Exception as exc:  # noqa: BLE001 - alias or model may simply not exist yet
        log.warning("no @%s alias on %s: %s", alias, model_name, exc)
        return None
    log.info("%s@%s resolves to version %s", model_name, alias, version.version)
    return f"models:/{model_name}@{alias}"


def register_champion(
    selection: SelectionResult,
    model_name: str,
    artifact_path: str = "model",
    dry_run: bool = False,
) -> RegisteredChampion | None:
    """Register and promote the selected champion — the end of the pipeline.

    Refuses to promote when the selector found no eligible candidate. That is the
    correct behaviour: a pipeline that always ships something will eventually ship
    a regression, and "no candidate cleared the bar" is a valid, useful outcome.
    """
    if selection.champion is None:
        log.error("selection produced no champion — nothing to register")
        return None

    champion = selection.champion
    if not champion.run_id:
        log.error("champion %s has no run_id — cannot register", champion.run_name)
        return None

    description = _build_description(selection)
    tags = {
        "method": champion.method,
        "utility": f"{champion.utility:.4f}",
        "quality": f"{champion.quality:.4f}",
        "latency_p95_ms": f"{champion.latency_ms:.1f}",
        "memory_mb": f"{champion.memory_mb:.1f}",
        "selected_by": "forgeml.optimization.selector",
    }

    if dry_run:
        log.info("[dry run] would register %s as %s", champion.run_name, model_name)
        log.info("[dry run] description:\n%s", description)
        return None

    version = register_model_from_run(
        run_id=champion.run_id,
        model_name=model_name,
        artifact_path=artifact_path,
        description=description,
        tags=tags,
    )
    promote_alias(model_name, version, CHAMPION_ALIAS)

    # The runner-up becomes the challenger, which is what makes A/B comparison or
    # a one-call rollback possible later.
    if selection.challenger and selection.challenger.run_id:
        try:
            challenger_version = register_model_from_run(
                run_id=selection.challenger.run_id,
                model_name=model_name,
                artifact_path=artifact_path,
                description=f"Challenger: {selection.challenger.run_name}",
            )
            promote_alias(model_name, challenger_version, CHALLENGER_ALIAS)
        except Exception as exc:  # noqa: BLE001 - challenger is a nice-to-have
            log.warning("could not register challenger: %s", exc)

    return RegisteredChampion(
        name=model_name,
        version=version,
        run_id=champion.run_id,
        alias=CHAMPION_ALIAS,
    )


def _build_description(selection: SelectionResult) -> str:
    """Human-readable justification, stored on the model version itself.

    Six months later, "why is this the production model?" is answerable from the
    registry UI alone, without archaeology through notebooks.
    """
    champion = selection.champion
    assert champion is not None

    lines = [
        f"Selected by ForgeML on utility {champion.utility:.4f}.",
        "",
        f"Method            : {champion.method}",
        f"Quality           : {champion.quality:.4f}",
        f"Latency (p95)     : {champion.latency_ms:.1f} ms",
        f"Peak memory       : {champion.memory_mb:.1f} MB",
        f"Model size        : {champion.model_size_mb:.1f} MB",
        f"Cost / 1k requests: ${champion.cost_per_1k_usd:.4f}",
        "",
        f"Weights: {selection.weights}",
    ]

    gain = selection.quality_gain_vs_baseline()
    speedup = selection.speedup_vs_baseline()
    if gain is not None:
        lines.append(f"Quality vs baseline: {gain:+.4f}")
    if speedup is not None:
        lines.append(f"Latency vs baseline: {speedup:.2f}x")
    if selection.challenger:
        lines.append(f"Runner-up: {selection.challenger.run_name}")
    if selection.rejected:
        lines.append("")
        lines.append("Rejected candidates:")
        for candidate in selection.rejected:
            if candidate.is_baseline:
                continue
            lines.append(f"  - {candidate.run_name}: {'; '.join(candidate.rejected_reasons)}")

    return "\n".join(lines)


def rollback(model_name: str, to_version: str) -> None:
    """Move @champion back to a known-good version. The whole point of aliases."""
    promote_alias(model_name, to_version, CHAMPION_ALIAS)
    log.warning("rolled back %s@%s to version %s", model_name, CHAMPION_ALIAS, to_version)
