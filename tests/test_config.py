"""Config composition and validation.

The point of a typed config is that bad experiments fail before they reach a GPU,
so these tests assert on the *rejections* as much as the successes.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from tests.conftest import CONFIG_DIR

from forgeml.config import ForgeConfig, LoRAConfig, load_config

ALL_CONFIGS = [
    "base.yaml",
    "baseline.yaml",
    "full_ft.yaml",
    "lora.yaml",
    "lora_r16.yaml",
    "qlora.yaml",
    "qlora_r16.yaml",
    "smoke.yaml",
]


@pytest.mark.parametrize("name", ALL_CONFIGS)
def test_every_shipped_config_is_valid(name: str) -> None:
    """A broken config in the repo is a broken experiment for whoever clones it."""
    config = load_config(CONFIG_DIR / name)
    assert config.project == "forgeml"
    assert config.run_slug()


def test_extends_chain_inherits_and_overrides(lora_config: ForgeConfig) -> None:
    # inherited from base.yaml
    assert lora_config.data.source == "databricks/databricks-dolly-15k"
    # overridden in lora.yaml
    assert lora_config.training.method == "lora"
    assert lora_config.training.learning_rate == pytest.approx(2e-4)


def test_extends_is_transitive() -> None:
    """lora_r16 -> lora -> base. The grandparent's values must survive."""
    config = load_config(CONFIG_DIR / "lora_r16.yaml")
    assert config.lora is not None
    assert config.lora.r == 16  # from lora_r16
    assert config.lora.dropout == pytest.approx(0.05)  # from lora
    assert config.model.max_seq_length == 1024  # from base


def test_dotted_overrides_win() -> None:
    config = load_config(
        CONFIG_DIR / "lora.yaml",
        overrides={"training.epochs": 5, "lora.r": 64, "model.name": "test/model"},
    )
    assert config.training.epochs == 5
    assert config.lora is not None
    assert config.lora.r == 64
    assert config.model.name == "test/model"


def test_env_overrides_are_applied(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FORGEML__training__epochs", "7")
    config = load_config(CONFIG_DIR / "lora.yaml")
    assert config.training.epochs == 7


def test_env_overrides_survive_windows_case_folding(monkeypatch: pytest.MonkeyPatch) -> None:
    """Windows upper-cases os.environ keys; Linux does not.

    Without case folding this override works on a Databricks job cluster and
    silently does nothing on a Windows laptop, which is the worst possible
    failure mode for a config mechanism.
    """
    monkeypatch.setenv("FORGEML__TRAINING__LEARNING_RATE", "0.005")
    config = load_config(CONFIG_DIR / "lora.yaml")
    assert config.training.learning_rate == pytest.approx(0.005)


def test_qlora_without_quantization_is_rejected() -> None:
    with pytest.raises(ValidationError, match=r"quantization\.enabled"):
        ForgeConfig.model_validate(
            {
                "training": {"method": "qlora"},
                "lora": {"r": 8},
                "quantization": {"enabled": False},
            }
        )


def test_lora_method_without_lora_block_is_rejected() -> None:
    with pytest.raises(ValidationError, match="requires a `lora:` section"):
        ForgeConfig.model_validate({"training": {"method": "lora"}})


def test_full_ft_with_lora_block_is_rejected() -> None:
    """Silently ignoring the lora block would produce a mislabelled experiment."""
    with pytest.raises(ValidationError, match="must not define"):
        ForgeConfig.model_validate({"training": {"method": "full"}, "lora": {"r": 8}})


def test_split_ratios_must_sum_to_one() -> None:
    with pytest.raises(ValidationError, match=r"must sum to 1\.0"):
        ForgeConfig.model_validate(
            {"data": {"train_ratio": 0.8, "val_ratio": 0.3, "test_ratio": 0.1}}
        )


def test_both_precision_flags_is_rejected() -> None:
    with pytest.raises(ValidationError, match="exactly one"):
        ForgeConfig.model_validate({"training": {"bf16": True, "fp16": True}})


def test_effective_batch_size_is_the_product() -> None:
    config = ForgeConfig.model_validate(
        {"training": {"per_device_train_batch_size": 8, "gradient_accumulation_steps": 4}}
    )
    assert config.training.effective_batch_size == 32


def test_lora_scaling_is_alpha_over_r() -> None:
    assert LoRAConfig(r=8, alpha=16).scaling == pytest.approx(2.0)
    assert LoRAConfig(r=16, alpha=32).scaling == pytest.approx(2.0)
    assert LoRAConfig(r=64, alpha=16).scaling == pytest.approx(0.25)


def test_run_slug_distinguishes_the_training_arms() -> None:
    """Two arms sharing a slug would be indistinguishable in the MLflow run list.

    base/baseline/full_ft legitimately collide — same model, same method, same
    dataset — and are separated by the `arm` tag instead. The four PEFT arms are
    the ones that must never collide.
    """
    arms = ["lora.yaml", "lora_r16.yaml", "qlora.yaml", "qlora_r16.yaml"]
    slugs = [load_config(CONFIG_DIR / name).run_slug() for name in arms]
    assert len(set(slugs)) == len(arms)
    assert "lora-r8a16" in slugs[0]
    assert "qlora-r16a32-4bit" in slugs[3]


def test_flat_params_are_mlflow_safe(lora_config: ForgeConfig) -> None:
    params = lora_config.flat_params()
    assert params["training.method"] == "lora"
    assert params["lora.r"] == 8
    assert params["training.effective_batch_size"] == 16
    # MLflow rejects param values over 500 characters
    assert all(len(str(v)) <= 500 for v in params.values())
    # lists must be JSON-encoded, not repr'd
    assert params["lora.target_modules"].startswith("[")


def test_dump_config_round_trips(lora_config: ForgeConfig, tmp_path) -> None:
    from forgeml.config import dump_config

    path = dump_config(lora_config, tmp_path / "resolved.yaml")
    reloaded = load_config(path)
    assert reloaded.model_dump() == lora_config.model_dump()


def test_total_optimizer_steps_divides_out_accumulation() -> None:
    """Warmup is expressed in absolute steps on transformers 5.x, so this count
    determines the actual schedule — not just a progress bar."""
    from forgeml.training.trainer import _total_optimizer_steps

    config = ForgeConfig.model_validate(
        {
            "training": {
                "per_device_train_batch_size": 4,
                "gradient_accumulation_steps": 4,  # effective batch 16
                "epochs": 2,
            }
        }
    )
    # 100 examples / 16 per step = 7 steps per epoch (rounded up), x2 epochs
    assert _total_optimizer_steps(config, num_examples=100) == 14


def test_total_optimizer_steps_respects_max_steps() -> None:
    from forgeml.training.trainer import _total_optimizer_steps

    config = ForgeConfig.model_validate({"training": {"max_steps": 4, "epochs": 99}})
    assert _total_optimizer_steps(config, num_examples=100_000) == 4


def test_total_optimizer_steps_never_returns_zero() -> None:
    """A zero would make warmup_steps zero and, worse, divide-by-zero a scheduler."""
    from forgeml.training.trainer import _total_optimizer_steps

    config = ForgeConfig.model_validate({"training": {"epochs": 1}})
    assert _total_optimizer_steps(config, num_examples=1) >= 1


def test_circular_extends_is_detected(tmp_path) -> None:
    (tmp_path / "a.yaml").write_text("extends: b.yaml\nproject: a\n", encoding="utf-8")
    (tmp_path / "b.yaml").write_text("extends: a.yaml\nproject: b\n", encoding="utf-8")
    with pytest.raises(ValueError, match="circular"):
        load_config(tmp_path / "a.yaml")


def test_workspace_experiment_path_survives_a_remote_gpu(monkeypatch) -> None:
    """Training on Colab while tracking to Databricks must keep the absolute path.

    Databricks rejects a non-absolute experiment name outright, so mangling
    /Shared/forgeml to Shared-forgeml because the GPU is elsewhere fails the run.
    Compute location and tracking location are independent.
    """
    import mlflow

    from forgeml.tracking import mlflow_utils

    monkeypatch.delenv("DATABRICKS_RUNTIME_VERSION", raising=False)  # not on DB compute
    monkeypatch.setattr(mlflow, "get_tracking_uri", lambda: "databricks")
    monkeypatch.setattr(mlflow, "set_experiment", lambda name: None)
    monkeypatch.setattr(mlflow, "set_tracking_uri", lambda uri: None)

    config = load_config(CONFIG_DIR / "lora.yaml")
    assert mlflow_utils.setup_mlflow(config) == "/Shared/forgeml"


def test_workspace_path_is_flattened_for_a_local_store(monkeypatch) -> None:
    """Against ./mlruns the same path would create a directory named "/Shared/forgeml"."""
    import mlflow

    from forgeml.tracking import mlflow_utils

    monkeypatch.delenv("DATABRICKS_RUNTIME_VERSION", raising=False)
    monkeypatch.setattr(mlflow, "get_tracking_uri", lambda: "file:///tmp/mlruns")
    monkeypatch.setattr(mlflow, "set_experiment", lambda name: None)
    monkeypatch.setattr(mlflow, "set_tracking_uri", lambda uri: None)

    config = load_config(CONFIG_DIR / "lora.yaml")
    assert mlflow_utils.setup_mlflow(config) == "Shared-forgeml"


def test_torchao_version_constraint_is_pinned() -> None:
    """torchao must be pinned to >=0.16.0 to avoid version conflicts with bitsandbytes.

    This is a regression test: if someone removes the constraint from pyproject.toml,
    Colab installs will fail with ImportError about torchao version.
    """
    from pathlib import Path

    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    content = pyproject.read_text(encoding="utf-8")

    # Find the train dependencies section and check for torchao constraint.
    # The section is between train = [ and the closing ].
    train_section_start = content.find('train = [')
    train_section_end = content.find(']', train_section_start)
    train_section = content[train_section_start:train_section_end]

    assert "torchao>=0.16" in train_section, (
        "torchao>=0.16.0 must be pinned in pyproject.toml [project.optional-dependencies] "
        "train to avoid version conflicts on Colab"
    )
