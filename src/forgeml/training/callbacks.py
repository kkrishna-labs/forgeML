"""Trainer callbacks that stream metrics into MLflow while training runs.

Transformers ships its own MLflow integration, but it logs whatever key names the
Trainer happens to emit. The selector downstream depends on stable metric names,
so we own the mapping here instead of inheriting it.
"""

from __future__ import annotations

from typing import Any

from forgeml.logging_utils import get_logger

log = get_logger(__name__)


def _import_callback_base() -> Any:
    from transformers import TrainerCallback

    return TrainerCallback


def build_mlflow_callback(prefix: str = "train") -> Any:
    """Construct the callback class lazily so importing this module needs no torch."""
    TrainerCallback = _import_callback_base()

    class MLflowStreamingCallback(TrainerCallback):  # type: ignore[misc, valid-type]
        """Forward every Trainer log line to the active MLflow run.

        Also tracks peak GPU memory as training proceeds. Sampling it here rather
        than once at the end matters: peak memory usually occurs during the first
        few optimizer steps, and a reading taken after training has finished
        reports a number far below the true requirement.
        """

        def __init__(self) -> None:
            self.peak_memory_mb = 0.0
            self._step_losses: list[float] = []

        def on_log(self, args: Any, state: Any, control: Any, logs: dict | None = None, **kw: Any):
            if not logs or not state.is_world_process_zero:
                return control

            import mlflow

            payload: dict[str, float] = {}
            for key, value in logs.items():
                if not isinstance(value, (int, float)):
                    continue
                # `loss` -> `train/loss`, `eval_loss` -> `eval/loss`
                if key.startswith("eval_"):
                    name = f"eval/{key.removeprefix('eval_')}"
                elif key in ("loss", "grad_norm", "learning_rate", "epoch"):
                    name = f"{prefix}/{key}"
                else:
                    name = f"{prefix}/{key}"
                payload[name] = float(value)

            if "loss" in logs:
                self._step_losses.append(float(logs["loss"]))

            peak = _peak_gpu_memory_mb()
            if peak:
                self.peak_memory_mb = max(self.peak_memory_mb, peak)
                payload["system/gpu_peak_mb"] = self.peak_memory_mb

            if payload:
                try:
                    mlflow.log_metrics(payload, step=int(state.global_step))
                except Exception as exc:  # noqa: BLE001 - never kill a run over logging
                    log.warning("mlflow.log_metrics failed at step %s: %s", state.global_step, exc)
            return control

        @property
        def final_train_loss(self) -> float | None:
            return self._step_losses[-1] if self._step_losses else None

    return MLflowStreamingCallback()


def _peak_gpu_memory_mb() -> float:
    try:
        import torch

        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.max_memory_allocated() / 1024**2
    except Exception:  # noqa: BLE001
        return 0.0
