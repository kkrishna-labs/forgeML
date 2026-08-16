"""The predictor used by both the API and the demo.

Two implementations behind one interface:

* :class:`Predictor` — loads real weights, from the MLflow registry or a local
  checkpoint directory.
* :class:`StubPredictor` — echoes a canned response with no model at all.

The stub is not a testing afterthought; it is what lets the API's contract tests
run in CI in two seconds, and what keeps the Hugging Face Space responsive while
a cold container is still pulling weights. Same schema, same code path, no torch.
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from forgeml.logging_utils import get_logger
from forgeml.prompts import format_prompt

log = get_logger(__name__)


@dataclass
class ModelInfo:
    """What ``GET /model`` reports."""

    name: str = "forgeml-champion"
    method: str = "unknown"
    base_model: str = "unknown"
    version: str = "0"
    source: str = "stub"
    quality: float | None = None
    latency_p95_ms: float | None = None
    parameters: int | None = None
    quantization: str | None = None
    loaded: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "method": self.method,
            "base_model": self.base_model,
            "version": self.version,
            "source": self.source,
            "quality": self.quality,
            "latency_p95_ms": self.latency_p95_ms,
            "parameters": self.parameters,
            "quantization": self.quantization,
            "loaded": self.loaded,
            **self.extra,
        }


@dataclass
class Prediction:
    """What ``POST /predict`` returns."""

    text: str
    latency_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = "forgeml-champion"
    finish_reason: str = "stop"

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "prediction": self.text,
            "latency_ms": round(self.latency_ms, 2),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "finish_reason": self.finish_reason,
        }


class BasePredictor(ABC):
    """The interface the API depends on. Keep it small."""

    @abstractmethod
    def predict(self, prompt: str, context: str | None = None, **kwargs: Any) -> Prediction: ...

    @abstractmethod
    def info(self) -> ModelInfo: ...

    def healthy(self) -> bool:
        return True

    def warmup(self) -> None:
        """Run one throwaway generation so the first real request is not the slow one."""
        return None


class StubPredictor(BasePredictor):
    """A predictor with no model behind it.

    Used by CI, by ``forgeml serve --stub``, and as the fallback when weights fail
    to load — an API that returns a clearly-labelled stub response is far more
    debuggable than one that will not start.
    """

    def __init__(self, reason: str = "stub mode") -> None:
        self.reason = reason
        log.warning("StubPredictor active (%s) — responses are not model output", reason)

    def predict(self, prompt: str, context: str | None = None, **kwargs: Any) -> Prediction:
        started = time.perf_counter()
        text = (
            "[stub] No model is loaded, so this is a placeholder response. "
            f"The prompt received was: {prompt[:200]}"
        )
        return Prediction(
            text=text,
            latency_ms=(time.perf_counter() - started) * 1000,
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(text.split()),
            model="stub",
            finish_reason="stub",
        )

    def info(self) -> ModelInfo:
        return ModelInfo(name="stub", source="stub", loaded=False, extra={"reason": self.reason})


class Predictor(BasePredictor):
    """Loads real weights and generates.

    ``model_uri`` accepts either an MLflow reference (``models:/name@champion``) or
    a local checkpoint directory. Adapter directories are detected by the presence
    of ``adapter_config.json`` and loaded on top of their recorded base model.
    """

    def __init__(
        self,
        model_uri: str,
        base_model: str | None = None,
        device: str | None = None,
        max_new_tokens: int = 256,
        load_in_4bit: bool = False,
    ) -> None:
        self.model_uri = model_uri
        self.max_new_tokens = max_new_tokens
        self._info = ModelInfo(source=model_uri)

        import torch

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.tokenizer = self._load(model_uri, base_model, load_in_4bit)
        self.model.eval()

        self._info.loaded = True
        self._info.parameters = sum(p.numel() for p in self.model.parameters())
        log.info(
            "predictor ready: %s on %s (%.1fM params)",
            model_uri,
            self.device,
            self._info.parameters / 1e6,
        )

    # -- loading -----------------------------------------------------------

    def _load(self, model_uri: str, base_model: str | None, load_in_4bit: bool) -> tuple[Any, Any]:
        local_path = self._materialize(model_uri)

        if (Path(local_path) / "adapter_config.json").exists():
            return self._load_adapter(local_path, base_model, load_in_4bit)
        return self._load_full(local_path, load_in_4bit)

    def _materialize(self, model_uri: str) -> str:
        """Resolve an MLflow URI to a local directory; pass paths through unchanged."""
        if not model_uri.startswith(("models:/", "runs:/")):
            return model_uri

        import mlflow

        log.info("downloading %s from the model registry", model_uri)
        return mlflow.artifacts.download_artifacts(artifact_uri=model_uri)

    def _load_full(self, path: str, load_in_4bit: bool) -> tuple[Any, Any]:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(path)
        kwargs: dict[str, Any] = {}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            kwargs["device_map"] = "auto"
            self._info.quantization = "4-bit nf4"
        else:
            kwargs["torch_dtype"] = torch.bfloat16 if self.device == "cuda" else torch.float32

        model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
        if "device_map" not in kwargs:
            model = model.to(self.device)

        self._prepare_tokenizer(tokenizer)
        self._info.method = "full"
        self._info.base_model = path
        return model, tokenizer

    def _load_adapter(
        self, path: str, base_model: str | None, load_in_4bit: bool
    ) -> tuple[Any, Any]:
        import json

        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        adapter_config = json.loads(
            (Path(path) / "adapter_config.json").read_text(encoding="utf-8")
        )
        base = base_model or adapter_config.get("base_model_name_or_path")
        if not base:
            raise ValueError("adapter checkpoint does not record its base model; pass base_model=")

        log.info("loading adapter %s on base %s", path, base)

        kwargs: dict[str, Any] = {}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            kwargs["device_map"] = "auto"
            self._info.quantization = "4-bit nf4"
        else:
            kwargs["torch_dtype"] = torch.bfloat16 if self.device == "cuda" else torch.float32

        model = AutoModelForCausalLM.from_pretrained(base, **kwargs)
        model = PeftModel.from_pretrained(model, path)

        # Merging removes the adapter's per-layer overhead at inference. Not
        # possible on a 4-bit base, so that path keeps the adapter attached.
        if not load_in_4bit:
            model = model.merge_and_unload()
            log.info("merged adapter into base weights")

        if "device_map" not in kwargs:
            model = model.to(self.device)

        tokenizer_path = path if (Path(path) / "tokenizer_config.json").exists() else base
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self._prepare_tokenizer(tokenizer)

        self._info.method = "qlora" if load_in_4bit else "lora"
        self._info.base_model = base
        return model, tokenizer

    @staticmethod
    def _prepare_tokenizer(tokenizer: Any) -> None:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"  # required for correct batched generation

    # -- inference ---------------------------------------------------------

    def predict(self, prompt: str, context: str | None = None, **kwargs: Any) -> Prediction:
        import torch

        max_new_tokens = int(kwargs.get("max_new_tokens") or self.max_new_tokens)
        temperature = float(kwargs.get("temperature", 0.0) or 0.0)
        do_sample = temperature > 0

        text = format_prompt(prompt, context, tokenizer=self.tokenizer)
        encoded = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        prompt_tokens = int(encoded["input_ids"].shape[1])

        started = time.perf_counter()
        with torch.no_grad():
            output = self.model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=float(kwargs.get("top_p", 0.9)) if do_sample else None,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        if self.device == "cuda":
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - started) * 1000

        generated = output[0][prompt_tokens:]
        completion = self.tokenizer.decode(generated, skip_special_tokens=True).strip()

        return Prediction(
            text=completion,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=int(generated.shape[0]),
            model=self._info.name,
            finish_reason="length" if int(generated.shape[0]) >= max_new_tokens else "stop",
        )

    def info(self) -> ModelInfo:
        return self._info

    def warmup(self) -> None:
        log.info("warming up the model")
        self.predict("Say hello.", max_new_tokens=4)
        log.info("warmup complete")


def load_predictor(
    model_uri: str | None = None,
    base_model: str | None = None,
    allow_stub: bool = True,
) -> BasePredictor:
    """Build the right predictor from the environment.

    Resolution order: explicit argument, then ``FORGEML_MODEL_URI``, then stub.
    A load failure falls back to the stub when ``allow_stub`` — a demo that says
    "model unavailable" beats a container that crash-loops.
    """
    uri = model_uri or os.getenv("FORGEML_MODEL_URI")

    if os.getenv("FORGEML_STUB", "").lower() in ("1", "true", "yes"):
        return StubPredictor("FORGEML_STUB is set")

    if not uri:
        if not allow_stub:
            raise ValueError("no model_uri and FORGEML_MODEL_URI is unset")
        return StubPredictor("no FORGEML_MODEL_URI configured")

    try:
        return Predictor(
            model_uri=uri,
            base_model=base_model or os.getenv("FORGEML_BASE_MODEL"),
            load_in_4bit=os.getenv("FORGEML_LOAD_4BIT", "").lower() in ("1", "true", "yes"),
        )
    except Exception as exc:
        if not allow_stub:
            raise
        log.exception("failed to load %s, falling back to stub", uri)
        return StubPredictor(f"load failed: {type(exc).__name__}: {exc}")
