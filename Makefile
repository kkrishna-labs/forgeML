# ---------------------------------------------------------------------------
# ForgeML — common tasks.
# Works with GNU make on Linux/macOS and Git Bash on Windows.
# ---------------------------------------------------------------------------

PY      ?= python
VENV    ?= .venv
BIN     := $(VENV)/bin
ifeq ($(OS),Windows_NT)
BIN     := $(VENV)/Scripts
endif

.DEFAULT_GOAL := help
.PHONY: help venv install install-train lint format typecheck test test-cov smoke data train-all select serve demo docker clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

venv:  ## Create the virtualenv with uv
	uv venv $(VENV) --python 3.12

install:  ## Install the light dev environment (no torch)
	uv pip install -e ".[tracking,api,dev]"

install-train:  ## Install everything, including torch (GPU box only)
	uv pip install -e ".[all]"

lint:  ## Run ruff
	$(BIN)/ruff check src tests api demo

format:  ## Auto-fix and format
	$(BIN)/ruff check --fix src tests api demo
	$(BIN)/ruff format src tests api demo

typecheck:  ## Run mypy
	$(BIN)/mypy

test:  ## Run the test suite
	$(BIN)/pytest

test-cov:  ## Run tests with a coverage report
	$(BIN)/pytest --cov=forgeml --cov-report=term-missing --cov-report=html

smoke:  ## End-to-end pipeline on a tiny model — run this before any GPU job
	$(BIN)/forgeml data prepare --config configs/smoke.yaml
	$(BIN)/forgeml train --config configs/smoke.yaml --no-mlflow

data:  ## Prepare the real dataset
	$(BIN)/forgeml data prepare --config configs/base.yaml

train-all:  ## Train every arm locally (slow without a GPU)
	$(BIN)/forgeml train --config configs/baseline.yaml
	$(BIN)/forgeml train --config configs/lora.yaml
	$(BIN)/forgeml train --config configs/lora_r16.yaml
	$(BIN)/forgeml train --config configs/qlora.yaml
	$(BIN)/forgeml train --config configs/qlora_r16.yaml

select:  ## Rank the runs and print the champion decision
	$(BIN)/forgeml select --output reports/selection.json

serve:  ## Run the API in stub mode
	$(BIN)/forgeml serve --stub --reload

demo:  ## Run the Gradio demo
	$(BIN)/python demo/app.py

docker:  ## Build the inference image
	docker build -f deployment/Dockerfile -t forgeml:latest .

clean:  ## Remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage dist build
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
