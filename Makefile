.PHONY: help setup test lint format synthetic data diagnostics baselines train mlflow report clean

CONFIG ?= configs/baselines.yaml

PY := uv run python

help:  ## List targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

setup:  ## Install dependencies into .venv
	uv sync
	mkdir -p data/raw/optionsdx data/processed data/synthetic

test:  ## Run unit tests
	uv run pytest

lint:  ## Lint with ruff
	uv run ruff check src tests scripts

format:  ## Format with ruff
	uv run ruff format src tests scripts
	uv run ruff check --fix src tests scripts

synthetic:  ## Generate synthetic surfaces with labelled mispricings
	$(PY) scripts/make_synthetic.py --config configs/synthetic.yaml

data:  ## Extract OptionsDX archives and build processed SPX quotes and forwards
	$(PY) scripts/build_spx.py --config configs/data_spx.yaml

diagnostics:  ## Data-quality figures and tables for the processed SPX data
	$(PY) scripts/spx_diagnostics.py

baselines:  ## Evaluate the baseline surface models on the validation years
	$(PY) scripts/evaluate_models.py --config configs/baselines.yaml

penalized:  ## Evaluate the frozen arbitrage-penalized network against the baselines
	$(PY) scripts/evaluate_models.py --config configs/phase4_final.yaml

train:  ## Evaluate one model from a config: make train MODEL=mlp [CONFIG=...]
	$(PY) scripts/evaluate_models.py --config $(CONFIG) --models $(MODEL)

mlflow:  ## Launch the local MLflow UI
	uv run mlflow ui --backend-store-uri sqlite:///mlflow.db

report:  ## Compile the LaTeX report
	cd report && tectonic main.tex

clean:  ## Remove caches and build outputs
	rm -rf .pytest_cache .ruff_cache report/*.pdf
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
