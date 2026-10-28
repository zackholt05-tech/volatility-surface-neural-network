.PHONY: help setup test lint format synthetic mlflow report clean

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

mlflow:  ## Launch the local MLflow UI
	uv run mlflow ui --backend-store-uri ./mlruns

report:  ## Compile the LaTeX report
	cd report && tectonic main.tex

clean:  ## Remove caches and build outputs
	rm -rf .pytest_cache .ruff_cache report/*.pdf
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
