.PHONY: help setup test lint format synthetic data diagnostics baselines penalized constrained conditional cross-day gaps forecast mispricing-validation mispricing-test detector-tuning heston-multi heston-multi-test walk-forward walk-forward-test phase7b-validation phase7b-test train mlflow report clean

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

constrained:  ## Evaluate the hard-constrained eSSVI models and compare them with the Phase 4 network
	$(PY) scripts/evaluate_models.py --config configs/phase5_final.yaml
	$(PY) scripts/compare_results.py --experiments phase5-final phase4-final phase4-clean \
		--models ssvi svi spline mlp pnn essvi nmix1 mix4 nmix4 --out phase5-comparison

conditional:  ## Train the cross-day conditional models (SPX, synthetic Heston)
	$(PY) scripts/train_conditional.py --config configs/phase6_train_spx.yaml
	$(PY) scripts/train_conditional.py --config configs/phase6_train_heston.yaml

cross-day:  ## Evaluate the trained conditional models and compare them with Phases 4 and 5
	$(PY) scripts/evaluate_models.py --config configs/phase6_final.yaml
	$(PY) scripts/compare_results.py --experiments phase6-final phase5-final phase4-final \
		--models ssvi svi pnn mix4 nmix4 cond cond_refined --out phase6-comparison
	$(PY) scripts/conditional_shift.py

gaps:  ## Gap-filling: refit every model with more of each chain held out
	for f in 0.5 0.75 0.9; do \
		$(PY) scripts/evaluate_models.py --config configs/phase6_gaps.yaml \
			--experiment phase6-gap-$$f --set eval.holdout.interior_frac=$$f; \
	done
	$(PY) scripts/evaluate_models.py --config configs/phase6_gaps.yaml \
		--experiment phase6-gap-slices --set eval.holdout.slice_frac=0.3

forecast:  ## Forecast surfaces from the latent dynamics against random walks
	$(PY) scripts/forecast_conditional.py --checkpoint artifacts/conditional/spx_d8.pt
	$(PY) scripts/forecast_conditional.py --checkpoint artifacts/conditional/heston_d8.pt \
		--source synthetic:heston --train-years 2020 --eval-years 2021

HESTON_CKPT := artifacts/conditional/heston_d8.pt

mispricing-validation:  ## Fit the detector on the validation years and report flags, reversion, backtest
	$(PY) scripts/fit_panel.py --source synthetic:heston --years 2021 --checkpoint $(HESTON_CKPT) \
		--seed-cache results/phase6-forecast/synthetic_heston/mix4_fits.pkl
	$(PY) scripts/fit_panel.py --source spx --years 2022 \
		--seed-cache results/phase6-forecast/spx/mix4_fits.pkl
	$(PY) scripts/mispricing_report.py --split validation

mispricing-test:  ## The same on the test years (SPX 2023, synthetic 2022), with configs/phase7.yaml frozen
	$(PY) scripts/fit_panel.py --source synthetic:heston --years 2022 --checkpoint $(HESTON_CKPT)
	$(PY) scripts/fit_panel.py --source spx --years 2023
	$(PY) scripts/mispricing_report.py --split test

detector-tuning:  ## Choose the run-pooled robust detector on synthetic Heston 2021 (Phase 7b)
	$(PY) scripts/tune_detector.py

heston-multi:  ## Train the synthetic conditional model on eight more generator years; validate it
	$(PY) scripts/make_synthetic.py --only heston_train heston_fresh
	$(PY) scripts/train_conditional.py --config configs/phase6_train_heston_multi.yaml
	$(PY) scripts/evaluate_models.py --config configs/heston_multi_validation.yaml

heston-multi-test:  ## The multi-year synthetic model on Heston 2022 and the fresh panel, frozen config
	$(PY) scripts/evaluate_models.py --config configs/heston_multi_test.yaml

walk-forward:  ## Quarterly refits of the SPX conditional model, scored on 2022 (where 40 epochs won)
	$(PY) scripts/walk_forward.py --split validation
	$(PY) scripts/evaluate_models.py --config configs/walkforward_validation.yaml
	$(PY) scripts/conditional_shift.py --years 2022 --out results/walkforward-shift/spx \
		--checkpoint "artifacts/conditional/walkforward/spx_e40_{quarter}.pt"

walk-forward-test:  ## The 2023 quarterly refits with the chosen epoch count, scored with mix4, frozen
	$(PY) scripts/walk_forward.py --split test
	$(PY) scripts/evaluate_models.py --config configs/walkforward_test.yaml
	$(PY) scripts/conditional_shift.py --years 2023 --out results/walkforward-shift/spx-test \
		--checkpoint "artifacts/conditional/walkforward/spx_e40_{quarter}.pt"

PHASE7B := --config configs/phase7b.yaml --out results/phase7b-fits \
	--checkpoint cond=$(HESTON_CKPT) cond_multi=artifacts/conditional/heston_multi_d8.pt

phase7b-validation:  ## Phase 7 with the pooled detector and the multi-year model, validation years
	$(PY) scripts/fit_panel.py $(PHASE7B) --source synthetic:heston --years 2021 \
		--seed-cache results/phase7-fits/synthetic_heston/fits.pkl
	$(PY) scripts/fit_panel.py --config configs/phase7b.yaml --out results/phase7b-fits \
		--source spx --years 2022 --seed-cache results/phase7-fits/spx/fits.pkl
	$(PY) scripts/mispricing_report.py --config configs/phase7b.yaml --split validation

phase7b-test:  ## The same on the test panels, with configs/phase7b.yaml frozen
	$(PY) scripts/fit_panel.py $(PHASE7B) --source synthetic:heston --years 2022 \
		--seed-cache results/phase7-fits/synthetic_heston/fits.pkl
	$(PY) scripts/fit_panel.py $(PHASE7B) --source synthetic:heston_fresh --years 2023
	$(PY) scripts/fit_panel.py --config configs/phase7b.yaml --out results/phase7b-fits \
		--source spx --years 2023 --seed-cache results/phase7-fits/spx/fits.pkl
	$(PY) scripts/mispricing_report.py --config configs/phase7b.yaml --split test

train:  ## Evaluate one model from a config: make train MODEL=mlp [CONFIG=...]
	$(PY) scripts/evaluate_models.py --config $(CONFIG) --models $(MODEL)

mlflow:  ## Launch the local MLflow UI
	uv run mlflow ui --backend-store-uri sqlite:///mlflow.db

report:  ## Compile the LaTeX report
	cd report && tectonic main.tex

clean:  ## Remove caches and build outputs
	rm -rf .pytest_cache .ruff_cache report/*.pdf
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
