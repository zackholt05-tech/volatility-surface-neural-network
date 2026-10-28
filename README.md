# Volatility Surface Neural Network

Arbitrage-aware neural models of the SPX implied-volatility surface. The models learn total implied
variance $w(k, T)$ from end-of-day option chains. Static no-arbitrage conditions (calendar,
butterfly, wing asymptotics) are either penalized in the loss or guaranteed by construction. The
resulting arbitrage-free surface is used to flag candidate mispricings across strikes and
maturities, which are then validated statistically and in a cost-aware, delta-hedged backtest.

## Components

| Area | Contents |
|------|----------|
| Pricing | Black-76 pricing and vectorized implied-vol inversion, Heston (COS) pricer |
| Arbitrage | Calendar / Durrleman butterfly / Lee wing checks on $w(k,T)$, model-free quote audits |
| Data | OptionsDX SPX ingestion, put-call-parity forwards, IV re-inversion; synthetic Heston/SSVI surfaces with labelled mispricings |
| Models | SVI, SSVI, spline baselines; soft-penalty MLP; hard-constrained neural SSVI; cross-day conditional encoder–decoder |
| Evaluation | Fit/arbitrage metrics, mispricing flags, residual mean-reversion tests, delta-hedged backtest with costs and placebo |

## Setup

```bash
make setup      # uv sync + data directories
make test
make help       # list all targets
```

Requires [uv](https://docs.astral.sh/uv/) and, for the report, [Tectonic](https://tectonic-typesetting.github.io/).

## Data

Real-data experiments use SPX end-of-day option chains from [OptionsDX](https://www.optionsdx.com/)
(free account; data is downloaded manually, one year per order). Place the extracted monthly `.txt`
files under `data/raw/optionsdx/`. Raw and processed data are not committed.

Synthetic surfaces need no external data: `make synthetic`.

## Layout

```
src/volsurf/   pricing, arbitrage, data, models, training, eval, viz
scripts/       CLI entry points used by the Makefile
configs/       YAML configs for data, models, and experiments
dashboard/     Streamlit surface explorer
report/        LaTeX technical report
tests/         unit tests
```

## Experiment tracking

Runs are logged to a local MLflow store (`./mlruns`). Browse with `make mlflow`.

## Roadmap

- [x] Project scaffold
- [x] Pricing core (Black-76, Heston COS), static-arbitrage checks, synthetic chain generator
- [ ] OptionsDX SPX ingestion: cleaning, put-call-parity forwards, IV re-inversion
- [ ] Baselines (SVI, SSVI, spline, unconstrained MLP) and evaluation harness
- [ ] Soft-penalty arbitrage-aware network
- [ ] Hard-constrained neural SSVI
- [ ] Cross-day conditional surface model
- [ ] Mispricing detection, residual reversion tests, delta-hedged backtest
- [ ] Streamlit dashboard
- [ ] Technical report
