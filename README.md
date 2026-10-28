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

Real-data experiments use SPX end-of-day option chains (2018–2023, 16:00 ET snapshots) from
[OptionsDX](https://www.optionsdx.com/) (free account; data is downloaded manually). Place the
downloaded `.7z` archives in `data/raw/optionsdx/`, then:

```bash
make data          # extract, estimate forwards, audit, clean -> data/processed/spx/
make diagnostics   # rate/carry history, filter funnel, sample surfaces
```

The pipeline:

- derives each expiry's forward and discount factor from put-call parity, with a per-date rate
  curve estimated from longer maturities, so no external rate or dividend data is needed;
- re-inverts Black-76 implied volatilities from bid, ask and mid of out-of-the-money options;
- flags quotes that are legs of tradable vertical or butterfly arbitrage (model-free, at bid/ask);
- writes `quotes`, `forwards` and per-day quality (`days`) tables as monthly Parquet partitions.

Synthetic surfaces need no external data: `make synthetic`. Both sources share the column contract in
`src/volsurf/data/schema.py`. Raw and processed data are not committed.

## Layout

```
src/volsurf/   pricing, arbitrage, data, models, training, eval, viz
scripts/       CLI entry points used by the Makefile
configs/       YAML configs for data, models, and experiments
dashboard/     Streamlit surface explorer
report/        LaTeX technical report
tests/         unit tests
```

## Evaluation protocol

`make baselines` fits every model in `configs/baselines.yaml` on the validation years (SPX 2022;
synthetic 2021) under one fixed protocol (`src/volsurf/eval/`):

- **Days:** blocks of two consecutive trading days every ten days. Days with heavy model-free
  arbitrage or poor forward coverage are screened out, and quotes flagged as arbitrage legs are
  excluded from fits.
- **Holdouts:** per slice, the outer 10% of strikes on each side (wing extrapolation) and a random
  15% of interior strikes (interpolation). The split is seeded by the date, so every model sees
  the same one.
- **Quote metrics:** IV RMSE/MAE in vol points, RMS residual in bid-ask half-spreads, share of
  model IVs inside the bid-ask, by maturity and moneyness bucket. Synthetic panels also report
  the error against the true surface.
- **Arbitrage metrics:** calendar and butterfly (Durrleman) violation rates and magnitudes on dense
  grids over the quoted domain and a 50%-widened domain.
- **Stability:** day-over-day change of fitted IVs on a fixed (k, T) probe grid.

`make train MODEL=<name>` runs a single model. Results go to `results/<experiment>/` and to a local
MLflow store (`mlflow.db`); browse runs with `make mlflow`. The 2023 SPX and 2022 synthetic test
years are held back until models are frozen.

## Roadmap

- [x] Project scaffold
- [x] Pricing core (Black-76, Heston COS), static-arbitrage checks, synthetic chain generator
- [x] OptionsDX SPX ingestion: cleaning, put-call-parity forwards, IV re-inversion, quote audit
- [x] Baselines (SVI, SSVI, spline, unconstrained MLP) and evaluation harness
- [ ] Soft-penalty arbitrage-aware network
- [ ] Hard-constrained neural SSVI
- [ ] Cross-day conditional surface model
- [ ] Mispricing detection, residual reversion tests, delta-hedged backtest
- [ ] Streamlit dashboard
- [ ] Technical report
