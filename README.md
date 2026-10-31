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

## Arbitrage-penalized network

The soft-constrained model (`src/volsurf/models/mlp.py`, `src/volsurf/training/`) multiplies a
per-day SSVI fit by a bounded neural correction, $w = w_\text{SSVI}\,\exp(3\tanh(f/3))$, with
inputs $k/\sqrt{\theta_\text{SSVI}(T)}$ and $\log T$. The loss is the spread-normalized IV fit plus
hinge penalties on calendar, Durrleman butterfly and wing conditions, evaluated by autograd on a
collocation grid that covers every quoted maturity, the midpoints between them, and strikes well
beyond the quoted range. Fits run in float64 on CPU (Adam, then L-BFGS) and warm-start from the
previous trading day.

```bash
uv run python scripts/evaluate_models.py --config configs/phase4_sweep.yaml     # penalty-weight sweep
uv run python scripts/evaluate_models.py --config configs/phase4_ablation.yaml  # ablations
uv run python scripts/plot_tradeoff.py --results results/phase4-ablation        # fit vs violations
make penalized                                                                  # frozen model, full protocol
```

## Hard-constrained eSSVI surfaces

The hard-constrained models (`src/volsurf/models/essvi.py`, `src/volsurf/models/neural_essvi.py`)
are free of static arbitrage by construction rather than by penalty. Each eSSVI slice is written in
wing coordinates $u_\pm = \psi(1 \pm \rho)$, where the sufficient calendar conditions ($\theta$
increasing, $u_\pm$ nondecreasing, $\psi/\theta$ nonincreasing) and the Gatheral–Jacquier butterfly
bounds become monotonicity and bound constraints. Knots are therefore generated in maturity order
from unconstrained rates (wings first, then $\theta$ above its lower bounds), so every parameter
vector is feasible; between knots the parameters are linear in $T$. The headline model, `mix4`,
mixes four such surfaces in forward-normalized price space with weights that do not depend on
maturity, which preserves the absence of arbitrage. Its implied variance comes from a bracketed
log-price Newton solve, and fits differentiate through that inversion. Knots are either free per
quoted maturity (`mix4`, `essvi`) or generated by a network of $\log T$ (`nmix4`, `nmix1`).

```bash
uv run python scripts/evaluate_models.py --config configs/phase5_ablation.yaml  # components, warm start, generator
make constrained  # full protocol, merged with the Phase 4 results (needs `make penalized` and configs/phase4_clean.yaml first)
```

SPX validation year 2022, 52 days; violation rates are on the 50%-widened domain, and the Phase 4
rows come from `phase4-final` (the deterministic anchors reproduce it exactly on every day):

| model | fit RMSE (vp) | within spread | held-out wings (vp) | calendar viol. | butterfly viol. | day-over-day (vp) |
|---|---|---|---|---|---|---|
| SSVI | 0.79 | 22% | 3.80 | 0 | 0 | 0.53 |
| SVI | 0.20 | 62% | 5.07 | 21.9% | 7.9% | 0.62 |
| spline | 0.13 | 64% | 3.95 | 21.6% | 0.14% | 0.70 |
| penalized network (`pnn`) | 0.125 | 75% | **0.86** | 0.38% | 0.12% | 0.56 |
| eSSVI (`essvi`) | 0.42 | 40% | 3.24 | 0 | 0 | 0.55 |
| eSSVI mixture (`mix4`) | **0.121** | **81%** | 1.31 | 0 | 0 | 0.54 |
| network-generated mixture (`nmix4`) | 0.41 (median 0.26) | 49% | 1.79 | 0 | 0 | 0.56 |

- Every eSSVI-family fit has zero calendar and butterfly violations on every day of every panel
  (SPX and three synthetic panels, 211 days), in the data domain and the widened one, with no failed
  fits. `mix4` matches or beats the penalized network on fit (better on 33 of 52 SPX days) at about a
  quarter of its fit time when the two run side by side (35 s against 160 s per day).
- The cost is wing extrapolation on SPX: `mix4` is worse than the penalized network on 42 of 52 days,
  but better on 40 of 53 days on both the synthetic Heston and SSVI panels. On four sampled SPX days
  the sufficient calendar conditions bind at the fitted knots ($u_+$ nondecreasing at about half of
  them, $\psi/\theta$ nonincreasing at a third to a half of the short-dated ones) while the butterfly
  bounds never do: the wing gap is the price of conditions that are sufficient rather than necessary,
  and mixing components recovers part of it (single eSSVI: 3.24 vp).
- Truth error on the synthetic panels is comparable to the penalized network (Heston / SSVI / clean
  Heston: 0.094 / 0.104 / 0.053 vp against 0.095 / 0.092 / 0.051) and, as with that network, it
  roughly doubles when mispricings are injected.
- Network-generated knots keep the guarantee but fit worse per day than free knots and are sensitive
  to initialization: `nmix4` beats a single eSSVI on 49 of 52 SPX days but gets stuck on two, and
  changing the optimizer schedule moves which days get stuck. Free knots win per-day fits; the
  generator is meant for the cross-day model, where it is trained across days.
- Warm-starting from the previous day lowers day-over-day surface changes (ablation: 0.52 against
  0.71 vp cold) at unchanged fit.

## Roadmap

- [x] Project scaffold
- [x] Pricing core (Black-76, Heston COS), static-arbitrage checks, synthetic chain generator
- [x] OptionsDX SPX ingestion: cleaning, put-call-parity forwards, IV re-inversion, quote audit
- [x] Baselines (SVI, SSVI, spline, unconstrained MLP) and evaluation harness
- [x] Soft-penalty arbitrage-aware network
- [x] Hard-constrained eSSVI surfaces (arbitrage-free by construction)
- [ ] Cross-day conditional surface model
- [ ] Mispricing detection, residual reversion tests, delta-hedged backtest
- [ ] Streamlit dashboard
- [ ] Technical report
