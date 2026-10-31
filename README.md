# Volatility Surface Neural Network

Arbitrage-aware neural models of the SPX implied-volatility surface. The models learn total implied
variance $w(k, T)$ from end-of-day option chains. Static no-arbitrage conditions (calendar,
butterfly, wing asymptotics) are either penalized in the loss or guaranteed by construction. The
resulting arbitrage-free surface is used to flag candidate mispricings across strikes and
maturities, which are then validated statistically and in a cost-aware, delta-hedged backtest.

## Results at a glance

The best model on each task in the SPX validation year (2022). The 2023 test year is untouched.

| task | best model | result |
|---|---|---|
| fit a day's chain | eSSVI price mixture `mix4` | 0.121 vp RMSE, 81% of model IVs inside the bid-ask, zero arbitrage |
| extrapolate the wings | penalized network `pnn` | 0.86 vp (0.38% calendar violations left); best arbitrage-free model: `mix4`, 1.31 vp |
| fill missing strikes | `mix4` | 0.155 vp with 90% of interior strikes held out |
| fill missing maturities | conditional model, refined latent `cond_refined` | 0.64 vp with 30% of inner maturities held out (`mix4` 1.18; SVI 0.65 with calendar arbitrage) |
| fit fast | conditional model, amortized `cond` | 0.06 s per day, against about 30 s for `mix4`, at 0.80 vp |
| forecast the surface | none | no model beats the random walk significantly at 1, 5 or 20 days |

On the synthetic Heston panel, whose regime does not change between years, the cross-day
conditional model wins every held-out task: it is closest to the true surface on held-out strikes
(0.062 vp against 0.094 for `mix4`) and has the lowest error on held-out wings and on chains with
missing strikes or maturities. Per-day fits stay closer to the noisy quotes they are fitted to.

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

## Cross-day conditional model

One network for all days (`src/volsurf/models/conditional.py`, `src/volsurf/training/conditional.py`).
A set encoder reads any subset of a day's quotes, each described by $k/\sqrt{T}$, $\log T$, IV and
log half-spread (shared per-quote network, mean and max pooling), and summarizes the day as a
latent $z \in \mathbb{R}^8$. The decoder maps $z$ to the rates of four eSSVI surfaces on a fixed
grid of 26 maturities (5 to 900 days) and their mixture weights, built exactly as `mix4`. Every
latent therefore decodes to a surface free of static arbitrage, including latents that were never
fitted, such as forecasts. Training is neural-process style: each step encodes a random partial
chain (outer wings cut, a random share of interior strikes dropped, sometimes whole maturities)
and scores the decoded surface on all of that day's quotes. A day is then fitted either by the
encoder alone (`cond`) or by refining its latent against the quotes with L-BFGS (`cond_refined`).

SPX training uses 2018 to September 2021, with October–December 2021 for early stopping; the
synthetic Heston model trains on 2020. The latent size was chosen on the early-stopping loss
(d = 4 / 8 / 16: 14.8 / 7.6 / 8.1 squared half-spreads).

```bash
make conditional  # train both models (the SPX one runs for over an hour) -> artifacts/conditional/
make cross-day    # full protocol, merged with Phases 4 and 5, plus the regime diagnostic
make gaps         # refit every model with more of each chain held out
make forecast     # latent VAR(1) forecasts against random walks
```

SPX 2022, 52 days (violation rates on the widened domain; other rows as in the previous table):

| model | fit RMSE (vp) | within spread | held-out wings (vp) | calendar viol. | butterfly viol. | fit time (s) |
|---|---|---|---|---|---|---|
| SVI | 0.20 | 62% | 5.07 | 21.9% | 7.9% | 5.3 |
| penalized network (`pnn`) | 0.125 | 75% | **0.86** | 0.38% | 0.12% | 109 |
| eSSVI mixture (`mix4`) | **0.121** | **81%** | 1.31 | 0 | 0 | 33 |
| conditional, amortized (`cond`) | 0.80 | 16% | 1.86 | 0 | 0 | **0.06** |
| conditional, refined (`cond_refined`) | 0.61 | 21% | 2.20 | 0 | 0 | 3.6 |

- The guarantee holds throughout: zero calendar and butterfly violations, no non-finite values and
  no failed fits on all 105 SPX and Heston days, in this run and in every gap-filling run.
- On SPX the conditional model fits 5–7 times worse than `mix4`, because 2022 left the family of
  surfaces it learned from 2018–2021 (`make cross-day` writes the quarterly table to
  `results/phase6-shift/`). Its amortized error is 0.31–0.44 vp in every sampled quarter of
  2019 and 2021, but 0.73 vp in 2022Q1 and 1.46 vp by 2022Q4, while the ATM skew $\partial\sigma/\partial k$
  flattens from −0.91 to −0.45 (−0.70 to −0.95 in those quarters). Neither the encoder nor
  the maturity grid is the cause. Thinning the extra 2022 short-dated expiries back to the
  training count leaves the error unchanged, and `mix4` restricted to the decoder's grid still fits
  2022 to 0.19–0.29 vp. Refining the latent recovers only 3–11%. The error already doubles in
  2022Q1, before the skew moves, so the skew is the clearest symptom rather than the whole story.
  A latent model is only as good as the regimes it was trained on, and this year sits outside them.
- On synthetic Heston, where the regime is stationary, the same design is the best model: held-out
  truth error 0.062 vp (`mix4` 0.094, `pnn` 0.095) and held-out wings 0.25 vp (`mix4` 0.30), from a
  12 ms forward pass or a 1.4 s refinement.

**Gap-filling.** Every model is refitted with more of each chain held out (`interp` RMSE, vp):

| held out | SPX `mix4` | SPX `cond_refined` | SPX SVI | Heston `mix4` | Heston `cond_refined` | Heston SVI |
|---|---|---|---|---|---|---|
| 15% of interior strikes | **0.123** | 0.62 | 0.22 | 0.165 | 0.167 | 0.18 |
| 50% | **0.125** | 0.61 | 0.22 | 0.18 | **0.17** | 0.29 |
| 75% | **0.132** | 0.61 | 0.24 | 0.22 | **0.17** | 1.10 |
| 90% | **0.155** | 0.61 | 0.35 | 0.42 | **0.17** | fails |
| 30% of inner maturities | 1.18 | **0.64** | 0.65 | 0.88 | **0.16** | 0.19 |

- Per-day `mix4` degrades slowly as strikes go missing and stays best on SPX. Missing maturities
  are where it breaks (the cause is not yet diagnosed; SVI, which also bridges each gap by
  interpolation, does not break), and the conditional model beats it on 45 of 52 SPX days. SVI matches the conditional model there only by
  violating calendar arbitrage on a fifth of the widened grid.
- The conditional model's error barely moves with the share held out, on both panels. With 90% of
  strikes missing, the error against the true Heston surface is 0.073 vp, against 0.40 for `mix4`.
  At 90%, SVI cannot fit any Heston day: too few strikes remain per slice.

**Forecasting.** A VAR(1) on the latents of the training days forecasts $z$ h trading days ahead,
and the decoded surface is scored on every quote of the target day (all 2022 SPX days, all 2021
Heston days; RMSE in vp, Diebold–Mariano p-values with Newey–West variance):

| h | SPX VAR | SPX latent RW | SPX `mix4` RW | SPX decoded target day | Heston VAR | Heston `mix4` RW |
|---|---|---|---|---|---|---|
| 1 | 1.73 | 1.73 | **1.35** | 1.19 | 0.64 | **0.60** |
| 5 | 2.70 | 2.70 | **2.53** | 1.20 | 1.30 | **1.15** |
| 20 | **3.90** | 4.25 | 4.27 | 1.22 | 2.82 | **1.96** |

- On SPX the VAR loses to today's `mix4` surface at one day (p < 0.001): the decoded surface's own
  2022 error (1.19 vp, last SPX column) is almost as large as a one-day move, so forecasts built on
  decoded surfaces start behind today's `mix4` fit. At 20 days it has 16% lower MSE than
  both random walks, which is not significant (p = 0.27 and 0.29).
- The Heston panel is the positive control. The latent encodes the true log-variance almost
  linearly (out-of-sample R² 0.995), and forecasting that state with the generator's own dynamics
  beats the random walk by 24% at 20 days (p = 0.03). Yet an AR(1) fitted to the true state over
  the single training year loses by 80%, and the latent VAR by a similar margin (MSE ratio 2.1). One
  year of a process with a 35-day half-life is too short to estimate its mean: 2020 averaged nearly
  twice the variance of 2021. The representation carries the forecastable signal; estimating the
  dynamics from one year cannot recover it. No nonlinear dynamics model (such as a GRU) was
  tried, since the linear one did not beat the random walk.

## Roadmap

- [x] Project scaffold
- [x] Pricing core (Black-76, Heston COS), static-arbitrage checks, synthetic chain generator
- [x] OptionsDX SPX ingestion: cleaning, put-call-parity forwards, IV re-inversion, quote audit
- [x] Baselines (SVI, SSVI, spline, unconstrained MLP) and evaluation harness
- [x] Soft-penalty arbitrage-aware network
- [x] Hard-constrained eSSVI surfaces (arbitrage-free by construction)
- [x] Cross-day conditional surface model
- [ ] Mispricing detection, residual reversion tests, delta-hedged backtest
- [ ] Streamlit dashboard
- [ ] Technical report
