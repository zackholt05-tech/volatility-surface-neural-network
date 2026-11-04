# Volatility Surface Neural Network

Arbitrage-aware neural models of the SPX implied-volatility surface. The models learn total implied
variance $w(k, T)$ from end-of-day option chains. Static no-arbitrage conditions (calendar,
butterfly, wing asymptotics) are either penalized in the loss or guaranteed by construction. The
resulting arbitrage-free surface is used to flag candidate mispricings across strikes and
maturities, which are then validated statistically and in a cost-aware, delta-hedged backtest.

## Results at a glance

The best model on each task, in the SPX validation year (2022) unless a row names a test year. The
test years (SPX 2023, synthetic Heston 2022) were first run in Phase 7, with every setting frozen
beforehand. The later refinements were chosen on the validation years in the same way and frozen
before being run on the test years once more, and on a fresh synthetic panel. They are quarterly
refits of the SPX conditional model, the same model trained on 2010–2021, a synthetic model
trained on more years and a pooled robust detector. SPX 2016–2017 was added after all of these
choices and held out from them; the whole comparison was run on it once
([replication](#replication-on-spx-20162017)). Last, every mispricing panel was refitted with a start
from the previous day's surface, which avoids poor local minima of the default `mix4` fit. The rule was
frozen on the validation years before a fourth look at SPX 2023, a third at Heston 2022 and a
second at the fresh panel and at 2016–2017 ([refitted panels](#refitted-mispricing-panels)).

| task | best model | result |
|---|---|---|
| fit a day's chain | eSSVI price mixture `mix4` | 0.121 vp RMSE, 81% of model IVs inside the bid-ask, zero arbitrage (2016–2017: 0.068 vp, 99%) |
| extrapolate the wings | penalized network `pnn` | 0.86 vp (0.38% calendar violations left; 2016–2017: 0.32 vp). Best arbitrage-free: no single model. The conditional model trained on 2010–2021 and refitted quarterly is about even with `mix4` on 2022 (1.23 against 1.31 vp) and ahead on 2023 (0.70 against 0.82); on 2016–2017 `mix4` leads (0.35 against 0.55) |
| fill missing strikes | `mix4` | 0.155 vp with 90% of interior strikes held out (2016–2017: 0.13 vp) |
| fill missing maturities | `mix4` | 0.34 vp with 30% of inner maturities held out (conditional model 0.64; SVI 0.65 with calendar arbitrage; 2016–2017: `mix4` 0.18) |
| fit fast | conditional model, amortized, trained on 2010–2021 and refitted quarterly | 0.07 s per day, against about 30 s for `mix4`, at 0.58 vp (SPX 2023 test year: 0.38 vp, `mix4` 0.071) |
| forecast the surface | none | no model beats the random walk significantly at 1, 5 or 20 days |
| detect mispricings (synthetic test panels) | conditional model trained on nine years | 0.996 / 0.65 precision / recall on the fresh panel (robust `mix4` 0.95 / 0.45, pooled robust `mix4` 0.96 / 0.54); finds 66% of mispriced segments against 29% and 43%; Heston 2022 alike |
| trade them (SPX 2023 test year) | none | on refitted surfaces, the traded detector's flagged quotes revert no faster than unflagged ones the next day (p = 0.72), and their market prices move towards the model no more than unflagged ones at any horizon; next-day trades net +$292k, but matched placebos net +$233k (excess p = 0.31). On 2016–2017 the refits leave almost nothing to flag (3 trades) |

On the synthetic Heston panels, whose regime does not change between years, the cross-day
conditional model wins every held-out task. Trained on nine generator years, it is closest to the
true surface on held-out strikes (0.045 vp on both test panels against 0.098–0.099 for `mix4`)
and on held-out wings. Trained on one year, it already had the lowest error on chains with missing
strikes or maturities.
Per-day fits stay closer to the noisy quotes they are fitted to.

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

Real-data experiments use SPX end-of-day option chains (2010–2023, 16:00 ET snapshots; 3,491
trading days) from [OptionsDX](https://www.optionsdx.com/) (free account; data is downloaded
manually). The project was built on 2018–2023; 2010–2017 was added later, for a longer training
history and the replication window. Older chains are thinner and wider: a median 570 quotes in 12.5
expiries a day in 2010, about 2,400 in 27–31 in 2016–2017 and 6,300 in 37 in 2022, with median
half-spreads of 0.75, 0.14–0.20 and 0.10 vol points. Place the downloaded `.7z` archives in
`data/raw/optionsdx/`, then:

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
dashboard/     Streamlit dashboard: surfaces, arbitrage maps, mispricing flags
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
years are held back until models are frozen, and SPX 2016–2017 is a later, second test window.

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
make walk-forward       # quarterly SPX refits scored on 2022 (about an hour of training)
make walk-forward-test  # the same on 2023 with the chosen epoch count
make full-history       # the SPX model trained on 2010-2021, refitted quarterly, scored on 2022
make full-history-test  # the same on 2023 with the chosen epoch count
make heston-multi       # the synthetic model trained on nine years, scored on 2021
make heston-multi-test  # the same on Heston 2022 and the fresh panel
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
| 30% of inner maturities, plus 15% of strikes | **0.34** | 0.64 | 0.65 | 0.17 | **0.16** | 0.19 |

- Per-day `mix4` degrades slowly as strikes or whole maturities go missing and stays best on SPX:
  with 30% of inner maturities missing it beats the conditional model on 50 of 52 days. Missing
  maturities are interpolated between the observed ones; no model is given parameters at a
  maturity without quotes. SVI, the closest runner-up there, violates calendar arbitrage on a
  fifth of the widened grid.
- The conditional model's error barely moves with the share held out, on both panels, and on
  Heston it is best at every level. With 90% of strikes missing, its error against the true Heston
  surface is 0.073 vp, against 0.40 for `mix4`; with maturities missing, 0.057 against 0.087.
  At 90%, SVI cannot fit any Heston day: too few strikes remain per slice.

**Forecasting.** A VAR(1) on the latents of the training days forecasts $z$ h trading days ahead,
and the decoded surface is scored on every quote of the target day (all 2022 SPX days, all 2021
Heston days; RMSE in vp, Diebold–Mariano p-values with Newey–West variance):

| h | SPX VAR | SPX latent RW | SPX `mix4` RW | SPX decoded target day | Heston VAR | Heston `mix4` RW |
|---|---|---|---|---|---|---|
| 1 | 1.73 | 1.73 | **1.35** | 1.19 | 0.64 | **0.60** |
| 5 | 2.70 | 2.70 | **2.53** | 1.20 | 1.30 | **1.15** |
| 20 | 3.90 (n.s.) | 4.25 | 4.27 | 1.22 | 2.82 | **1.96** |

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

**Walk-forward refits** (`scripts/walk_forward.py`, `configs/walkforward_spx.yaml`). The SPX model
above stops learning in September 2021. Before each quarter, a copy of it is now fine-tuned on
every screened day up to the quarter's start, from 2018 on (979 days for 2022Q1, 1,416 for
2023Q4). Fine-tuning runs at a constant learning rate of 3·10⁻⁴ with no early stopping, and the
quarter is then fitted with the refreshed model. Every refit starts from the base model, not from
the previous quarter's, and takes about a quarter of an hour. The epoch count (10, 20 or 40) was
chosen on 2022 by the lowest mean amortized fit error (0.65 / 0.63 / 0.63 vp; 40 won) and frozen
before 2023 was run. Fit RMSE (vp) on the harness days:

| model | 2022Q1 | Q2 | Q3 | Q4 | 2022 | 2023Q1 | Q2 | Q3 | Q4 | 2023 (test) |
|---|---|---|---|---|---|---|---|---|---|---|
| `cond`, frozen | 0.49 | 0.83 | 0.75 | 1.21 | 0.80 | 0.84 | 0.46 | 0.55 | 0.59 | 0.61 |
| `cond`, walk-forward | 0.45 | 0.77 | 0.52 | 0.83 | 0.63 | 0.53 | 0.34 | 0.38 | 0.33 | 0.40 |
| walk-forward, refined | 0.36 | 0.52 | 0.45 | 0.65 | 0.49 | 0.48 | 0.30 | 0.30 | 0.27 | 0.34 |
| `mix4` | 0.12 | 0.12 | 0.11 | 0.14 | 0.12 | 0.11 | 0.06 | 0.06 | 0.06 | 0.071 |

- On 2023, the test year, the refreshed model beats the frozen one on all 50 days. It cuts the
  amortized error by a third and the held-out wing error from 1.44 to 0.85 vp. On the wings it is
  about even with `mix4`: 0.85 against 0.82 vp in 2023 (better on 19 of 50 days) and 1.25 against
  1.31 in 2022 (28 of 52). These are the first surface-fit metrics computed on 2023. Phase 7
  fitted `mix4` on every 2023 day for reversion and the backtest, but never scored the fits, and
  nothing about the conditional model was tuned on 2023.
- `mix4` still fits each day about five times more closely, and 80% of its model IVs fall inside the
  bid-ask against 20% for the refreshed model. Refitting can at best bring the conditional model
  back to its in-sample error (0.31–0.44 vp in 2019 and 2021). On the quarterly shift diagnostic
  (`results/walkforward-shift/`) the refits close about half of the 2022 gap to that range,
  from 0.73 / 1.19 / 1.19 / 1.46 to 0.68 / 0.94 / 0.72 / 0.90 vp, and in 2023 the error is
  0.42–0.69 vp.
- The guarantee carries over: zero calendar and butterfly violations, no failed fits.

**Full history** (`configs/phase6_train_spx_full.yaml`, `configs/walkforward_spx_full.yaml`). With
2010–2017 added, the same model was retrained on 2010 to September 2021 (2,906 days, early-stopped
on 2021Q4 as before) and refitted each quarter on every screened day from 2010 (2,970 days for
2022Q1, 3,407 for 2023Q4). The epoch count was chosen on 2022 by the same rule (0.584 / 0.585 /
0.595 vp for 10 / 20 / 40 epochs; 10 won) and frozen before 2023 was run. That run is the third
look at 2023, made because the new refits beat the old ones on 2022, amortized and refined. RMSE
(vp) on the same harness days:

| model | fit 2022 | fit 2023 (test) | held-out wings 2022 | wings 2023 |
|---|---|---|---|---|
| trained from 2018, frozen | 0.80 | 0.61 | 1.86 | 1.44 |
| trained from 2010, frozen | 0.77 | 0.48 | 1.88 | 1.25 |
| from 2018, walk-forward (40 epochs) | 0.63 | 0.40 | 1.25 | 0.85 |
| from 2010, walk-forward (10 epochs) | 0.58 | 0.38 | 1.23 | 0.70 |
| from 2018, walk-forward, refined | 0.49 | 0.34 | 1.38 | 0.88 |
| from 2010, walk-forward, refined | 0.48 | 0.32 | 1.40 | 0.78 |
| `mix4` | 0.12 | 0.071 | 1.31 | 0.82 |

- The longer history lowers the fit error everywhere. Amortized walk-forward error falls by 7% on
  2022 (better on 38 of 52 days, p < 0.001) and 5% on 2023 (36 of 50, p = 0.001). Refined error
  falls by 2% on 2022 (not significant, p = 0.25) and 5% on 2023 (p < 0.001). The frozen model
  gains most on 2023: 22%, better on 47 of 50 days.
- Held-out wings on 2023 improve by 18% (0.85 to 0.70 vp, better on 49 of 50 days), ahead of
  `mix4` (0.82 vp; p = 0.009, though better on only 27 of 50 days). On 2022 they are unchanged.
  Twenty epochs would have done better there (1.12 vp, p < 0.001), but the rule fixed beforehand
  selects on fit error alone.
- The 2018-base refits, rescored in both runs, reproduce 0.63 / 0.49 vp on 2022 and 0.40 / 0.34
  on 2023 exactly.
- `mix4` still fits each day about five times more closely. The frozen model's 2022 error is still
  twice its early-stopping error (0.39 vp), so 2022 also lies outside 2010–2021.

**More synthetic training years** (`configs/phase6_train_heston_multi.yaml`). The Heston model
above trains on a single year, and Phase 7 found its precision collapsing once 2022 reached
variance levels that year never visited. A second panel from the same generator
(`synthetic:heston_train`, seed 10, 2012–2019, 2,087 days, independent of every evaluation
panel) spans the generator's stationary 1st–99th variance percentiles. `cond_multi` trains on it
plus 2020 (2,282 days) with the same architecture and early-stopping days. A further
panel (`synthetic:heston_fresh`, seed 20, 2023) was generated for the test and not looked at
beforehand. It is this model's clean test: Heston 2022 is where the single-year model's failure
motivated it. Truth error on held-out strikes (vp):

| panel | `cond` | `cond_refined` | `cond_multi` | `cond_multi`, refined | `mix4` |
|---|---|---|---|---|---|
| Heston 2021 (validation) | 0.083 | 0.062 | 0.040 | 0.040 | 0.094 |
| Heston 2022 (test) | 0.138 | 0.078 | 0.045 | 0.040 | 0.098 |
| fresh panel, 2023 (test) | 0.152 | 0.082 | 0.045 | 0.042 | 0.099 |

- With nine training years, the amortized model is less than half as far from the true surface as
  `mix4` on every panel, in an 11 ms forward pass against 7.7 s. Its held-out wings are also
  closest to the truth (0.11–0.13 vp, `mix4` 0.24–0.25), and 96% of its model IVs fall inside the
  bid-ask (single-year model 83–85% on the test panels).
- `mix4` stays closest to the quotes themselves (fit RMSE 0.13 against 0.16 vp), because it also
  fits each day's quote noise.

## Mispricing detection and backtest

`src/volsurf/eval/mispricing.py`, `reversion.py`, `backtest.py`; every setting is in
`configs/phase7.yaml`. The settings were chosen on the validation years (SPX 2022, synthetic Heston
2021) and committed before the test years (SPX 2023, synthetic 2022) were run once. The SPX results
of this section are first looks on cached `mix4` fits, many of which stopped in poor local minima;
[refitted panels](#refitted-mispricing-panels) reruns them and replaces their conclusions. The
synthetic results are unchanged by the refits.

```bash
make mispricing-validation  # fit every validation day, then detection, reversion, backtest tables
make mispricing-test        # the same on the test years (about an hour on 8 cores)
```

**Detector.** Each day's `mix4` surface is refitted twice by iteratively reweighted least squares
for a Cauchy loss on spread-normalized residuals (scale one half-spread), and every quote is
scored against it. A quote is flagged when the model price lies outside the bid-ask by more than a
margin, in half-spreads; rich quotes are sold and cheap ones bought. Least-squares fits absorb part
of each mispricing into the surface. On Heston 2021 they absorb a median 15% of an isolated
mispriced strike and 55% of a mispriced segment of 3–6 neighbouring strikes, which one expiry's
smile can bend to fit. The robust refit halves the first figure (7%) but barely moves the second
(51%). The cross-day conditional model, whose surfaces move only through the latent, absorbs 1%
and 4% and is scored alongside on the synthetic panels (`absorption.csv`). On SPX its error (0.6–1.5 vp in 2022, Phase 6) is
several times the half-spread (about 0.1 vp), so it is not used there.

**Detection on the synthetic test year** (Heston 2022; labels are injected shifts above one
half-spread; "visible" recall counts labels whose true IV lies outside the quoted bid-ask):

| detector, margin 0.5 | precision | recall | recall, visible | recall, isolated | recall, segments |
|---|---|---|---|---|---|
| least-squares `mix4` | 0.90 | 0.39 | 0.45 | 0.53 | 0.21 |
| robust `mix4` | **0.95** | 0.44 | 0.52 | 0.58 | 0.27 |
| conditional model | 0.64 | **0.63** | **0.74** | **0.63** | **0.63** |

- Up to margin 2, the robust refit has higher precision and recall than least squares on both
  synthetic years.
- The conditional model finds segments that every per-day surface absorbs. On the validation year it
  was also the most precise (0.96 against 0.95 for robust `mix4`) and was chosen as the traded
  synthetic detector. On the test year its precision is 0.99 in 2022Q1–Q2 but 0.52 and 0.35 in
  Q3–Q4, while its recall stays at 0.62–0.64: the second half of 2022 reaches variance levels below
  anything in its single training year (79% of 2022Q4 days are below the 2020 minimum), where its
  error against the true surface triples (0.07 to 0.21 vp). This is the same regime limit as on SPX
  in Phase 6.

**Do flagged residuals revert?** For each day and horizon h, the residual change of every contract
quoted on both days is regressed on its residual, separately for flagged and unflagged quotes, and
the daily slopes are averaged (Newey–West errors). A slope of −1 is full reversion.

- On synthetic data, flagged slopes of the conditional detector (−0.49 / −0.89 / −0.95 at 1 / 3 / 5
  days) match the injected decay (−0.50 / −0.875 / −0.97). Unflagged synthetic residuals are
  mostly quote noise, which reverts completely, so their one-day slope is steeper still. On SPX,
  unflagged residuals are mostly persistent model misfit, and flagged ones revert faster.
- SPX 2023, margin 4 (0.4% of quotes flagged): flagged residuals revert −0.57 / −0.71 / −0.72 at 1 /
  3 / 5 days, against −0.34 / −0.44 / −0.50 for unflagged ones (differences significant, p ≤ 0.003;
  144 days with at least five flags): 57% of a flagged residual is gone the next day. The model-free
  arbitrage legs of Phase 2 (0.5% of quotes) are mostly flagged at a zero margin (67%) but rarely at
  margin 4 (6%).

**Backtest.** Each flag is traded with $1,000 of vega, crossing the spread on entry and exit ($1 per
contract per side in fees). It is delta-hedged daily with its expiry's put-call-parity forward, and
closed once the residual is below half a spread or after five days. Options that move into the money
are marked through put-call parity. Entries are restricted to contracts that stay inside the cleaned
panel until their latest possible exit, so 0.7% of SPX trades lose their quote. Next-day execution
(signal at one close, trade at the next) is the main case; same-day execution is an optimistic
bound. Each trade has 20 placebos: random unflagged contracts from the same day, type, direction and
maturity–moneyness bucket, held for the same days.

| | trades | gross | costs | net | Sharpe | placebo gross | net minus placebo (p) |
|---|---|---|---|---|---|---|---|
| SPX 2023, next day | 2,595 | +$162k | $354k | **−$192k** | −0.66 | +$129k | +$100k (0.09) |
| SPX 2023, same day (bound) | 2,883 | +$421k | $351k | +$70k | 0.35 | +$176k | +$423k (0.002) |
| SPX 2022 (validation; best of 12 cells), next day | 3,978 | +$906k | $894k | +$12k | 0.01 | +$471k | +$641k (0.03) |
| Heston 2022, conditional, next day | 3,091 | +$998k | $1,582k | −$584k | −5.4 | +$37k | +$940k (< 0.001) |
| Heston 2022, conditional, same day | 3,329 | +$2,122k | $1,740k | +$383k | 3.1 | +$61k | +$2,068k (< 0.001) |
| Heston 2022, true labels, same day | 2,786 | +$1,747k | $1,541k | +$206k | 1.9 | −$28k | +$1,752k (< 0.001) |

- The accounting passes its check: trading the true labels at the signal's close makes money net
  of costs (last row).
- On the cached fits of SPX 2023 the signal looked real but not tradable at end-of-day prices
  (refitted, it is neither; see [refitted panels](#refitted-mispricing-panels)). Flagged contracts earn more
  than matched placebos, significantly with same-day execution. But half of the reversion happens
  before a next-day trade, and the spread, paid twice, costs more than what remains: the 2023
  next-day strategy loses $192k, and its edge over the placebo is not significant (p = 0.09).
  End-of-day data also makes even the same-day bound optimistic, since it assumes a fill at the
  closing quotes that produced the signal.
- Placebos earn a positive gross on SPX, likely the variance risk premium collected by the sold
  options (58–59% of SPX trades are sales). The excess over the placebo is the part the signal
  adds.
- Hit rates are 46% (next day) and 52% (same day) on SPX 2023, maximum drawdowns $409k and $268k,
  and turnover 66 and 77 contracts a day. The validation row is the best of the 12 cells it was
  selected from, so it is biased upward.

**Follow-up detectors** (`configs/phase7b.yaml`). These detectors were chosen on the validation years
and frozen with the configs. They were then run on the test years: a second look at Heston 2022
and SPX 2023, and a first at the fresh synthetic panel.

```bash
make detector-tuning     # choose the pooled detector on Heston 2021
make phase7b-validation  # detection and reversion tables on the validation years
make phase7b-test        # the same on Heston 2022, the fresh panel and SPX 2023 (about two hours)
```

- *Pooled robust refit.* A mispriced segment survives the robust refit because each quote's Cauchy
  weight reads only its own residual, and a smooth bump spread over 3–6 strikes leaves each one
  small. With `pool = h`, a quote's weight uses the larger of its own residual and the mean
  residual over up to h strikes on each side in its expiry. A run of same-signed residuals is then
  down-weighted as a whole, while an isolated quote keeps its own weight. The setting was chosen
  among 24 cells (pool 0–2, scale 0.125–1 half-spread, 4–16 passes) on every sixth day of Heston
  2021. The rule was the highest recall at margin 0.5 among cells at least as precise as the
  Phase 7 detector and with at least its recall on isolated strikes. The chosen detector, pooled
  robust `mix4` (pool 2, scale 0.25, 16 passes), gains most from the stronger, longer
  reweighting. The same cell without pooling reaches recall 0.48, against 0.50 with it and 0.42
  for the Phase 7 detector.
- *Multi-year conditional model.* `cond_multi` from Phase 6, the conditional model trained on
  nine generator years instead of one, is scored as a detector.

Detection at margin 0.5, Heston 2022 / fresh panel (2023; the multi-year model's clean test):

| detector | precision | recall | recall, isolated | recall, segments | segment absorbed |
|---|---|---|---|---|---|
| least-squares `mix4` | 0.90 / 0.90 | 0.39 / 0.40 | 0.53 / 0.55 | 0.21 / 0.23 | 56% / 55% |
| robust `mix4` (Phase 7) | 0.95 / 0.95 | 0.44 / 0.46 | 0.58 / 0.59 | 0.27 / 0.30 | 52% / 52% |
| pooled robust `mix4` | 0.96 / 0.96 | 0.52 / 0.54 | 0.62 / 0.64 | 0.40 / 0.43 | 34% / 33% |
| conditional, one training year | 0.64 / 0.58 | 0.63 / 0.65 | 0.63 / 0.65 | 0.63 / 0.65 | 4% / 4% |
| conditional, nine training years | **0.997 / 0.996** | 0.63 / 0.65 | 0.63 / 0.64 | 0.62 / 0.66 | 4% / 3% |

- The multi-year conditional model is the best detector on both panels. Its precision is
  0.995–0.998 in every quarter, and it keeps the single-year model's recall, including 62–66% of
  mispriced segments. The single-year model loses precision in two quarters of each panel: 0.52
  and 0.35 in 2022Q3–Q4, where the variance falls below its training year, and 0.39 and 0.47 in
  the first half of the fresh panel.
- The pooled refit is the best per-day detector. At the Phase 7 detector's precision, it adds 8
  points of recall and 13 of segment recall, and the median share of a mispriced segment absorbed
  into the surface falls from 52% to 34%. The least-squares and robust rows reproduce Phase 7
  exactly.
- Flagged residuals of the multi-year detector revert −0.52 / −0.95 / −0.99 at 1 / 3 / 5 days on
  Heston 2022, against an injected decay of −0.50 / −0.875 / −0.97.
- SPX 2023, margin 4: the pooled detector flags 0.9% of quotes (robust `mix4` 0.4%), with at least
  five flags on 211 days (144). Its flagged residuals revert −0.63 / −0.73 / −0.75 against −0.39 /
  −0.49 / −0.55 for unflagged ones (p < 0.001 at every horizon; on SPX 2022 the gaps were −0.27 /
  −0.23 / −0.32). That is more than twice as many flags, with a gap over unflagged residuals at
  least as large as the Phase 7 detector's.
- The backtest keeps the Phase 7 detector, as frozen before these runs, so its rows above are
  unchanged. On SPX the conditional model is still not a detector: even refitted each quarter,
  its 2023 error (0.3–0.5 vp) is several half-spreads.

## Replication on SPX 2016–2017

SPX 2016–2017 was added after every choice above and is a second test window that nothing was tuned
on. Every model, gap level and detector was run on it once with the settings chosen on 2022,
committed before any model was scored on it. The conditional models were retrained on 2010 to
September 2015 (early-stopped on 2015Q4) and refitted before each quarter for 40 epochs on every
day from 2010, the walk-forward setting chosen on 2022. The harness samples 101 days.

```bash
make replication             # retrain and refit the conditional model, every model, two gap levels
make replication-mispricing  # detectors, reversion and backtest
```

| model | fit RMSE (vp) | within spread | held-out wings (vp) | calendar viol. | butterfly viol. | fit time (s) |
|---|---|---|---|---|---|---|
| SVI | 0.128 | 90% | 2.67 | 15.6% | 10.9% | 8.4 |
| spline | 0.130 | 83% | 2.23 | 7.4% | 0.66% | 0.003 |
| SSVI | 0.61 | 37% | 1.83 | 0 | 0 | 0.12 |
| penalized network (`pnn`) | 0.085 | 96% | **0.32** | 0.08% | 0.004% | 194 |
| eSSVI mixture (`mix4`) | **0.068** | **99%** | 0.35 | 0 | 0 | 32 |
| conditional, frozen | 0.51 | 34% | 0.75 | 0 | 0 | 0.11 |
| conditional, frozen, refined | 0.37 | 46% | 0.71 | 0 | 0 | 2.6 |
| conditional, walk-forward | 0.38 | 43% | 0.55 | 0 | 0 | **0.07** |
| conditional, walk-forward, refined | 0.29 | 55% | 0.52 | 0 | 0 | 1.6 |

- The rankings of 2022 and 2023 hold. `mix4` fits closest and is arbitrage-free. `pnn` follows, with
  the best wings, at six times `mix4`'s fit time and with a few violations left. SVI and the spline
  fit almost as closely but violate calendar arbitrage on 16% and 7% of the widened grid.
- Every model sits inside the bid-ask more often than in 2022, because the spreads are wider:
  median half-spreads are 0.14–0.20 vp, against 0.10 in 2022 and 0.07 in 2023.
- Walk-forward refits cut the conditional model's amortized error by 27% (22% on 2022, 35% on
  2023). The refreshed model is also the most stable day to day (0.24 vp, `mix4` 0.28, SVI 0.50).
- Gap filling repeats too. With 90% of interior strikes held out, `mix4` misses the held-out
  strikes by 0.13 vp, against 0.29 for the refined walk-forward model, 0.52 for SVI and 0.60 for
  SSVI. With 30% of inner maturities held out, `mix4` misses by 0.18 vp, SVI by 0.26 (calendar
  violations on 16% of the grid) and the refined walk-forward model by 0.30.
- One 2023 finding does not replicate. The walk-forward model is not even with `mix4` on the wings
  here (0.55 against 0.35 vp). Only with 90% of strikes held out are they close: 0.53 vp for the
  refined model against 0.52.

The mispricing tests (`configs/replication_mispricing.yaml`, the Phase 7 and 7b settings unchanged)
use all 501 days, with `mix4` fitted on each. On the cached fits, almost every flag was a fit failure:
92% of them fell on four days of October 2017, on which the fits miss whole expiries by about
2 vol points. Refitted from the previous day's surface
([refitted panels](#refitted-mispricing-panels)), the panel has next to nothing to flag:

| | trades | gross | costs | net | Sharpe | placebo gross | net minus placebo (p) |
|---|---|---|---|---|---|---|---|
| SPX 2016–2017, cached fits, next day | 364 | +$9k | $130k | −$121k | −1.42 | +$7k | −$16k (0.32) |
| SPX 2016–2017, cached fits, same day (bound) | 370 | −$15k | $128k | −$143k | −1.34 | −$15k | −$7k (0.63) |
| SPX 2016–2017, refits, next day | 3 | +$2 | $2.2k | −$2.2k | −1.03 | +$365 | −$781 (0.45) |
| SPX 2016–2017, refits, same day (bound) | 4 | −$2.7k | $1.9k | −$4.5k | −1.06 | −$1.6k | −$741 (0.51) |

- `mix4` sits inside the wider spreads, and margins are in half-spreads, so margin 4 is a two to
  three times larger deviation in vol points than in 2023. On the refits, the robust detector
  flags 0.003% of quotes at margin 4 (cached fits 0.04%; 2023 refits 0.30%) and the pooled detector
  0.006% (0.05%); even at a zero margin the robust detector flags 1.0% (2023: 18%).
- There is nothing to test reversion on: at most 2 days have five flags of any detector (cached
  fits: 5 robust, 8 pooled). Unflagged residuals revert −0.79 / −0.85 / −0.88 at 1 / 3 / 5 days,
  against −0.33 / −0.45 / −0.51 in 2023. This is closer to the synthetic panels than to 2023,
  suggesting that residuals inside the spread are mostly quote noise rather than persistent misfit.
- The cached-fit backtest lost money even as a bound, and did no better than its placebos, but it
  traded the October 2017 fit failures. The 2016–2017 window says only that, with spreads this wide,
  an arbitrage-free `mix4` fit leaves almost no quote four half-spreads outside the bid-ask.

## Refitted mispricing panels

`configs/phase7_warm.yaml`, `configs/replication_warm.yaml`. The default least-squares `mix4` fit
sometimes stops in a poor local minimum, and a day fitted that way flags whole expiries. Every
mispricing panel was refitted (`scripts/fit_panel.py --warm-start`): each day's least-squares
surface is the better, by its own objective, of the cached fit and a fit started from the previous
day's chosen surface, so the days run in date order, and the robust refits start from it. The rule
and every other setting were frozen on the validation years, with a written prediction
(`results/phase7-warm-validation/notes_before_test.md`, commit 30c766c), before the test panels were
run once: a fourth look at SPX 2023, a third at Heston 2022 and a second at the fresh panel and at
SPX 2016–2017.

```bash
make warm-validation   # SPX 2022 and Heston 2021, then the backtest selection rule reapplied
make warm-test         # SPX 2023, Heston 2022 and the fresh panel
make replication-warm  # SPX 2016-2017
```

**The cached fits failed often.** The chosen fit's objective relative to the cached one, which is
the fit behind every earlier mispricing result:

| panel | previous-day start won | median ratio | days > 10% lower | > 2× lower | worst day |
|---|---|---|---|---|---|
| SPX 2022 (validation) | 154 / 253 | 0.92 | 54 | 0 | 0.47 → 0.25 (Sep 9) |
| SPX 2023 | 205 / 247 | 0.69 | 172 | 69 | 0.53 → 0.12 (Jul 7) |
| SPX 2016 | 105 / 251 | 1.00 | 20 | 0 | 0.149 → 0.115 (Oct 27) |
| SPX 2017 | 144 / 250 | 0.97 | 83 | 12 | 5.51 → 0.048 (Oct 24) |
| Heston 2021 / 2022 / fresh 2023 | 41 / 48 / 42 of 261 / 234 / 260 | 1.00 | 0 | 0 | |

- On 69 of the 247 SPX 2023 days the cached fit's objective was more than twice what the
  surface can reach. The synthetic surfaces, drawn from one smooth model, have no such minima.
- On the validation years the refits change little: the backtest rule again picks margin 4 and
  hold 5 (next-day net Sharpe 0.14, against 0.01), and the Heston 2021 detection table is all but
  identical. Every setting of the original configs was kept.
- The model comparisons of Phases 3–6 fit `mix4` on their own sampled days and were not rerun.
  Minima there would overstate `mix4`'s errors.

**SPX 2023, robust `mix4` at margin 4.** The refits flag 0.30% of quotes (cached fits 0.44%), with
at least five flags on 67 days (144).

| flagged minus unflagged reversion slope | 1 day | 3 days | 5 days |
|---|---|---|---|
| cached fits (Phase 7) | −0.23 (p = 0.002) | −0.23 (< 0.001) | −0.18 (0.003) |
| refits | −0.05 (0.72) | −0.24 (0.006) | −0.17 (0.048) |

| | trades | gross | costs | net | Sharpe | placebo gross | net minus placebo (p) |
|---|---|---|---|---|---|---|---|
| SPX 2023, next day | 1,482 | +$476k | $184k | **+$292k** | 0.99 | +$459k | +$59k (0.31) |
| SPX 2023, same day (bound) | 1,586 | +$545k | $176k | +$368k | 1.63 | +$326k | +$320k (0.019) |
| SPX 2022 (validation; rule reapplied), next day | 3,420 | +$948k | $771k | +$178k | 0.14 | +$478k | +$632k (0.02) |

- The next-day profit is not the signal's. Placebos, matched on day, type, direction and bucket,
  net +$233k over the same holding days; 68% of trades are sales (59% on the cached fits), so this
  is likely the variance risk premium of a year in which implied volatility fell. The excess over
  the placebos is smaller than on the cached fits ($59k against $100k) and not significant.
- The pooled detector flags 0.65% of quotes (0.94%), and its flags still revert faster than
  unflagged ones at every horizon (−0.24 / −0.24 / −0.29, p < 0.001, 132 days); least squares
  alike. The next paragraph shows why that does not make them tradable.

**Which side moves.** The residual is x = (market IV − model IV) / s, with s the origin day's
half-spread in IV, so its slope on x splits into the slope of the market's change minus that of
the model's (`reversion.decomposed_slopes`, `reversion_parts.csv`). Only the market part is a
mispricing that can be traded. This basis differs from the reversion slopes above, which use price
residuals and each day's spread: on SPX 2022 the robust one-day flagged slope is −0.46 in prices,
−0.33 in IV and −0.21 in IV with the origin day's spread, so each part is compared with its own
total. Slopes at 1 / 3 / 5 days, SPX 2023 refits, margin 4:

| | flagged, market (p) | flagged, model | unflagged, market | flagged minus unflagged market (p) |
|---|---|---|---|---|
| robust `mix4` | −0.28 (0.17) / −0.53 (0.12) / −0.93 (0.02) | +0.01 / −0.04 / −0.57 | −0.34 / −0.59 / −0.74 | 0.89 / 0.36 / 0.72 |
| pooled robust `mix4` | −0.19 (0.13) / −0.08 (0.65) / −0.21 (0.36) | +0.14 / +0.40 / +0.41 | −0.33 / −0.70 / −1.10 | 0.39 / < 0.001 / < 0.001 |

A positive model part is the surface moving towards the quote. The prediction written before the
test run had two parts:

- *"Next-day residual reversion of flags is mostly not the market moving."* Not borne out at one
  day. Most of the flagged residuals' one-day change is the market's (robust −0.28 of −0.29, pooled
  −0.19 of −0.33), though neither part is significant. At 3 and 5 days it holds for the pooled
  detector: its flags revert mostly through the surface moving (market −0.08 of −0.48, −0.21 of
  −0.62).
- *"The flagged market-side move is not distinguishable from the unflagged one."* Holds for robust
  `mix4` at every horizon, and for the pooled detector at one day. At 3 and 5 days the pooled
  detector's flagged prices move significantly less towards the model than unflagged ones (t =
  +3.96): distinguishable, but in the direction against a tradable mispricing.
- The same-day edge over placebos (p = 0.019) and the slope test measure different things.
  Placebos are contracts with small residuals. A flag sits at least four half-spreads outside the
  bid-ask, so if it reverts by the same fraction as an unflagged quote, it still moves several times
  as many dollars. The slope test asks whether it reverts by a larger fraction, and at one day it
  does not.
- So on SPX 2023 the flags do not mark prices that return to the arbitrage-free surface faster than
  other quotes. This was the fourth look at 2023; any further change to the detector has to be
  chosen on the validation years.

**Synthetic panels.** Heston 2022 and the fresh panel are unchanged. The previous-day start won on
48 and 42 days but lowered no objective by 10%. Every figure of the Phase 7b detection table is the
same at two decimals, except robust `mix4` on the fresh panel (recall 0.45 and segment recall
0.29, against 0.46 and 0.30) and segment absorption (up to 2 points). The multi-year conditional
model remains the best detector (precision 0.997 / 0.996, recall 0.63 / 0.65).

**Compute.** The validation panels were fitted on macOS arm64, the test panels on Linux x86_64
containers on Modal (`scripts/modal_run.py`, `scripts/modal_sync.sh`) with the same Python and
locked dependencies. Six SPX 2022 days fitted both ways agree to rounding, not bit for bit. The
conditional detectors do not use `mix4`, and their per-day refinement differs between the
platforms by at most 0.034 vp a quote (4 of 7,480 one-year-model flags on Heston 2022).
`scripts/fit_panel.py` saves each day's fits as they finish, so an interrupted run, such as a
preempted container, resumes where it stopped. A run killed during the date-ordered fits and again
during the robust refits, then resumed, reproduces the uninterrupted run's tables exactly.

## Dashboard

```bash
make dashboard   # http://localhost:8501
```

The sidebar picks a panel, a detector, a flag margin and a day. Each panel opens on its frozen
detector and margin, on a busy test-year day. There are three pages:

- **Surface**: the day's fitted surface in 3D over the quoted strikes, with the quotes and flags;
  each detector's fit to the day's quotes; and one expiry's smile against the bid-ask with every
  detector's IVs (and the true IV on synthetic panels). The surface is a saved `mix4` fit, or the
  conditional model (the walk-forward checkpoint of the date's quarter), SVI or the spline fitted
  on the day's quotes.
- **Arbitrage**: where a surface breaks the butterfly (Durrleman $g < 0$) and calendar conditions,
  on the evaluation's grids over the quoted or the extended domain. The violation shares are the
  evaluation's own (`tests/test_viz.py` checks they match). The maps of `mix4` and the conditional
  model are empty by construction; those of SVI and the spline show where they fail.
- **Mispricings**: flags per day across the panel, and the day's flags ranked by residual, with
  each residual 1, 3 and 5 trading days later. Selecting a flag shows its smile and its residual
  path. On synthetic panels the injected mispricings are listed beside the flags.

The dashboard reads files that are not committed, the refitted per-quote tables and fits under
`results/` and the checkpoints under `artifacts/conditional/`, and leaves out any panel whose files
are missing. The refits start from the cached fits, so those targets run first:

| panel | per-quote tables and fits | conditional model |
|---|---|---|
| SPX 2022–2023 | `make phase7b-validation phase7b-test`, then `make warm-validation warm-test` | `make full-history full-history-test` |
| SPX 2016–2017 | `make replication-mispricing replication-warm` | `make replication` |
| Heston 2021–2022, Heston fresh 2023 | `make conditional heston-multi`, then `make phase7b-validation phase7b-test`, then `make warm-validation warm-test` | `make heston-multi` |

## Roadmap

- [x] Project scaffold
- [x] Pricing core (Black-76, Heston COS), static-arbitrage checks, synthetic chain generator
- [x] OptionsDX SPX ingestion: cleaning, put-call-parity forwards, IV re-inversion, quote audit
- [x] Baselines (SVI, SSVI, spline, unconstrained MLP) and evaluation harness
- [x] Soft-penalty arbitrage-aware network
- [x] Hard-constrained eSSVI surfaces (arbitrage-free by construction)
- [x] Cross-day conditional surface model
- [x] Mispricing detection, residual reversion tests, delta-hedged backtest
- [x] Streamlit dashboard
- [ ] Technical report
