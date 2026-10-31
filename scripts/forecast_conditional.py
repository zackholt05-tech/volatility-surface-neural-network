"""Surface forecasts h trading days ahead from the conditional model's latent dynamics.

Every screened day is encoded from all of its quotes. A VAR(1) fitted on the training years'
latents forecasts z h days ahead, and the decoded surface is scored on every quote of the target
day. It is compared with two random walks: the decoded surface of today's latent, which isolates
the dynamics, and today's per-day mix4 fit evaluated at the target day's (k, T), the strongest
practical baseline. The decoded latent of the target day itself ("oracle") shows how much of the
error is representation rather than forecasting. Horizons count screened trading days.
"""

from __future__ import annotations

import argparse
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from volsurf.config import deep_merge, load_yaml
from volsurf.data.synthetic import TRADING_DAYS, SyntheticConfig
from volsurf.eval.dayset import FitData
from volsurf.eval.forecast import diebold_mariano, fit_var1, var_forecast
from volsurf.models.conditional import ConditionalNet, batch_iv
from volsurf.models.neural_essvi import NeuralESSVIConfig, NeuralESSVISurface
from volsurf.training.conditional import TrainDay, make_batch, prepare_days

MIX4 = NeuralESSVIConfig(components=4, generator="free", init_scales=(1.0, 1.8, 0.6, 1.3))
CHUNK = 16


def encode(net: ConditionalNet, days: list[TrainDay]) -> np.ndarray:
    grid = net.grid.numpy()
    out = []
    with torch.no_grad():
        for i in range(0, len(days), CHUNK):
            chunk = days[i : i + CHUNK]
            q = make_batch(chunk, [np.ones(len(d.k), bool) for d in chunk], grid, None, None)
            out.append(net.encode(q).numpy())
    return np.concatenate(out)


def decoded_errors(net: ConditionalNet, z: np.ndarray, targets: list[TrainDay]) -> np.ndarray:
    """Squared IV errors (mean per target day) and spread-normalized mean squares."""
    grid = net.grid.numpy()
    rows = []
    with torch.no_grad():
        for i in range(0, len(targets), CHUNK):
            chunk = targets[i : i + CHUNK]
            q = make_batch(chunk, [np.ones(len(d.k), bool) for d in chunk], grid, None, None)
            rates, logits = net.decoder(torch.as_tensor(z[i : i + CHUNK]))
            iv = batch_iv(rates, logits, net.grid, q).numpy()
            for j, day in enumerate(chunk):
                m = q.day.numpy() == j
                rows.append(_errors(iv[m], day))
    return np.array(rows)


def _errors(iv: np.ndarray, day: TrainDay) -> tuple[float, float]:
    return float(np.mean((iv - day.iv) ** 2)), float(np.mean(((iv - day.iv) / day.s) ** 2))


def _fit_mix4(day: TrainDay):
    torch.set_num_threads(1)
    data_T = np.unique(day.T)
    model = NeuralESSVISurface(MIX4).fit(FitData(day.k, day.T, day.iv, day.s, day.slice_id, data_T))
    return model.knot_T, model.knots, model.weights


def mix4_fits(days: list[TrainDay], cache: Path, workers: int) -> dict:
    fits = pickle.loads(cache.read_bytes()) if cache.exists() else {}
    todo = [d for d in days if d.date not in fits]
    if todo:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for day, fit in zip(todo, pool.map(_fit_mix4, todo), strict=True):
                fits[day.date] = fit
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(pickle.dumps(fits))
    return fits


def mix4_errors(fit, target: TrainDay) -> tuple[float, float]:
    model = NeuralESSVISurface(MIX4)
    model.knot_T, model.knots, model.weights = fit
    return _errors(model.iv(target.k, target.T), target)


def state_check(source: str, z_train, train_days, z_eval, eval_days, horizons) -> pd.DataFrame:
    """Synthetic positive control on the true state x = log v, which follows a known OU process.

    Reports how linearly the latent encodes x out of sample, and how much an h-step forecast of x
    itself beats the random walk, once with the generator's own dynamics (the predictability
    there is to find) and once with an AR(1) estimated on the training days, as the latent VAR is
    (what estimation from those days alone can recover)."""
    name = source.split(":")[-1]
    states = pd.read_parquet(Path("data/synthetic") / f"{name}_states.parquet")
    log_v = np.log(states.set_index("date")["v"])
    y_train = log_v.loc[[d.date for d in train_days]].to_numpy()
    y_eval = log_v.loc[[d.date for d in eval_days]].to_numpy()
    design = np.c_[np.ones(len(z_train)), z_train]
    coef, *_ = np.linalg.lstsq(design, y_train, rcond=None)
    pred = np.c_[np.ones(len(z_eval)), z_eval] @ coef
    r2 = 1.0 - np.sum((y_eval - pred) ** 2) / np.sum((y_eval - y_eval.mean()) ** 2)

    spec = load_yaml("configs/synthetic.yaml")
    state = SyntheticConfig.from_dict(deep_merge(spec["defaults"], spec["datasets"][name])).state
    mean, phi = 2.0 * np.log(state.vol_mean), 1.0 - state.kappa / TRADING_DAYS
    c, A = fit_var1(y_train[:, None], np.ones(len(y_train), dtype=bool))
    rows = []
    for h in horizons:
        o = np.arange(len(y_eval) - h)
        rw = (y_eval[o + h] - y_eval[o]) ** 2
        forecasts = {
            "true_dynamics": mean + (y_eval[o] - mean) * phi**h,
            "estimated_ar1": var_forecast(c, A, y_eval[o][:, None], h)[:, 0],
        }
        for label, f in forecasts.items():
            err = (y_eval[o + h] - f) ** 2
            stat, p = diebold_mariano(rw - err, h)
            rows.append({"h": h, "forecast": label, "mse_ratio_vs_rw": err.mean() / rw.mean(),
                         "dm_stat": stat, "p_value": p, "log_v_r2_eval": r2})  # fmt: skip
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source", default="spx")
    parser.add_argument("--train-years", type=int, nargs="+", default=[2018, 2019, 2020, 2021])
    parser.add_argument("--eval-years", type=int, nargs="+", default=[2022])
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 20])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", default="results/phase6-forecast")
    args = parser.parse_args()

    out = Path(args.out) / args.source.replace(":", "_")
    net = ConditionalNet.load(args.checkpoint)
    train_days = prepare_days(args.source, args.train_years)
    eval_days = prepare_days(args.source, args.eval_years)
    z_train, z_eval = encode(net, train_days), encode(net, eval_days)
    c, A = fit_var1(z_train, np.ones(len(z_train), dtype=bool))
    print(f"{len(train_days)} training days, {len(eval_days)} evaluation days; "
          f"VAR(1) spectral radius {max(abs(np.linalg.eigvals(A))):.3f}")  # fmt: skip
    fits = mix4_fits(eval_days, out / "mix4_fits.pkl", args.workers)

    rows, tests = [], []
    for h in args.horizons:
        origins = np.arange(len(eval_days) - h)
        targets = [eval_days[t + h] for t in origins]
        errors = {
            "var1": decoded_errors(net, var_forecast(c, A, z_eval[origins], h), targets),
            "latent_rw": decoded_errors(net, z_eval[origins], targets),
            "oracle": decoded_errors(net, z_eval[origins + h], targets),
            "mix4_rw": np.array(
                [mix4_errors(fits[eval_days[t].date], eval_days[t + h]) for t in origins]
            ),
        }
        for name, err in errors.items():
            for t, (mse, norm) in zip(origins, err, strict=True):
                rows.append({"h": h, "model": name, "origin": eval_days[t].date,
                             "mse": mse, "norm_ms": norm})  # fmt: skip
        for base in ("latent_rw", "mix4_rw"):
            model_mse, base_mse = errors["var1"][:, 0], errors[base][:, 0]
            stat, p = diebold_mariano(base_mse - model_mse, h)
            tests.append({"h": h, "baseline": base, "dm_stat": stat, "p_value": p,
                          "mse_ratio": model_mse.mean() / base_mse.mean()})  # fmt: skip

    frame = pd.DataFrame(rows)
    summary = frame.groupby(["h", "model"]).agg(
        rmse_vp=("mse", lambda x: 100.0 * np.sqrt(x.mean())),
        rms_norm=("norm_ms", lambda x: np.sqrt(x.mean())),
        days=("mse", "size"),
    )
    tests = pd.DataFrame(tests)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "errors.csv", index=False)
    summary.to_csv(out / "summary.csv")
    tests.to_csv(out / "tests.csv", index=False)
    print(summary.round(4).to_string())
    print(tests.round(4).to_string(index=False))
    if args.source.startswith("synthetic:"):
        state = state_check(args.source, z_train, train_days, z_eval, eval_days, args.horizons)
        state.to_csv(out / "state_check.csv", index=False)
        print(state.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
