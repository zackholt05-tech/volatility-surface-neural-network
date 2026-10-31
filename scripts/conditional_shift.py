"""Where the conditional model stops generalizing: quarterly errors against the market state.

For a sample of days per quarter it reports the amortized and latent-refined errors on every quote,
the error after thinning the context's short-dated expiries back to the training years' count (a
change in chain composition), and the error of a free per-day mix4 fit, before and after it is
moved onto the decoder's maturity grid (whether the grid can express the surface at all). The
market state is summarized by the ATM skew, dIV/dk over |k| < 0.05.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from volsurf.eval.dayset import FitData
from volsurf.models.conditional import batch_iv, load_cached, per_day_loss
from volsurf.models.essvi import interp_knots, rates_from_knots
from volsurf.models.neural_essvi import NeuralESSVIConfig, NeuralESSVISurface, _minimize_lbfgs
from volsurf.training.conditional import TrainDay, make_batch, prepare_days

MIX4 = NeuralESSVIConfig(components=4, generator="free", init_scales=(1.0, 1.8, 0.6, 1.3))
SHORT_T, SHORT_KEEP = 30 / 365, 10  # training years had about ten expiries under 30 days


def _rmse(iv, target) -> float:
    return 100.0 * float(np.sqrt(np.mean((np.asarray(iv) - np.asarray(target)) ** 2)))


def _day_row(args) -> dict:
    checkpoint, day, seed = args
    torch.set_num_threads(1)
    net = load_cached(checkpoint)
    grid = net.grid
    q = make_batch([day], [np.ones(len(day.k), bool)], grid.numpy(), None, None)
    T = np.unique(day.T)
    short = np.flatnonzero(T < SHORT_T)
    rng = np.random.default_rng(seed)
    drop = rng.choice(short, max(len(short) - SHORT_KEEP, 0), replace=False)
    thin = make_batch([day], [~np.isin(day.slice_id, drop)], grid.numpy(), None, None)
    with torch.no_grad():
        z, z_thin = net.encode(q), net.encode(thin)
        amortized, thinned = _rmse(net.iv(z, q), day.iv), _rmse(net.iv(z_thin, q), day.iv)
    z = z.clone().requires_grad_(True)
    _minimize_lbfgs([z], lambda: per_day_loss(net.iv(z, q), q)[0], 100)
    with torch.no_grad():
        refined = _rmse(net.iv(z, q), day.iv)

    fit = NeuralESSVISurface(MIX4).fit(FitData(day.k, day.T, day.iv, day.s, day.slice_id, T))
    g = grid.numpy()
    rates = torch.as_tensor(
        np.stack([rates_from_knots(g, *interp_knots(g, fit.knot_T, *kn)) for kn in fit.knots])[None]
    ).requires_grad_(True)
    logits = torch.log(torch.as_tensor(fit.weights[None])).requires_grad_(True)
    _minimize_lbfgs(
        [rates, logits], lambda: per_day_loss(batch_iv(rates, logits, grid, q), q)[0], 300
    )
    with torch.no_grad():
        on_grid = _rmse(batch_iv(rates, logits, grid, q), day.iv)

    atm = np.abs(day.k) < 0.05
    return {
        "date": day.date,
        "quarter": f"{day.date.year}Q{(day.date.month - 1) // 3 + 1}",
        "atm_skew": np.polyfit(day.k[atm], day.iv[atm], 1)[0],
        "short_expiries": len(short),
        "amortized_vp": amortized,
        "thinned_context_vp": thinned,
        "refined_vp": refined,
        "mix4_free_vp": _rmse(fit.iv(day.k, day.T), day.iv),
        "mix4_on_grid_vp": on_grid,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="artifacts/conditional/spx_d8.pt")
    parser.add_argument("--years", type=int, nargs="+", default=[2019, 2021, 2022])
    parser.add_argument("--every", type=int, default=8, help="take every n-th day of each quarter")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", default="results/phase6-shift/spx")
    args = parser.parse_args()

    days: list[TrainDay] = []
    for year in args.years:
        by_quarter: dict[int, list[TrainDay]] = {}
        for d in prepare_days("spx", [year]):
            by_quarter.setdefault((d.date.month - 1) // 3, []).append(d)
        days += [d for quarter in sorted(by_quarter) for d in by_quarter[quarter][:: args.every]]
    jobs = [(args.checkpoint, d, i) for i, d in enumerate(days)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        frame = pd.DataFrame(list(pool.map(_day_row, jobs)))

    summary = frame.drop(columns="date").groupby("quarter").mean()
    summary.insert(0, "days", frame.groupby("quarter").size())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "days.csv", index=False)
    summary.to_csv(out / "quarterly.csv")
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
