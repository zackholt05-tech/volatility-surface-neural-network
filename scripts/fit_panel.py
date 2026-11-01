"""Fit the mispricing detector on every screened day of a panel and score every quote.

Writes results/phase7-fits/<source>/<year>.parquet (one row per quote: market quote, least-squares
and robust mix4 IVs, those of the config's further robust ``variants``, and each conditional model's
given with --checkpoint) and caches the fitted surfaces in fits.pkl, so reruns only fit missing
days. --seed-cache, when the file exists, reuses the least-squares mix4 fits of the same days saved
by scripts/forecast_conditional.py or by an earlier run of this script (fitted on the same quotes
with the same settings), which spares the cold fits.
"""

from __future__ import annotations

import argparse
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from volsurf.config import deep_merge, from_dict, load_yaml
from volsurf.eval.mispricing import RobustConfig, SurfaceState, score_day
from volsurf.eval.runner import EvalConfig, SamplingConfig, ScreenConfig, load_eval_days


def _job(args):
    date, quotes, cfg, cold, checkpoints, variants = args
    table, states, seconds = score_day(quotes, cfg, cold, checkpoints, variants=variants)
    return date, table, [(s.knot_T, s.knots, s.weights) for s in states], seconds


def _least_squares(fit):
    """A cached least-squares fit: a state, or the list of passes this script caches."""
    return fit[0] if isinstance(fit, list) else fit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase7.yaml")
    parser.add_argument("--source", default="spx")
    parser.add_argument("--years", type=int, nargs="+", required=True)
    parser.add_argument("--checkpoint", nargs="*", default=[], metavar="[NAME=]PATH",
                        help="also score conditional models (synthetic panels); NAME defaults "
                        "to cond")  # fmt: skip
    parser.add_argument("--seed-cache", help="pickle of least-squares mix4 fits by date")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-days", type=int, help="first n days only (smoke runs)")
    parser.add_argument("--out", default="results/phase7-fits")
    args = parser.parse_args()

    spec = load_yaml(args.config)
    robust = spec.get("robust", {})
    cfg = from_dict(RobustConfig, robust)
    variants = {
        name: from_dict(RobustConfig, deep_merge(robust, over))
        for name, over in (spec.get("variants") or {}).items()
    }
    checkpoints = dict(c.split("=", 1) if "=" in c else ("cond", c) for c in args.checkpoint)
    out = Path(args.out) / args.source.replace(":", "_")
    out.mkdir(parents=True, exist_ok=True)
    cache_path = out / "fits.pkl"
    fits = pickle.loads(cache_path.read_bytes()) if cache_path.exists() else {}
    seeded = args.seed_cache and Path(args.seed_cache).exists()
    seed = pickle.loads(Path(args.seed_cache).read_bytes()) if seeded else {}

    screen = EvalConfig(
        sampling=SamplingConfig(every=1, block=1), screen=ScreenConfig(drop_flagged_quotes=False)
    )
    for year in args.years:
        days, _, info = load_eval_days(args.source, [year], screen)
        dates = sorted(days)[: args.max_days]
        jobs = []
        for d in dates:
            cold = fits[d][0] if d in fits else seed.get(d)
            cold = _least_squares(cold) if cold else None
            jobs.append(
                (d, days[d], cfg, SurfaceState(*cold) if cold else None, checkpoints, variants)
            )
        start = time.perf_counter()
        tables, seconds = [], []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for date, table, states, sec in pool.map(_job, jobs):
                tables.append(table)
                fits[date] = states
                seconds.append(sec)
        cache_path.write_bytes(pickle.dumps(fits))
        panel = pd.concat(tables, ignore_index=True)
        panel.to_parquet(out / f"{year}.parquet", index=False)
        print(
            f"{args.source} {year}: {len(dates)} of {info['sampled_days']} days, {len(panel)} "
            f"quotes, {sum(job[3] is not None for job in jobs)} seeded, "
            f"median fit {pd.Series(seconds).median():.1f}s, {time.perf_counter() - start:.0f}s"
        )


if __name__ == "__main__":
    main()
