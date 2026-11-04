"""Fit the mispricing detector on every screened day of a panel and score every quote.

Writes results/phase7-fits/<source>/<year>.parquet (one row per quote: market quote, least-squares
and robust mix4 IVs, those of the config's further robust ``variants``, and each conditional model's
given with --checkpoint) and caches the fitted surfaces in fits.pkl. --seed-cache, when the file
exists, reuses the least-squares mix4 fits of the same days saved by scripts/forecast_conditional.py
or by an earlier run of this script (fitted on the same quotes with the same settings), which
spares the cold fits.

With --warm-start, each day's least-squares surface is also fitted from the previous day's chosen
one and the lower objective is kept (`fit_least_squares`), so the days run in date order before
the robust refits run in parallel. The previous day is the latest cached day of the output panel
within a week, so a run of the test year continues the chain of the validation year. The start
that won and both objectives are written to ls_starts_<year>.csv.

Each day's fits are saved as they finish (chain_<year>.pkl, parts_<year>/), so a run that is
interrupted, as a preempted cloud container is, resumes where it stopped and gives what an
uninterrupted run gives. They are removed once the year is written; a run with other settings
must not resume from them.
"""

from __future__ import annotations

import argparse
import multiprocessing
import pickle
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import torch

from volsurf.config import deep_merge, from_dict, load_yaml
from volsurf.eval.mispricing import (
    MIX4,
    RobustConfig,
    SurfaceState,
    day_data,
    fit_least_squares,
    ls_objective,
    score_day,
)
from volsurf.eval.runner import EvalConfig, SamplingConfig, ScreenConfig, load_eval_days
from volsurf.models.neural_essvi import NeuralESSVISurface


def _job(args):
    date, quotes, cfg, cold, checkpoints, variants = args
    table, states, seconds = score_day(quotes, cfg, cold, checkpoints, variants=variants)
    return date, table, [(s.knot_T, s.knots, s.weights) for s in states], seconds


def _least_squares(fit):
    """A cached least-squares fit: a state, or the list of passes this script caches."""
    return fit[0] if isinstance(fit, list) else fit


def _save(path: Path, obj) -> None:
    """Pickle ``obj`` to ``path`` by way of a temporary file, so an interruption never leaves a
    partly written file behind."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(pickle.dumps(obj))
    tmp.replace(path)


def _load(path: Path, default=None):
    """The pickle at ``path``, one this pipeline wrote, or ``default`` when there is none."""
    return pickle.loads(path.read_bytes()) if path.exists() else default


def _part(parts: Path, date) -> Path:
    return parts / f"{pd.Timestamp(date):%Y-%m-%d}.pkl"


def _previous(fits: dict, date, max_days: int = 7) -> SurfaceState | None:
    """The chosen least-squares fit of the latest cached day before ``date``, if within a week."""
    earlier = [d for d in fits if d < date]
    if not earlier or (date - max(earlier)).days > max_days:
        return None
    return SurfaceState(*_least_squares(fits[max(earlier)]))


def chain(days: dict, dates: list, fits: dict, seed: dict, threads: int, checkpoint: Path):
    """Least-squares fits in date order, each the better of the default start and the previous
    day's chosen fit. Returns the fits, and a frame of the start that won and both objectives.
    Each day is saved to ``checkpoint`` once fitted, and the days found there are not refitted."""
    torch.set_num_threads(threads)
    done = _load(checkpoint, {})  # date -> (chosen fit, row)
    prev = _previous(fits, dates[0])
    for d in dates:
        if d in done:
            prev = done[d][0]
            continue
        _, _, data = day_data(days[d])
        cold = SurfaceState(*_least_squares(seed[d])) if d in seed else None
        cold = cold or SurfaceState.of(NeuralESSVISurface(MIX4).fit(data))
        best, start = fit_least_squares(data, cold, prev)
        row = {"date": d, "start": start, "objective": ls_objective(best, data),
               "cold_objective": ls_objective(cold, data)}  # fmt: skip
        done[d] = (best, row)
        _save(checkpoint, done)
        prev = best
    return {d: done[d][0] for d in dates}, pd.DataFrame([done[d][1] for d in dates])


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
    parser.add_argument(
        "--warm-start",
        action="store_true",
        help="also fit each day from the previous day's fit; keep the better",
    )
    parser.add_argument("--threads", type=int, default=4, help="torch threads of that chain")
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
    fits = _load(cache_path, {})
    seed = _load(Path(args.seed_cache), {}) if args.seed_cache else {}

    screen = EvalConfig(
        sampling=SamplingConfig(every=1, block=1), screen=ScreenConfig(drop_flagged_quotes=False)
    )
    for year in args.years:
        days, _, info = load_eval_days(args.source, [year], screen)
        dates = sorted(days)[: args.max_days]
        start = time.perf_counter()
        jobs, starts = [], None
        chain_path, parts = out / f"chain_{year}.pkl", out / f"parts_{year}"
        if args.warm_start:
            chosen, starts = chain(days, dates, fits, seed, args.threads, chain_path)
            starts.to_csv(out / f"ls_starts_{year}.csv", index=False)
            print(f"{args.source} {year}: least-squares chain done, previous-day start won on "
                  f"{(starts['start'] == 'warm').sum()} of {len(starts)} days, "
                  f"{time.perf_counter() - start:.0f}s", flush=True)  # fmt: skip
        for d in dates:
            if args.warm_start:
                jobs.append((d, days[d], cfg, chosen[d], checkpoints, variants))
                continue
            cold = fits[d][0] if d in fits else seed.get(d)
            cold = _least_squares(cold) if cold else None
            jobs.append(
                (d, days[d], cfg, SurfaceState(*cold) if cold else None, checkpoints, variants)
            )
        parts.mkdir(exist_ok=True)
        todo = [job for job in jobs if not _part(parts, job[0]).exists()]
        # spawn, as on macOS: forking after the chain's threaded torch fits can hang the workers
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
            for future in as_completed([pool.submit(_job, job) for job in todo]):
                date, table, states, sec = future.result()
                _save(_part(parts, date), (table, states, sec))
        tables, seconds = [], []
        for d in dates:
            table, fits[d], sec = _load(_part(parts, d))
            tables.append(table)
            seconds.append(sec)
        _save(cache_path, fits)
        panel = pd.concat(tables, ignore_index=True)
        if starts is not None:
            panel["ls_start"] = panel["date"].map(starts.set_index("date")["start"])
        panel.to_parquet(out / f"{year}.parquet", index=False)
        shutil.rmtree(parts)
        chain_path.unlink(missing_ok=True)
        print(
            f"{args.source} {year}: {len(dates)} of {info['sampled_days']} days, {len(panel)} "
            f"quotes, {sum(job[3] is not None for job in jobs)} seeded, "
            f"median fit {pd.Series(seconds).median():.1f}s, {time.perf_counter() - start:.0f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
