"""Evaluation protocol: day sampling and screening, per-day fits, aggregation.

The protocol is fixed in config and identical for every model: the same sampled days, the same
date-seeded holdout splits, the same residual floor and the same arbitrage grids.
"""

from __future__ import annotations

import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import pandas as pd

from volsurf.data import store
from volsurf.eval import metrics
from volsurf.eval.dayset import RESID_FLOOR, HoldoutConfig, date_seed, make_day
from volsurf.eval.metrics import ArbEvalConfig

THREAD_ENV = (
    "OMP_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
)


@dataclass
class SamplingConfig:
    every: int = 10  # start a block of consecutive trading days every `every` days
    block: int = 2  # consecutive days per block (pairs give day-over-day stability)
    max_days: int | None = None


@dataclass
class ScreenConfig:
    max_arb_butterfly: float = 0.10  # drop days with more quotes in butterfly violations
    min_forward_valid: float = 0.8  # drop days where fewer slices have a valid forward
    drop_flagged_quotes: bool = True  # exclude legs of model-free arbitrage from fits


@dataclass
class EvalConfig:
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    screen: ScreenConfig = field(default_factory=ScreenConfig)
    holdout: HoldoutConfig = field(default_factory=HoldoutConfig)
    arbitrage: ArbEvalConfig = field(default_factory=ArbEvalConfig)
    resid_floor: float = RESID_FLOOR


def sample_dates(dates, sampling: SamplingConfig) -> tuple[list, list[tuple]]:
    """Blocks of consecutive trading dates; returns the dates and consecutive pairs."""
    dates = sorted(pd.to_datetime(pd.Series(dates).unique()))
    chosen, pairs = [], []
    for start in range(0, len(dates), sampling.every):
        block = dates[start : start + sampling.block]
        chosen += block
        pairs += list(zip(block[:-1], block[1:], strict=True))
    if sampling.max_days:
        chosen = chosen[: sampling.max_days]
        pairs = [p for p in pairs if p[1] in set(chosen)]
    return chosen, pairs


def load_eval_days(source: str, years, cfg: EvalConfig) -> tuple[dict, list[tuple], dict]:
    """Quotes per sampled, screened date, plus consecutive pairs and screening counts."""
    quotes = store.load_quotes(source, years=list(years))
    dates, pairs = sample_dates(quotes["date"], cfg.sampling)
    info = {"sampled_days": len(dates)}

    if not source.startswith("synthetic:"):
        days = store.load_table(source, "days", years=list(years))
        bad = days[
            (days["arb_butterfly"] > cfg.screen.max_arb_butterfly)
            | (days["forward_valid"] < cfg.screen.min_forward_valid)
        ]["date"]
        dates = [d for d in dates if d not in set(bad)]
        if cfg.screen.drop_flagged_quotes and "arb_vertical" in quotes:
            quotes = quotes[~(quotes["arb_vertical"] | quotes["arb_butterfly"])]
    info["screened_days"] = len(dates)
    kept = set(dates)
    pairs = [p for p in pairs if p[0] in kept and p[1] in kept]

    quotes = quotes[quotes["date"].isin(kept)]
    return {d: g for d, g in quotes.groupby("date")}, pairs, info


def _init_worker() -> None:
    import torch

    torch.set_num_threads(1)


def _evaluate(model, day, cfg: EvalConfig, row: dict, init=None) -> dict:
    start = time.perf_counter()
    model.fit(day.fit_data()) if init is None else model.fit(day.fit_data(), init=init)
    row["fit_seconds"] = time.perf_counter() - start
    row["warm_start"] = init is not None
    row["n_failed"] = model.n_failed
    row["n_quotes"] = len(day.k)
    row["n_slices"] = len(day.slice_T)
    for name, value in getattr(model, "penalties", {}).items():
        row[f"loss/{name}"] = value

    iv = model.iv(day.k, day.T)
    for name, mask in (("fit", day.fit), ("interp", day.interp), ("wing", day.wing)):
        row |= metrics.quote_metrics(day, iv, mask, name)
    row |= metrics.bucket_metrics(day, iv, day.fit)
    bounds = day.slice_bounds()
    arb = cfg.arbitrage
    row |= metrics.arbitrage_metrics(model, bounds, arb, 0.0, "arb_data")
    row |= metrics.arbitrage_metrics(model, bounds, arb, arb.widen, "arb_ext")
    row["_probe"] = metrics.probe(model, bounds)
    return row


def evaluate_block(spec: dict, block: list[pd.DataFrame], cfg: EvalConfig) -> list[dict]:
    """Fit one model on a block of consecutive days, in order.

    With ``warm_start: true`` in the spec, each day after the first is initialized from the
    previous day's fitted model, as in daily production refits. Seeds are derived from the date
    plus an optional ``seed_offset``, so configs are compared on paired seeds.
    """
    from volsurf.models.registry import build_model

    rows, previous = [], None
    for quotes in block:
        date = quotes["date"].iloc[0]
        row = {"model": spec["name"], "date": date}
        try:
            day = make_day(quotes, cfg.holdout, cfg.resid_floor)
            model = build_model(spec, seed=date_seed(date) + spec.get("seed_offset", 0))
            init = previous if spec.get("warm_start") else None
            rows.append(_evaluate(model, day, cfg, row, init))
            previous = model
        except Exception:  # one bad day must not stop the run
            row["error"] = traceback.format_exc(limit=3)
            rows.append(row)
            previous = None
    return rows


def blocks_from_pairs(dates, pairs: list[tuple]) -> list[list]:
    """Group sampled dates into runs of consecutive days linked by ``pairs``."""
    linked = {d1: d0 for d0, d1 in pairs}
    blocks: list[list] = []
    for d in sorted(dates):
        if d in linked and blocks and blocks[-1][-1] == linked[d]:
            blocks[-1].append(d)
        else:
            blocks.append([d])
    return blocks


def run(specs: list[dict], days: dict, pairs: list[tuple], cfg: EvalConfig, workers: int):
    """Evaluate every (model, block of days); returns per-day metrics and day-over-day change."""
    for var in THREAD_ENV:
        os.environ[var] = "1"
    blocks = [[days[d] for d in b] for b in blocks_from_pairs(days.keys(), pairs)]
    task_specs = [spec for spec in specs for _ in blocks]
    task_blocks = [b for _ in specs for b in blocks]
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
        results = pool.map(evaluate_block, task_specs, task_blocks, [cfg] * len(task_specs))
        rows = [row for block_rows in results for row in block_rows]

    daily = pd.DataFrame(rows)
    probes = daily.pop("_probe") if "_probe" in daily else pd.Series(dtype=object)
    stability = []
    for name, idx in daily.groupby("model").groups.items():
        p = {daily.at[i, "date"]: probes[i] for i in idx if isinstance(probes.get(i), tuple)}
        dod = metrics.day_over_day(p, pairs)
        stability.append(
            pd.DataFrame({"model": name, "date": dod.index, "dod_rmse_vp": dod.values})
        )
    stability = pd.concat(stability, ignore_index=True) if stability else pd.DataFrame()
    return daily, stability


def summarize(daily: pd.DataFrame, stability: pd.DataFrame) -> pd.DataFrame:
    """Mean of each metric per model, with error counts and median day-over-day change."""
    ok = daily[daily["error"].isna()] if "error" in daily else daily
    numeric = ok.drop(columns=["date"]).groupby("model").mean(numeric_only=True)
    numeric["days"] = ok.groupby("model").size()
    numeric["errors"] = daily.groupby("model").size() - numeric["days"]
    if len(stability):
        numeric["dod_rmse_vp"] = stability.groupby("model")["dod_rmse_vp"].median()
    return numeric


def flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for key, value in d.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out |= flatten(value, f"{name}.")
        else:
            out[name] = value if not isinstance(value, list | tuple) else str(list(value))
    return out


def key_columns(summary: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "fit/rmse_vp", "fit/within_spread", "interp/rmse_vp", "wing/rmse_vp",
        "interp/truth_rmse_vp", "dte_7_30/rmse_vp",
        "arb_data/calendar_frac", "arb_data/calendar_frac_material", "arb_data/calendar_max_vp",
        "arb_data/butterfly_frac_quoted",
        "arb_data/butterfly_frac_between", "arb_ext/calendar_frac", "arb_ext/butterfly_frac",
        "dod_rmse_vp", "fit_seconds", "n_failed", "days", "errors",
    ]  # fmt: skip
    return summary[[c for c in cols if c in summary]].astype(float).round(4)
