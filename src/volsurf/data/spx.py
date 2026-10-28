"""End-to-end processing of OptionsDX SPX months: parse, forwards, audit, clean, store.

Each month writes three partitions: ``quotes`` (filtered OTM quotes), ``forwards`` (per-slice
parity forwards and rates) and ``days`` (per-day quality statistics). Quotes that are legs of
model-free arbitrage are kept and flagged, so later stages can choose to exclude them or the
days where they cluster.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from volsurf.arbitrage.quotes import audit_quotes
from volsurf.config import from_dict
from volsurf.data import optionsdx, store
from volsurf.data.clean import FilterConfig, build_quotes
from volsurf.data.forwards import ForwardConfig, estimate_forwards

SOURCE = "spx"


@dataclass
class SpxConfig:
    raw_dir: str = "data/raw/optionsdx"
    extract_dir: str = "data/raw/optionsdx_txt"
    years: tuple[int, ...] | None = None
    workers: int = 6
    forwards: ForwardConfig = field(default_factory=ForwardConfig)
    filters: FilterConfig = field(default_factory=FilterConfig)

    @classmethod
    def from_dict(cls, d: dict) -> SpxConfig:
        return from_dict(cls, d)


def day_quality(quotes: pd.DataFrame, forwards: pd.DataFrame, cfg: SpxConfig) -> pd.DataFrame:
    """Per-day quality statistics used to screen out unreliable snapshots."""
    in_window = forwards[forwards["dte"].between(cfg.filters.min_dte, cfg.filters.max_dte)]
    fwd = in_window.groupby("date").agg(
        expiries_listed=("expiry", "size"), forward_valid=("valid", "mean")
    )
    q = quotes.groupby("date").agg(
        quotes=("k", "size"),
        slices=("expiry", "nunique"),
        arb_vertical=("arb_vertical", "mean"),
        arb_butterfly=("arb_butterfly", "mean"),
        median_half_spread_iv=("half_spread_iv", "median"),
    )
    return fwd.join(q, how="left").reset_index()


def process_month(path: str | Path, cfg: SpxConfig, root: Path = store.DATA_ROOT) -> dict:
    """Process one monthly file and write its ``quotes``, ``forwards`` and ``days`` partitions.

    Returns a summary row: filter funnel counts, forward and audit statistics, and timing.
    """
    start = time.perf_counter()
    key = optionsdx.month_key(path)
    wide = audit_quotes(optionsdx.read_month(path))
    forwards = estimate_forwards(wide, cfg.forwards)
    quotes, funnel = build_quotes(wide, forwards, cfg.filters)

    store.write_partition(forwards, SOURCE, "forwards", key, root)
    store.write_partition(quotes, SOURCE, "quotes", key, root)
    store.write_partition(day_quality(quotes, forwards, cfg), SOURCE, "days", key, root)

    in_window = forwards["dte"].between(cfg.filters.min_dte, cfg.filters.max_dte)
    return {
        "month": key,
        **funnel,
        "days": quotes["date"].nunique(),
        "slices": quotes.groupby(["date", "expiry"]).ngroups,
        "forward_valid": float(forwards.loc[in_window, "valid"].mean()),
        "rate_fallback": float((forwards["rate_source"] != "curve").mean()),
        "arb_vertical": float(quotes["arb_vertical"].mean()),
        "arb_butterfly": float(quotes["arb_butterfly"].mean()),
        "seconds": round(time.perf_counter() - start, 2),
    }
