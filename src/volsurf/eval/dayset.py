"""One quote date prepared for fitting and evaluation, with deterministic holdout splits.

Every model sees exactly the same split for a given date: the random interior holdout is seeded
from the date itself, not from a loop-level generator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Floor on the IV half-spread used to normalize residuals. Shared by fitting losses, evaluation
# metrics and (later) mispricing scores so "residual in half-spreads" means the same everywhere.
RESID_FLOOR = 0.001


@dataclass
class HoldoutConfig:
    wing_frac: float = 0.1  # outermost share of strikes held out on each side of every slice
    interior_frac: float = 0.15  # random share of the remaining strikes held out
    min_slice: int = 10  # slices with fewer quotes are used for fitting only
    slice_frac: float = 0.0  # share of inner maturities held out whole (maturity gaps)


@dataclass
class FitData:
    """Arrays a surface model is fitted on. ``s`` is the floored IV half-spread."""

    k: np.ndarray
    T: np.ndarray
    iv: np.ndarray
    s: np.ndarray
    slice_id: np.ndarray  # 0..n_slices-1, ordered by maturity
    slice_T: np.ndarray  # maturity of each slice

    @property
    def w(self) -> np.ndarray:
        return self.iv**2 * self.T


@dataclass
class Day:
    date: pd.Timestamp
    quotes: pd.DataFrame  # sorted by (T, k); index aligned with the arrays below
    k: np.ndarray
    T: np.ndarray
    iv: np.ndarray
    iv_bid: np.ndarray
    iv_ask: np.ndarray
    s: np.ndarray
    z: np.ndarray  # k / (ATM iv sqrt T), ATM iv interpolated from the market per slice
    dte: np.ndarray
    slice_id: np.ndarray
    slice_T: np.ndarray
    fit: np.ndarray  # boolean masks partitioning the quotes
    interp: np.ndarray
    wing: np.ndarray
    iv_true: np.ndarray | None = None  # synthetic ground truth when available

    def fit_data(self, mask: np.ndarray | None = None) -> FitData:
        m = self.fit if mask is None else mask
        return FitData(self.k[m], self.T[m], self.iv[m], self.s[m], self.slice_id[m], self.slice_T)

    def slice_bounds(self) -> pd.DataFrame:
        """Quoted log-moneyness range of each slice (all quotes, holdouts included)."""
        frame = pd.DataFrame({"slice": self.slice_id, "k": self.k})
        bounds = frame.groupby("slice")["k"].agg(["min", "max"])
        bounds["T"] = self.slice_T[bounds.index]
        return bounds.reset_index(drop=True)


def date_seed(date) -> int:
    return int(pd.Timestamp(date).strftime("%Y%m%d"))


def _holdout_masks(slice_id: np.ndarray, cfg: HoldoutConfig, rng: np.random.Generator):
    n = len(slice_id)
    wing = np.zeros(n, dtype=bool)
    interp = np.zeros(n, dtype=bool)
    starts = np.flatnonzero(np.r_[True, slice_id[1:] != slice_id[:-1]])
    ends = np.r_[starts[1:], n]
    for a, b in zip(starts, ends, strict=True):
        size = b - a
        if size < cfg.min_slice:
            continue
        n_wing = max(1, int(round(cfg.wing_frac * size)))
        wing[a : a + n_wing] = True
        wing[b - n_wing : b] = True
        inner = np.arange(a + n_wing, b - n_wing)
        n_int = int(round(cfg.interior_frac * len(inner)))
        interp[rng.choice(inner, size=n_int, replace=False)] = True
    if cfg.slice_frac > 0:  # drawn last, so the other holdouts match runs without slice gaps
        inner_slices = np.arange(1, len(starts) - 1)  # interpolation in T, not extrapolation
        n_gap = int(round(cfg.slice_frac * len(inner_slices)))
        gaps = rng.choice(inner_slices, size=n_gap, replace=False)
        interp |= np.isin(slice_id, gaps) & ~wing
    return ~(wing | interp), interp, wing


def make_day(
    quotes: pd.DataFrame,
    holdout: HoldoutConfig | None = None,
    resid_floor: float = RESID_FLOOR,
) -> Day:
    """Build a ``Day`` from one date's quotes (core columns; ``iv_true`` used if present)."""
    holdout = holdout or HoldoutConfig()
    q = quotes.sort_values(["T", "k"]).reset_index(drop=True)
    date = q["date"].iloc[0]
    slice_T, slice_id = np.unique(q["T"].to_numpy(), return_inverse=True)

    atm = (
        q.groupby(slice_id)
        .apply(lambda g: np.interp(0.0, g["k"], g["iv_mid"]), include_groups=False)
        .to_numpy()
    )
    z = q["k"].to_numpy() / (atm[slice_id] * np.sqrt(q["T"].to_numpy()))

    fit, interp, wing = _holdout_masks(slice_id, holdout, np.random.default_rng(date_seed(date)))
    return Day(
        date=date,
        quotes=q,
        k=q["k"].to_numpy(),
        T=q["T"].to_numpy(),
        iv=q["iv_mid"].to_numpy(),
        iv_bid=q["iv_bid"].to_numpy(),
        iv_ask=q["iv_ask"].to_numpy(),
        s=np.maximum(q["half_spread_iv"].to_numpy(), resid_floor),
        z=z,
        dte=q["dte"].to_numpy(),
        slice_id=slice_id,
        slice_T=slice_T,
        fit=fit,
        interp=interp,
        wing=wing,
        iv_true=q["iv_true"].to_numpy() if "iv_true" in q else None,
    )
