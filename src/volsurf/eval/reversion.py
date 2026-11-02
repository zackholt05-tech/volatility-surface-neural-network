"""Do flagged residuals revert? Day-by-day regressions of residual changes on residuals.

For each origin day t and horizon h, contracts quoted on both t and t + h (same expiry, strike and
type; h counts screened trading days) give a slope of r_{t+h} - r_t on r_t, separately for flagged
and unflagged quotes. A slope of -1 is full reversion within h days and 0 none. Quote noise makes
residuals revert mechanically even where nothing is mispriced, so unflagged quotes are the control.
Inference is Fama-MacBeth: one slope per day, with Newey-West standard errors on the daily series,
because pooled regressions over thousands of quotes a day would overstate significance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from volsurf.eval.forecast import diebold_mariano

KEYS = ["expiry", "strike", "option_type"]


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    x = x - x.mean()
    return float(x @ (y - y.mean()) / (x @ x)) if x @ x > 0 else np.nan


def daily_slopes(
    table: pd.DataFrame, r: np.ndarray, flags: np.ndarray, h: int, min_n: int = 5
) -> pd.DataFrame:
    """One row per origin day: slopes and counts for flagged and unflagged quotes."""
    dates = np.sort(table["date"].unique())
    t = np.searchsorted(dates, table["date"].to_numpy())
    now = table[KEYS].assign(t=t, r=np.asarray(r), flagged=np.asarray(flags) != 0)
    later = now[KEYS + ["t", "r"]].assign(t=now["t"] - h).rename(columns={"r": "r_h"})
    pairs = now.merge(later, on=KEYS + ["t"])
    pairs["dr"] = pairs["r_h"] - pairs["r"]

    rows = []
    for day, g in pairs.groupby("t"):
        row = {"date": dates[day]}
        for name, sel in (("flagged", g["flagged"]), ("unflagged", ~g["flagged"])):
            x, y = g.loc[sel, "r"].to_numpy(), g.loc[sel, "dr"].to_numpy()
            row[f"n_{name}"] = len(x)
            row[f"slope_{name}"] = _slope(x, y) if len(x) >= min_n else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_slopes(daily: pd.DataFrame, h: int) -> dict:
    """Mean daily slopes, the implied half-life of flagged residuals, and a Newey-West test that
    flagged quotes revert faster than unflagged ones (negative difference)."""
    both = daily.dropna(subset=["slope_flagged", "slope_unflagged"])
    diff = (both["slope_flagged"] - both["slope_unflagged"]).to_numpy()
    stat, p = diebold_mariano(diff, h) if len(diff) > h + 2 else (np.nan, np.nan)
    flagged = float(daily["slope_flagged"].mean())
    persistence = (1.0 + flagged) ** (1.0 / h) if -1.0 < flagged < 0.0 else np.nan
    return {
        "h": h,
        "days": len(both),
        "flagged_per_day": float(daily["n_flagged"].mean()),
        "slope_flagged": flagged,
        "slope_unflagged": float(daily["slope_unflagged"].mean()),
        "difference": float(diff.mean()) if len(diff) else np.nan,
        "t_stat": stat,
        "p_value": p,
        "half_life_days": float(np.log(0.5) / np.log(persistence))
        if np.isfinite(persistence)
        else np.nan,
    }


def decomposed_slopes(
    table: pd.DataFrame, iv_col: str, flags: np.ndarray, h: int, min_n: int = 5
) -> pd.DataFrame:
    """Which side moves when a residual reverts: the market quote or the fitted surface.

    The residual x_t = (market IV - model IV) / s_t changes by (market change - model change) /
    s_t, so its daily slope on x_t is the slope of the market part minus that of the model part.
    A model refitted each day has errors of its own that revert as it is refitted; selecting large
    residuals selects them, and the unflagged control does not remove them. Only the market part
    is a mispricing that can be traded. One row per origin day, flagged and unflagged quotes apart.
    """
    dates = np.sort(table["date"].unique())
    t = np.searchsorted(dates, table["date"].to_numpy())
    now = table[KEYS].assign(
        t=t, mkt=table["iv_mid"].to_numpy(), mod=table[iv_col].to_numpy(),
        s=table["s"].to_numpy(), flagged=np.asarray(flags) != 0,
    )  # fmt: skip
    later = now[KEYS + ["t", "mkt", "mod"]].assign(t=now["t"] - h)
    pairs = now.merge(later, on=KEYS + ["t"], suffixes=("", "_h"))
    x = ((pairs["mkt"] - pairs["mod"]) / pairs["s"]).to_numpy()
    d_mkt = ((pairs["mkt_h"] - pairs["mkt"]) / pairs["s"]).to_numpy()
    d_mod = ((pairs["mod_h"] - pairs["mod"]) / pairs["s"]).to_numpy()

    flagged = pairs["flagged"].to_numpy()
    rows = []
    for day, idx in pd.Series(np.arange(len(pairs))).groupby(pairs["t"].to_numpy()):
        i = idx.to_numpy()
        row = {"date": dates[day]}
        for group, j in (("flagged", i[flagged[i]]), ("unflagged", i[~flagged[i]])):
            ok = len(j) >= min_n
            row[f"n_{group}"] = len(j)
            row[f"market_{group}"] = _slope(x[j], d_mkt[j]) if ok else np.nan
            row[f"model_{group}"] = _slope(x[j], d_mod[j]) if ok else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_decomposed(daily: pd.DataFrame, h: int) -> dict:
    """Mean daily slopes of each part, and Newey-West tests that flagged quotes' market prices
    move towards the model (negative market slope) and more than unflagged ones."""
    out = {"h": h}
    for col in ("market_flagged", "model_flagged", "market_unflagged", "model_unflagged"):
        out[col] = float(daily[col].mean())
    flagged = daily["market_flagged"].dropna().to_numpy()
    stat, p = diebold_mariano(flagged, h) if len(flagged) > h + 2 else (np.nan, np.nan)
    out |= {"days": len(flagged), "market_flagged_t": stat, "market_flagged_p": p}
    both = daily.dropna(subset=["market_flagged", "market_unflagged"])
    diff = (both["market_flagged"] - both["market_unflagged"]).to_numpy()
    stat, p = diebold_mariano(diff, h) if len(diff) > h + 2 else (np.nan, np.nan)
    return out | {"market_difference_t": stat, "market_difference_p": p}
