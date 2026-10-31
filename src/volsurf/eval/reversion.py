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
