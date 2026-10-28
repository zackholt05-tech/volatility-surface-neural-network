"""Forward prices and discount factors implied by put-call parity.

For a European call C and put P with the same strike and expiry, C - P = D (F - K). Regressing
mid(C - P) on K over near-the-money strikes gives D (minus the slope) and F (intercept / D),
so no external rate or dividend data is needed.

The slope is weakly identified at short maturities (D ~ 1 - rT), so estimation runs in three
stages per quote date:

1. rate:    weighted regression per expiry with T >= ``min_T_rate``; r = -log(D) / T,
2. curve:   robust kernel smoothing of r across maturities, flat below ``min_T_rate``,
3. forward: D = exp(-r(T) T) for every expiry, then F = weighted median of K + (C - P) / D.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from volsurf.data.optionsdx import two_sided

SLICE = ["date", "expiry"]


@dataclass
class ForwardConfig:
    min_T_rate: float = 30 / 365  # shortest maturity used to estimate rates
    n_pairs: int = 20  # call/put pairs nearest the forward used per expiry
    min_pairs: int = 5
    rate_bounds: tuple[float, float] = (-0.01, 0.10)
    bandwidth: float = 0.5  # kernel bandwidth of the rate curve, in log T
    outlier_mads: float = 4.0  # rate estimates further than this from the date median are dropped
    fallback_rate: float = 0.02
    max_dispersion: float = 0.002  # max relative MAD of per-strike forward estimates


def parity_pairs(wide: pd.DataFrame, n_pairs: int) -> pd.DataFrame:
    """Strikes with two-sided call and put quotes, restricted to the ``n_pairs`` nearest the
    forward (smallest |C - P|) in each slice."""
    ok = two_sided(wide["c_bid"], wide["c_ask"]) & two_sided(wide["p_bid"], wide["p_ask"])
    w = wide.loc[ok]
    pairs = w[SLICE + ["T", "strike"]].copy()
    pairs["y"] = 0.5 * (w["c_bid"] + w["c_ask"]) - 0.5 * (w["p_bid"] + w["p_ask"])
    pairs["w"] = 1.0 / ((w["c_ask"] - w["c_bid"]) + (w["p_ask"] - w["p_bid"])) ** 2
    rank = pairs["y"].abs().groupby([pairs["date"], pairs["expiry"]]).rank(method="first")
    return pairs[rank <= n_pairs]


def _slice_regression(pairs: pd.DataFrame) -> pd.DataFrame:
    """Weighted least squares of y on K per slice, via grouped sufficient statistics."""
    p = pairs.assign(
        wk=pairs["w"] * pairs["strike"],
        wy=pairs["w"] * pairs["y"],
        wkk=pairs["w"] * pairs["strike"] ** 2,
        wky=pairs["w"] * pairs["strike"] * pairs["y"],
    )
    s = p.groupby(SLICE).agg(
        T=("T", "first"),
        n=("w", "size"),
        sw=("w", "sum"),
        swk=("wk", "sum"),
        swy=("wy", "sum"),
        swkk=("wkk", "sum"),
        swky=("wky", "sum"),
    )
    det = s["sw"] * s["swkk"] - s["swk"] ** 2
    slope = (s["sw"] * s["swky"] - s["swk"] * s["swy"]) / det
    D = -slope
    with np.errstate(invalid="ignore", divide="ignore"):
        s["r_raw"] = -np.log(D) / s["T"]
    return s[["T", "n", "r_raw"]]


def _rate_curves(reg: pd.DataFrame, cfg: ForwardConfig) -> pd.DataFrame:
    """Smoothed r(T) per slice, with the source of each rate."""
    lo, hi = cfg.rate_bounds
    usable = (
        (reg["T"] >= cfg.min_T_rate) & (reg["n"] >= cfg.min_pairs) & reg["r_raw"].between(lo, hi)
    )
    out = reg.assign(r=np.nan, rate_source="fallback")
    prev_level = None
    for _, day in reg.groupby(level="date", sort=True):
        pts = day[usable.loc[day.index]]
        if len(pts) >= 2:
            med = pts["r_raw"].median()
            mad = max((pts["r_raw"] - med).abs().median(), 5e-4)
            pts = pts[(pts["r_raw"] - med).abs() <= cfg.outlier_mads * mad]
        if len(pts) == 0:
            level = prev_level if prev_level is not None else cfg.fallback_rate
            out.loc[day.index, "r"] = level
            out.loc[day.index, "rate_source"] = (
                "prev_date" if prev_level is not None else "fallback"
            )
            continue
        x = np.log(pts["T"].to_numpy())
        wts = pts["T"].to_numpy() ** 2  # error in r scales like 1/T
        r_pts = pts["r_raw"].to_numpy()
        x_eval = np.log(np.maximum(day["T"].to_numpy(), cfg.min_T_rate))
        kern = np.exp(-0.5 * ((x_eval[:, None] - x[None, :]) / cfg.bandwidth) ** 2) * wts
        out.loc[day.index, "r"] = (kern @ r_pts) / kern.sum(axis=1)
        out.loc[day.index, "rate_source"] = "curve"
        prev_level = float(np.average(r_pts, weights=wts))
    return out


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    cw = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cw, 0.5 * cw[-1])])


def estimate_forwards(wide: pd.DataFrame, cfg: ForwardConfig | None = None) -> pd.DataFrame:
    """Per-slice forward table: T, n_pairs, r_raw, r, D, F, dispersion, rate_source, valid."""
    cfg = cfg or ForwardConfig()
    pairs = parity_pairs(wide, cfg.n_pairs)
    curves = _rate_curves(_slice_regression(pairs), cfg)

    pairs = pairs.join(curves[["r"]], on=SLICE)
    pairs["D"] = np.exp(-pairs["r"] * pairs["T"])
    pairs["F_i"] = pairs["strike"] + pairs["y"] / pairs["D"]

    rows = []
    for (date, expiry), g in pairs.groupby(SLICE, sort=False):
        f_i, w = g["F_i"].to_numpy(), g["w"].to_numpy()
        F = _weighted_median(f_i, w)
        rows.append((date, expiry, F, _weighted_median(np.abs(f_i - F), w) / F))
    fwd = pd.DataFrame(rows, columns=SLICE + ["F", "dispersion"]).set_index(SLICE)

    out = curves.join(fwd).rename(columns={"n": "n_pairs"})
    out["D"] = np.exp(-out["r"] * out["T"])
    out["valid"] = (out["n_pairs"] >= cfg.min_pairs) & (out["dispersion"] <= cfg.max_dispersion)
    meta = wide.groupby(SLICE)[["spot", "dte", "is_third_friday"]].first()
    return out.join(meta).reset_index()
