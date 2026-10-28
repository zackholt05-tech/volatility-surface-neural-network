"""Evaluation metrics shared by every surface model.

* quote metrics: IV errors in vol points, spread-normalized RMS, share inside bid/ask, on the fit
  set and on the interior and wing holdouts, plus breakdowns by maturity and moneyness;
* arbitrage metrics: static-arbitrage checks of the fitted surface on dense grids over the data
  domain (quoted k-range per maturity) and an extended domain (wings);
* stability probes: fitted IVs on a fixed (k, T) grid, compared across consecutive days.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from volsurf.arbitrage.conditions import durrleman_g, fd_slice_derivatives
from volsurf.eval.dayset import Day

DTE_BUCKETS = [(7, 30), (31, 90), (91, 365), (366, 730)]
Z_BUCKETS = [(-np.inf, -2.0), (-2.0, -0.5), (-0.5, 0.5), (0.5, np.inf)]
PROBE_T = np.array([30, 60, 91, 182, 365]) / 365.0
PROBE_K = np.linspace(-0.3, 0.1, 9)


def quote_metrics(day: Day, iv_model: np.ndarray, mask: np.ndarray, prefix: str) -> dict:
    if not mask.any():
        return {}
    err = iv_model[mask] - day.iv[mask]
    inside = (iv_model[mask] >= day.iv_bid[mask]) & (iv_model[mask] <= day.iv_ask[mask])
    out = {
        f"{prefix}/rmse_vp": 100.0 * np.sqrt(np.mean(err**2)),
        f"{prefix}/mae_vp": 100.0 * np.mean(np.abs(err)),
        f"{prefix}/rms_norm": float(np.sqrt(np.mean((err / day.s[mask]) ** 2))),
        f"{prefix}/within_spread": float(np.mean(inside)),
    }
    if day.iv_true is not None:
        truth = iv_model[mask] - day.iv_true[mask]
        out[f"{prefix}/truth_rmse_vp"] = 100.0 * np.sqrt(np.mean(truth**2))
    return out


def bucket_metrics(day: Day, iv_model: np.ndarray, mask: np.ndarray) -> dict:
    out = {}
    err = iv_model - day.iv
    for lo, hi in DTE_BUCKETS:
        m = mask & (day.dte >= lo) & (day.dte <= hi)
        if m.any():
            out[f"dte_{lo}_{hi}/rmse_vp"] = 100.0 * np.sqrt(np.mean(err[m] ** 2))
    for lo, hi in Z_BUCKETS:
        m = mask & (day.z > lo) & (day.z <= hi)
        if m.any():
            out[f"z_{lo:g}_{hi:g}/rmse_vp"] = 100.0 * np.sqrt(np.mean(err[m] ** 2))
    return out


@dataclass
class ArbEvalConfig:
    n_T: int = 40  # maturities on the evaluation grid (log-spaced over the quoted range)
    n_k: int = 101  # points per maturity for the butterfly check
    n_k_cal: int = 801  # common k-grid for the calendar check
    widen: float = (
        0.5  # extended domain: quoted k-range widened by this share of its width per side
    )
    tol_g: float = 1e-5  # Durrleman g below -tol_g is a butterfly violation
    tol_cal: float = 1e-7  # total-variance decrease beyond this is a calendar violation


def domain(bounds: pd.DataFrame, T: np.ndarray, widen: float = 0.0):
    """Quoted k-range interpolated in T, optionally widened on each side."""
    lo = np.interp(T, bounds["T"], bounds["min"])
    hi = np.interp(T, bounds["T"], bounds["max"])
    width = hi - lo
    return lo - widen * width, hi + widen * width


def arbitrage_metrics(model, bounds: pd.DataFrame, cfg: ArbEvalConfig, widen: float, prefix: str):
    T_grid = np.geomspace(bounds["T"].min(), bounds["T"].max(), cfg.n_T)
    lo, hi = domain(bounds, T_grid, widen)

    # Butterfly: a uniform k-grid per maturity spanning that maturity's domain.
    frac = np.linspace(0.0, 1.0, cfg.n_k)
    K = lo[:, None] + frac[None, :] * (hi - lo)[:, None]
    W = model.w(K, np.broadcast_to(T_grid[:, None], K.shape))
    g = np.concatenate([durrleman_g(*fd_slice_derivatives(W[i], K[i])) for i in range(cfg.n_T)])
    h = (hi - lo) / (cfg.n_k - 1)
    slope_lo = np.abs(-3 * W[:, 0] + 4 * W[:, 1] - W[:, 2]) / (2 * h)
    slope_hi = np.abs(3 * W[:, -1] - 4 * W[:, -2] + W[:, -3]) / (2 * h)

    # Calendar: a common k-grid, compared between consecutive maturities inside both domains.
    kc = np.linspace(lo.min(), hi.max(), cfg.n_k_cal)
    Wc = model.w(
        np.broadcast_to(kc, (cfg.n_T, kc.size)),
        np.broadcast_to(T_grid[:, None], (cfg.n_T, kc.size)),
    )
    inside = (kc[None, :] >= lo[:, None]) & (kc[None, :] <= hi[:, None])
    both = inside[:-1] & inside[1:]
    dW = np.diff(Wc, axis=0)[both]

    return {
        f"{prefix}/calendar_frac": float(np.mean(dW < -cfg.tol_cal)) if dW.size else 0.0,
        f"{prefix}/calendar_max": float(max(0.0, -dW.min())) if dW.size else 0.0,
        f"{prefix}/butterfly_frac": float(np.mean(g < -cfg.tol_g)),
        f"{prefix}/butterfly_min_g": float(g.min()),
        f"{prefix}/positivity_frac": float(np.mean(W <= 0.0)),
        f"{prefix}/wing_max_slope": float(np.max(np.r_[slope_lo, slope_hi])),
    }


def probe(model, bounds: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Fitted IVs on the fixed stability grid and the mask of points inside the data domain."""
    lo, hi = domain(bounds, PROBE_T)
    inside = (PROBE_K[None, :] >= lo[:, None]) & (PROBE_K[None, :] <= hi[:, None])
    inside &= (PROBE_T >= bounds["T"].min())[:, None] & (PROBE_T <= bounds["T"].max())[:, None]
    iv = model.iv(
        np.broadcast_to(PROBE_K, inside.shape), np.broadcast_to(PROBE_T[:, None], inside.shape)
    )
    return iv, inside


def day_over_day(probes: dict, pairs: list[tuple]) -> pd.Series:
    """RMS change (vol points) of fitted IV on the probe grid between consecutive sampled days."""
    out = {}
    for d0, d1 in pairs:
        if d0 in probes and d1 in probes:
            (iv0, m0), (iv1, m1) = probes[d0], probes[d1]
            m = m0 & m1
            if m.any():
                out[d1] = 100.0 * np.sqrt(np.mean((iv1[m] - iv0[m]) ** 2))
    return pd.Series(out, dtype=float)
