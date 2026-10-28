"""Out-of-the-money quote selection, implied-vol re-inversion and filtering of SPX chains."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from volsurf.data.schema import CORE_COLUMNS
from volsurf.pricing import black76

SLICE = ["date", "expiry"]
EXTRA_COLUMNS = [
    "bid_size",
    "ask_size",
    "volume",
    "iv_vendor",
    "arb_vertical",
    "arb_butterfly",
    "is_third_friday",
    "atm_iv",
    "z",
]


@dataclass
class FilterConfig:
    min_dte: int = 7
    max_dte: int = 730
    max_rel_spread: float = 1.0  # (ask - bid) / mid
    k_range: tuple[float, float] = (-1.5, 0.75)  # hard log-moneyness band
    # band in ATM standard deviations, z = k / (atm_iv sqrt T)
    z_range: tuple[float, float] = (-6.0, 4.0)
    iv_range: tuple[float, float] = (0.01, 3.0)
    min_slice_quotes: int = 8


def _pick_side(df: pd.DataFrame, is_call: np.ndarray, field: str) -> np.ndarray:
    return np.where(is_call, df[f"c_{field}"].to_numpy(float), df[f"p_{field}"].to_numpy(float))


def _atm_iv(df: pd.DataFrame) -> pd.Series:
    """ATM implied vol per slice by linear interpolation of iv_mid at k = 0."""
    atm = df.groupby(SLICE, sort=False).apply(
        lambda g: float(np.interp(0.0, g["k"].to_numpy(), g["iv_mid"].to_numpy())),
        include_groups=False,
    )
    return df.join(atm.rename("atm_iv"), on=SLICE)["atm_iv"]


def build_quotes(
    wide: pd.DataFrame, forwards: pd.DataFrame, cfg: FilterConfig | None = None
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Long table of filtered OTM quotes, plus the row count after each filter step.

    ``wide`` is an audited chain (``audit_quotes``) and ``forwards`` the output of
    ``estimate_forwards``.
    """
    cfg = cfg or FilterConfig()
    funnel = {"contracts": len(wide)}

    df = wide[wide["dte"].between(cfg.min_dte, cfg.max_dte)]
    funnel["dte"] = len(df)

    good = forwards.loc[forwards["valid"], SLICE + ["F", "D"]]
    df = df.merge(good, on=SLICE, how="inner")
    funnel["forward"] = len(df)

    is_call = (df["strike"] >= df["F"]).to_numpy()
    q = df[SLICE + ["dte", "T", "strike", "spot", "F", "D", "is_third_friday"]].copy()
    q["option_type"] = np.where(is_call, "C", "P")
    for field in ("bid", "ask", "bid_size", "ask_size", "volume", "iv_vendor"):
        q[field] = _pick_side(df, is_call, field)
    for field in ("arb_vertical", "arb_butterfly"):
        q[field] = _pick_side(df, is_call, field).astype(bool)

    q = q[(q["bid"] > 0.0) & (q["ask"] > q["bid"])]
    funnel["two_sided"] = len(q)

    q["mid"] = 0.5 * (q["bid"] + q["ask"])
    q = q[(q["ask"] - q["bid"]) / q["mid"] <= cfg.max_rel_spread]
    funnel["spread"] = len(q)

    q["k"] = np.log(q["strike"] / q["F"])
    q = q[q["k"].between(*cfg.k_range)]
    funnel["k_range"] = len(q)

    args = [q[c].to_numpy() for c in ("F", "strike", "T")]
    call, D = q["option_type"].eq("C").to_numpy(), q["D"].to_numpy()
    for side in ("bid", "ask", "mid"):
        q[f"iv_{side}"] = black76.implied_vol(q[side].to_numpy(), *args, call, D)
    lo, hi = cfg.iv_range
    ivs = q[["iv_bid", "iv_ask", "iv_mid"]]
    q = q[ivs.notna().all(axis=1) & (ivs >= lo).all(axis=1) & (ivs <= hi).all(axis=1)]
    funnel["iv"] = len(q)

    q["atm_iv"] = _atm_iv(q)
    q["z"] = q["k"] / (q["atm_iv"] * np.sqrt(q["T"]))
    q = q[q["z"].between(*cfg.z_range)]
    funnel["z_range"] = len(q)

    sizes = q.groupby(SLICE)["strike"].transform("size")
    q = q[sizes >= cfg.min_slice_quotes]
    funnel["slice_size"] = len(q)

    q["half_spread_iv"] = 0.5 * (q["iv_ask"] - q["iv_bid"])
    return q[CORE_COLUMNS + EXTRA_COLUMNS].reset_index(drop=True), funnel
