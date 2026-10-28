"""Synthetic SPX-like option chains with known, arbitrage-free ground truth.

Each simulated business day draws the underlying and a variance state from a stochastic-volatility
path (log-variance Ornstein-Uhlenbeck, correlated with spot), then quotes an arbitrage-free surface
on a realistic listing of weekly and monthly expiries:

* ``heston``: Heston prices (COS method) with v0 set to the day's variance state;
* ``ssvi``:   SSVI with power-law curvature, its ATM total variance driven by the same state.

The regime path is kept separate from the surface parameters: realistic SPX smiles need a Heston
vol-of-vol far beyond the Feller bound, and simulating v with those parameters pins it at zero.

Observed quotes add bid-ask spreads, noise inside the spread, and tick rounding. They also carry
labelled mispricings, either isolated strikes or contiguous strike segments, which persist and
decay across days. The quotes share the long format of the real-data pipeline, so downstream code
does not depend on the data source.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from volsurf.models.ssvi import power_law_phi, ssvi
from volsurf.pricing import black76, heston
from volsurf.pricing.heston import HestonParams

STRIKE_INCREMENTS = (5.0, 10.0, 25.0, 50.0, 100.0)
TRADING_DAYS = 252
QUOTE_COLUMNS = [
    "date", "expiry", "dte", "T", "strike", "option_type", "spot", "F", "D", "k",
    "bid", "ask", "mid", "iv_bid", "iv_ask", "iv_mid", "iv_true", "w_true",
    "half_spread_iv", "shift", "label", "injection",
]  # fmt: skip


@dataclass
class ListingConfig:
    min_dte: int = 7
    max_dte: int = 730
    weekly_horizon: int = 56  # weeklies listed out to this many days, monthlies beyond
    z_range: tuple[float, float] = (-4.0, 2.5)  # strike range in ATM standard deviations
    target_strikes: int = 40  # approximate strikes per expiry


@dataclass
class QuoteConfig:
    half_spread: float = 0.0025  # IV half-spread at the money and 30 DTE
    wing_slope: float = 0.35  # relative widening per ATM standard deviation of moneyness
    noise: float = 0.3  # IV noise std as a fraction of the half-spread
    tick_small: float = 0.05
    tick_large: float = 0.10
    tick_cutoff: float = 3.0  # option price above which the large tick applies


@dataclass
class InjectionConfig:
    point_prob: float = 0.01  # per quote per day
    segment_prob: float = 0.05  # per expiry per day
    segment_len: tuple[int, int] = (3, 6)
    size: tuple[float, float] = (2.0, 5.0)  # magnitude in half-spreads
    decay: float = 0.5  # daily persistence factor of an existing mispricing
    drop_below: float = 0.5  # removed once its magnitude falls below this


@dataclass
class StateConfig:
    vol_mean: float = 0.18  # long-run level of sqrt(v), in log terms
    kappa: float = 5.0  # mean-reversion speed of log-variance
    vol_of_logvar: float = 2.0  # diffusion of log-variance (VIX-like daily moves)
    rho: float = -0.7  # spot / variance shock correlation
    v_floor: float = 0.002


@dataclass
class SyntheticConfig:
    surface: str = "heston"
    n_days: int = 756
    start: str = "2020-01-02"
    seed: int = 0
    spot0: float = 4000.0
    rate: float = 0.03
    div_yield: float = 0.015
    state: StateConfig = field(default_factory=StateConfig)
    heston: HestonParams = field(default_factory=lambda: HestonParams(2.0, 0.04, 0.6, -0.7, 0.03))
    ssvi_rho: float = -0.7
    ssvi_eta: float = 1.0
    ssvi_gamma: float = 0.4
    listing: ListingConfig = field(default_factory=ListingConfig)
    quotes: QuoteConfig = field(default_factory=QuoteConfig)
    injection: InjectionConfig = field(default_factory=InjectionConfig)

    @classmethod
    def from_dict(cls, d: dict) -> SyntheticConfig:
        nested = {
            "heston": HestonParams,
            "state": StateConfig,
            "listing": ListingConfig,
            "quotes": QuoteConfig,
            "injection": InjectionConfig,
        }

        def tuples(x: dict) -> dict:
            return {k: tuple(v) if isinstance(v, list) else v for k, v in x.items()}

        kwargs = {k: nested[k](**tuples(v)) if k in nested else v for k, v in d.items()}
        return cls(**kwargs)


@dataclass(frozen=True)
class DayState:
    date: pd.Timestamp
    spot: float
    v: float  # variance state used for quoting (floored)


def simulate_states(cfg: SyntheticConfig, rng: np.random.Generator) -> list[DayState]:
    """Daily path of spot and variance: log-variance OU with correlated spot shocks."""
    sc, dt = cfg.state, 1.0 / TRADING_DAYS
    s, x = cfg.spot0, np.log(cfg.heston.v0)
    x_mean = 2.0 * np.log(sc.vol_mean)
    states = []
    for date in pd.bdate_range(cfg.start, periods=cfg.n_days):
        v = max(np.exp(x), sc.v_floor)
        states.append(DayState(date, s, v))
        z1, z = rng.standard_normal(2)
        z2 = sc.rho * z1 + np.sqrt(1.0 - sc.rho**2) * z
        s *= np.exp((cfg.rate - cfg.div_yield - 0.5 * v) * dt + np.sqrt(v * dt) * z1)
        x += sc.kappa * (x_mean - x) * dt + sc.vol_of_logvar * np.sqrt(dt) * z2
    return states


def true_iv(cfg: SyntheticConfig, v: float, k, T: float):
    """Ground-truth implied volatility at log forward-moneyness ``k`` for one maturity."""
    params = replace(cfg.heston, v0=v)
    k = np.asarray(k, dtype=float)
    if cfg.surface == "heston":
        return heston.implied_vol_slice(1.0, np.exp(k), T, params)  # scale-invariant in F
    if cfg.surface == "ssvi":
        theta = heston.expected_total_variance(T, params)
        phi = power_law_phi(theta, cfg.ssvi_eta, cfg.ssvi_gamma)
        return np.sqrt(ssvi(k, theta, cfg.ssvi_rho, phi) / T)
    raise ValueError(f"unknown surface {cfg.surface!r}")


def true_total_variance(cfg: SyntheticConfig, v: float, k, T):
    """Ground-truth total variance on the grid T x k, shape (len(T), len(k))."""
    return np.stack([true_iv(cfg, v, k, t) ** 2 * t for t in np.atleast_1d(T)])


def _expiry_calendar(start: pd.Timestamp, end: pd.Timestamp):
    return pd.date_range(start, end, freq="W-FRI"), pd.date_range(start, end, freq="WOM-3FRI")


def _listed_expiries(date, fridays, monthlies, lc: ListingConfig) -> pd.DatetimeIndex:
    dte_w = (fridays - date).days
    dte_m = (monthlies - date).days
    weekly = fridays[(dte_w >= lc.min_dte) & (dte_w <= lc.weekly_horizon)]
    monthly = monthlies[(dte_m >= lc.min_dte) & (dte_m <= lc.max_dte)]
    return weekly.union(monthly)


def _listed_strikes(F: float, atm_std: float, lc: ListingConfig) -> np.ndarray:
    lo, hi = F * np.exp(lc.z_range[0] * atm_std), F * np.exp(lc.z_range[1] * atm_std)
    raw = (hi - lo) / lc.target_strikes
    top = STRIKE_INCREMENTS[-1]
    inc = next((i for i in STRIKE_INCREMENTS if i >= raw), top * np.ceil(raw / top))
    return np.arange(np.ceil(lo / inc) * inc, hi + 1e-9, inc)


def _chain_skeleton(cfg: SyntheticConfig, state: DayState, fridays, monthlies) -> pd.DataFrame:
    """Listed contracts for one day with forwards, discount factors and ground-truth IVs."""
    params = replace(cfg.heston, v0=state.v)
    parts = []
    for expiry in _listed_expiries(state.date, fridays, monthlies, cfg.listing):
        dte = (expiry - state.date).days
        T = dte / 365.0
        F = state.spot * np.exp((cfg.rate - cfg.div_yield) * T)
        atm_std = np.sqrt(heston.expected_total_variance(T, params))
        K = _listed_strikes(F, atm_std, cfg.listing)
        k = np.log(K / F)
        iv = true_iv(cfg, state.v, k, T)
        ok = np.isfinite(iv)
        parts.append(
            pd.DataFrame(
                {
                    "expiry": expiry,
                    "dte": dte,
                    "T": T,
                    "strike": K[ok],
                    "F": F,
                    "D": np.exp(-cfg.rate * T),
                    "k": k[ok],
                    "z": k[ok] / atm_std,
                    "iv_true": iv[ok],
                }
            )
        )
    chain = pd.concat(parts, ignore_index=True)
    chain.insert(0, "date", state.date)
    chain["spot"] = state.spot
    return chain


def _update_book(book: dict, chain: pd.DataFrame, date, ic: InjectionConfig, rng) -> None:
    """Decay existing mispricings, then inject new point and segment mispricings in place."""
    for key in list(book):
        size, kind = book[key]
        size *= ic.decay
        if abs(size) < ic.drop_below or key[0] <= date:
            del book[key]
        else:
            book[key] = (size, kind)

    def draw(n):
        return rng.choice([-1.0, 1.0], n) * rng.uniform(*ic.size, n)

    keys = list(zip(chain["expiry"], chain["strike"], strict=True))
    free = np.array([key not in book for key in keys])
    hit = free & (rng.random(len(keys)) < ic.point_prob)
    for key, size in zip(np.array(keys, dtype=object)[hit], draw(hit.sum()), strict=True):
        book[tuple(key)] = (size, "point")

    for _, slice_ in chain.groupby("expiry", sort=False):
        lo, hi = ic.segment_len
        if rng.random() >= ic.segment_prob or len(slice_) <= hi:
            continue
        length = rng.integers(lo, hi + 1)
        start = rng.integers(0, len(slice_) - length + 1)
        size = draw(1)[0]
        for expiry, strike in slice_.iloc[start : start + length][["expiry", "strike"]].itertuples(
            index=False
        ):
            book[(expiry, strike)] = (size, "segment")


def _quote(chain: pd.DataFrame, qc: QuoteConfig, rng) -> pd.DataFrame:
    """Turn ground truth plus injected shifts into tick-rounded bid/ask quotes and their IVs."""
    F, K, T, D = (chain[c].to_numpy() for c in ("F", "strike", "T", "D"))
    is_call = K >= F
    hs = qc.half_spread * (1.0 + qc.wing_slope * np.abs(chain["z"].to_numpy()))
    hs *= (30.0 / np.maximum(chain["dte"].to_numpy(), 7)) ** 0.25
    iv_obs = chain["iv_true"].to_numpy() + hs * (qc.noise * rng.standard_normal(len(chain)))
    iv_obs += hs * chain["shift"].to_numpy()

    theo = black76.price(F, K, T, iv_obs, is_call, D)
    tick = np.where(theo > qc.tick_cutoff, qc.tick_large, qc.tick_small)
    bid = black76.price(F, K, T, np.maximum(iv_obs - hs, 1e-4), is_call, D)
    ask = black76.price(F, K, T, iv_obs + hs, is_call, D)
    bid = np.round(np.floor(bid / tick + 1e-9) * tick, 2)
    ask = np.round(np.ceil(ask / tick - 1e-9) * tick, 2)
    mid = 0.5 * (bid + ask)

    out = chain.assign(
        option_type=np.where(is_call, "C", "P"),
        bid=bid,
        ask=ask,
        mid=mid,
        iv_bid=black76.implied_vol(bid, F, K, T, is_call, D),
        iv_ask=black76.implied_vol(ask, F, K, T, is_call, D),
        iv_mid=black76.implied_vol(mid, F, K, T, is_call, D),
        half_spread_iv=hs,
    )
    return out[(out["bid"] > 0.0) & out["iv_mid"].notna()]


def generate(cfg: SyntheticConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate a panel of daily option chains.

    Returns the quotes (one row per OTM option per day) and the daily states (spot, variance)
    from which the ground-truth surface can be rebuilt with ``true_total_variance``.
    """
    rng = np.random.default_rng(cfg.seed)
    states = simulate_states(cfg, rng)
    end = states[-1].date + pd.Timedelta(days=cfg.listing.max_dte + 7)
    fridays, monthlies = _expiry_calendar(states[0].date, end)

    book: dict[tuple, tuple[float, str]] = {}
    days = []
    for state in states:
        chain = _chain_skeleton(cfg, state, fridays, monthlies)
        _update_book(book, chain, state.date, cfg.injection, rng)
        keys = zip(chain["expiry"], chain["strike"], strict=True)
        entries = [book.get(key, (0.0, "none")) for key in keys]
        chain["shift"] = [e[0] for e in entries]
        chain["injection"] = [e[1] for e in entries]
        days.append(_quote(chain, cfg.quotes, rng))

    quotes = pd.concat(days, ignore_index=True)
    quotes["w_true"] = quotes["iv_true"] ** 2 * quotes["T"]
    quotes["label"] = np.where(np.abs(quotes["shift"]) > 1.0, np.sign(quotes["shift"]), 0).astype(
        np.int8
    )
    quotes.loc[quotes["shift"] == 0.0, "injection"] = "none"
    state_frame = pd.DataFrame(
        {
            "date": [s.date for s in states],
            "spot": [s.spot for s in states],
            "v": [s.v for s in states],
        }
    )
    return quotes[QUOTE_COLUMNS], state_frame
