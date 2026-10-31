"""Delta-hedged backtest of mispricing signals on end-of-day quotes.

A signal on day s (+1 rich: sell, -1 cheap: buy) is executed at the close of day s + lag, crossing
the spread, in a size of ``target_vega`` dollars per vol point. The position is marked to mid
every close and delta-hedged with the forward of its expiry (delta from the day's market IV), and
it is closed, again crossing the spread, ``lag`` days after the first close at which its exit
condition holds (the residual has reverted) or after ``hold`` days in the trade. A contract is
traded at most once at a time.

The panel holds out-of-the-money options only, so an option that moves into the money is marked
through put-call parity from the out-of-the-money option at the same strike, C - P = D (F - K),
with the same spread. Cleaning filters move with the market, so a held contract can drop out of
the panel while it still trades: for up to ``max_gap`` days it keeps its hedge and is marked when
it is quoted again, and beyond that it is closed at its last quote and counted.

The placebo enters, for every trade, a random contract that was not flagged on the same day, of
the same type, direction and maturity and moneyness bucket, for the same days: it earns what the
strategy's exposure earns (for example the variance risk premium on sold options) without its
signal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from volsurf.eval.forecast import diebold_mariano
from volsurf.pricing import black76

DTE_EDGES = np.array([7, 30, 90, 365])
MONEYNESS_EDGES = np.array([-0.4, -0.15, 0.15])  # k / sqrt(T)


@dataclass
class BacktestConfig:
    margin: float = 0.5  # flag when the model price is this many half-spreads outside the bid-ask
    exit_r: float = 0.5  # exit once |residual| is below this, in half-spreads
    hold: int = 5  # maximum trading days in a trade
    lag: int = 1  # trading days from a signal to its execution (0: at the signal's close)
    min_vega: float = 0.0  # minimum dollar vega per vol point of one contract
    max_gap: int = 3  # trading days a held contract may go unquoted before it is closed
    min_dte: int = 7  # the panel's shortest maturity; entries must stay above it until exit
    z_range: tuple[float, float] = (-6.0, 4.0)  # the panel's k / (atm iv sqrt T) band
    fee: float = 1.0  # dollars per contract per side
    target_vega: float = 1000.0  # dollars per vol point per trade
    multiplier: float = 100.0
    placebo_draws: int = 20


class Market:
    """Quotes indexed by (trading day, expiry, strike), with parity marks for either type."""

    def __init__(self, table: pd.DataFrame) -> None:
        self.dates = np.sort(table["date"].unique())
        t = np.searchsorted(self.dates, table["date"].to_numpy())
        self.t = t
        self.cols = {
            c: table[c].to_numpy()
            for c in ("strike", "F", "D", "T", "dte", "bid", "ask", "mid", "iv_mid")
        }
        self.is_call = table["option_type"].to_numpy() == "C"
        self.expiry = table["expiry"].to_numpy()
        atm = table.groupby([t, self.expiry]).apply(
            lambda g: np.interp(0.0, g["k"], g["iv_mid"]), include_groups=False
        )
        atm_row = atm.reindex(pd.MultiIndex.from_arrays([t, self.expiry])).to_numpy()
        self.z = table["k"].to_numpy() / (atm_row * np.sqrt(self.cols["T"]))
        keys = zip(t.tolist(), self.expiry.tolist(), self.cols["strike"].tolist(), strict=True)
        self.index = {key: i for i, key in enumerate(keys)}

    def row(self, t: int, expiry, strike) -> int | None:
        return self.index.get((t, expiry, strike))

    def quote(self, i: int, is_call: bool) -> tuple[float, float, float]:
        """Bid, ask and mid of the option of type ``is_call`` at row i's strike and expiry."""
        c = self.cols
        shift = 0.0
        if self.is_call[i] != is_call:  # parity: C = P + D (F - K)
            parity = c["D"][i] * (c["F"][i] - c["strike"][i])
            shift = parity if is_call else -parity
        return c["bid"][i] + shift, c["ask"][i] + shift, c["mid"][i] + shift

    def delta(self, i: int, is_call: bool) -> float:
        c = self.cols
        return float(
            black76.delta(c["F"][i], c["strike"][i], c["T"][i], c["iv_mid"][i], is_call, c["D"][i])
        )

    def dollar_vega(self, i: int, multiplier: float) -> float:
        """Dollars per vol point of one contract."""
        c = self.cols
        vega = black76.vega(c["F"][i], c["strike"][i], c["T"][i], c["iv_mid"][i], c["D"][i])
        return float(vega) * 0.01 * multiplier


def stays_quoted(mkt: Market, cfg: BacktestConfig) -> np.ndarray:
    """Rows whose contract stays inside the panel (above the shortest maturity and within the
    z band, which widens in |z| as T falls) until the latest exit a trade entered there can have."""
    last = len(mkt.dates) - 1
    latest = np.minimum(mkt.t + 2 * cfg.lag + cfg.hold + cfg.max_gap, last)
    days = (mkt.dates[latest] - mkt.dates[mkt.t]) / np.timedelta64(1, "D")
    T = mkt.cols["T"]
    T_exit = T - days / 365.0
    z_exit = mkt.z * np.sqrt(T / np.maximum(T_exit, 1e-6))
    lo, hi = cfg.z_range
    return (mkt.cols["dte"] - days > cfg.min_dte) & (z_exit > lo) & (z_exit < hi)


def buckets(table: pd.DataFrame) -> np.ndarray:
    m = table["k"].to_numpy() / np.sqrt(table["T"].to_numpy())
    return np.digitize(table["dte"].to_numpy(), DTE_EDGES) * 10 + np.digitize(m, MONEYNESS_EDGES)


def _run_trade(mkt: Market, cfg: BacktestConfig, i_signal, direction, revert, exit_after=None):
    """Simulate one trade from its signal row; returns per-day PnL records or None if it cannot
    be entered. ``exit_after`` forces the holding length (placebo)."""
    t0, expiry, strike = mkt.t[i_signal], mkt.expiry[i_signal], mkt.cols["strike"][i_signal]
    is_call = bool(mkt.is_call[i_signal])
    e = t0 + cfg.lag
    i = mkt.row(e, expiry, strike)
    if i is None:
        return None
    vega = mkt.dollar_vega(i_signal, cfg.multiplier)
    n = cfg.target_vega / vega
    units = n * cfg.multiplier * direction  # signed option units (+ long)
    bid, ask, mid = mkt.quote(i, is_call)
    records = [(e, 0.0, n * cfg.multiplier * (ask - bid) / 2 + n * cfg.fee)]
    exit_signal, t, missing, gap_days = None, e, False, 0
    while True:
        if exit_after is not None:
            stop = t - e >= exit_after
        else:
            if exit_signal is None and (revert[i] or t - e >= cfg.hold):
                exit_signal = t
            stop = exit_signal is not None and t >= exit_signal + cfg.lag
        if stop or t == len(mkt.dates) - 1:  # exit, or closed at the panel's last close
            break
        step = next(
            (g for g in range(1, cfg.max_gap + 2) if mkt.row(t + g, expiry, strike) is not None),
            None,
        )
        if step is None:  # closed at the last quote
            missing = True
            break
        j = mkt.row(t + step, expiry, strike)
        _, _, mid_next = mkt.quote(j, is_call)
        F0, F1 = mkt.cols["F"][i], mkt.cols["F"][j]
        gross = units * ((mid_next - mid) - mkt.delta(i, is_call) * (F1 - F0))
        records.append((t + step, gross, 0.0))
        gap_days += step - 1
        i, t, mid = j, t + step, mid_next
    bid, ask, _ = mkt.quote(i, is_call)
    records[-1] = (
        records[-1][0],
        records[-1][1],
        records[-1][2] + n * cfg.multiplier * (ask - bid) / 2 + n * cfg.fee,
    )
    return {"t_signal": t0, "t_entry": e, "t_exit": t, "contracts": n, "records": records,
            "missing": missing, "gap_days": gap_days, "direction": direction}  # fmt: skip


def run_signals(
    table: pd.DataFrame,
    signal: np.ndarray,
    revert: np.ndarray,
    cfg: BacktestConfig,
    tradable: np.ndarray | None = None,
    mkt: Market | None = None,
) -> pd.DataFrame:
    """Trades for every tradable signal (+1 sell, -1 buy), one open trade per contract."""
    mkt = mkt or Market(table)
    signal = np.asarray(signal)
    ok = signal != 0
    if tradable is not None:
        ok &= tradable
    ok &= stays_quoted(mkt, cfg)
    rows = np.flatnonzero(ok)
    rows = rows[np.argsort(mkt.t[rows], kind="stable")]
    busy: dict = {}
    trades = []
    for i in rows:
        key = (mkt.expiry[i], mkt.cols["strike"][i])
        if busy.get(key, -1) >= mkt.t[i]:
            continue
        if mkt.dollar_vega(i, cfg.multiplier) < cfg.min_vega:
            continue
        trade = _run_trade(mkt, cfg, i, -int(signal[i]), revert)
        if trade is None:
            continue
        busy[key] = trade["t_exit"]
        trades.append(trade | {"row": i})
    return pd.DataFrame(trades)


def run_placebo(
    table: pd.DataFrame,
    trades: pd.DataFrame,
    pool: np.ndarray,
    cfg: BacktestConfig,
    seed: int,
    mkt: Market | None = None,
) -> pd.DataFrame:
    """For each trade, a random contract from ``pool`` (unflagged, tradable) on the same signal
    day with the same type and bucket, entered in the same direction for the same days."""
    mkt = mkt or Market(table)
    rng = np.random.default_rng(seed)
    pool = pool & stays_quoted(mkt, cfg)
    bucket = buckets(table)
    groups = (
        pd.Series(np.flatnonzero(pool))
        .groupby([mkt.t[pool], mkt.is_call[pool], bucket[pool]])
        .groups
    )
    candidates = {key: np.flatnonzero(pool)[idx] for key, idx in groups.items()} if groups else {}
    never = np.zeros(len(table), dtype=bool)
    out = []
    for tr in trades.itertuples():
        i = tr.row
        cands = candidates.get((mkt.t[i], mkt.is_call[i], bucket[i]))
        if cands is None or not len(cands):
            continue
        j = int(rng.choice(cands))
        trade = _run_trade(mkt, cfg, j, tr.direction, never, exit_after=tr.t_exit - tr.t_entry)
        if trade is not None:
            out.append(trade | {"row": j})
    return pd.DataFrame(out)


def daily_pnl(trades: pd.DataFrame, n_days: int) -> pd.DataFrame:
    gross, cost = np.zeros(n_days), np.zeros(n_days)
    for records in trades.get("records", []):
        for t, g, c in records:
            gross[t] += g
            cost[t] += c
    return pd.DataFrame({"gross": gross, "cost": cost, "net": gross - cost})


def trade_pnl(trades: pd.DataFrame) -> pd.DataFrame:
    gross = trades["records"].map(lambda r: sum(g for _, g, _ in r))
    cost = trades["records"].map(lambda r: sum(c for _, _, c in r))
    return pd.DataFrame({"gross": gross, "cost": cost, "net": gross - cost})


def performance(trades: pd.DataFrame, n_days: int) -> dict:
    if trades.empty:
        return {"trades": 0}
    daily = daily_pnl(trades, n_days)
    per = trade_pnl(trades)
    cum = daily["net"].cumsum().to_numpy()
    std = daily["net"].std()
    return {
        "trades": len(trades),
        "share_sold": float((trades["direction"] < 0).mean()),
        "mean_days": float((trades["t_exit"] - trades["t_entry"]).mean()),
        "gross": float(per["gross"].sum()),
        "cost": float(per["cost"].sum()),
        "net": float(per["net"].sum()),
        "net_per_trade": float(per["net"].mean()),
        "gross_per_trade": float(per["gross"].mean()),
        "hit_rate": float((per["net"] > 0).mean()),
        "sharpe": float(daily["net"].mean() / std * np.sqrt(252)) if std > 0 else np.nan,
        "max_drawdown": float(np.max(np.maximum.accumulate(np.r_[0.0, cum])[1:] - cum)),
        "contracts_per_day": float(2 * trades["contracts"].sum() / n_days),
        "missing_marks": int(trades["missing"].sum()),
        "gap_days": int(trades["gap_days"].sum()),
    }


def against_placebo(trades: pd.DataFrame, placebos: list[pd.DataFrame], n_days: int, h: int):
    """Newey-West test of daily strategy-minus-placebo net PnL (positive: the signal adds)."""
    strategy = daily_pnl(trades, n_days)["net"].to_numpy()
    placebo = np.mean([daily_pnl(p, n_days)["net"].to_numpy() for p in placebos], axis=0)
    stat, p = diebold_mariano(strategy - placebo, h)
    return {"placebo_net": float(placebo.sum()), "excess_net": float((strategy - placebo).sum()),
            "t_stat": stat, "p_value": p}  # fmt: skip
