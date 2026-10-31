import numpy as np
import pandas as pd
import pytest

from volsurf.eval.backtest import (
    BacktestConfig,
    Market,
    performance,
    run_placebo,
    run_signals,
    stays_quoted,
    trade_pnl,
)
from volsurf.eval.mispricing import cauchy_spreads, flag, precision_recall
from volsurf.eval.reversion import daily_slopes, summarize_slopes
from volsurf.pricing import black76


def _panel(F_path, iv=0.2, K=(90.0, 100.0, 110.0), dte0=60, half=0.05, D=0.99):
    """One expiry quoted at three strikes over len(F_path) days at a constant IV, OTM only."""
    rows = []
    expiry = pd.Timestamp("2022-03-01")
    for t, F in enumerate(F_path):
        dte = dte0 - t
        T = dte / 365
        for k in K:
            is_call = k >= F
            mid = float(black76.price(F, k, T, iv, is_call, D))
            date = pd.Timestamp("2022-01-03") + pd.Timedelta(days=t)
            option_type = "C" if is_call else "P"
            rows.append({"date": date, "expiry": expiry, "strike": k, "option_type": option_type,
                         "F": F, "D": D, "T": T, "dte": dte, "k": np.log(k / F), "iv_mid": iv,
                         "bid": mid - half, "ask": mid + half, "mid": mid})  # fmt: skip
    return pd.DataFrame(rows)


def test_parity_marks_an_option_that_moved_into_the_money():
    table = _panel(
        [100.0, 115.0]
    )  # the 110 call is OTM on day 0 and ITM (quoted as a put) on day 1
    mkt = Market(table)
    i = mkt.row(1, table["expiry"].iloc[0], 110.0)
    assert not mkt.is_call[i]
    bid, ask, mid = mkt.quote(i, is_call=True)
    expected = black76.price(115.0, 110.0, table["T"].iloc[3], 0.2, True, 0.99)
    assert mid == pytest.approx(float(expected), abs=1e-10)
    assert ask - bid == pytest.approx(0.1)


def test_trade_accounting_costs_hedging_and_lag():
    cfg = BacktestConfig(hold=2, lag=1, fee=1.0, target_vega=100.0, min_dte=0)
    F_path = [100.0, 100.5, 99.8, 100.3, 100.1]
    table = _panel(F_path)
    signal = np.zeros(len(table), dtype=int)
    first = (table["date"] == table["date"].min()) & (table["strike"] == 100.0)
    signal[np.flatnonzero(first)] = -1  # cheap: buy the ATM call on day 0
    never = np.zeros(len(table), dtype=bool)
    trades = run_signals(table, signal, never, cfg)
    assert len(trades) == 1
    tr = trades.iloc[0]
    assert tr.t_entry == 1 and tr.t_exit == 1 + cfg.hold + cfg.lag  # exit signal at entry + hold
    per = trade_pnl(trades).iloc[0]
    n = tr.contracts
    assert per["cost"] == pytest.approx(2 * n * (100 * 0.05 + 1.0))  # half-spread + fee each side
    # hedged gross is small next to the unhedged move of the option over the same days
    mkt = Market(table)
    i0, i1 = (mkt.row(t, table["expiry"].iloc[0], 100.0) for t in (tr.t_entry, tr.t_exit))
    unhedged = n * 100 * (mkt.quote(i1, True)[2] - mkt.quote(i0, True)[2])
    assert abs(per["gross"]) < 0.25 * abs(unhedged)


def test_sizing_targets_vega_and_contracts_trade_once_at_a_time():
    cfg = BacktestConfig(hold=3, lag=0, target_vega=1000.0, min_dte=0)
    table = _panel([100.0] * 6)
    signal = np.where(table["strike"] == 100.0, 1, 0)  # rich every day: sell
    never = np.zeros(len(table), dtype=bool)
    trades = run_signals(table, signal, never, cfg)
    assert list(trades["t_entry"]) == [0, 4]  # the second signal waits for the first exit (day 3)
    mkt = Market(table)
    assert trades["contracts"].iloc[0] * mkt.dollar_vega(
        trades["row"].iloc[0], 100.0
    ) == pytest.approx(1000.0)
    assert (trades["direction"] == -1).all()
    perf = performance(trades, len(mkt.dates))
    assert perf["trades"] == 2 and perf["share_sold"] == 1.0


def test_placebo_matches_day_type_bucket_and_holding_period():
    cfg = BacktestConfig(hold=2, lag=0, min_dte=0)
    table = pd.concat(
        [_panel([100.0] * 5, K=(86.0, 88.0, 90.0, 92.0, 100.0, 105.0))], ignore_index=True
    )
    signal = np.where((table["strike"] == 90.0) & (table["dte"] == 60), 1, 0)
    never = np.zeros(len(table), dtype=bool)
    trades = run_signals(table, signal, never, cfg)
    pool = signal == 0
    placebo = run_placebo(table, trades, pool, cfg, seed=0)
    assert len(placebo) == len(trades) == 1
    mkt = Market(table)
    j, i = placebo["row"].iloc[0], trades["row"].iloc[0]
    assert mkt.t[j] == mkt.t[i] and mkt.is_call[j] == mkt.is_call[i] and j != i
    assert (placebo["t_exit"] - placebo["t_entry"]).iloc[0] == (
        trades["t_exit"] - trades["t_entry"]
    ).iloc[0]


def test_reversion_slopes_recover_ar1_persistence():
    rng = np.random.default_rng(0)
    n_days, n_contracts, phi = 200, 300, 0.5
    x = np.zeros((n_days, n_contracts))
    x[0] = rng.normal(size=n_contracts)
    for t in range(1, n_days):
        x[t] = phi * x[t - 1] + rng.normal(size=n_contracts)
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    table = pd.DataFrame({"date": np.repeat(dates, n_contracts),
                          "expiry": pd.Timestamp("2023-01-20"),
                          "strike": np.tile(np.arange(n_contracts, dtype=float), n_days),
                          "option_type": "C"})  # fmt: skip
    r = x.ravel()
    flags = np.abs(r) > 1.5
    for h in (1, 3):
        summary = summarize_slopes(daily_slopes(table, r, flags, h), h)
        assert summary["slope_unflagged"] == pytest.approx(phi**h - 1, abs=0.05)
        assert summary["slope_flagged"] == pytest.approx(phi**h - 1, abs=0.08)
        assert summary["p_value"] > 0.01  # no extra reversion for flagged quotes in an AR(1)


def test_flags_weights_and_detection_scores():
    assert list(flag(np.array([0.5, 1.6, -2.0, 1.4]), 0.5)) == [0, 1, -1, 0]
    s = np.array([1.0, 1.0])
    np.testing.assert_allclose(cauchy_spreads(s, np.array([0.0, 2.0]), 1.0), [1.0, np.sqrt(5.0)])
    table = pd.DataFrame({"label": [1, 0, -1, 0], "shift": [3.0, 0.5, -2.0, 0.0],
                          "iv_true": [0.2] * 4, "iv_bid": [0.21, 0.19, 0.19, 0.19],
                          "iv_ask": [0.22, 0.21, 0.21, 0.21],
                          "injection": ["point", "point", "segment", "none"]})  # fmt: skip
    pr = precision_recall(np.array([1, 1, 0, 0]), table)
    assert pr["precision"] == 0.5 and pr["precision_ex_weak"] == 1.0
    assert pr["recall"] == 0.5 and pr["recall_visible"] == 1.0  # only the first label is visible
    assert pr["recall_point"] == 1.0 and pr["recall_segment"] == 0.0


def test_held_contract_bridges_a_quote_gap_and_entries_stay_inside_the_panel():
    cfg = BacktestConfig(hold=3, lag=0, min_dte=0, z_range=(-50.0, 50.0))
    table = _panel([100.0, 100.2, 100.4, 100.1, 99.9, 100.0])
    gap = (table["date"] == table["date"].unique()[2]) & (table["strike"] == 100.0)
    table = table[~gap].reset_index(drop=True)  # the ATM contract is unquoted on day 2
    signal = np.where((table["strike"] == 100.0) & (table["date"] == table["date"].min()), 1, 0)
    never = np.zeros(len(table), dtype=bool)
    trades = run_signals(table, signal, never, cfg)
    tr = trades.iloc[0]
    assert not tr.missing and tr.gap_days == 1 and tr.t_exit == 3
    assert [t for t, _, _ in tr.records] == [0, 1, 3]  # marked again when quoted on day 3

    # an entry whose expiry would fall below the panel's shortest maturity before exit is skipped
    short = BacktestConfig(hold=3, lag=0, min_dte=58, z_range=(-50.0, 50.0))
    assert run_signals(table, signal, never, short).empty
    assert stays_quoted(Market(table), short).sum() == 0
