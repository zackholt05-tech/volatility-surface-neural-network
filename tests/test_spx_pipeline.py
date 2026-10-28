"""Phase 2 pipeline tests on fabricated chains written in the OptionsDX text format."""

import numpy as np
import pandas as pd
import pytest

from volsurf.arbitrage.quotes import audit_quotes
from volsurf.config import from_dict
from volsurf.data import optionsdx, store
from volsurf.data.clean import FilterConfig, build_quotes
from volsurf.data.forwards import ForwardConfig, estimate_forwards
from volsurf.data.spx import SpxConfig, process_month
from volsurf.pricing import black76

HEADER = (
    "[QUOTE_UNIXTIME], [QUOTE_READTIME], [QUOTE_DATE], [QUOTE_TIME_HOURS], [UNDERLYING_LAST], "
    "[EXPIRE_DATE], [EXPIRE_UNIX], [DTE], [C_DELTA], [C_GAMMA], [C_VEGA], [C_THETA], [C_RHO], "
    "[C_IV], [C_VOLUME], [C_LAST], [C_SIZE], [C_BID], [C_ASK], [STRIKE], [P_BID], [P_ASK], "
    "[P_SIZE], [P_LAST], [P_DELTA], [P_GAMMA], [P_VEGA], [P_THETA], [P_RHO], [P_IV], [P_VOLUME], "
    "[STRIKE_DISTANCE], [STRIKE_DISTANCE_PCT]"
)
SPOT, RATE, DIV = 4000.0, 0.03, 0.015
DTES = [3, 10, 18, 45, 120, 365]  # 2022-01-03 + 18 days is a third Friday
DATES = ["2022-01-03", "2022-01-04"]


def smile(k):
    return 0.18 - 0.25 * k + 0.4 * k**2


def _quote(price):
    half = np.maximum(0.05, 0.01 * price)
    bid = np.floor((price - half) / 0.05) * 0.05
    ask = np.ceil((price + half) / 0.05) * 0.05
    return np.maximum(bid, 0.0), ask


def write_month(path, rng):
    """Write a fabricated two-day chain with known forwards, rates and smile."""
    lines = [HEADER]
    for date in DATES:
        t0 = pd.Timestamp(f"{date} 16:00", tz="America/New_York")
        for dte in DTES:
            t1 = t0 + pd.Timedelta(days=dte)
            T = (t1 - t0).total_seconds() / (365 * 86400)
            F, D = SPOT * np.exp((RATE - DIV) * T), np.exp(-RATE * T)
            K = np.arange(3000.0, 5001.0, 25.0)
            sig = smile(np.log(K / F))
            c_bid, c_ask = _quote(black76.price(F, K, T, sig, True, D))
            p_bid, p_ask = _quote(black76.price(F, K, T, sig, False, D))
            for i, strike in enumerate(K):
                c_iv = "" if rng.random() < 0.1 else f"{sig[i]:.6f}"
                lines.append(
                    f"{int(t0.timestamp())}, {date} 16:00, {date}, 16.000000, {SPOT:.6f}, "
                    f"{t1.date()}, {int(t1.timestamp())}, {dte:.6f}, 0.5, 0.0, 0.0, 0.0, 0.0, "
                    f"{c_iv}, 3.000000, 0.000000, 10 x 12, {c_bid[i]:.6f}, {c_ask[i]:.6f}, "
                    f"{strike:.6f}, {p_bid[i]:.6f}, {p_ask[i]:.6f}, 7 x 9, 0.000000, -0.5, 0.0, "
                    f"0.0, 0.0, 0.0, {sig[i]:.6f}, , 0.0, 0.0"
                )
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.fixture(scope="module")
def month_file(tmp_path_factory):
    return write_month(
        tmp_path_factory.mktemp("raw") / "spx_eod_202201.txt", np.random.default_rng(0)
    )


@pytest.fixture(scope="module")
def wide(month_file):
    return audit_quotes(optionsdx.read_month(month_file))


def test_parser_types_and_derived_columns(wide):
    assert len(wide) == len(DATES) * len(DTES) * 81
    assert wide["c_bid_size"].eq(10).all() and wide["p_ask_size"].eq(9).all()
    assert wide["c_iv_vendor"].isna().any() and wide["p_volume"].isna().all()
    np.testing.assert_allclose(wide["T"] * 365, wide["dte"], atol=1e-9)
    third = wide[wide["expiry"] == pd.Timestamp("2022-01-21")]["is_third_friday"]
    assert len(third) and third.all()
    assert optionsdx.month_key("x/spx_eod_202201.txt") == "202201"


def test_forwards_recover_parity_inputs(wide):
    fwd = estimate_forwards(wide)
    assert fwd["valid"].all()
    long = fwd[fwd["dte"] >= 30]
    np.testing.assert_allclose(long["r"], RATE, atol=5e-4)
    F_true = SPOT * np.exp((RATE - DIV) * fwd["T"])
    np.testing.assert_allclose(fwd["F"], F_true, rtol=2e-4)


def test_clean_quotes_recover_smile(wide):
    quotes, funnel = build_quotes(wide, estimate_forwards(wide), FilterConfig())
    steps = list(funnel.values())
    assert steps == sorted(steps, reverse=True) and steps[-1] == len(quotes) > 0
    assert quotes["dte"].min() >= 7
    assert ((quotes["option_type"] == "C") == (quotes["strike"] >= quotes["F"])).all()
    tight = quotes[quotes["half_spread_iv"] < 0.01]
    err = tight["iv_mid"] - smile(
        np.log(tight["strike"] / (SPOT * np.exp((RATE - DIV) * tight["T"])))
    )
    assert err.abs().median() < 2e-3
    assert not quotes["arb_vertical"].any() and not quotes["arb_butterfly"].any()


def _chain(bids, asks, side="c"):
    n = len(bids)
    other = "p" if side == "c" else "c"
    return pd.DataFrame(
        {
            "date": pd.Timestamp("2022-01-03"),
            "expiry": pd.Timestamp("2022-02-18"),
            "strike": 100.0 + 5.0 * np.arange(n),
            f"{side}_bid": bids,
            f"{side}_ask": asks,
            f"{other}_bid": np.nan,
            f"{other}_ask": np.nan,
        }
    )


def test_audit_flags_call_butterfly():
    # Convex chain except the 110 body: its bid 6.2 exceeds the wing cost 0.5 * (8.6 + 3.1).
    flags = audit_quotes(_chain([11.9, 8.4, 6.2, 2.9, 0.9], [12.1, 8.6, 6.4, 3.1, 1.1]))
    assert flags["c_arb_butterfly"].tolist() == [False, True, True, True, False]
    assert not flags["c_arb_vertical"].any()


def test_audit_flags_put_vertical():
    # Put bid at 105 is above the put ask at 110: puts must not get cheaper with strike.
    flags = audit_quotes(_chain([1.0, 3.0, 2.5, 6.0], [1.2, 3.2, 2.8, 6.4], side="p"))
    assert flags["p_arb_vertical"].tolist() == [False, True, True, False]


def test_audit_passes_clean_convex_chain():
    K = 100.0 + 5.0 * np.arange(12)
    mid = black76.price(110.0, K, 0.25, 0.25, True)
    flags = audit_quotes(_chain(np.maximum(mid - 0.05, 0.0), mid + 0.05))
    assert not flags[["c_arb_vertical", "c_arb_butterfly"]].any().any()


def test_process_month_writes_partitions(month_file, tmp_path):
    cfg = SpxConfig(forwards=ForwardConfig(), filters=FilterConfig())
    row = process_month(month_file, cfg, root=tmp_path)
    assert row["month"] == "202201" and row["days"] == len(DATES)
    for table in ("quotes", "forwards", "days"):
        assert (store.table_dir("spx", table, tmp_path) / "202201.parquet").exists()
    quotes = store.load_quotes("spx", root=tmp_path)
    assert len(quotes) == row["slice_size"]
    assert store.load_table("spx", "days", root=tmp_path)["forward_valid"].eq(1.0).all()


def test_config_rejects_unknown_keys():
    with pytest.raises(KeyError):
        from_dict(SpxConfig, {"filters": {"min_dte": 7, "typo": 1}})
    cfg = from_dict(SpxConfig, {"filters": {"z_range": [-5, 3]}, "years": [2018]})
    assert cfg.filters.z_range == (-5, 3) and cfg.years == (2018,)
