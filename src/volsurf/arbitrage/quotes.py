"""Model-free static-arbitrage audit of raw option quotes.

Calls and puts are checked separately within each (date, expiry) at executable prices (buy at the
ask, sell at the bid), so no forward, discount factor or model enters. A quote is flagged when it
is a leg of a tradable violation between adjacent listed strikes K1 < K2 < K3:

* vertical: prices must be monotone in strike and a vertical spread can be worth at most the
  strike gap (D <= 1). For calls, a violation is bid(K2) > ask(K1) or bid(K1) - ask(K2) > K2 - K1;
  puts mirror this;
* butterfly: prices must be convex in strike, so buying the wings must cost at least the body,
  w ask(K1) + (1 - w) ask(K3) >= bid(K2) with w = (K3 - K2) / (K3 - K1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SLICE = ["date", "expiry"]
EPS = 1e-6


def _side_flags(wide: pd.DataFrame, bid_col: str, ask_col: str, is_call: bool):
    bid, ask = wide[bid_col], wide[ask_col]
    ok = np.isfinite(bid) & np.isfinite(ask) & (ask > 0) & (bid >= 0) & (ask >= bid)
    v = wide.loc[ok, SLICE + ["strike"]].assign(bid=bid[ok], ask=ask[ok])
    g = v.groupby(SLICE, sort=False)
    k_next, bid_next, ask_next = (g[c].shift(-1) for c in ("strike", "bid", "ask"))
    k_prev, ask_prev = g["strike"].shift(1), g["ask"].shift(1)

    gap = k_next - v["strike"]
    if is_call:
        pair = (bid_next > v["ask"] + EPS) | (v["bid"] - ask_next > gap + EPS)
    else:
        pair = (v["bid"] > ask_next + EPS) | (bid_next - v["ask"] > gap + EPS)
    pair = pair.fillna(False)
    vertical = pair | pair.groupby([v["date"], v["expiry"]]).shift(1, fill_value=False)

    w = (k_next - v["strike"]) / (k_next - k_prev)
    center = (w * ask_prev + (1.0 - w) * ask_next < v["bid"] - EPS).fillna(False)
    gc = center.groupby([v["date"], v["expiry"]])
    butterfly = center | gc.shift(1, fill_value=False) | gc.shift(-1, fill_value=False)

    out = pd.DataFrame(False, index=wide.index, columns=["vertical", "butterfly"])
    out.loc[v.index, "vertical"] = vertical.astype(bool)
    out.loc[v.index, "butterfly"] = butterfly.astype(bool)
    return out["vertical"], out["butterfly"]


def audit_quotes(wide: pd.DataFrame) -> pd.DataFrame:
    """Add ``{c,p}_arb_vertical`` and ``{c,p}_arb_butterfly`` flags to a wide chain.

    ``wide`` must be sorted by (date, expiry, strike).
    """
    out = wide.copy()
    out["c_arb_vertical"], out["c_arb_butterfly"] = _side_flags(wide, "c_bid", "c_ask", True)
    out["p_arb_vertical"], out["p_arb_butterfly"] = _side_flags(wide, "p_bid", "p_ask", False)
    return out
