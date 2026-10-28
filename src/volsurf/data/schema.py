"""Column contract shared by the real-data pipeline and the synthetic generator.

Every quotes table has one row per out-of-the-money option per quote date. Models and evaluation
code should rely only on ``CORE_COLUMNS``; sources may append extra columns.
"""

CORE_COLUMNS = [
    "date",  # quote date (16:00 ET snapshot)
    "expiry",
    "dte",  # calendar days to expiry
    "T",  # year fraction to expiry, ACT/365
    "strike",
    "option_type",  # "C" if strike >= F else "P"
    "spot",  # vendor index print, not synchronized with the option snapshot: use F instead
    "F",  # forward to expiry from put-call parity; the reference for moneyness and hedging
    "D",  # discount factor to expiry
    "k",  # log forward-moneyness log(K / F)
    "bid",
    "ask",
    "mid",
    "iv_bid",
    "iv_ask",
    "iv_mid",
    "half_spread_iv",  # observed 0.5 * (iv_ask - iv_bid)
]
