"""Ranked mispricing flags of the day, what became of their residuals, and the panel's flags."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from common import current, get_flag_counts

from volsurf.eval.mispricing import residuals
from volsurf.viz.panels import DETECTORS, follow
from volsurf.viz.surfaces import flags_figure, path_figure, smile_figure

HORIZONS = (1, 2, 3, 4, 5)


def _ranked(sel, flags: np.ndarray, later: pd.DataFrame) -> pd.DataFrame:
    day = sel.day
    res = residuals(day, sel.detector)
    table = pd.DataFrame(
        {
            "expiry": day["expiry"].dt.date, "days": day["dte"], "strike": day["strike"],
            "type": day["option_type"], "bid": day["bid"], "ask": day["ask"],
            "model price": res["model_price"], "side": np.where(flags > 0, "rich", "cheap"),
            "residual (half-spreads)": sel.r, "residual (vol pts)": res["resid_vp"],
            "after 1 day": later["r_1d"], "after 3 days": later["r_3d"],
            "after 5 days": later["r_5d"], "arbitrage leg": day["arb_leg"],
        },
        index=day.index,
    )  # fmt: skip
    if "label" in day:
        table["injected"] = np.where(day["label"] != 0, day["injection"], "")
        table["injected shift (IV half-spreads)"] = day["shift"]
    table = table[flags != 0]
    return table.iloc[np.argsort(-np.abs(table["residual (half-spreads)"].to_numpy()))]


def render() -> None:
    sel = current()
    panel = sel.panel
    st.header(f"Mispricing flags, {sel.name}, {sel.date.date()}")
    counts = get_flag_counts(sel.name, sel.detector, sel.margin)
    st.plotly_chart(flags_figure(panel.dates, counts, sel.date, panel.spec.test), width="stretch")

    flags = sel.flags
    later = follow(panel, sel.r_all, sel.i, HORIZONS)
    table = _ranked(sel, flags, later)
    c = st.columns(4)
    c[0].metric("flagged quotes", len(table))
    c[1].metric("rich / cheap", f"{int((flags > 0).sum())} / {int((flags < 0).sum())}")
    c[2].metric("share of the day's quotes", f"{100 * np.mean(flags != 0):.2f}%")
    if "injected" in table:
        hit = (table["injected"] != "").mean() if len(table) else np.nan
        c[3].metric("flags on injected mispricings", f"{100 * hit:.0f}%" if len(table) else "–")
    else:
        back = later.loc[table.index, "r_1d"].abs() < np.abs(table["residual (half-spreads)"])
        c[3].metric("smaller the next day", f"{100 * back.mean():.0f}%" if len(table) else "–")

    if table.empty:
        st.info(
            f"No quote is more than {sel.margin:g} half-spreads outside the bid-ask under the "
            f"{DETECTORS[sel.detector]} surface on this day. Lower the margin or pick a day with "
            "flags on the chart above."
        )
        return
    st.caption(
        "Flags, not trades: the backtest trades only flags with at least 7 days to expiry, inside "
        "the listing's moneyness band, quoted until their latest exit and not legs of a quote "
        "arbitrage. Residuals are market minus model in price half-spreads (positive: rich); "
        "a residual inside ±1 is inside the bid-ask. Select a row to follow it."
    )
    event = st.dataframe(
        table, hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row",
        key=f"flags:{sel.name}:{sel.i}:{sel.detector}:{sel.margin}",
        column_config={
            c: st.column_config.NumberColumn(format="%.2f")
            for c in ("bid", "ask", "model price", "residual (half-spreads)", "residual (vol pts)",
                      "after 1 day", "after 3 days", "after 5 days")
        },
    )  # fmt: skip
    rows = event.selection.rows if event is not None else []
    pick = table.index[rows[0]] if rows else table.index[0]

    row = sel.day.loc[pick]
    on = (sel.day["expiry"] == row["expiry"]).to_numpy()
    q = sel.day[on]
    curves = {sel.detector: q[sel.detector].to_numpy()}
    if "iv_true" in q:
        curves["iv_true"] = q["iv_true"].to_numpy()
    where = int(np.flatnonzero(q.index == pick)[0])
    left, right = st.columns(2)
    left.markdown(
        f"**{row['option_type']} {row['strike']:.0f}, expiry {row['expiry'].date()}**: its smile"
    )
    left.plotly_chart(smile_figure(q, curves, flags[on], highlight=where), width="stretch")
    path = np.r_[sel.r[sel.day.index.get_loc(pick)], later.loc[pick, [f"r_{h}d" for h in HORIZONS]]]
    right.markdown("**Its residual over the next five trading days**")
    right.plotly_chart(path_figure([0, *HORIZONS], path, sel.margin), width="stretch")


render()
