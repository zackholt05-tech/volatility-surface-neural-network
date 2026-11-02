"""Surface explorer: the day's fitted implied-volatility surface, its fit to the quotes by model,
and one expiry's smile against the bid-ask."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from common import MODEL_HELP, current, get_model

from volsurf.eval.metrics import domain
from volsurf.viz.panels import DETECTORS, LIVE_MODELS, slice_bounds
from volsurf.viz.surfaces import smile_figure, surface_figure, surface_grid


def _fit_table(day: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Each model column against the quotes it was fitted on (every quote but arbitrage legs)."""
    q = day[~day["arb_leg"].to_numpy()]
    rows = []
    for c in columns:
        err = q[c] - q["iv_mid"]
        row = {
            "model": DETECTORS.get(c, c),
            "RMSE (vol pts)": 100.0 * np.sqrt(np.mean(err**2)),
            "inside bid-ask": np.mean((q[c] >= q["iv_bid"]) & (q[c] <= q["iv_ask"])),
        }
        if "iv_true" in q:
            row["RMSE to true IV (vol pts)"] = 100.0 * np.sqrt(np.mean((q[c] - q["iv_true"]) ** 2))
        rows.append(row)
    return pd.DataFrame(rows)


def render() -> None:
    sel = current()
    day, flags = sel.day, sel.flags
    st.header(f"{sel.name}, {sel.date.date()}")

    c = st.columns(5)
    c[0].metric("quotes", f"{len(day):,}")
    c[1].metric("expiries", day["expiry"].nunique())
    c[2].metric("days to expiry", f"{day['dte'].min()}–{day['dte'].max()}")
    c[3].metric("median half-spread", f"{100 * day['s'].median():.2f} vp")
    c[4].metric(f"flags at margin {sel.margin:g}", int((flags != 0).sum()))

    kind = st.segmented_control(
        "Surface", list(LIVE_MODELS), default="mix4", format_func=LIVE_MODELS.get,
        key="surface_model", help=MODEL_HELP,
    ) or "mix4"  # fmt: skip
    try:
        model = get_model(sel.name, sel.i, kind)
    except (KeyError, FileNotFoundError) as err:
        st.warning(str(err))
        return
    K, T, iv = surface_grid(model, slice_bounds(day))
    st.plotly_chart(surface_figure(K, T, iv, day, flags), width="stretch")
    st.caption(
        "Each maturity is drawn over its quoted strikes only. Red and green points are the "
        f"quotes flagged rich and cheap by the {DETECTORS[sel.detector]} detector."
    )

    st.subheader("Fit to the quotes")
    st.dataframe(
        _fit_table(day, sel.panel.detectors), hide_index=True, width="stretch",
        column_config={
            "RMSE (vol pts)": st.column_config.NumberColumn(format="%.3f"),
            "RMSE to true IV (vol pts)": st.column_config.NumberColumn(format="%.3f"),
            "inside bid-ask": st.column_config.NumberColumn(format="percent"),
        },
    )  # fmt: skip
    st.caption(
        "Every quote of the day but arbitrage legs, scored with the IVs of the mispricing runs. "
        "The README's fit errors are lower because the evaluation harness holds the outer strikes "
        "out of the fit and scores them separately, as wings."
    )

    st.subheader("Smile")
    expiries = day.groupby("expiry")["dte"].first()
    labels = [f"{e.date()} ({d} days)" for e, d in expiries.items()]
    near_month = int(np.argmin(np.abs(expiries.to_numpy() - 30)))
    pick = st.selectbox(
        "Expiry", range(len(labels)), index=near_month, format_func=labels.__getitem__,
        key="smile_expiry",
    )  # fmt: skip
    on = (day["expiry"] == expiries.index[pick]).to_numpy()
    q = day[on]
    curves = {c: q[c].to_numpy() for c in sel.panel.detectors}
    if "iv_true" in q:
        curves["iv_true"] = q["iv_true"].to_numpy()
    Tq = float(q["T"].iloc[0])
    lo, hi = domain(slice_bounds(day), np.array([Tq]))
    k = np.linspace(lo[0], hi[0], 200)
    dense = (k, model.iv(k, np.full_like(k, Tq)), LIVE_MODELS[kind])
    st.plotly_chart(smile_figure(q, curves, flags[on], dense), width="stretch")


render()
